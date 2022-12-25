import argparse
import json
import os

import torch
from dataset import (CLEFCaptionLoader, Datasets, FlickrCaptionLoader,
                     ImageDataset, custom_collate)
from filters import POSRelationFilter, RuleBasedRelationFilter
from preproc import adjust_learning_rate
from torch import nn
from torch.backends import cudnn
from train import CaptionTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

hyperparameters = {
    'beam_size': 3,
    'smooth': False,
}

# Training parameters
training_parameters = {
    'epochs': 120,  # number of epochs to train for (if early stopping is not triggered)
    'batch_size': 32,
    'batch_size_val': 5,
    'workers': 1,  # for data-loading; right now, only 1 works with h5py
    'encoder_lr': 1e-4,  # learning rate for encoder if fine-tuning
    'decoder_lr': 4e-4,  # learning rate for decoder
    'grad_clip': 5.,  # clip gradients at an absolute value of
    'alpha_c': 1.,  # regularization parameter for 'doubly stochastic attention', as in the paper
    'print_freq': 100,  # print training/validation stats every __ batches
    'fine_tune_encoder': False,  # fine-tune encoder?
}

def load_wordmap(word_map_path):
    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    return word_map

def load_data(dataset: str, data_root_dir: str, word_map: dict, number_of_images: int, unknown_filter: float):
    if dataset == Datasets.CLEF:
        clef_root_dir = os.path.join(data_root_dir, 'iaprtc12/')
        clef_image_dir = os.path.join(clef_root_dir, 'images/')
        clef_annotation_dir = os.path.join(clef_root_dir, 'annotations_complete_eng/')
        return ImageDataset(image_directory=clef_image_dir,
                            relation_filter=POSRelationFilter(),  # swap for the one below for a better filter
                            #relation_filter=RuleBasedRelationFilter(),
                            caption_loader=CLEFCaptionLoader(clef_annotation_dir),
                            number_images=number_of_images,
                            word_map=word_map,
                            min_frequency=1,
                            concat_captions=False,
                            unk_filter=unknown_filter)
    elif dataset == Datasets.FLICKR:
        flickr_root_dir = os.path.join(data_root_dir, 'flickr_8k/')
        flickr_image_dir = os.path.join(flickr_root_dir, 'Images/')
        return ImageDataset(image_directory=flickr_image_dir,
                            relation_filter=POSRelationFilter(),  # swap for the one below for a better filter
                            #relation_filter=RuleBasedRelationFilter(),
                            caption_loader=FlickrCaptionLoader(flickr_root_dir),
                            number_images=number_of_images, 
                            word_map=word_map,
                            min_frequency=1,
                            unk_filter=unknown_filter)
    else:
        raise ValueError('dataset must be one of the following: clef, flickr')

def split_dataset(dataset):
    # splitting the dataset by Maria
    # remove the last optional argument for random splits, this way the seed is fixed so results are reproducible
    # QUESTION: does this need to be done any prettier?

    # TODO relative values not working
    len_train = int(0.8 * len(dataset))
    len_val = int(0.1 * len(dataset))
    len_test = len(dataset) - len_train - len_val
    print(f'{"Dataset lengths":-^30}')
    print(f'train: {len_train}, val: {len_val}, test: {len_test}')
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(25))
    return train_set, val_set, test_set

def train(dataset: Datasets, train_set, val_set, word_map: dict, saved_root_dir: str, model_path: str):
    """
    Training and validation.
    """
    checkpoint = torch.load(model_path, map_location=torch.device(device))
    start_epoch = checkpoint['epoch'] + 1
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    best_bleu4 = checkpoint['bleu-4']
    decoder = checkpoint['decoder']
    decoder_optimizer = checkpoint['decoder_optimizer']
    encoder = checkpoint['encoder']
    encoder_optimizer = checkpoint['encoder_optimizer']
    if training_parameters['fine_tune_encoder'] is True and encoder_optimizer is None:
        encoder.fine_tune(training_parameters['fine_tune_encoder'])
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=training_parameters['encoder_lr'])

    trainer = CaptionTrainer(decoder, encoder, word_map, device, training_parameters)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_set,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=True,
        batch_size=training_parameters['batch_size'],
        num_workers=training_parameters['workers'],
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        shuffle=True,
        collate_fn=custom_collate,
        drop_last=True,
        batch_size=training_parameters['batch_size_val'],
        num_workers=training_parameters['workers'],
        pin_memory=True)

    # Epochs
    for epoch in range(start_epoch, training_parameters['epochs']):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if training_parameters['fine_tune_encoder']:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        trainer.train(train_loader=train_loader,
                      criterion=criterion,
                      encoder_optimizer=encoder_optimizer,
                      decoder_optimizer=decoder_optimizer,
                      epoch=epoch)

        # One epoch's validation
        recent_bleu4 = trainer.validate(val_loader=val_loader,
                                        criterion=criterion)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print(f'\nEpochs since last improvement: {epochs_since_improvement}\n')
        else:
            epochs_since_improvement = 0

        # Save checkpoint - needed since the original function did not allow for a path
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'bleu-4': recent_bleu4,
                 'encoder': encoder,
                 'decoder': decoder,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_optimizer': decoder_optimizer}
        filename = f'checkpoint_{dataset}.pth.tar'
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, saved_root_dir + 'BEST_' + filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--on_server', type=bool, default=False)
    parser.add_argument('--dataset', type=Datasets, default=Datasets.CLEF)
    parser.add_argument('--number_of_images', type=int, default=100)
    parser.add_argument('--unknown_filter', type=float, default=0.2)


    arguments = parser.parse_args()

    if arguments.on_server:
        data_root_dir = '/srv/data/guskunkdo/'
        saved_root_dir = os.path.join(data_root_dir, 'saved/')
        model_path = '/srv/data/aics/03-image-captioning/data/BEST_checkpoint_flickr8k_5_10.pth.tar'  # model path updated
        word_map_path = '/srv/data/aics/03-image-captioning/data/out/wordmap_flickr8k_5_10.json'  # wordmap path updated
    else:
        data_root_dir = '../data/'
        saved_root_dir = '../saved/'
        model_path = '../data/BEST_checkpoint_flickr8k_5_10.pth.tar'  # model path updated
        word_map_path = '../data/wordmap_flickr8k_5_10.json'  # wordmap path updated

    # Read word map
    word_map = load_wordmap(word_map_path)
    dataset = load_data(arguments.dataset, data_root_dir, word_map, arguments.number_of_images, arguments.unknown_filter)
    train_set, val_set, test_set = split_dataset(dataset)

    train(dataset=arguments.dataset,
          train_set=train_set,
          val_set=val_set,
          word_map=word_map,
          saved_root_dir=saved_root_dir,
          model_path=model_path)
