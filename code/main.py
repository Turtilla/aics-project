import json
import os
from cgi import test
from collections import Counter
from dataclasses import dataclass
from glob import glob
from pprint import pprint
from test import CaptionTester
from xml.etree import ElementTree
from xml.etree.ElementTree import ParseError

import matplotlib.pyplot as plt
import nltk
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from dataset import (END_TOKEN, PADDING_TOKEN, START_TOKEN, UNKNOWN_TOKEN,
                     CLEFCaptionLoader, FlickrCaptionLoader, ImageDataset,
                     Sample)
from filters import POSRelationFilter, RelationFilter, RuleBasedRelationFilter
from PIL import Image
from preproc import adjust_learning_rate, save_checkpoint
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from train import CaptionTrainer

on_server = False

if on_server:
    data_root_dir = '/srv/data/guskunkdo/'
    saved_root_dir = os.path.join(data_root_dir, 'saved/')
    model_path = '/srv/data/aics/03-image-captioning/data/BEST_checkpoint_flickr8k_5_10.pth.tar'  # model path updated
    word_map_path = '/srv/data/aics/03-image-captioning/data/out/wordmap_flickr8k_5_10.json'  # wordmap path updated
else:
    data_root_dir = '../data/'
    saved_root_dir = '../saved/'
    model_path = '../data/BEST_checkpoint_flickr8k_5_10.pth.tar'  # model path updated
    word_map_path = '../data/wordmap_flickr8k_5_10.json'  # wordmap path updated

clef_root_dir = os.path.join(data_root_dir, 'iaprtc12/')
clef_image_dir = os.path.join(clef_root_dir, 'images/')
clef_annotation_dir = os.path.join(clef_root_dir, 'annotations_complete_eng/')

flickr_root_dir = os.path.join(data_root_dir, 'flickr_8k/')
flickr_image_dir = os.path.join(flickr_root_dir, 'Images/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

hyperparameters = {
    'number_of_images': 100,
    
    'beam_size': 3,
    'smooth': False,
    
    'unk_filter': 0.2
}

# Training parameters
training_parameters = {
    'epochs': 120,  # number of epochs to train for (if early stopping is not triggered)
    'batch_size': 32,
    'workers': 1,  # for data-loading; right now, only 1 works with h5py
    'encoder_lr': 1e-4,  # learning rate for encoder if fine-tuning
    'decoder_lr': 4e-4,  # learning rate for decoder
    'grad_clip': 5.,  # clip gradients at an absolute value of
    'alpha_c': 1.,  # regularization parameter for 'doubly stochastic attention', as in the paper
    'print_freq': 100,  # print training/validation stats every __ batches
    'fine_tune_encoder': False,  # fine-tune encoder?
}

def custom_collate(samples: list[Sample]) -> dict:
    # by Dominik
    image_ids = []
    captions = []
    image_paths = []
    tokenized_captions = []
    caption_lengths = []
    encoded_captions = []
    images = []

    for sample in samples:
        image_ids.append(sample.image_id)
        captions.append(sample.caption)
        image_paths.append(sample.image_path)
        tokenized_captions.append(sample.tokenized_caption)
        caption_lengths.append(sample.caption_length)
        encoded_captions.append(sample.encoded_caption)
        images.append(sample.image)
    
    return {
        'image_ids': image_ids,
        'captions': captions,
        'caption_lengths': caption_lengths,
        'tokenized_captions': tokenized_captions,
        'encoded_captions': pad_sequence(encoded_captions, batch_first=True),
        'image_paths': image_paths,
        'images': images
    }


def load_wordmap(word_map_path):
    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    return word_map

def load_data(dataset: str, word_map):
    if dataset == 'clef':
        return ImageDataset(image_directory=clef_image_dir,
                            relation_filter=POSRelationFilter(),  # swap for the one below for a better filter
                            #relation_filter=RuleBasedRelationFilter(),
                            caption_loader=CLEFCaptionLoader(clef_annotation_dir),
                            number_images=hyperparameters['number_of_images'], 
                            word_map=word_map,
                            min_frequency=1,
                            concat_captions=False,
                            unk_filter=hyperparameters['unk_filter'])
    elif dataset == 'flickr':
        return ImageDataset(image_directory=flickr_image_dir,
                            relation_filter=POSRelationFilter(),  # swap for the one below for a better filter
                            #relation_filter=RuleBasedRelationFilter(),
                            caption_loader=FlickrCaptionLoader(flickr_root_dir),
                            number_images=hyperparameters['number_of_images'], 
                            word_map=word_map,
                            min_frequency=1,
                            unk_filter=hyperparameters['unk_filter'])
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

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(25))
    return train_set, val_set, test_set

def main(checkpoint_path, model_name):
    """
    Training and validation.
    """

    # Read word map
    word_map = load_wordmap(word_map_path)
    clef_dataset = load_data('clef', word_map)
    train_set, val_set, test_set = split_dataset(clef_dataset)

    #removing initializing model

    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        train_set,  
        shuffle=True, 
        collate_fn=custom_collate, 
        drop_last=True,
        batch_size=training_parameters['batch_size'],
        num_workers=training_parameters['workers'],
        pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        train_set,  # TODO change to val_set later!!!  
        shuffle=True, 
        collate_fn=custom_collate, 
        drop_last=True,
        batch_size=training_parameters['batch_size'],
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
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
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
        filename = 'checkpoint_' + model_name + '.pth.tar'
        torch.save(state, filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, saved_root_dir + 'BEST_' + filename)

main(model_path, 'imageCLEF')