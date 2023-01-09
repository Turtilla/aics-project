# This code was mostly taken from Nikolai Ilinykh's code, but adapted at times to suit our needs or to eliminate errors. Changes are marked in the code,
# the original documentation is preserved.

import time

import torch.optim
import torch.utils.data
from dataset import custom_collate
from nltk.translate.bleu_score import corpus_bleu
from preproc import AverageMeter, accuracy, adjust_learning_rate, clip_gradient
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

# Training parameters
training_parameters = {
    'epochs': 120,  # number of epochs to train for (if early stopping is not triggered)
    'batch_size': 32,
    'batch_size_val': 5,
    # workers need to be 0, otherwise 'OS Error: Too many open files' during training
    'workers': 0,  # for data-loading; right now, only 1 works with h5py
    'encoder_lr': 1e-4,  # learning rate for encoder if fine-tuning
    'decoder_lr': 4e-4,  # learning rate for decoder
    'grad_clip': 5.,  # clip gradients at an absolute value of
    'alpha_c': 1.,  # regularization parameter for 'doubly stochastic attention', as in the paper
    'print_freq': 100,  # print training/validation stats every __ batches
    'fine_tune_encoder': False,  # fine-tune encoder?
}

class CaptionTrainer:
    '''Most of the code by Nikolai Ilinykh, adapted into a class by Dominik.
    This class allows for defining how a model should be trained and then providing methods for carrying that training out.

    Attributes:
        encoder (nn.Module): The encoder model.
        decoder (nn.Module): The decoder model.
        word_map (dict): The word to index mapping.
        device (str): The kind of device the calculations are to be performed on.
        training_parameters (dict): The (hyper)parameters used for training.
    '''
    def __init__(self, decoder, encoder, word_map, device, training_parameters) -> None:
        self.encoder = encoder.to(device)
        self.encoder.train() # train mode (dropout and batchnorm is used)

        self.decoder = decoder.to(device)
        self.decoder.train() # train mode (dropout and batchnorm is used)

        self.word_map = word_map
        self.device = device
        self.training_parameters = training_parameters

    # everything below is from Nikolai Ilinykh's code.
    def train(self, train_loader, criterion, encoder_optimizer, decoder_optimizer, epoch):
        """
        Performs one epoch's training.
        :param train_loader: DataLoader for training data
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
        :param decoder_optimizer: optimizer to update decoder's weights
        :param epoch: epoch number
        """
        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss (per word decoded)
        top5accs = AverageMeter()  # top5 accuracy

        start = time.time()

        # Batches
        for i, content in enumerate(train_loader):  # changing to fit our dataloader
            images = torch.stack(content['images'])
            caps = content['encoded_captions']
            caplens = torch.stack(content['caption_lengths'])
            data_time.update(time.time() - start)

            # Move to GPU, if available
            images = images.to(self.device)
            caps = caps.to(self.device)
            caplens = caplens.to(self.device)

            # Forward prop.
            images = self.encoder(images)
            scores, caps_sorted, decode_lengths, alphas, _ = self.decoder(images, caps, caplens)

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            # useful tutorial: https://gist.github.com/HarshTrivedi/f4e7293e941b17d19058f6fb90ab0fec
            scores, *_ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, *_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

            # Calculate loss
            loss = criterion(scores, targets)

            # Add doubly stochastic attention regularization
            loss += self.training_parameters['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Back prop.
            decoder_optimizer.zero_grad()
            if encoder_optimizer is not None:
                encoder_optimizer.zero_grad()
            loss.backward()

            # Clip gradients
            if self.training_parameters['grad_clip'] is not None:
                clip_gradient(decoder_optimizer, self.training_parameters['grad_clip'])
                if encoder_optimizer is not None:
                    clip_gradient(encoder_optimizer, self.training_parameters['grad_clip'])

            # Update weights
            decoder_optimizer.step()
            if encoder_optimizer is not None:
                encoder_optimizer.step()

            # Keep track of metrics
            top5 = accuracy(scores, targets, 5)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i % self.training_parameters['print_freq'] == 0:
                print(f'Epoch: [{epoch}][{i}/{len(train_loader)}]\t'
                    f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    f'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                    f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})')


    def validate(self, val_loader, criterion):
        """
        Performs one epoch's validation.
        :param val_loader: DataLoader for validation data.
        :param encoder: encoder model
        :param decoder: decoder model
        :param criterion: loss layer
        :return: BLEU-4 score
        """
        self.decoder.eval()  # eval mode (no dropout or batchnorm)
        if self.encoder is not None:
            self.encoder.eval()

        batch_time = AverageMeter()
        losses = AverageMeter()
        top5accs = AverageMeter()

        start = time.time()

        references = list()  # references (true captions) for calculating BLEU-4 score
        hypotheses = list()  # hypotheses (predictions)

        # explicitly disable gradient calculation to avoid CUDA memory error
        # solves the issue #57
        with torch.no_grad():
            # Batches
            for i, content in enumerate(val_loader):  # changing to fit our dataloader
                images = torch.stack(content['images'])
                captions = content['encoded_captions']
                caplens = torch.stack(content['caption_lengths'])

                # Move to device, if available
                images = images.to(self.device)
                captions = captions.to(self.device)
                caplens = caplens.to(self.device)

                # Forward prop.
                if self.encoder is not None:
                    images = self.encoder(images)
                scores, captions_sorted, decode_lengths, alphas, sort_ind = self.decoder(images, captions, caplens)

                # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
                targets = captions_sorted[:, 1:]

                # Remove timesteps that we didn't decode at, or are pads
                # pack_padded_sequence is an easy trick to do this
                scores_copy = scores.clone()
                scores, *_ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
                targets, *_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

                # Calculate loss
                loss = criterion(scores, targets)

                # Add doubly stochastic attention regularization
                loss += self.training_parameters['alpha_c'] * ((1. - alphas.sum(dim=1)) ** 2).mean()

                # Keep track of metrics
                losses.update(loss.item(), sum(decode_lengths))
                top5 = accuracy(scores, targets, 5)
                top5accs.update(top5, sum(decode_lengths))
                batch_time.update(time.time() - start)

                start = time.time()

                if i % self.training_parameters['print_freq'] == 0:
                    print(f'Validation: [{i}/{len(val_loader)}]\t'
                        f'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Top-5 Accuracy {top5accs.val:.3f} ({top5accs.avg:.3f})\t')

                # Store references (true captions), and hypothesis (prediction) for each image
                # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
                # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]
                # In the CLEF dataset, we only have one caption per image hence the variables look like this:
                # references = [[ref1], [ref2], ...], hypotheses = [hyp1, hyp2, ...]

                # References           
                references.extend([
                    [[token for token in caption.tolist() if token not in {self.word_map['<start>'], self.word_map['<pad>']}]]
                    for caption in captions[sort_ind]
                ])

                # Hypotheses
                _, preds = torch.max(scores_copy, dim=2)
                preds = preds.tolist()
                temp_preds = []
                for j, p in enumerate(preds):
                    temp_preds.append(p[:decode_lengths[j]])  # remove pads
                preds = temp_preds
                hypotheses.extend(preds)

                assert len(references) == len(hypotheses)
            # print(references)
            # print(hypotheses)
            # Calculate BLEU-4 scores
            bleu4 = corpus_bleu(references, hypotheses)

            print(f'\n * LOSS - {losses.avg:.3f}, TOP-5 ACCURACY - {top5accs.avg:.3f}, BLEU-4 - {bleu4}\n')

        return bleu4

def train(checkpoint_name: str, train_set, val_set, word_map: dict, saved_root_dir: str, model_path: str, device):
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

        # Save checkpoint
        # This needed to be moved here in this fashion by Maria due to the original way of saving the model not allowing for a directory to be included.
        state = {'epoch': epoch,
                 'epochs_since_improvement': epochs_since_improvement,
                 'bleu-4': recent_bleu4,
                 'encoder': encoder,
                 'decoder': decoder,
                 'encoder_optimizer': encoder_optimizer,
                 'decoder_optimizer': decoder_optimizer}
        filename = f'checkpoint_{checkpoint_name}.pth.tar'
        torch.save(state,  saved_root_dir + filename)
        # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
        if is_best:
            torch.save(state, saved_root_dir + 'BEST_' + filename)
