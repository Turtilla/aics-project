import time

import torch.optim
import torch.utils.data
from nltk.translate.bleu_score import corpus_bleu
from preproc import AverageMeter, accuracy, clip_gradient
from torch.nn.utils.rnn import pack_padded_sequence


class CaptionTrainer:
    def __init__(self, decoder, encoder, word_map, device, training_parameters) -> None:
        self.encoder = encoder.to(device)
        self.encoder.train() # train mode (dropout and batchnorm is used)

        self.decoder = decoder.to(device)
        self.decoder.train() # train mode (dropout and batchnorm is used)

        self.word_map = word_map
        self.device = device
        self.training_parameters = training_parameters

    # this is all from Nikolai
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
        for i, content in enumerate(train_loader):  # changing to fit our dataloader, TODO same in validation
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
                # In the CLEF dataset, we only have one caption per image hence the variables looks like this:
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
