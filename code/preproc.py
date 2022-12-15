import os
import numpy as np
import h5py
import json
import torch

from PIL import Image

from tqdm import tqdm
from collections import Counter
from random import seed, choice, sample

import argparse

def create_input_files(args):
    
    '''
    Creates input files for training, validation, and test data.
    '''

    assert args.dataset in {'coco', 'flickr8k', 'flickr30k'}

    # Read Karpathy JSON
    with open(args.karpathy_json_path, 'r') as j:
        data = json.load(j)

    # Read image paths and captions for each image
    train_image_paths = []
    train_image_captions = []
    val_image_paths = []
    val_image_captions = []
    test_image_paths = []
    test_image_captions = []
    word_freq = Counter()

    for img in data['images']:
        captions = []
        for c in img['sentences']:
            # Update word frequency
            word_freq.update(c['tokens'])
            if len(c['tokens']) <= args.max_len:
                captions.append(c['tokens'])
            else:
                captions.append(c['tokens'][:args.max_len])
                
        # if there were no captions for this image       
        if len(captions) == 0:
            continue

        path = os.path.join(args.image_folder, img['filepath'], img['filename']) if args.dataset == 'coco' else os.path.join(
            args.image_folder, img['filename'])

        # create the splits
        # the splits are basically a combination of path-caption pairs
        if img['split'] in {'train', 'restval'}:
            train_image_paths.append(path)
            train_image_captions.append(captions)
        elif img['split'] in {'val'}:
            val_image_paths.append(path)
            val_image_captions.append(captions)
        elif img['split'] in {'test'}:
            test_image_paths.append(path)
            test_image_captions.append(captions)

    # Sanity check
    assert len(train_image_paths) == len(train_image_captions)
    assert len(val_image_paths) == len(val_image_captions)
    assert len(test_image_paths) == len(test_image_captions)


    # Create word map
    # It is always useful to keep the map somewhere saved,
    # because you might need this vocabulary later for re-training and testing
    words = [w for w in word_freq.keys() if word_freq[w] > args.min_word_freq]
    word_map = {k: v + 1 for v, k in enumerate(words)}
    word_map['<unk>'] = len(word_map) + 1
    word_map['<start>'] = len(word_map) + 1
    word_map['<end>'] = len(word_map) + 1
    word_map['<pad>'] = 0
    

    # Create a base/root name for your output files
    # Useful because if you are planning to run the scripts multiple times, you would need to make sure you don't mix data from different runs
    base_filename = args.dataset + '_' + str(args.captions_per_image)  + '_' + str(args.min_word_freq)

    # Save word map to a JSON
    with open(os.path.join(args.output_folder, 'wordmap_' + base_filename + '.json'), 'w') as j:
        json.dump(word_map, j)
        
    
    # Sample captions for each image, save images to HDF5 file, and captions and their lengths to JSON files
    # HDF5 is the format that is typically used to store large and sparse data in language-and-vision tasks
    # one important detail: if you re-create the hdf5 file, make sure you delete the older one
    seed(123)
    for impaths, imcaps, split in [(train_image_paths, train_image_captions, 'TRAIN'),
                                   (val_image_paths, val_image_captions, 'VAL'),
                                   (test_image_paths, test_image_captions, 'TEST')]:

        with h5py.File(os.path.join(args.output_folder, split + '_' + base_filename + '.hdf5'), 'a') as h:
            # Make a note of the number of captions we are sampling per image
            h.attrs['captions_per_image'] = args.captions_per_image

            # Create dataset inside HDF5 file to store images
            # num images x num of channels (red, blue, green) x image width x image height
            images = h.create_dataset('images', (len(impaths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            enc_captions = []
            caplens = []

            for i, path in enumerate(tqdm(impaths)):

                # Sample captions
                if len(imcaps[i]) < args.captions_per_image:
                    # we need to have N captions for each image,
                    # if there are less than N captions, one simple heuristics is to randomly sample one of the existing captions,
                    # what is the possible disadvantage of this approach?
                    captions = imcaps[i] + [choice(imcaps[i]) for _ in range(args.captions_per_image - len(imcaps[i]))]
                else:
                    # just take captions as they are
                    captions = imcaps[i]
                    #captions = sample(imcaps[i], k=args.captions_per_image)

                # Sanity check
                assert len(captions) == args.captions_per_image

                # Read images
                img = Image.open(impaths[i])
                img = img.resize((256, 256))
                img = np.transpose(img, (2, 0, 1))
                                
                # the values of elements in image representation should not be higher than 255
                assert np.max(img) <= 255

                # Save image to HDF5 file
                images[i] = img
                
                # encoding the captions
                for j, c in enumerate(captions):
                    # Encode captions
                    # first, it is the start token,
                    # then, it is the caption tokens if they are in the vocabulary and <unk> token if they are not in the vocabulary,
                    # lastly, it is the end token
                    # if the caption length is below some threshold, we pad it with <pad> token,
                    enc_c = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in c] + [
                        word_map['<end>']] + [word_map['<pad>']] * (args.max_len - len(c))

                    # Find caption lengths
                    # because of start/end tokens, it is +2
                    c_len = len(c) + 2

                    enc_captions.append(enc_c)
                    caplens.append(c_len)

            # Sanity check
            assert images.shape[0] * args.captions_per_image == len(enc_captions) == len(caplens)

            # Save encoded captions and their lengths to JSON files
            with open(os.path.join(args.output_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
                json.dump(enc_captions, j)
            with open(os.path.join(args.output_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
                json.dump(caplens, j)


def init_embedding(embeddings):
    """
    Fills embedding tensor with values from the uniform distribution.

    :param embeddings: embedding tensor
    """
    bias = np.sqrt(3.0 / embeddings.size(1))
    torch.nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(emb_file, word_map):
    """
    Creates an embedding tensor for the specified word map, for loading into the model.

    :param emb_file: file containing embeddings (stored in GloVe format)
    :param word_map: word map
    :return: embeddings in the same order as the words in the word map, dimension of embeddings
    """

    # Find embedding dimension
    with open(emb_file, 'r') as f:
        emb_dim = len(f.readline().split(' ')) - 1

    vocab = set(word_map.keys())

    # Create tensor to hold embeddings, initialize
    embeddings = torch.FloatTensor(len(vocab), emb_dim)
    init_embedding(embeddings)

    # Read embedding file
    print("\nLoading embeddings...")
    for line in open(emb_file, 'r'):
        line = line.split(' ')

        emb_word = line[0]
        embedding = list(map(lambda t: float(t), filter(lambda n: n and not n.isspace(), line[1:])))

        # Ignore word if not in train_vocab
        if emb_word not in vocab:
            continue

        embeddings[word_map[emb_word]] = torch.FloatTensor(embedding)

    return embeddings, emb_dim


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.

    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.

    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    '''
    :param dataset: name of the dataset,
                    one of 'coco', 'flickr8k', 'flickr30k'
    :param karpathy_json_path:
                    path of Karpathy JSON file with splits and captions,
                    Karpathy splits can be found here: https://cs.stanford.edu/people/karpathy/deepimagesent/
                    why Karpathy splits? Sort of a gold standard of splitting coco and flickr images into sets
    :param image_folder: folder with downloaded images,
                    images should be located under /srv/data
    :param captions_per_image: number of captions to sample per image,
                    for every image, there will be N captions to learn from,
                    bigger N would typically mean better generalisation and sensitivity to differences between captions
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>s,
                    we want to keep words, which are frequent enough, not too much or too less,
                    it will be hard to learn the meaning of less frequent words given that there is a limited data to learn from for these words, we normally do not want to learn representations for such words (e.g., too hard, not useful)
    :param output_folder: folder to save files,
                    where to keep processed images/captions,
                    better to keep it somewhere locally or in your personal division of the server
    :param max_len: don't sample captions longer than this length,
                    all captions need to be of the same size,
                    we basically cut them at some point if they are too long or pad them if they are too short
    '''
    
    parser.add_argument('--dataset', type=str, default='flickr8k')
    parser.add_argument('--karpathy_json_path', type=str, default='/srv/data/aics/03-image-captioning/data/dataset_flickr8k.json')
    parser.add_argument('--image_folder', type=str, default='/srv/data/aics/03-image-captioning/data/flickr8k/Images/')
    parser.add_argument('--captions_per_image', type=int, default=5)
    parser.add_argument('--min_word_freq', type=int, default=10)
    parser.add_argument('--output_folder', type=str, default='/srv/data/aics/03-image-captioning/data/out/')
    parser.add_argument('--max_len', type=int, default=100)

    arguments = parser.parse_args()
    
    create_input_files(arguments)