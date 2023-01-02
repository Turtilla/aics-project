import argparse
import json
import os
import time

import torch
from dataset import (CLEFCaptionLoader, Datasets, FlickrCaptionLoader,
                     ImageDataset)
from filters import POSRelationFilter, RuleBasedRelationFilter
from torch.backends import cudnn
from torch.utils.data import Dataset
from train import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


def load_wordmap(word_map_path: str):
    '''From Nikolai Ilinykh's code.
    A function that allows for the loading in of a saved wordmap.

    Args:
        word_map_path (str): The path to the file containing the wordmap.

    Returns:
        A dictionary containing the mappings between words and indices.
    '''
    # Load word map (word2ix)
    with open(word_map_path, 'r') as j:
        word_map = json.load(j)
    return word_map

def load_data(dataset: Datasets, data_root_dir: str, word_map: dict, number_of_images: int, unknown_filter: float):
    '''By Dominik.
    A function that allows for the loading in of the data from a specific dataset to be later used in the fine-tuning.
    
    Args:
        dataset (str): The name of the dataset that is to be loaded. Either clef or flickr.
        data_root_dir (str): The root directory in which both of the datasets are located.
        word_map (dict): A pre-loaded word to index mapping. Can be left empty, then a new one will be generated.
        number_of_images (int): The number of the captions that will be loaded after filtering. Does not fully correspond to the end number of samples due to image loading errors.
        unknown_filter (float): The maximum of unknown tokens relative to the length of the caption (e.g. 20% = 0.2).

    Return:
        An ImageDataset object with the data loaded in according to the given parameters.        
    '''
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

def split_dataset(dataset: Dataset):
    '''By Maria.
    A function that allows for the splitting of an ImageDataset object into subsets that can be used for training, validation, testing. Splitting is done according
    to preset proportions (8:1:1). Splitting is done using randomization, but with a fixed seed for reproducible results (elements of the dataset are shuffled,
    but the result of the shuffling is always the same). Also prints out the approximate sizes of the subsets.

    Args:
        dataset (Dataset): The Dataset object that is to be split. Technically does not have to be ImageDataset.

    Returns:
        Three subsets in the order of train, val, test, in the proportions of 8:1:1.
    '''

    # The printing out of the approximate lengths by Dominik.
    len_train = int(0.8 * len(dataset))
    len_val = int(0.1 * len(dataset))
    len_test = len(dataset) - len_train - len_val
    print(f'{"Dataset lengths":-^30}')
    print(f'train: {len_train}, val: {len_val}, test: {len_test}')
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [len_train, len_val, len_test], generator=torch.Generator().manual_seed(25))
    return train_set, val_set, test_set


if __name__ == '__main__':
    # This script is intended to be run from the command line as that makes the fine-tuning simpler.
    parser = argparse.ArgumentParser()
    parser.add_argument('--on_server', type=bool, default=False)
    parser.add_argument('--dataset', type=Datasets, default=Datasets.CLEF)
    parser.add_argument('--number_of_images', type=int, default=100)
    parser.add_argument('--unknown_filter', type=float, default=0.2)


    arguments = parser.parse_args()

    print(f'Start: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')

    if arguments.on_server:
        data_root_dir = '/srv/data/guskunkdo/'
        saved_root_dir = os.path.join(data_root_dir, 'saved/')
        model_path = '/srv/data/aics/03-image-captioning/data/BEST_checkpoint_flickr8k_5_10.pth.tar'  
        word_map_path = '/srv/data/aics/03-image-captioning/data/out/wordmap_flickr8k_5_10.json'  
    else:
        data_root_dir = '../data/'
        saved_root_dir = '../saved/'
        model_path = '../data/BEST_checkpoint_flickr8k_5_10.pth.tar'  
        word_map_path = '../data/wordmap_flickr8k_5_10.json'  

    # Read word map
    word_map = load_wordmap(word_map_path)
    dataset = load_data(arguments.dataset, data_root_dir, word_map, arguments.number_of_images, arguments.unknown_filter)
    train_set, val_set, test_set = split_dataset(dataset)

    # the training loop is imported from elsewhere (Nikolai Ilinykh's code)
    train(checkpoint_name=f'{arguments.dataset}_{arguments.number_of_images}_{arguments.unknown_filter}',
          train_set=train_set,
          val_set=val_set,
          word_map=word_map,
          saved_root_dir=saved_root_dir,
          model_path=model_path,
          device=device)

    print(f'End: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}')
