import os
import string
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from glob import glob
from xml.etree import ElementTree
from xml.etree.ElementTree import ParseError

import nltk
import torch
from filters import RelationFilter
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

# variables for unifying the non-word tokens
UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'

class Datasets(Enum):
    '''By Dominik
    An enum that lists all possible datasets and can be used instead of magic strings.
    '''
    CLEF = 'imageCLEF'
    FLICKR = 'flickr'

    def __str__(self):
        return self.value

@dataclass(slots=True, kw_only=True)
class Sample:
    '''By Dominik.
    A class which allows for the creation of picture-caption sample objects.

    Attributes:
        image_id (str): The ID of the image (file name without the .jpg extension).
        caption (str): The caption associated with the sample.
        image_path (str): The ID/file name of the image in the sample.
        tokenized_caption (list[str]): A tokenized version of the caption associated with the sample.
        caption_length (torch.CharTensor): The length of the encoded caption.
        encoded_caption (torch.LongTensor): The caption encoded using some wordmap.
        image (torch.FloatTensor): A tensor representation of the image associated with the sample.
    '''
    # by Dominik
    image_id: str
    caption: str
    image_path: str
    tokenized_caption: list[str] = None
    caption_length: torch.CharTensor = 0
    encoded_caption: torch.LongTensor = None
    image: torch.FloatTensor = None


class CaptionLoader(ABC):
    '''By Dominik
    An abstract CaptionLoader. All of its implementations need to implement the load_captions method
    '''
    @abstractmethod
    def load_captions(self, concat_captions: bool) -> list[Sample]:
        '''An abstract method that allows for loading in captions.

        Args:
            concat_captions (bool): Defines whether or not captions should be concatenated, in case they consist of more than one distinct phrase.

        Returns:
            A list of partially filled-out Sample objects.
        '''

class FlickrCaptionLoader(CaptionLoader):
    '''By Dominik.
    A class that allows for loading in the captions and names of associated images from the Flickr dataset.

    Attributes:
        annotation_directory (str): The directory in which the file containing the annotations is stored.
    '''
    def __init__(self, annotation_directory: str) -> None:
        '''The __init__ method of the class.
        
        Args:
            annotation_directory (str): The directory in which the file containing the annotations is stored.'''
        super().__init__()
        self.annotation_directory = annotation_directory

    def load_captions(self, concat_captions: bool) -> list[Sample]:
        '''A method that allows for loading in captions.

        Args:
            concat_captions (bool): Defines whether or not captions should be concatenated, in case they consist of more than one distinct phrase. Redundant in this class.

        Returns:
            A list of partially filled-out Sample objects.
        '''
        captions: list[Sample] = []

        meatdata_file = os.path.join(self.annotation_directory, 'captions.txt')
        with open(meatdata_file, 'r') as f:
            lines = f.readlines()[1:]
        for line in lines:
            image_path, caption = line.removesuffix('\n').split(',', maxsplit=1)
            image_id = image_path.removesuffix('.jpg')
            captions.append(Sample(
                image_id=image_id,
                caption=caption,
                image_path=image_path
            ))

        print(f'{len(captions)} captions loaded!')
        return captions



class CLEFCaptionLoader(CaptionLoader):
    '''By Dominik.
    A class that allows for loading in the captions and names of associated images from the imageCLEF dataset.

    Attributes:
        annotation_directory (str): The directory in which the file containing the annotations is stored.
    '''
    def __init__(self, annotation_directory: str) -> None:
        '''The __init__ method of the class.
        
        Args:
            annotation_directory (str): The directory in which the file containing the annotations is stored.'''
        super().__init__()
        self.annotation_directory = annotation_directory

    def load_captions(self, concat_captions: bool) -> list[Sample]:
        '''A method that allows for loading in captions.

        Args:
            concat_captions (bool): Defines whether or not captions should be concatenated, in case they consist of more than one distinct phrase.
            
        Returns:
            A list of partially filled-out Sample objects.
        '''
        captions: list[Sample] = []

        file_pattern = self.annotation_directory + '**/*.eng'
        for file in glob(file_pattern, recursive=True):
            try:
                root = ElementTree.parse(file).getroot()
                description = root.find('./DESCRIPTION').text

                # multiple captions option by Maria
                all_captions = [cleansed_caption
                                for caption in description.split(';')
                                if (cleansed_caption := caption.strip()) != '']
                if concat_captions is True:
                    caption = ' and '.join(all_captions)
                else:
                    caption = all_captions[0]


                image_path = root.find('./IMAGE').text.removeprefix('images/')
                image_id = image_path.removesuffix('.jpg')

                captions.append(Sample(
                    image_id=image_id,
                    caption=caption,
                    image_path=image_path
                ))

            except ParseError:
                continue

        print(f'{len(captions)} captions loaded!')  # added for clarity by Maria
        return captions

class ImageDataset(Dataset):
    '''By Dominik.
    A custom torch Dataset-based dataset class intended for storing and performing operations on Sample objects representing the information from a given
    dataset (imageCLEF or Flickr8k).

    Attributes:
        unknown_words (Counter): An object storing the unknown words encountered during the dataset generation.
        word_map (dict): A dictionary containing the mappings between words and indices used for encoding the captions. Can be pre-loaded or created from scratch.
        samples (list[Sample]): A list of fully filled-out Sample objects representing captions and corresponding images transformed or encoded in various ways.
    '''
    def __init__(
        self,
        image_directory: str,
        relation_filter: RelationFilter,
        caption_loader: CaptionLoader,
        number_images: int = 100,
        word_map: dict = None,
        min_frequency: int = 10,
        concat_captions: bool = False,  # added by Maria to allow the optional concatenation of multiple captions into one
        unk_filter: float = 0.2  # added by Maria for filtering out the captions with more than X% unknown tokens, only works with a pre-loaded wordmap
    ) -> None:
        '''The __init__ method of the class.

        Args:
            image_directory (str): The directory in which the corresponding image files are located.
            relation_filter (RelationFilter): The class that will be used for selecting the captions containing relations between entities.
            caption_loader (CaptionLoader): The class that will be used to load the captions.
            number_images (int): The number of captions that are to be selected (will not necessarily correspond to the number of the full samples, but will be close).
            word_map (dict): A preexisting wordmap. Can be left empty (None), in which case a new wordmap will be generated.
            concat_captions (bool): Determines whether multi-element captions will be concatenated or only the first element will be used. Only relevant in imageCLEF.
            unk_filter (float): Determines the maximum % of unknown tokens in a caption (e.g. 0.2 = 20% in this case).
        '''
        print(f'{caption_loader.__class__.__name__:-^30}')
        super().__init__()
        self.unknown_words = Counter()

        # This part (unk filtering) was initially implemented by Maria, with some re-working by Dominik.
        vocab = set(word_map.keys()) if word_map is not None else None

        captions = caption_loader.load_captions(concat_captions)
        filtered_captions = self._filter_captions(captions, number_images, relation_filter, vocab, unk_filter)
        samples = self._load_images(image_directory, filtered_captions)

        if word_map is None:
            self.word_map = self._create_word_map(samples, min_frequency)
        else:
            self.word_map = word_map

        self.samples = self._encode_captions(samples)

    def _filter_captions(self,
                         samples: list[Sample],
                         number_images: int,
                         relation_filter: RelationFilter,
                         vocab: set[str],
                         unk_filter: float) -> list[Sample]:
        '''A method that allows for the filtering of captions to only retain ones that possess specific, desirable features (e.g. contain relations or do
        not contain too many unknown tokens).

        Args:
            samples (list[Sample]): A list of partially filled-out samples (include only captions, image paths, IDs).
            number_images(int): The upper limit on how many captions get retained (does not 100% correspond to the number of images due to image loading errors).
            relation_filter (RelationFilter): The class that will be used for selecting the captions containing relations between entities.
            vocab (set): A set of all the words in the pre-loaded wordmap.
            unk_filter (float): Determines the maximum % of unknown tokens in a caption (e.g. 0.2 = 20% in this case).

        Returns:
            A list of captions filtered using the given filters.
        '''
        captions: list[Sample] = []

        for sample in samples:
            if len(captions) == number_images:
                break

            punctuations = string.punctuation + '``\'\''
            tokenized_caption = [word.lower()
                                 for word in nltk.word_tokenize(sample.caption)
                                 if word not in punctuations]
            
            # By Maria
            # if there is a word map then this will filter out the unks (hopefully)
            if vocab is not None:
                unk_counter = 0
                for word in tokenized_caption:
                    if word not in vocab:
                        unk_counter += 1

                unk_ratio = unk_counter / len(tokenized_caption)
                if unk_ratio >= unk_filter:
                    continue

            if relation_filter.has_relation(tokenized_caption):
                sample.tokenized_caption = tokenized_caption
                # +2 for start and end token
                sample.caption_length = torch.CharTensor([len(tokenized_caption) + 2])
                captions.append(sample)

        print(f'{len(captions)} captions filtered.')
        return captions

    def _load_images(self, directory: str, captions: list[Sample]) -> list[Sample]:
        '''A method of the class which allows for the loading in of the images associated with the captions.

        Args:
            directory (str): The directory in which the image files are located.
            captions (list[Sample]): A list of partially filled-out Sample objects (not containing the actual images yet).

        Returns:
            A list of samples now containing the loaded images.
        '''
        transform = transforms.ToTensor()

        samples: list[Sample] = []
        for sample in tqdm(captions, desc='Loading images...'):  # tqdm added because Maria is impatient
            image_path = os.path.join(directory, sample.image_path)

            # error-handling added by Maria
            try:
                with Image.open(image_path) as img:
                    image = img.resize((256, 256)).convert('RGB')
                    sample.image = transform(image)
                    samples.append(sample)
            except FileNotFoundError:
                continue

        print(f'{len(samples)} images loaded!\n')  # added for clarity by Maria

        return samples

    def _create_word_map(self, samples: list[Sample], min_frequency: int) -> dict:
        '''A class method which creates a new word map for the captions in the dataset.
        This method is based on https://github.com/sdobnik/aics/blob/master/tutorials/03-image-captioning/2022/preproc.py

        Args:
            samples (list[Sample]): A list of partially filled-out Sample objects. Contains the tokenized captions that are the basis for creating the map.
            min_frequency (int): The minimum frequency a word has to have to be included in the vocabulary / wordmap.

        Returns:
            A dict that is a representation of the given vocabulary and its mappings to indices.
        '''
        word_frequency = Counter()
        for sample in samples:
            word_frequency.update(sample.tokenized_caption)

        words = [word for word in word_frequency.keys() if word_frequency[word] >= min_frequency]

        word_map = {word: index for index, word in enumerate(words, start=1)}
        word_map[UNKNOWN_TOKEN] = len(word_map) + 1
        word_map[START_TOKEN] = len(word_map) + 1
        word_map[END_TOKEN] = len(word_map) + 1
        word_map[PADDING_TOKEN] = 0

        return word_map

    def _encode_captions(self, samples: list[Sample]) -> list[Sample]:
        '''A class method that allows for the encoding of the tokenized captions using a certain word to index mapping (wordmap).

        Args:
            samples (list[Sample]): A list of partially filled-out Sample objects.

        Returns:
            A list of Sample objects now containing the encoded versions of captions.
        '''
        encoded_samples: list[Sample] = []
        for sample in samples:
            encoding = [self.get_encoded_token(START_TOKEN),
                        *[self.get_encoded_token(token) for token in sample.tokenized_caption],
                        self.get_encoded_token(END_TOKEN)]
            sample.encoded_caption = torch.LongTensor(encoding)  # changing to LongTensor to match the model (Maria)
            encoded_samples.append(sample)
        return encoded_samples

    def get_encoded_token(self, token: str) -> int:
        '''A class method that allows for the retrieval of the index that a given word token is associated with in the Dataset's wordmap.
        
        Args:
            token (str): The token whose index is being searched for.
            
        Returns:
            An integer representing the index of the token in Dataset's wordmap.
        '''
        if token in self.word_map:
            index = self.word_map[token]
        else:
            self.unknown_words.update([token])
            index = self.word_map[UNKNOWN_TOKEN]

        return index

    def __getitem__(self, index: int) -> Sample:
        '''The __getitem__ magic method of the class.
        
        Args:
            index (int): The index that is being asked for.
            
        Returns:
            The Sample object from the internal samples attribute that is located under the given index.
        '''
        return self.samples[index]

    def __len__(self) -> int:
        '''The __len__ magic method of the class.

        Returns:
            The length of the list of samples stored in the class object.
        '''
        return len(self.samples)


def custom_collate(samples: list[Sample]) -> dict:
    '''By Dominik.
    A custom Collate function that will be used in the creation of Dataloaders out of the custom Datasets.

    Args:
        samples (list[Sample]): A list of Sample objects containing information about captions and corresponding images, encoded or transformed in various ways.

    Returns:
        A dictionary representation of the given list, where the values of a given attribute of the Sample objects are gathered in a list which is then stored as
        one of the dictionary's values. This essentially allows for the batching of the data. 
        The encoded captions are padded to the longest sequence in the batch.
    '''
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
