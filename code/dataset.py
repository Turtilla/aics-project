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

UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'

class Datasets(Enum):
    CLEF = 'clef'
    FLICKR = 'flickr'

    def __str__(self):
        return self.value

@dataclass(slots=True, kw_only=True)
class Sample:
    # by Dominik
    image_id: str
    caption: str
    image_path: str
    tokenized_caption: list[str] = None
    caption_length: torch.CharTensor = 0
    encoded_caption: torch.LongTensor = None
    image: torch.FloatTensor = None


class CaptionLoader(ABC):
    @abstractmethod
    def load_captions(self, concat_captions: bool) -> list[Sample]:
        pass

class FlickrCaptionLoader(CaptionLoader):
    def __init__(self, annotation_directory: str) -> None:
        super().__init__()
        self.annotation_directory = annotation_directory

    def load_captions(self, concat_captions: bool) -> list[Sample]:
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
    def __init__(self, annotation_directory: str) -> None:
        super().__init__()
        self.annotation_directory = annotation_directory

    def load_captions(self, concat_captions: bool) -> list[Sample]:
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
    # by Dominik, individual contributions by Maria marked with in-line comments or comments under specific methods
    def __init__(
        self,
        image_directory: str,
        relation_filter: RelationFilter,
        caption_loader: CaptionLoader,
        number_images=100,
        word_map: dict = None,
        min_frequency=10,
        concat_captions: bool = False,  # added by Maria to allow the optional concatenation of multiple captions into one
        unk_filter: float = 0.2  # added by Maria for filtering out the captions with more than X% unknown tokens, only works with a pre-loaded wordmap
    ) -> None:
        print(f'{caption_loader.__class__.__name__:-^30}')
        super().__init__()
        self.unknown_words = Counter()

         # this needs to be ahead for the UNK filtering, otherwise the word map is generated later
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
        captions: list[Sample] = []

        for sample in samples:
            if len(captions) == number_images:
                break

            punctuations = string.punctuation + '``\'\''
            tokenized_caption = [word.lower()
                                 for word in nltk.word_tokenize(sample.caption)
                                 if word not in punctuations]

            if vocab is not None:  # if there is a word map then this will filter out the unks (hopefully)
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
        transform = transforms.ToTensor()

        samples: list[Sample] = []
        for sample in tqdm(captions, desc='Loading images...'):  # tqdm added because Maria is impatient
            image_path = os.path.join(directory, sample.image_path)

            # TODO correct conversion?
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
        encoded_samples: list[Sample] = []
        for sample in samples:
            encoding = [self.get_encoded_token(START_TOKEN),
                        *[self.get_encoded_token(token) for token in sample.tokenized_caption],
                        self.get_encoded_token(END_TOKEN)]
            sample.encoded_caption = torch.LongTensor(encoding)  # changing to LongTensor to match the model (Maria)
            encoded_samples.append(sample)
        return encoded_samples

    def get_encoded_token(self, token: str) -> int:
        if token in self.word_map:
            index = self.word_map[token]
        else:
            self.unknown_words.update([token])
            index = self.word_map[UNKNOWN_TOKEN]

        return index

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)


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
