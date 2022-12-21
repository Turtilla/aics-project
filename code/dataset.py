import os
import string
from collections import Counter
from dataclasses import dataclass
from glob import glob
from xml.etree import ElementTree
from xml.etree.ElementTree import ParseError

import nltk
import torch
from filters import RelationFilter
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm

UNKNOWN_TOKEN = '<unk>'
START_TOKEN = '<start>'
END_TOKEN = '<end>'
PADDING_TOKEN = '<pad>'

@dataclass(slots=True, kw_only=True)
class CLEFSample:
    # by Dominik
    image_id: str
    caption: str
    caption_length: torch.CharTensor
    image_path: str
    encoded_caption: torch.LongTensor = None
    image: torch.FloatTensor = None


class CLEFDataset(Dataset):
    # by Dominik, individual contributions by Maria marked with in-line comments or comments under specific methods
    def __init__(
        self,
        annotation_directory: str,
        image_directory: str,
        relation_filter: RelationFilter,
        number_images=100,
        word_map: dict = None,
        min_frequency=10,
        concat_captions: bool = False,  # added by Maria to allow the optional concatenation of multiple captions into one
    ) -> None:
        super(CLEFDataset, self).__init__()
        self.unknown_words = Counter()
        
        captions = self._load_captions(annotation_directory, number_images, concat_captions, relation_filter)
        samples = self._load_images(image_directory, captions)

        if word_map is None:
            word_map = self._create_word_map(samples, min_frequency)
        self.word_map = word_map

        self.samples = self._encode_captions(samples)


    def _load_captions(self, directory: str, number_images: int, concat_captions: bool, relation_filter: RelationFilter) -> list[CLEFSample]:
        captions: list[CLEFSample] = []

        file_pattern = directory + '**/*.eng'
        for file in glob(file_pattern, recursive=True):
            if len(captions) == number_images:
                break
            try:
                root = ElementTree.parse(file).getroot()
                description = root.find('./DESCRIPTION').text
                # multiple captions option by Maria
                all_captions = [cleansed_caption
                                for caption in description.split(';')
                                if (cleansed_caption := caption.strip()) != '']
                if concat_captions is True:
                    first_caption = ' and '.join(all_captions)
                else:
                    first_caption = all_captions[0]

                punctuations = string.punctuation + '``\'\''
                tokenized_caption = [word.lower() 
                                     for word in nltk.word_tokenize(first_caption)
                                     if word not in punctuations]

                image_path = root.find('./IMAGE').text.removeprefix('images/')
                image_id = image_path.removesuffix('.jpg')

                if relation_filter.has_relation(tokenized_caption):
                    captions.append(CLEFSample(
                        image_id=image_id,
                        caption=tokenized_caption,
                        # +2 for start and end token
                        caption_length=torch.CharTensor([len(tokenized_caption) + 2]),
                        image_path=image_path
                    ))
                else:
                    continue

            except ParseError:
                continue

        print('Captions loaded!')  # added for clarity by Maria

        return captions

    def _load_images(self, directory: str, captions: list[CLEFSample]) -> list[CLEFSample]:
        transform = transforms.ToTensor()

        samples: list[CLEFSample] = []
        for sample in tqdm(captions, desc='Loading images...'):  # tqdm added because Maria is impatient
            image_path = os.path.join(directory, sample.image_path)

            # TODO correct conversion?
            # error-handling added by Maria
            try:
                image = Image.open(image_path).resize((256, 256)).convert('RGB')
                sample.image = transform(image)
                samples.append(sample)
            except FileNotFoundError:
                continue

        print('Images loaded!')  # added for clarity by Maria

        return samples

    def _create_word_map(self, samples: list[CLEFSample], min_frequency: int) -> dict:
        word_frequency = Counter()
        for sample in samples:
            word_frequency.update(sample.caption)

        words = [word for word in word_frequency.keys() if word_frequency[word] >= min_frequency]

        word_map = {word: index for index, word in enumerate(words, start=1)}
        word_map[UNKNOWN_TOKEN] = len(word_map) + 1
        word_map[START_TOKEN] = len(word_map) + 1
        word_map[END_TOKEN] = len(word_map) + 1
        word_map[PADDING_TOKEN] = 0

        return word_map

    def _encode_captions(self, samples: list[CLEFSample]) -> list[CLEFSample]:
        encoded_samples: list[CLEFSample] = []
        for sample in samples:
            encoding = [self.get_encoded_token(START_TOKEN), *[self.get_encoded_token(token)
                                                               for token in sample.caption], self.get_encoded_token(END_TOKEN)]
            sample.encoded_caption = torch.LongTensor(encoding)  # changing to LongTensor to match the model (Maria)
            encoded_samples.append(sample)
        return encoded_samples

    def get_encoded_token(self, token: str) -> int:
        if token in self.word_map:
            return self.word_map[token]
        else:
            self.unknown_words.update([token])
            return self.word_map[UNKNOWN_TOKEN]

    def __getitem__(self, index: int) -> CLEFSample:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)
