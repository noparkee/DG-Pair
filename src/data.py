import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import nltk



DOMAIN = ["photo", "cartoon", "art", "paint"]

class CUB_DG_Dataset(torch.utils.data.Dataset):
    def __init__(self, text_flag, i):       # i == 0: test, i == 1: train, i == 2: val
        self.data_path = "data/CUB-DG"
        
        ### for GCN
        self.descriptions = pd.read_pickle(os.path.join(self.data_path, "cub_dggcn_dataset.pkl"))
        self.descriptions = self.descriptions.loc[self.descriptions["split"] == i].reset_index(drop=True)

        self.data = []
        for dm in DOMAIN:
            self.data.append(self.descriptions[dm + "_images"])
        self.class_labels = self.descriptions["category_ids"]
        self.num_class = len(set(self.class_labels))

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()

        self.vocab = Vocabulary.load(os.path.join(self.data_path, "vocab_nlk.pkl"))
        
        self.text_flag = text_flag
        if self.text_flag:

            self.caption_tokens = []
            for d, dm in enumerate(DOMAIN):
                domain_caption = []
                for i in range(len(self.descriptions)):
                    lines = self.descriptions[dm + '_captions'][i]
                    domain_caption.append([line.strip().lower() for line in lines if len(line) > 5])
                self.caption_tokens.append(domain_caption)

    def __getitem__(self, idx):
        label = torch.tensor(self.class_labels[idx], dtype=torch.long)
        
        file_names, images = [], []
        for d, domain in enumerate(DOMAIN):
            file_name = self.data[d][idx]
            image = Image.open(os.path.join(self.data_path, "images_%s" % domain, file_name)).convert('RGB')
            image = self.transform_train(image) if self.train_flag else self.transform_eval(image)
            
            file_names.append(file_name)
            images.append(image)

        if self.text_flag:
            texts = []
            for d, domain in enumerate(DOMAIN):
                rand_idx = np.random.randint(len(self.caption_tokens[d][idx]))

                text = []
                text.append(self.vocab('<start>'))
                text.extend([self.vocab(token) for token in nltk.tokenize.word_tokenize(self.caption_tokens[d][idx][rand_idx])])
                text.append(self.vocab('<end>'))
                text = torch.Tensor(text)

                texts.append(text)
            
            return images[0], images[1], images[2], images[3], \
                    texts[0], texts[1], texts[2], texts[3], \
                     label, \
                      file_names[0], file_names[1], file_names[2], file_names[3]
        else:
            return images, \
                    label, \
                     file_names

    def __len__(self):
        return len(self.descriptions)

    def set_train_flag(self, train_flag):
        self.train_flag = train_flag


def get_datasets_and_iterators(text_flag, eval_flag=False):
    BATCH_SIZE = 32
    NUM_WORKERS = 16

    datasets, iterators_train, iterators_eval, names_eval = [], [], [], []
    
    train_data = CUB_DG_Dataset(text_flag, 1)
    val_data = CUB_DG_Dataset(text_flag, 2) 
    test_data = CUB_DG_Dataset(text_flag, 0)
    
    datasets.append(train_data)
    datasets.append(val_data)
    datasets.append(test_data)

    #### loader 다시 수정하자

    if not eval_flag:
        iterators_train.append(InfiniteDataLoader(_SplitDataset(train_data, range(len(train_data)), True), \
                                                    BATCH_SIZE, NUM_WORKERS, text_flag))
        iterators_eval.append(torch.utils.data.DataLoader(_SplitDataset(val_data, range(len(val_data)), False), \
                                                            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                            collate_fn=collate_fn if text_flag else None, shuffle=False, pin_memory=True))
        names_eval.append("validation") 
    
    iterators_eval.append(torch.utils.data.DataLoader(_SplitDataset(test_data, range(len(test_data)), False), \
                                                        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                        collate_fn=collate_fn if text_flag else None, shuffle=False, pin_memory=True))
    names_eval.append("evaluation")

    if not eval_flag:
        iterators_train = zip(*iterators_train)


    return datasets, iterators_train, iterators_eval, names_eval


def get_transforms():
    resize, cropsize = 512, 448

    transform_train = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomCrop(cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_eval = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(cropsize),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return transform_train, transform_eval


class _SplitDataset(torch.utils.data.Dataset):
    def __init__(self, underlying_dataset, keys, train_flag):
        super(_SplitDataset, self).__init__()
        self.underlying_dataset = underlying_dataset
        self.keys = keys
        self.train_flag = train_flag

    def __getitem__(self, key):
        self.underlying_dataset.set_train_flag(self.train_flag)
        return self.underlying_dataset[self.keys[key]]

    def __len__(self):
        return len(self.keys)


def split_dataset(dataset):
    total = len(dataset)
    n = int(total * 0.2)
    keys = list(range(total))

    keys_2, train_flag_2 = keys[total-n:], False
    keys_1, train_flag_1 = keys[total-2*n:total-n], False
    keys_0, train_flag_0 = keys[:total-2*n], True

    return _SplitDataset(dataset, keys_2, train_flag_2), _SplitDataset(dataset, keys_1, train_flag_1), _SplitDataset(dataset, keys_0, train_flag_0)


class _InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, num_workers, text_flag):
        super().__init__()

        batch_sampler = torch.utils.data.BatchSampler(
            torch.utils.data.RandomSampler(dataset, replacement=True),
            batch_size=batch_size,
            drop_last=True)

        self._infinite_iterator = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            pin_memory=True,
            batch_sampler=_InfiniteSampler(batch_sampler),
            collate_fn=collate_fn if text_flag else None
        )

        self.iterator = iter(self._infinite_iterator)

    def __iter__(self):
        while True:
            yield next(self.iterator)

    def __len__(self):
        raise ValueError


def collate_fn(data):
    photo_images, cartoon_images, art_images, paint_images, \
            photo_texts, cartoon_texts, art_texts, paint_texts, \
                label, \
                    photo_files, cartoon_files, art_files, paint_files = zip(*data)
    
    labels = torch.stack(label, 0)
    
    images = [photo_images, cartoon_images, art_images, paint_images]
    texts = [photo_texts, cartoon_texts, art_texts, paint_texts]
    file_names = [photo_files, cartoon_files, art_files, paint_files]

    for i in range(len(images)):
        images[i] = torch.stack(images[i], 0)
        
    lengths, texts_input, texts_target, max_lengths = [], [], [], []
    for i in range(len(texts)):
        length = torch.tensor([len(cap) - 1 for cap in texts[i]], dtype=torch.int64)
        lengths.append(length)
        max_length = torch.tensor([max(length) for cap in texts[i]], dtype=torch.int64)
        max_lengths.append(max_length)

        text_input = torch.zeros(len(texts[i]), max(length)).long()
        text_target = torch.zeros(len(texts[i]), max(length)).long()
        for j, cap in enumerate(texts[i]):
            end = length[j]
            text_input[j, :end] = cap[:-1]
            text_target[j, :end] = cap[1:]
        texts_input.append(text_input)
        texts_target.append(text_target)

    return images, labels, texts_input, texts_target, lengths, max_lengths, file_names



    
class Vocabulary(object):        # nltk
    """Simple vocabulary wrapper."""        
    def __init__(self):                     
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.unknown_token = '<unk>'
        self.start_token = '<start>'
        self.end_token = '<end>'

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    ### 추가
    def get_word_from_idx(self, idx):
        if not idx in self.idx2word:
            return self.unknown_token
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    ### 추가
    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
        assert isinstance(vocab, cls)
        return vocab
    
    ### 추가
    @classmethod
    def save(cls, vocab, path):
        assert isinstance(vocab, cls)
        with open(path, 'wb') as f:
            pickle.dump(vocab, f)
