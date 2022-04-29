import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import nltk



class CUB_DG_Dataset(torch.utils.data.Dataset):
    def __init__(self, domain, text_flag):       # i == 0: test, i == 1: train, i == 2: val
        # split = -1, 0, 1, 2
        # -1: test
        # 0, 1, 2: train
        self.data_path = "data/CUB-DG"
        self.domain = domain
        self.descriptions = pd.read_pickle(os.path.join(self.data_path, "cub_dggcn_dataset3.pkl"))
        self.num_classes = len(list(set(self.descriptions['category_ids'])))
        with open(os.path.join(self.data_path, "split_info.pkl"), 'rb') as f:
            self.split_info = pickle.load(f)

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()

        self.vocab = Vocabulary.load(os.path.join(self.data_path, "vocab_nlk3.pkl"))
        
        self.text_flag = text_flag
        if self.text_flag:
            self.caption_tokens = []
            for i in range(len(self.descriptions)):
                lines = self.descriptions['captions'][i]
                self.caption_tokens.append([line.strip().lower() for line in lines if len(line) > 5])

    def __getitem__(self, idx):
        file_name = self.descriptions['images'][idx]
        label = self.descriptions['category_ids'][idx]
        label = torch.tensor(label, dtype=torch.long)
        image = Image.open(os.path.join(self.data_path, "images_%s" % self.domain, file_name)).convert('RGB')
        image = self.transform_train(image) if self.train_flag else self.transform_eval(image)

        if self.text_flag:
            rand_idx = np.random.randint(len(self.caption_tokens[idx]))
            text = []
            text.append(self.vocab('<start>'))
            text.extend([self.vocab(token) for token in nltk.tokenize.word_tokenize(self.caption_tokens[idx][rand_idx])])
            text.append(self.vocab('<end>'))
            text = torch.Tensor(text)

            return image, text, label, file_name 
        else:
            return image, label, file_name

    def __len__(self):
        return len(self.descriptions)

    def set_train_flag(self, train_flag):
        self.train_flag = train_flag


def get_datasets_and_iterators(test_env, text_flag, eval_flag=False):
    DOMAINS = ["photo", "cartoon", "art", "paint"]
    BATCH_SIZE = 32
    NUM_WORKERS = 16

    source_num = 0
    datasets, iterators_train, iterators_eval, names_eval = [], [], [], []
    
    for d, domain in enumerate(DOMAINS):
        dataset = CUB_DG_Dataset(domain, text_flag)   
        datasets.append(dataset)

        # evaluation, validation 에서는 text를 사용하지 않으니까 collate_fn 따로 없어도 될 듯
        if not eval_flag and d != test_env:
            iterators_train.append(InfiniteDataLoader(
                _SplitDataset(dataset, dataset.split_info["split%d_train" % source_num], train_flag=True), BATCH_SIZE, NUM_WORKERS, text_flag))
            iterators_eval.append(torch.utils.data.DataLoader(
                _SplitDataset(dataset, dataset.split_info["split%d_val" % source_num], train_flag=True),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True))   
            names_eval.append("env%d_1" % d)
            source_num += 1
        if d == test_env:
            iterators_eval.append(torch.utils.data.DataLoader(
                _SplitDataset(dataset, dataset.split_info["split_test"], train_flag=False),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, pin_memory=True))
            names_eval.append("env%d_2" %d)
    
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
    """ used by split_dataset """
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
    images, texts, labels, file_names = zip(*data)
    
    images = torch.stack(images, 0)
    labels = torch.stack(labels, 0)
    
    lengths = [len(cap) - 1 for cap in texts]
    texts_input = torch.zero(len(texts), max(lengths)).long()
    texts_target = torch.zeros(len(texts), max(lengths)).long()

    for i, cap in enumerate(texts):
        end = lengths[i]
        texts_input[i, :end] = cap[:-1]
        texts_target[i, :end] = cap[1:]

    max_lengths = torch.tensor([max(lengths) for cap in texts], dtype=torch.int64)

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
