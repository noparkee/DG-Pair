import os
import pickle
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torchvision import transforms

import nltk



DOMAIN = ["photo", "cartoon", "art", "paint"]

class CUB_DG_trainset(torch.utils.data.Dataset):
    def __init__(self, test_env, text_flag):
        '''for training dataset'''

        self.data_path = "data/CUB-DG"

        self.domain = ["photo", "cartoon", "art", "paint"]
        del self.domain[test_env]

        self.test_domain = DOMAIN[test_env]
        
        self.descriptions = pd.read_pickle(os.path.join(self.data_path, "cub_dggcn_train.pkl"))
        self.num_class = len(set(self.descriptions['category_ids']))

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()

        self.vocab = Vocabulary.load(os.path.join(self.data_path, "vocab_nlk_gcn.pkl"))
        
        self.text_flag = text_flag
        if self.text_flag:
            self.caption_tokens = []
            for d, dm in enumerate(self.domain):
                domain_caption = []
                for i in range(len(self.descriptions)):
                    lines = self.descriptions['captions%d' %d][i]
                    domain_caption.append([line.strip().lower() for line in lines if len(line) > 5])
                self.caption_tokens.append(domain_caption)
    
    def __getitem__(self, idx):
        label = torch.tensor(self.descriptions['category_ids'][idx])

        file_names, images = [], []
        for d, dm in enumerate(self.domain):
            file_name = self.descriptions['images%d' %d][idx]
            image = Image.open(os.path.join(self.data_path, "images_%s" %dm, file_name)).convert('RGB')
            image = self.transform_train(image) if self.train_flag else self.transform_eval(image)

            file_names.append(file_name)
            images.append(image)
        
        if self.text_flag:
            texts = []
            for d, dm in enumerate(self.domain):
                rand_idx = np.random.randint(len(self.caption_tokens[d][idx]))

                text = []
                text.append(self.vocab('<start>'))
                text.extend([self.vocab(token) for token in nltk.tokenize.word_tokenize(self.caption_tokens[d][idx][rand_idx])])
                text.append(self.vocab('<end>'))
                text = torch.Tensor(text)

                texts.append(text)

            return images[0], images[1], images[2], \
                    texts[0], texts[1], texts[2], \
                     label, \
                      file_names[0], file_names[1], file_names[2]
        else:
            return images, \
                    label, \
                     file_names

    def __len__(self):
        return len(self.descriptions)
    
    def set_train_flag(self, train_flag):
        self.train_flag = train_flag



class CUB_DG_Dataset(torch.utils.data.Dataset):
    def __init__(self, domain):
        '''for validation/evaluation dataset'''

        self.data_path = "data/CUB-DG"
        self.domain = domain

        self.descriptions = pd.read_pickle(os.path.join(self.data_path, "cub_dggcn_data.pkl"))
        self.num_classes = len(list(set(self.descriptions['category_ids'])))
        with open(os.path.join(self.data_path, "split_info.pkl"), 'rb') as f:
            self.split_info = pickle.load(f)

        self.train_flag = False
        self.transform_train, self.transform_eval = get_transforms()

        self.vocab = Vocabulary.load(os.path.join(self.data_path, "vocab_nlk_gcn.pkl"))

    def __getitem__(self, idx):
        file_name = self.descriptions['images'][idx]
        label = self.descriptions['category_ids'][idx]
        label = torch.tensor(label, dtype=torch.long)
        image = Image.open(os.path.join(self.data_path, "images_%s" % self.domain, file_name)).convert('RGB')
        image = self.transform_train(image) if self.train_flag else self.transform_eval(image)

        return image, label, file_name

    def __len__(self):
        # 이 데이터셋에서는 train을 사용하지 않고, val, test만 사용하는데도 length를 이렇게 해도 되나?
        # _SplitDataset에서 적용한걸로 바뀌는건가...?
        return len(self.descriptions)

    def set_train_flag(self, train_flag):
        self.train_flag = train_flag


def get_datasets_and_iterators(test_env, text_flag, eval_flag=False):
    DOMAINS = ["photo", "cartoon", "art", "paint"]
    BATCH_SIZE = 32
    NUM_WORKERS = 16

    source_num = 0
    datasets, iterators_train, iterators_eval, names_eval = [], [], [], []
    
    train_data = CUB_DG_trainset(test_env, text_flag)
    train_data.set_train_flag(True)
    datasets.append(train_data)
    iterators_train.append(InfiniteDataLoader(train_data, BATCH_SIZE, NUM_WORKERS, text_flag))
                                                    
    for d, domain in enumerate(DOMAINS):
        dataset = CUB_DG_Dataset(domain)
        datasets.append(dataset)

        if not eval_flag and d != test_env:
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
    images0, images1, images2, \
            captions0, captions1, captions2, \
                label, \
                    files0, files1, files2 = zip(*data)
    
    labels = torch.stack(label, 0)
    
    images = [images0, images1, images2]
    texts = [captions0, captions1, captions2]
    file_names = [files0, files1, files2]

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
