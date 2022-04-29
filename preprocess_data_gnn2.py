import os
import hashlib
import numpy as np
import pandas as pd
import random
import nltk

from collections import Counter
from src.data import Vocabulary


def main():
    path = 'data/CUB-DG/'
    seed = 0        ### for debugging
    description, descriptions = make_description(path, seed)
    
    #total = len(descriptions)
    #n = int(total * 0.2)
    #keys = list(range(total))
    #keys_2, train_flag_2 = keys[total-n:], False
    #keys_1, train_flag_1 = keys[total-2*n:total-n], False
    #keys_0, train_flag_0 = keys[:total-2*n], True
    #train = descriptions.iloc[keys_0, :]
    #val = descriptions.iloc[keys_1, :]
    #eval = descriptions.iloc[keys_2, :]

    #print(train)
    #print(val)
    #print(eval)
    
    #print(len(list(set(train.category_ids))))
    #print(len(list(set(val.category_ids))))
    #print(len(list(set(eval.category_ids))))

    #total = len(descriptions)
    #n = int(total * 0.2)
    #train = descriptions.iloc[:total-2*n, :]
    #val = descriptions.iloc[total-2*n:total-n, :]
    #eval = descriptions.iloc[total-n:, :]

    print("Making a vocab file ...")

    vocab = build_vocab(description)
    Vocabulary.save(vocab, os.path.join(path, "vocab_nlk.pkl"))
    
    print("Saved the vocab file")


def make_description(path, seed):
    print("Making a description file ...")
    
    description = get_data(path)                                # image, text 짝을 짓고
    train, val, test = split_description(seed, description)     # split을 통해서 test, val, eval 나눔
    descriptions = merge_descriptions(train, val, test)   # 나눠진 애들을 가지고 domain 마다 추가
    descriptions.to_pickle(os.path.join(path, 'cub_dggcn_dataset.pkl'))

    print("Saved the description file")

    return description, descriptions


def get_data(path):
    ### image-text pair
    category_ids, categories, images, captions = [], [], [], []
    image_ids, split = [], []
    
    image_file = open(os.path.join(path, "images.txt"), "r")
    while True:
        image_line = image_file.readline()
        if not image_line:
            break
        image_id, image_name = image_line.split()
        image_ids.append(image_id)
        images.append(image_name)
        categories.append(image_name.split('/')[0])
    image_file.close()

    split_file = open(os.path.join(path, "train_test_split.txt"), "r")      # 1이 training
    while True:
        split_line = split_file.readline()
        if not split_line:
            break
        image_id, split_flag = split_line.split()
        split.append(int(split_flag))
    split_file.close()

    class_file = open(os.path.join(path, "image_class_labels.txt"), "r")
    while True:
        class_line = class_file.readline()
        if not class_line:
            break
        image_id, category_id = class_line.split()
        category_ids.append(int(category_id)-1)
    class_file.close

    for file in images:
        f = file.replace(".jpg", ".txt")
        lines = open(os.path.join(path, "captions", f)).readlines()
        lines = [line.strip().lower() for line in lines if len(line.strip().split()) > 5]
        
        captions.append(lines)

    descriptions = pd.DataFrame({"category_ids": category_ids, "categories": categories, "images": images, "captions": captions, "split": split})

    return descriptions


def split_description(seed, description):
    random.seed(0)
    SPLIT_TRAIN_VAL = int(0.1 * 30)     # train dataset에서 한 class 당 약 30개의 데이터 존재 + train과 val의 비율은 9:1 (임의로)

    train = description.loc[description.split == 1].reset_index(drop=True)
    test = description.loc[description.split == 0].reset_index(drop=True)

    offset = 0
    for i in range(200):
        l = len(train.loc[train.category_ids == i])
        val_idcs = random.sample(range(l), SPLIT_TRAIN_VAL)
        for v in val_idcs:
            train.iloc[offset+v, -1] = 2

        offset += l

    val = train.loc[train.split == 2].reset_index(drop=True)
    train = train.loc[train.split == 1].reset_index(drop=True)

    return train, val, test


def merge_descriptions(train, val, test):
    datasets = []
    
    num_class = 200

    for i, domain in enumerate((["photo","cartoon", "art", "paint"])):
        if domain == 'photo':
            shuffled_data = train.sort_values(by='categories').reset_index(drop=True)
        else:
            val = val.loc[:, ['images', 'captions']]
            test = test.loc[:, ['images', 'captions']]
            for c in range(num_class):
                tmp = train.loc[train['category_ids']==c, :].sample(frac=1, random_state=i).sort_values(by='category_ids').reset_index(drop=True).loc[:, ['images', 'captions']]
                if c == 0:
                    shuffled_data = tmp
                else:
                    shuffled_data = pd.concat([shuffled_data, tmp])
            
        shuffled_data = pd.concat([shuffled_data, val, test]).reset_index(drop=True)
        shuffled_data.rename(columns={"images": domain+'_images', 'captions': domain+'_captions'}, inplace=True)
        datasets.append(shuffled_data)
    
    for d, domain in enumerate(["photo", "cartoon", "art", "paint"]):
        if d == 0:
            descriptions = datasets[d]
        else:
            descriptions = pd.concat([descriptions, datasets[d]], axis=1)

    descriptions = descriptions.iloc[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 4]]
    '''
    Index(['category_ids', 'categories', 'photo_images', 'photo_captions', 'split',
       'cartoon_images', 'cartoon_captions', 'art_images', 'art_captions',
       'paint_images', 'paint_captions'],
      dtype='object')
    '''

    return descriptions


def build_vocab(description, threshold=1):
    """Build a simple vocabulary wrapper."""
   
    counter = Counter()
    for i in range(len(description)):
        for line in description['captions'][i]:
            if len(line) < 10: continue
            tokens = nltk.tokenize.word_tokenize(line.strip().lower())
            counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)
    return vocab


if __name__ == "__main__":
    main()
