import os
import hashlib
import numpy as np
import pandas as pd
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


def make_description(path, seed):
    print("Making description files ...")
    
    description = get_data(path)                                # image, text 짝을 짓고
    train, val, eval = split_description(seed, description)     # split을 통해서 test, val, eval 나눔
    descriptions = merge_descriptions(seed, train, val, eval)   # 나눠진 애들을 가지고 domain 마다 추가
    descriptions.to_pickle(os.path.join(path, 'dggnn_descriptions.pkl'))
    
    print("Saved description files")

    return description, descriptions


def get_data(path):
    ### image-text pair
    category_ids, categories, images, captions = [], [], [], []

    caption_path = os.path.join(path, "captions")
    for c, cls in enumerate(sorted(os.listdir(caption_path))):
        if cls.startswith("."): continue

        cls_path = os.path.join(caption_path, cls)
        for n, name in enumerate(sorted(os.listdir(cls_path))):
            if name.startswith("."): continue
            file_name = os.path.join(cls, name)

            lines = open(os.path.join(caption_path, file_name)).readlines()
            lines = [line.strip().lower() for line in lines if len(line.strip().split()) > 5]

            captions.append(lines)
            images.append(file_name.replace(".txt", ".jpg"))

            categories.append(cls)
            category_ids.append(c)

    descriptions = pd.DataFrame({"category_ids": category_ids, "categories": categories, "images": images, "captions": captions})
    return descriptions


def split_description(seed, description):
    total = len(description)
    n = int(total * 0.2)
    keys = list(range(len(description)))
    np.random.RandomState(seed).shuffle(keys)

    keys_2 = keys[:n]
    keys_1 = keys[n:2*n]
    keys_0 = keys[2*n:]

    train = description.iloc[keys_0, :]
    val = description.iloc[keys_1, :]
    eval = description.iloc[keys_2, :]
     
    return train, val, eval


def merge_descriptions(seed, train, val, eval):
    trains = []
    
    num_class = len(set(list(set(train['category_ids'])) + list(set(val['category_ids'])) + list(set(eval['category_ids']))))

    for domain in (["photo","cartoon", "art", "paint"]):
        if domain == 'photo':
            shuffled_train = train.sort_values(by='categories').reset_index(drop=True)
        else:
            val = val.loc[:, ['images', 'captions']]
            eval = eval.loc[:, ['images', 'captions']]
            for c in range(num_class):
                tmp = train.loc[train['category_ids']==c, :].sample(frac=1, random_state=seed).sort_values(by='category_ids').reset_index(drop=True).loc[:, ['images', 'captions']]
                if c == 0:
                    shuffled_train = tmp
                else:
                    shuffled_train = pd.concat([shuffled_train, tmp])
            
        shuffled_train = pd.concat([shuffled_train, val, eval]).reset_index(drop=True)
        shuffled_train.rename(columns={"images": domain+'_images', 'captions': domain+'_captions'}, inplace=True)
        trains.append(shuffled_train)

    
    for d, domain in enumerate(["photo", "cartoon", "art", "paint"]):
        if d == 0:
            descriptions = trains[d]
        else:
            descriptions = pd.concat([descriptions, trains[d]], axis=1)
    
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
