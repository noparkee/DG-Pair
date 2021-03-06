from copyreg import pickle
import os
import hashlib
import numpy as np
import pandas as pd
import pickle
import random
import nltk
import tqdm

from collections import Counter
from src.gcndata import Vocabulary
from sentence_transformers import SentenceTransformer



def main():
    path = 'data/CUB-DG/'
    seed = 0        ### for debugging
    description, data = make_description(path, seed)

    print("Making a vocab file ...")

    #vocab = build_vocab(description)
    texts, tokenized_texts, vocab = build_vocab(description)
    #Vocabulary.save(vocab, os.path.join(path, "vocab_nlk_gcn.pkl"))

    print("Embedding texts ...")
    embed_texts(path, texts, tokenized_texts, vocab)
    
    print("Saved the vocab file")


def make_description(path, seed):
    print("Making a description file ...")
    
    description = get_data(path)                                # image, text 짝을 짓고
    data = split_description(seed, description)     # split을 통해서 test, val, eval 나눔
    #data.to_pickle(os.path.join(path, 'cub_dggcn_data.pkl'))

    split_info = get_split_index(data)
    #with open(os.path.join(path, 'split_info.pkl'),'wb') as f:
    #    pickle.dump(split_info, f)

    gcn_train = make_gcn_description(data)
    #gcn_train.to_pickle(os.path.join(path, 'cub_dggcn_train.pkl'))
    
    print("Saved the description files")

    return description, data


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

    descriptions = pd.DataFrame({"category_ids": category_ids, "categories": categories, "images": images, "captions": captions, "train_flag": split})

    return descriptions


def split_description(seed, description):
    '''
    1. train dataset을 3개의 group으로 나눔
    2. 각 group은 대게로, 하나의 class 당 약 10개의 data 존재
    3. train:val 비율을 8:2
    '''
    random.seed(seed)
    SPLIT_TRAIN_VAL = int(0.2 * 10)

    train = description.loc[description.train_flag == 1].reset_index(drop=True)
    test = description.loc[description.train_flag == 0].reset_index(drop=True)
    
    train["split"] = [0 for _ in range(len(train))]
    ###
    offset = 0
    for i in range(200):
        l = len(train.loc[train.category_ids == i])                 # i class 갯수
        shuffle_idcs = random.sample(range(offset, offset+l), l)    # i class 순서 셔플
        offset += l

        in_offset, split_num = 0, 0
        for i in range(3, 0, -1):
            ans = l // i
            l -= ans

            train.iloc[shuffle_idcs[in_offset:in_offset+ans], -1] = split_num

            split_num += 1
            in_offset += ans
    
    # train -> train-val
    split0 = train.loc[train.split == 0].reset_index(drop=True)
    split1 = train.loc[train.split == 1].reset_index(drop=True)
    split2 = train.loc[train.split == 2].reset_index(drop=True)
    split = [split0, split1, split2]
    
    for i in range(3):
        offset = 0
        for c in range(200):
            l = len(split[i].loc[split[i].category_ids == c])
            val_idcs = random.sample(range(l), int(0.2 * l))
            for v in val_idcs:
                split[i].iloc[offset+v, -2] = 2     # validation: train_flag = 2
            offset += l

    train = pd.concat(split).reset_index(drop=True)
    test["split"] = [-1 for _ in range(len(test))]

    data = pd.concat([train, test]).reset_index(drop=True)

    return data


def get_split_index(data):
    split0_train = data.loc[(data['split'] == 0) & (data['train_flag'] == 1)].index
    split0_val = data.loc[(data['split'] == 0) & (data['train_flag'] == 2)].index
    split1_train = data.loc[(data['split'] == 1) & (data['train_flag'] == 1)].index
    split1_val = data.loc[(data['split'] == 1) & (data['train_flag'] == 2)].index
    split2_train = data.loc[(data['split'] == 2) & (data['train_flag'] == 1)].index
    split2_val = data.loc[(data['split'] == 2) & (data['train_flag'] == 2)].index
    split_test = data[data.split == -1].index

    split_info = {"split0_train": split0_train, "split0_val": split0_val, \
        "split1_train": split1_train, "split1_val": split1_val, \
            "split2_train": split2_train, "split2_val": split2_val, \
                "split_test": split_test}
    
    return split_info


def make_gcn_description(data):       # ['category_ids', 'categories', 'images', 'captions', 'train_flag', 'split']
    split0_train = data.loc[(data['split'] == 0) & (data['train_flag'] == 1), ["category_ids", "categories", "images", "captions"]]
    split0_train = split0_train.sort_values(by=['images']).rename(columns={"images": "images0", "captions": "captions0"}).reset_index(drop=True)
    split1_train = data.loc[(data['split'] == 1) & (data['train_flag'] == 1), ["images", "captions"]]
    split1_train = split1_train.sort_values(by=['images']).rename(columns={"images": "images1", "captions": "captions1"}).reset_index(drop=True)
    split2_train = data.loc[(data['split'] == 2) & (data['train_flag'] == 1), ["images", "captions"]]
    split2_train = split2_train.sort_values(by=['images']).rename(columns={"images": "images2", "captions": "captions2"}).reset_index(drop=True)

    train = pd.concat([split0_train, split1_train], axis=1)
    train = pd.concat([train, split2_train], axis=1)

    return train
    


'''def build_vocab(description, threshold=1):
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
    return vocab'''



def embed_texts(path, texts, tokenized_texts, vocab):
    model = SentenceTransformer('clip-ViT-B-32')

    texts_s, texts_w, lengths = [], [], []
    for text in texts:
        try:
            e = model.encode(text)
        except:
            e = model.encode(".".join(text.split(".")[:-2]))
        texts_s.append(e)

    MaxLength = 64 if ("CUB-DG" in path or "domain_net" in path) else 32
    for text in tokenized_texts:
        lengths.append(len(text) + 1)
        if len(text) < MaxLength - 2:
            text += ["<end>"] * (MaxLength - 2 - len(text))
        elif len(text) > MaxLength - 2:
            text = text[:MaxLength - 2]
        texts_w.append(np.array([vocab(vocab.start_token)] + [vocab(word) for word in text] + [vocab(vocab.end_token)]))

    if not os.path.exists(os.path.join(path, "texts/")):
        os.makedirs(os.path.join(path, "texts/"), exist_ok=True)
        np.save(os.path.join(path, "texts", "texts_s.npy"), np.stack(texts_s, 0))
        np.save(os.path.join(path, "texts", "texts_w.npy"), np.stack(texts_w, 0))
        np.save(os.path.join(path, "texts", "lengths.npy"), np.stack(lengths, 0))

def build_vocab(description, threshold=1):
    """Build a simple vocabulary wrapper."""
    
    texts, tokenized_texts = [], []
    counter = Counter()
    for i in range(len(description)):
        for line in description['captions'][i]:
            ###
            texts.append(line)
            tokens = nltk.tokenize.word_tokenize(line.strip().lower())

            #if len(line) < 10: continue
            #tokens = nltk.tokenize.word_tokenize(line.strip().lower())
            
            ###
            tokenized_texts.append(tokens)
            counter.update(tokens)
    
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    for word in words:
        vocab.add_word(word)

    return texts, tokenized_texts, vocab



if __name__ == "__main__":
    main()
