from collections import OrderedDict, Counter
from re import L

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src.networks import ResNet, Explainer, SentenceClassifier, GCNNet

import os
import numpy as np
import random



MODELS = ['GVE', 'ERM', 'ERM_GVE', 'GCN']


def get_model(model_cfg, vocab, num_classes=200):
    """ get model supporting different model types """
    if model_cfg.name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(model_cfg.name))
    return globals()[model_cfg.name](model_cfg, vocab, num_classes)


def get_optimizer(params):
    """ configure optim and scheduler """
    LEARNING_RATE = 5e-5
    WEIGHT_DECAY = 0
    optimizer = torch.optim.Adam(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    return optimizer



class GVE(torch.nn.Module):
    """ Generating Visual Explanations (GVE) """
    def __init__(self, model_cfg, vocab, num_classes):
        super(GVE, self).__init__()
        
        self.featurizer = ResNet(model_cfg.attn)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        
        self.explainer = Explainer(model_cfg, vocab, num_classes, self.featurizer.n_outputs)
        self.sentence_classifier = SentenceClassifier(model_cfg, vocab, num_classes)
        
        ### for Projection
        embed_size, hidden_size, perceptron_size = model_cfg.explainer_embed_size, model_cfg.explainer_hidden_size, model_cfg.perceptron_size
        self.img_perceptron = nn.Linear(self.featurizer.n_outputs, perceptron_size)      # 512
        self.txt_perceptron = nn.Linear(embed_size, perceptron_size)

        #self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "sd_loss", "ed_loss"]
        self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss"]

        ### for Attention
        self.attn = model_cfg.attn
        if self.attn:
            self.query_conv = nn.Conv2d(in_channels = 2048, out_channels = 2048//8, kernel_size=1)
            self.key_conv  = nn.Conv2d(in_channels = 2048, out_channels = 2048//8, kernel_size=1) 
            self.value_conv   = nn.Conv2d(in_channels = 2048, out_channels = 2048, kernel_size=1)         
            self.gamma = nn.Parameter(torch.zeros(1))
            self.softmax  = nn.Softmax(dim=-1)
            self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.optimizer = get_optimizer(self.parameters())

    def add_attention(self, image_before_pooling):
        n_batchsize, n_channel, width, height = image_before_pooling.size()

        proj_query  = self.query_conv(image_before_pooling).view(n_batchsize,-1,width*height).permute(0,2,1)
        proj_key =  self.key_conv(image_before_pooling).view(n_batchsize,-1,width*height)
        energy =  torch.bmm(proj_query,proj_key)
        attention = self.softmax(energy)
                
        proj_value = self.value_conv(image_before_pooling).view(n_batchsize,-1,width*height)
        
        out = torch.bmm(proj_value,attention.permute(0,2,1))        # (32, 2048, 196)
        out = out.view(n_batchsize,n_channel,width,height)          # (32, 2048, 14, 14)
        sum_out = self.gamma*out + image_before_pooling             # (32, 2048, 14, 14)
        image_features = self.avgpool(sum_out).squeeze()            # (32, 2048)
        
        return image_features

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, tis, tts, ls , mls, fs = minibatch

        del xs[test_env]
        del tis[test_env]
        del tts[test_env]
        del ls[test_env]
        del mls[test_env]
        del fs[test_env]
        
        num_domains = len(xs)

        cls_loss, rel_loss, dis_loss, sd_loss, ed_loss = 0, 0, 0, 0, 0
        for i in range(num_domains):
            x, ti, tt, l, ml, f = xs[i], tis[i], tts[i], ls[i], mls[i], fs[i]
            
            if self.attn:
                image_before_pooling = self.featurizer(x)
                image_features = self.add_attention(image_before_pooling)
            else:
                image_features = self.featurizer(x)

            cls_outputs = self.classifier(image_features)
            cls_loss += F.cross_entropy(cls_outputs, y)
            
            l = l.cpu()
            exp_outputs = self.explainer(ti, l, ml, image_features, cls_outputs)
            tt = pack_padded_sequence(tt, l, batch_first=True, enforce_sorted=False)
            rel_loss += F.cross_entropy(exp_outputs, tt[0])

            sampled_t, log_ps, sampled_l, states1 = self.explainer.module.sample(image_features, cls_outputs)
            sc_outputs = self.sentence_classifier(sampled_t, sampled_l)
            rewards = F.softmax(sc_outputs, dim=1).gather(1, y.view(-1, 1)).squeeze()
            dis_loss += -(log_ps.sum(dim=1) * rewards).sum() / len(y)

            ### sd loss
            #sd_loss += 0.1 * (cls_outputs ** 2).mean()

            ### projection loss
            #text_features = torch.squeeze(states1[0])
            #txt_features = self.txt_perceptron(text_features)
            #img_features = self.img_perceptron(image_features)
            #img_features = img_features / img_features.norm(dim=1, keepdim=True)
            #txt_features = txt_features / txt_features.norm(dim=1, keepdim=True)
            #ed_loss += F.mse_loss(img_features, txt_features)
        
        cls_loss /= num_domains
        rel_loss /= num_domains
        dis_loss /= num_domains
        #sd_loss /= num_domains
        #ed_loss /= num_domains

        loss = cls_loss + rel_loss + dis_loss #+ sd_loss + ed_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "rel_loss": rel_loss, "dis_loss": dis_loss})#, "sd_loss": sd_loss, "ed_loss": ed_loss})

    def evaluate(self, minibatch, test_env):
        xs, y, tis, tts, ls , mls, fs = minibatch
        x = xs[test_env]

        f = fs[test_env]

        with open("in_evaluate_%s.txt" %str(test_env), "a") as file:
            for fn in f:
                file.write(fn + '\n')

        if self.attn:
            image_before_pooling = self.featurizer(x)
            image_features = self.add_attention(image_before_pooling)
        else:
            image_features = self.featurizer(x)

        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))
        
        return correct, total
    
    def set_parallel(self, data_parallel):
        if data_parallel:
            self.featurizer = nn.DataParallel(self.featurizer)
            self.classifier = nn.DataParallel(self.classifier)

            self.explainer = nn.DataParallel(self.explainer)
            self.sentence_classifier = nn.DataParallel(self.sentence_classifier)

            self.img_perceptron = nn.DataParallel(self.img_perceptron)
            self.txt_perceptron = nn.DataParallel(self.txt_perceptron)



class GCN(torch.nn.Module):
    def __init__(self, model_cfg, vocab, num_classes, n_masking=2):
        super(GCN, self).__init__()
        
        self.featurizer = ResNet(model_cfg.attn)
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)
        
        self.explainer = Explainer(model_cfg, vocab, num_classes, self.featurizer.n_outputs)
        self.sentence_classifier = SentenceClassifier(model_cfg, vocab, num_classes)

        ### for Projection
        embed_size, hidden_size, perceptron_size = model_cfg.explainer_embed_size, model_cfg.explainer_hidden_size, model_cfg.perceptron_size
        self.img_perceptron = nn.Linear(self.featurizer.n_outputs, perceptron_size)      # 512
        self.txt_perceptron = nn.Linear(embed_size, perceptron_size)
        
        ### for gcn
        self.n_masking = n_masking
        self.gcn = GCNNet(n_block=1, n_layer=1, in_dim=1024, hidden_dim=512, out_dim=256, n_feat=6)

        self.gcn_classifier = nn.Linear(256, num_classes)   # gcn의 out_dim

        #self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "sd_loss", "ed_loss"]
        self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "gcn_cls_loss", "gcn_feat_loss"]

        self.optimizer = get_optimizer(self.parameters())
    
    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, tis, tts, ls , mls, fs = minibatch

        del xs[test_env]
        del tis[test_env]
        del tts[test_env]
        del ls[test_env]
        del mls[test_env]
        del fs[test_env]
        
        num_domains = len(xs)
        
        remove_idx = random.sample([0, 1, 2, 3, 4, 5], self.n_masking)
        feat, masking_feat = None, None

        cls_loss, rel_loss, dis_loss, sd_loss, gcn_cls_loss, gcn_feat_loss = 0, 0, 0, 0, 0, 0
        for i in range(num_domains):
            x, ti, tt, l, ml, f = xs[i], tis[i], tts[i], ls[i], mls[i], fs[i]
            
            image_features = self.featurizer(x)     # (B, 2048)
            
            cls_outputs = self.classifier(image_features)
            cls_loss += F.cross_entropy(cls_outputs, y)
            
            l = l.cpu()
            exp_outputs = self.explainer(ti, l, ml, image_features, cls_outputs)
            tt = pack_padded_sequence(tt, l, batch_first=True, enforce_sorted=False)
            rel_loss += F.cross_entropy(exp_outputs, tt[0])

            sampled_t, log_ps, sampled_l, states1 = self.explainer.module.sample(image_features, cls_outputs)      # states1[0]: (1, B, 1024)
            sc_outputs = self.sentence_classifier(sampled_t, sampled_l)
            rewards = F.softmax(sc_outputs, dim=1).gather(1, y.view(-1, 1)).squeeze()
            dis_loss += -(log_ps.sum(dim=1) * rewards).sum() / len(y)

            #sd_loss += 0.1 * (cls_outputs ** 2).mean()

            ###
            img_features = self.img_perceptron(image_features)              # (32, perceptron_size)
            txt_features = self.txt_perceptron(torch.squeeze(states1[0]))   # (32, perceptron_size)

            ### make gcn features without maksing
            if feat == None:
                feat = img_features
            if feat.shape == img_features.shape:
                feat = torch.stack([feat, txt_features], dim=1)
            else:
                feat = torch.cat([feat, torch.unsqueeze(img_features, dim=1)], dim=1)
                feat = torch.cat([feat, torch.unsqueeze(txt_features, dim=1)], dim=1)

        # 방법 1. adjaceney matrix에 random masking -> GCN
        adj = torch.ones(feat.shape[0], 6, 6).cuda().float()
        random_adj = torch.randint(2, (feat.shape[0], 6, 6)).cuda().float()
        for i in range(feat.shape[1]):
            random_adj[:, i, i] = 1
        out = self.gcn(feat, adj)
        masking_out = self.gcn(feat, random_adj)

        # 방법 2. feature 자체를 random masking -> GCN2
        #adj = torch.ones(6, 6).cuda().float()      # adj의 shape이 [6, 6]이 맞나? [32, 6, 6] 되어야 할 것 같아서..
        #out = self.gcn(feat, adj)
        #masking_feat = feat.clone()
        #for i in range(feat.shape[0]):
        #    masking_idcs = random.sample(range(6), 2)
        #    for mi in masking_idcs:
        #        masking_feat[i, mi] = 0.
        #masking_out = self.gcn(feat, adj)


        gcn_cls_out = self.gcn_classifier(out)
        gcn_cls_out2 = self.gcn_classifier(masking_out)

        gcn_cls_loss = (F.cross_entropy(gcn_cls_out, y) + F.cross_entropy(gcn_cls_out2, y)) / 2
        gcn_feat_loss = F.mse_loss(gcn_cls_out, gcn_cls_out2)
        
        cls_loss /= num_domains
        rel_loss /= num_domains
        dis_loss /= num_domains

        loss = cls_loss + rel_loss + dis_loss + gcn_cls_loss + gcn_feat_loss#+ sd_loss + ed_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "rel_loss": rel_loss, "dis_loss": dis_loss, "gcn_cls_loss": gcn_cls_loss, "gcn_feat_loss": gcn_feat_loss })#, "sd_loss": sd_loss, "ed_loss": ed_loss})
    
    def evaluate(self, minibatch, test_env):
        xs, y, tis, tts, ls , mls, fs = minibatch
        x = xs[test_env]

        image_features = self.featurizer(x)

        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))
        
        return correct, total
    
    def set_parallel(self, data_parallel):
        if data_parallel:
            self.featurizer = nn.DataParallel(self.featurizer)
            self.classifier = nn.DataParallel(self.classifier)

            self.explainer = nn.DataParallel(self.explainer)
            self.sentence_classifier = nn.DataParallel(self.sentence_classifier)

            self.img_perceptron = nn.DataParallel(self.img_perceptron)
            self.txt_perceptron = nn.DataParallel(self.txt_perceptron)

            self.gcn = nn.DataParallel(self.gcn)
            self.gcn_classifier = nn.DataParallel(self.gcn_classifier)



###
class ERM(torch.nn.Module):
    """ Empirical Risk Minimization (ERM) """
    def __init__(self, model_cfg, vocab, num_classes):
        super(ERM, self).__init__()
        self.featurizer = ResNet()
        self.classifier = nn.Linear(self.featurizer.n_outputs, num_classes)

        self.loss_names = ["loss", "cls_loss"]
        self.optimizer = get_optimizer(self.parameters())

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, fs = minibatch

        del xs[test_env]
        del fs[test_env]

        num_domains = len(xs)

        cls_loss = 0
        for i in range(num_domains):
            x, f = xs[i], fs[i]

            image_features = self.featurizer(x)
            cls_outputs = self.classifier(image_features)
            cls_loss += F.cross_entropy(cls_outputs, y)
                
        cls_loss /= num_domains
        loss = cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return OrderedDict({'loss': loss, 'cls_loss': cls_loss})

    def evaluate(self, minibatch, test_env):
        xs, y, fs = minibatch
        x = xs[test_env]
        
        image_features = self.featurizer(x)
        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        return correct, total

    def set_parallel(self, data_parallel):
        if data_parallel:
            self.featurizer = nn.DataParallel(self.featurizer)
            self.classifier = nn.DataParallel(self.classifier)