from collections import OrderedDict, Counter
from re import L
from jax import mask

import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from src.networks import ResNet, Explainer, SentenceClassifier, GCNNet

import os
import numpy as np
import random



MODELS = ['GVE', 'ERM', 'GCN']


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
        
        self.featurizer = nn.DataParallel(ResNet(model_cfg.attn))
        self.classifier = nn.Linear(self.featurizer.module.n_outputs, num_classes)
        
        self.explainer = Explainer(model_cfg, vocab, num_classes, self.featurizer.module.n_outputs)
        self.sentence_classifier = SentenceClassifier(model_cfg, vocab, num_classes)
        
        ### for Projection
        embed_size, hidden_size, perceptron_size = model_cfg.explainer_embed_size, model_cfg.explainer_hidden_size, model_cfg.perceptron_size
        self.img_perceptron = nn.Linear(self.featurizer.module.n_outputs, perceptron_size)      # 512
        self.txt_perceptron = nn.Linear(embed_size, perceptron_size)

        self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "sd_loss"]#, "ed_loss"]
        #self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss"]

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

            sampled_t, log_ps, sampled_l, states1 = self.explainer.sample(image_features, cls_outputs)
            sc_outputs = self.sentence_classifier(sampled_t, sampled_l)
            rewards = F.softmax(sc_outputs, dim=1).gather(1, y.view(-1, 1)).squeeze()
            dis_loss += -(log_ps.sum(dim=1) * rewards).sum() / len(y)

            ### sd loss
            sd_loss += 0.1 * (cls_outputs ** 2).mean()

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
        sd_loss /= num_domains
        #ed_loss /= num_domains

        loss = cls_loss + rel_loss + dis_loss + sd_loss #+ ed_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "rel_loss": rel_loss, "dis_loss": dis_loss, "sd_loss": sd_loss})#, "ed_loss": ed_loss})

    def evaluate(self, minibatch, test_env):
        x, y, f = minibatch

        if self.attn:
            image_before_pooling = self.featurizer(x)
            image_features = self.add_attention(image_before_pooling)
        else:
            image_features = self.featurizer(x)

        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))
        
        return correct, total


class PAIR(torch.nn.Module):
    def __init__(self, model_cfg, vocab, num_classes):
        super(GCN, self).__init__()
        
        self.featurizer = nn.DataParallel(ResNet(model_cfg.attn))
        self.classifier = nn.Linear(self.featurizer.module.n_outputs, num_classes)
        
        #self.explainer = Explainer(model_cfg, vocab, num_classes, self.featurizer.module.n_outputs)
        #self.sentence_classifier = SentenceClassifier(model_cfg, vocab, num_classes)

        ### for Projection
        embed_size, hidden_size, perceptron_size = model_cfg.explainer_embed_size, model_cfg.explainer_hidden_size, model_cfg.perceptron_size
        self.img_perceptron = nn.Linear(self.featurizer.module.n_outputs, perceptron_size)      # 512
        #self.txt_perceptron = nn.Linear(embed_size, perceptron_size)
        
        ### for gcn
        self.gcn = GCNNet(n_block=1, n_layer=1, in_dim=1024, hidden_dim=512, out_dim=256, n_feat=6)
        self.gcn_classifier = nn.Linear(256, num_classes)   # gcn의 out_dim

        ### for contrastive-loss
        self.margin = 10
        self.sample_num = 4

        #self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "sd_loss"]#, "ed_loss"]
        #self.loss_names = ["loss", "cls_loss", "rel_loss", "dis_loss", "gcn_cls_loss", "gcn_feat_loss"]
        #self.loss_names = ["loss", "cls_loss", "gcn_cls_loss", "pair_loss"]
        self.loss_names = ["loss", "cls_loss", "pair_loss", "gcn_feat_loss"]

        self.optimizer = get_optimizer(self.parameters())
    
    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, tis, tts, ls , mls, fs = minibatch

        num_domains = len(xs)
        
        features = None
        cls_loss, rel_loss, dis_loss, sd_loss, gcn_cls_loss, gcn_feat_loss, pair_loss = 0, 0, 0, 0, 0, 0, 0
        for i in range(num_domains):
            x, ti, tt, l, ml, f = xs[i], tis[i], tts[i], ls[i], mls[i], fs[i]
            
            image_features = self.featurizer(x)     # (B, 2048)
            
            cls_outputs = self.classifier(image_features)
            cls_loss += F.cross_entropy(cls_outputs, y)
            
            #l = l.cpu()
            #exp_outputs = self.explainer(ti, l, ml, image_features, cls_outputs)
            #tt = pack_padded_sequence(tt, l, batch_first=True, enforce_sorted=False)
            #rel_loss += F.cross_entropy(exp_outputs, tt[0])

            #sampled_t, log_ps, sampled_l, states1 = self.explainer.sample(image_features, cls_outputs)      # states1[0]: (1, B, 1024)
            #sc_outputs = self.sentence_classifier(sampled_t, sampled_l)
            #rewards = F.softmax(sc_outputs, dim=1).gather(1, y.view(-1, 1)).squeeze()
            #dis_loss += -(log_ps.sum(dim=1) * rewards).sum() / len(y)

            #sd_loss += 0.1 * (cls_outputs ** 2).mean()

            ###
            img_features = self.img_perceptron(image_features)              # (32, perceptron_size)
            #txt_features = self.txt_perceptron(torch.squeeze(states1[0]))   # (32, perceptron_size)

            ### make gcn features without maksing
            if features == None:
                features = img_features.unsqueeze(dim=1)
            else:
                features = torch.cat([features, torch.unsqueeze(img_features, dim=1)], dim=1)

        ### reshape features (2개 이상의 class 씩 묶을 수 있게)
        #print(features[1][0])
        reshape_features = features.reshape(features.shape[0]//2, -1, features.shape[2])        # [16, 6, 1024]
        #print(features[0][3])
        
        ### make adjacent matrix
        adj = torch.ones(reshape_features.shape[0], reshape_features.shape[1], reshape_features.shape[1]).cuda().float()
        for i in range(len(reshape_features)):
            if y[2*i] != y[2*i + 1]:
                adj[i][0][4] = 0
                adj[i][0][5] = 0
                adj[i][1][3] = 0
                adj[i][1][5] = 0
                adj[i][2][3] = 0
                adj[i][2][4] = 0

                adj[i][4][0] = 0
                adj[i][5][0] = 0
                adj[i][3][1] = 0
                adj[i][5][1] = 0
                adj[i][3][2] = 0
                adj[i][4][2] = 0
        
        masking_adj = torch.randint(2, (adj.shape)).cuda().float()
        for i in range(reshape_features.shape[1]):
            masking_adj[:, i, i] = 1
        
        ##########################################################################################
        # features.shape = [32, n_features, 1024]
        # 방법 1. adjaceney matrix에 random masking -> GCN
        #adj = torch.ones(features.shape[0], 3, 3).cuda().float()
        #random_adj = torch.randint(2, (features.shape[0], 3, 3)).cuda().float()
        #for i in range(features.shape[1]):
        #    random_adj[:, i, i] = 1
        #gcn_features = self.gcn(features, adj)                       # gcn_features.shape = [32, 256]
        #masking_gcn_features = self.gcn(features, random_adj)        # masking_gcn_features.shape = [32, 256]

        # 방법 2. feature 자체를 random masking -> GCN2
        #adj = torch.ones(features.shape[0], 3, 3).cuda().float()
        #gcn_features = self.gcn(features, adj)
        #masking_feat = features.clone()
        #for i in range(features.shape[0]):
        #    masking_idcs = random.sample(range(3), 1)
        #    for mi in masking_idcs:
        #        masking_feat[i, mi] = 0.
        #masking_gcn_features = self.gcn(features, adj)
        ##########################################################################################

        gcn_features = self.gcn(reshape_features, adj)
        maksing_gcn_features = self.gcn(reshape_features, masking_adj)
        gcn_feat_loss = F.mse_loss(maksing_gcn_features, gcn_features)

        #gcn_cls_out = self.gcn_classifier(gcn_features)
        #gcn_cls_loss = F.cross_entropy(gcn_cls_out, y)

        '''
        ### applying pair loss to gcn features
        random_idx = random.sample(range(32), 4)
        for i in range(len(random_idx)-1):
            for j in range(i+1, len(random_idx)):
                g1 = gcn_features[i]
                g2 = gcn_features[j]
                
                # normalization
                g1_var_mean = torch.var_mean(g1)
                g2_var_mean = torch.var_mean(g2)
                g1 = ((g1 - g1_var_mean[1]) / g1_var_mean[0]**(0.5)).reshape(1, -1)
                g2 = ((g2 - g2_var_mean[1]) / g2_var_mean[0]**(0.5)).reshape(1, -1)
                
                diff = F.mse_loss(g1, g2)

                if y[random_idx[i]] != y[random_idx[j]]:        # negative-pair
                    pair_loss += F.relu(self.margin - diff)
                else:                                           # positive-pair
                    pair_loss += diff
        print(pair_loss)'''

        random_idx = random.sample(range(32), 4)
        random_feat = features[random_idx]
        random_feat = (random_feat - random_feat.mean(dim=0)) / random_feat.var(dim=0)**0.5     # normalization     [4, 3, 1024] - class마다 정규화

        ### applying pair loss to image features
        # positive loss
        for i in range(len(random_idx)):
            diff = F.mse_loss(random_feat[i][0], random_feat[i][1])
            diff += F.mse_loss(random_feat[i][0], random_feat[i][2])
            diff += F.mse_loss(random_feat[i][1], random_feat[i][2])
            pair_loss += diff
        # negative loss
        for i in range(len(random_idx)-1):
            for j in range(i+1, len(random_idx)):
                if y[random_idx[i]] != y[random_idx[j]]:
                    diff = F.mse_loss(random_feat[i][0], random_feat[j][0])
                    diff += F.mse_loss(random_feat[i][0], random_feat[j][1])
                    diff += F.mse_loss(random_feat[i][0], random_feat[j][2])
                    diff += F.mse_loss(random_feat[i][1], random_feat[j][0])
                    diff += F.mse_loss(random_feat[i][1], random_feat[j][1])
                    diff += F.mse_loss(random_feat[i][1], random_feat[j][2])
                    diff += F.mse_loss(random_feat[i][2], random_feat[j][0])
                    diff += F.mse_loss(random_feat[i][2], random_feat[j][1])
                    diff += F.mse_loss(random_feat[i][2], random_feat[j][2])
                    pair_loss += F.relu(self.margin - diff)
        
        cls_loss /= num_domains
        pair_loss *= 1e-1
        gcn_feat_loss *= 1e2
        #rel_loss /= num_domains
        #dis_loss /= num_domains

        loss = cls_loss + pair_loss + gcn_feat_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        #return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "rel_loss": rel_loss, "dis_loss": dis_loss, "gcn_cls_loss": gcn_cls_loss, "gcn_feat_loss": gcn_feat_loss, "sd_loss": sd_loss})#, "ed_loss": ed_loss})
        #return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "gcn_cls_loss": gcn_cls_loss, "pair_loss": pair_loss})
        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "pair_loss": pair_loss, "gcn_feat_loss": gcn_feat_loss})
    
    def evaluate(self, minibatch, test_env):
        x, y, f = minibatch

        image_features = self.featurizer(x)

        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))
        
        return correct, total



class StyleFeature(torch.nn.Module):
    def __init__(self, model_cfg, vocab, num_classes):
        super(StyleFeature, self).__init__()
        self.featurizer = nn.DataParallel(ResNet(model_cfg.attn))       # output.shape = (B, 2048)
        self.classifier = nn.Linear(self.featurizer.module.n_outputs, num_classes)

        ###
        # DG: source domain은 3개
        self.style_featurizer = nn.Sequential(
                                            nn.Linear(self.featurizer.module.n_outputs, self.featurizer.module.n_outputs // 2),
                                            nn.ReLU(),
                                            nn.Linear(self.featurizer.module.n_outputs // 2, self.featurizer.module.n_outputs))
        self.style_classifier = nn.Linear(self.featurizer.module.n_outputs, 3)
        ### linear extractor 말고 conv...

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.loss_names = ["loss", "cls_loss", "style_cls_loss"]
        self.optimizer = get_optimizer(self.parameters())

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, fs = minibatch

        num_domains = len(xs)
        style_y = []
        cls_loss, style_cls_loss = 0, 0
        for i in range(num_domains):
            x, f = xs[i], fs[i]
            s_y = [i] * (x.shape[0])
            #style_y += s_y
            s_y = torch.tensor(s_y).to(self.device)

            image_features = self.featurizer(x)
            style_features = self.style_featurizer(image_features)

            style_cls_outputs = self.style_classifier(style_features)
            style_cls_loss += F.cross_entropy(style_cls_outputs, s_y)

            cls_outputs = self.classifier(image_features-style_features)    # only use class feature
            cls_loss += F.cross_entropy(cls_outputs, y)
        
        #feature_y = torch.tensor(feature_y)
        

        cls_loss /= num_domains
        style_cls_loss /= num_domains
        loss = cls_loss + style_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return OrderedDict({'loss': loss, 'cls_loss': cls_loss, "style_cls_loss": style_cls_loss})

    def evaluate(self, minibatch, test_env):
        x, y, f = minibatch

        image_features = self.featurizer(x)
        style_features = self.style_featurizer(image_features)

        cls_outputs = self.classifier(image_features-style_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        return correct, total




###
class ERM(torch.nn.Module):
    """ Empirical Risk Minimization (ERM) """
    def __init__(self, model_cfg, vocab, num_classes):
        super(ERM, self).__init__()
        self.featurizer = nn.DataParallel(ResNet(model_cfg.attn))
        self.classifier = nn.Linear(self.featurizer.module.n_outputs, num_classes)

        self.loss_names = ["loss", "cls_loss"]
        self.optimizer = get_optimizer(self.parameters())

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, fs = minibatch

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
        x, y, f = minibatch

        image_features = self.featurizer(x)
        cls_outputs = self.classifier(image_features)

        correct = (cls_outputs.argmax(1).eq(y).float()).sum().item()
        total = float(len(x))

        return correct, total



class CORAL(ERM):
    """ ERM while matching the pair-wise domain feature distributions using mean and covariance difference """
    def __init__(self, model_cfg, vocab, num_classes):
        super(CORAL, self).__init__(model_cfg, vocab, num_classes)
        self.loss_names = ["loss", "cls_loss", "mmd_loss"]

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, fs = minibatch

        num_domains = len(xs)
        num_domains_pair = (num_domains * (num_domains - 1) / 2)
        x_mb = []

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for i in range(num_domains):
            x, f = xs[i], fs[i]

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["cls_loss"] += F.cross_entropy(y_hat, y) / num_domains
            x_mb.append(x)

        for i in range(num_domains):
            for j in range(i + 1, num_domains):
                loss_dict["mmd_loss"] += self.mmd(x_mb[i], x_mb[j]) / num_domains_pair

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def mmd(self, x, y):
        mean_x = x.mean(0, keepdim=True)
        mean_y = y.mean(0, keepdim=True)
        cent_x = x - mean_x
        cent_y = y - mean_y
        cova_x = (cent_x.t() @ cent_x) / (len(x) - 1)
        cova_y = (cent_y.t() @ cent_y) / (len(y) - 1)

        mean_diff = (mean_x - mean_y).pow(2).mean()
        cova_diff = (cova_x - cova_y).pow(2).mean()

        return mean_diff + cova_diff


class SD(ERM):
    """ Gradient starvation: a learning proclivity in neural networks """
    def __init__(self, model_cfg, vocab, num_classes):
        super(SD, self).__init__(model_cfg, vocab, num_classes)
        self.loss_names = ["loss", "cls_loss", "sd_loss"]

    def update(self, minibatch, test_env):
        minibatch = minibatch[0]
        xs, y, fs = minibatch

        num_domains = len(xs)

        loss_dict = OrderedDict({loss_name: 0 for loss_name in self.loss_names})
        for i in range(num_domains):
            x, f = xs[i], fs[i]

            x = self.featurizer(x)
            y_hat = self.classifier(x)
            loss_dict["cls_loss"] += F.cross_entropy(y_hat, y) / num_domains
            loss_dict["sd_loss"] += 0.1 * (y_hat ** 2).mean() / num_domains

        for loss_name in self.loss_names:
            if loss_name != "loss":
                loss_dict["loss"] += loss_dict[loss_name]

        self.optimizer.zero_grad()
        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

