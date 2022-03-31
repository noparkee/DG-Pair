import torch
import torch.nn as nn
import torchvision.models
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResNet(torch.nn.Module):
    """ ResNet with the softmax chopped off and the batchnorm frozen """
    def __init__(self, attn=False):
        super(ResNet, self).__init__()
        self.network = torchvision.models.resnet50(pretrained=True)
        self.n_outputs = 2048
        
        del self.network.fc
        self.network.fc = Identity()

        self.attn = attn
        if self.attn:
            del self.network.avgpool
            self.network.avgpool = Identity()
            
        # resnet50 = list(torchvision.models.resnet50(pretrained=True).children())
        # self.network = nn.Sequential(*resnet50[:-2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.n_outputs = 2048

        self.freeze_bn()

    def forward(self, x):
        """ encode x into a feature vector of size n_outputs """
        if self.attn:
            return self.network(x).view(len(x), -1, 14, 14)        # (batch_size, 401408) --> (batch_size, 2048, 14, 14)
        else:
            return self.network(x)

    def train(self, mode=True):
        """ override the default train() to freeze the BN parameters """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class Identity(nn.Module):
    """ identity layer """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Explainer(torch.nn.Module):
    """ LSTM-based Explainer based on image_features and classifier outputs """
    def __init__(self, model_cfg, vocab, num_classes, image_feature_size):
        super(Explainer, self).__init__()
        embed_size, hidden_size = model_cfg.explainer_embed_size, model_cfg.explainer_hidden_size
        self.image_embed = nn.Linear(image_feature_size, hidden_size)
        self.text_embed = nn.Embedding(len(vocab), embed_size)
        self.lstm1 = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size*2 + num_classes, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, len(vocab))

        self.start_word = torch.tensor([vocab('<start>')], dtype=torch.long)
        self.end_word = torch.tensor([vocab('<end>')], dtype=torch.long)

    def forward(self, x, l, ml, image_features, cls_outputs):
        """ generate explanation sentences from image_features and cls_outputs with teacher forcing """
        l = l.cpu()
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()

        image_features = self.image_embed(image_features).unsqueeze(1).expand(-1, x.size(1), -1)
        cls_outputs = cls_outputs.unsqueeze(1).expand(-1, x.size(1), -1)
        
        x = self.text_embed(x)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm1(x)
        x, _ = pad_packed_sequence(x, total_length=ml[0], batch_first=True)

        x = torch.cat((x, image_features, cls_outputs), 2)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm2(x)
        x = self.linear(x[0])

        return x

    def sample(self, image_features, cls_outputs, max_length=80):
        image_features = self.image_embed(image_features).unsqueeze(1)
        cls_outputs = cls_outputs.unsqueeze(1)

        word = self.text_embed(self.start_word.to(image_features.device)).unsqueeze(0)
        word = word.expand(image_features.size(0), -1, -1)

        end_word = self.end_word.to(image_features.device).squeeze().expand(image_features.size(0))
        reached_end = torch.zeros_like(end_word.data).bool()
        lengths = torch.zeros_like(end_word.data).long()

        sampled_ids, log_ps = [], []
        states1, states2 = None, None
        for i in range(max_length):
            if reached_end.all(): break
            x, states1 = self.lstm1(word, states1)
            x = torch.cat((x, image_features, cls_outputs), 2)
            x, states2 = self.lstm2(x, states2)
            x = self.linear(x.squeeze(1))

            prob = Categorical(logits=x)
            sampled_id = prob.sample()
            log_p = prob.log_prob(sampled_id) * (~reached_end).float()

            sampled_ids.append(sampled_id)
            log_ps.append(log_p)
            lengths += (~reached_end).long()

            reached_end = reached_end | sampled_id.eq(end_word).data
            word = self.text_embed(sampled_id).unsqueeze(1)

        sampled_ids = torch.stack(sampled_ids, 1)
        log_ps = torch.stack(log_ps, 1)
        lengths = lengths.cpu()

        ### 
        '''captions = []
        for i in range(len(sampled_ids)):
            caption = []
            for sampled_id in sampled_ids[i]:
                word = self.vocab.get_word_from_idx(sampled_id.data.item())
                if word == self.vocab('<end>'):         # self.vocab.end_token:
                    break
                elif word != self.vocab('<start>'):     #self.vocab.start_token: 
                    caption.append(word)   
            captions.append(' '.join(caption))

        return sampled_ids, log_ps, lengths, states1, captions'''

        return sampled_ids, log_ps, lengths, states2


class SentenceClassifier(torch.nn.Module):
    """ LSTM-based sentence classifier """
    def __init__(self, model_cfg, vocab, num_classes):
        super(SentenceClassifier, self).__init__()
        embed_size, hidden_size = model_cfg.sc_embed_size, model_cfg.sc_hidden_size
        self.text_embed = nn.Embedding(len(vocab), embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, num_classes)

    def forward(self, x, l):
        """ classify given sentences """
        l = l.cpu()
        self.lstm.flatten_parameters()

        x = self.text_embed(x)
        x = pack_padded_sequence(x, l, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(x)
        x, _ = pad_packed_sequence(x, batch_first=True)

        last_idxs = (l - 1).view(-1, 1, 1).expand(-1, -1, x.size(2))
        x = x.gather(1, last_idxs.to(x.device)).squeeze()
        x = self.linear(x)

        return x


'''GCN Network
    reference: https://github.com/heartcored98/Standalone-DeepLearning/blob/master/Lec9/Lab11_logP_Prediction_with_GCN.ipynb'''

class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_feat, act=None, bn=False):
        super(GCNLayer, self).__init__()
        self.use_bn = bn
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = act
        self.bn = nn.BatchNorm1d(n_feat)
        
    def forward(self, x, adj):
        out = self.linear(x)
        out = torch.matmul(adj, out)
        if self.use_bn:
            out = self.bn(out)
        if self.activation != None:
            out = self.activation(out)

        return out

class SkipConnection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        out = in_x + out_x
        return out

class GatedSkipConnection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GatedSkipConnection, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.linear_coef_in = nn.Linear(out_dim, out_dim)
        self.linear_coef_out = nn.Linear(out_dim, out_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, in_x, out_x):
        if (self.in_dim != self.out_dim):
            in_x = self.linear(in_x)
        z = self.gate_coefficient(in_x, out_x)
        out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
        return out
            
    def gate_coefficient(self, in_x, out_x):
        x1 = self.linear_coef_in(in_x)
        x2 = self.linear_coef_out(out_x)
        return self.sigmoid(x1+x2)

class GCNBlock(nn.Module):
    def __init__(self, n_layer, in_dim, hidden_dim, out_dim, n_feat, bn, sc='gsc'):
        super(GCNBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layer):
            self.layers.append(GCNLayer(in_dim if i==0 else hidden_dim,
                                        out_dim if i==n_layer-1 else hidden_dim,
                                        n_feat,
                                        nn.ReLU() if i!=n_layer-1 else None,
                                        bn))
        self.relu = nn.ReLU()
        if sc=='gsc':
            self.sc = GatedSkipConnection(in_dim, out_dim)
        elif sc=='sc':
            self.sc = SkipConnection(in_dim, out_dim)
        elif sc=='no':
            self.sc = None
        else:
            assert False, "Wrong sc type."
        
    def forward(self, x, adj):
        residual = x
        for i, layer in enumerate(self.layers):
            out = layer((x if i==0 else out), adj)
        if self.sc != None:
            out = self.sc(residual, out)
        out = self.relu(out)
        return out

class ReadOut(nn.Module):
    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()
        self.in_dim = in_dim
        self.out_dim= out_dim
        
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        self.activation = act

    def forward(self, x):
        out = self.linear(x)
        out = torch.sum(out, 1)
        if self.activation != None:
            out = self.activation(out)
        return out

class GCNNet(nn.Module):
    def __init__(self, n_block, n_layer, in_dim, hidden_dim, out_dim, n_feat=6, bn=True, sc='gsc'):
        super(GCNNet, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(n_block):
            self.blocks.append(GCNBlock(n_layer,
                                        in_dim if i==0 else hidden_dim,
                                        hidden_dim,
                                        hidden_dim,
                                        n_feat,
                                        bn,
                                        sc))
        self.readout = ReadOut(hidden_dim, 
                               out_dim,
                               act=nn.ReLU())
        
    def forward(self, x, adj):
        for i, block in enumerate(self.blocks):
            out = block((x if i==0 else out), adj)
        out = self.readout(out)

        return out
