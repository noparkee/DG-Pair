import os
import json
import numpy as np
from collections import OrderedDict

import torch

from src.utils import Print


class Trainer():
    """ train / eval helper class """
    def __init__(self, model, device, data_parallel):
        self.model = model.to(device)
        # self.model.set_parallel(data_parallel)
        self.device = device
        self.data_parallel = data_parallel

        # initialize logging parameters
        self.logger_train = Logger()
        self.logger_eval = Logger()

    def train(self, minibatches, test_env):
        # training of the model
        minibatches = set_device(minibatches, self.device)

        self.model.train()
        loss_dict = self.model.module.update(minibatches, test_env)

        # logging
        self.logger_train.loss_update(loss_dict)

    def evaluate(self, minibatch, test_env, name, text_flag=False, eval_flag=False):
        # evaluation of the model
        minibatch = set_device(minibatch, self.device)

        self.model.eval()
        with torch.no_grad():
            correct, total = self.model.module.evaluate(minibatch, test_env)

        self.logger_eval.acc_update({name: [correct, total]})
        
    def save_model(self, step, save_prefix):
        # save a state_dict to checkpoint """
        if save_prefix is None: return
        elif not os.path.exists(os.path.join(save_prefix, "checkpoints/")):
            os.makedirs(os.path.join(save_prefix, "checkpoints/"), exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_prefix, "checkpoints/%d.pt" % step))

    def load_model(self, checkpoint, output):
        # load a state_dict from checkpoint """
        if checkpoint is None: return
        Print('loading a model state_dict from the checkpoint', output)
        checkpoint = torch.load(checkpoint, map_location=self.device)
        state_dict = OrderedDict()
        for k, v in checkpoint.items(): 
            if not self.data_parallel:
                k = k.replace("module.", "")
            state_dict[k] = v
        self.model.load_state_dict(state_dict)

    def headline(self, idx, loss_names, eval_names, output):
        # get a headline for logging
        if idx == "step":
            headline = [idx] + loss_names + eval_names
        else:
            headline = [idx] + eval_names

        Print("\t".join(headline), output)

    def log(self, step, output, writer, save_prefix=None):
        # logging
        self.logger_train.aggregate()
        self.logger_eval.aggregate()

        if writer is not None:
            log = ["%04d" % step] + self.logger_train.log + self.logger_eval.log
            Print("\t".join(log), output)
            for k, v in self.logger_train.log_dict.items():
                writer.add_scalar(k, v, step)
            for k, v in self.logger_eval.log_dict.items():
                writer.add_scalar(k, v, step)
            writer.flush()

        else:
            log = [str(step)] + self.logger_eval.log
            Print("\t".join(log), output)

        self.log_reset()

    def log_reset(self):
        # reset logging parameters
        self.logger_train.reset()
        self.logger_eval.reset()


class Logger():
    """ Logger class """
    def __init__(self):
        self.loss_dict = OrderedDict()
        self.acc_dict = OrderedDict()
        self.result_dict = OrderedDict()
        self.log_dict = OrderedDict()
        self.log = []

    def loss_update(self, loss_dict):
        # update loss_dict for current minibatch
        for k, v in loss_dict.items():
            if k not in self.loss_dict:
                self.loss_dict[k] = []
            self.loss_dict[k].append(v.item())

    def acc_update(self, acc_dict):
        # update acc_dict for current minibatch
        for k, v in acc_dict.items():
            if k not in self.acc_dict:
                self.acc_dict[k] = [0, 0]
            self.acc_dict[k][0] += v[0]
            self.acc_dict[k][1] += v[1]

    def result_update(self, result_dict):
        # update result_dict for current minibatch
        for k, v in result_dict.items():
            if k not in self.result_dict:
                self.result_dict[k] = []
            if isinstance(v, np.ndarray):
                self.result_dict[k].append(v)
            else:
                self.result_dict[k] += v

    def aggregate(self):
        # aggregate logger dicts
        if len(self.log) == 0:
            for k, v in self.loss_dict.items():
                loss = np.mean(v)
                self.log_dict[k] = loss
                self.log.append("%.4f" % loss)

            for k, v in self.acc_dict.items():
                acc = v[0] / v[1]
                self.log_dict[k] = acc
                self.log.append("%.4f" % acc)

        for k, v in self.result_dict.items():
            if isinstance(v, list) and isinstance(v[0], np.ndarray):
                self.result_dict[k] = np.concatenate(v, axis=0)

    def reset(self):
        # reset logger
        self.loss_dict = OrderedDict()
        self.acc_dict = OrderedDict()
        self.result_dict = OrderedDict()
        self.log_dict = OrderedDict()
        self.log = []


def set_device(batch, device):
    """ recursive function for setting device for batch """
    if isinstance(batch, tuple) or isinstance(batch, list):
        return [set_device(t, device) for t in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch
