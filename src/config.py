# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import os
import sys
import json

import torch

from src.utils import Print


class ModelConfig():
    def __init__(self, file=None, idx="model_config"):
        """ model configurations """
        self.idx = idx
        self.name = None
        self.explainer_embed_size = None
        self.explainer_hidden_size = None
        self.sc_embed_size = None
        self.sc_hidden_size = None
        self.perceptron_size = None

        # load config from json file
        if file is not None:
            if not os.path.exists(file): sys.exit("model-config [%s] does not exists" % file)
            else: cfg = json.load(open(file, "r"))

            self.attn = 0
            for key, value in cfg.items():
                if key == "name":                       self.name = value
                elif key == "explainer_embed_size":     self.explainer_embed_size = value
                elif key == "explainer_hidden_size":    self.explainer_hidden_size = value
                elif key == "sc_embed_size":            self.sc_embed_size = value
                elif key == "sc_hidden_size":           self.sc_hidden_size = value
                elif key == "perceptron_size":          self.perceptron_size = value
                elif key == "attn":                     self.attn = value
                else: sys.exit("# ERROR: invalid key [%s] in model-config file" % key)

    def get_config(self):
        configs = []
        configs.append(["name", self.name])
        if self.explainer_embed_size is not None:
            configs.append(["explainer_embed_size", self.explainer_embed_size])
            configs.append(["explainer_hidden_size", self.explainer_hidden_size])
        if self.sc_embed_size is not None:
            configs.append(["sc_embed_size", self.sc_embed_size])
            configs.append(["sc_hidden_size", self.sc_hidden_size])

        return configs

def print_configs(args, cfgs, device, output):
    Print(" ".join(['##### arguments #####']), output)
    for cfg in cfgs:
        Print(" ".join(['%s:' % cfg.idx, str(args[cfg.idx])]), output)
        for c, v in cfg.get_config():
            Print(" ".join(['-- %s: %s' % (c, v)]), output)
    Print(" ".join(['device: %s (%d GPUs)' % (device, torch.cuda.device_count())]), output)
    Print(" ".join(['output_path:', str(args["output_path"])]), output)
    Print(" ".join(['log_file:', str(output.name)]), output, newline=True)
