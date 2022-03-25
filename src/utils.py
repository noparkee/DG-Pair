import os
import sys
import time
import random
import hashlib
import dominate
import datetime
import subprocess
import numpy as np
from gpuinfo import GPUInfo
from collections import OrderedDict
from dominate.tags import h3, table, tr, td, p, a, img, br

import torch
from torch.utils.tensorboard import SummaryWriter


import requests
import pickle



def Print(string, output, newline=False, timestamp=True):
    """ print to stdout and a file (if given) """
    if timestamp:
        time = datetime.datetime.now()
        line = '\t'.join([str(time.strftime('%m-%d %H:%M:%S')), string])
    else: 
        time = None
        line = string

    if not output == sys.stdout:
        print(line, file=output)
        if newline: print("", file=output)
    else:
        print(line, file=sys.stderr)
        if newline: print("", file=sys.stderr)

    output.flush()
    return time


def seed_hash(*args):
    """ derive an integer hash from all args, for use as a random seed """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def set_seeds(model, test_env, seed):
    """ set random seeds """
    seed = seed_hash(model, test_env, seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_output(args, string, test_env):
    """ set output configurations """
    output, writer, save_prefix = sys.stdout, None, None
    if args["output_path"] is not None:
        save_prefix = args["output_path"]
        if not os.path.exists(os.path.join(save_prefix, "codes/")):
            os.makedirs(os.path.join(save_prefix, "codes/"), exist_ok=True)
        
        # 코드 저장
        os.system("cp * %s/codes/.  > /dev/null 2>&1"  % save_prefix)
        os.system("cp -r src %s/codes/.  > /dev/null 2>&1"  % save_prefix)
        
        # 결과 저장
        output = open(os.path.join(args["output_path"], "%s.txt" % string), "a")
        
        # tensorboard 저장
        if "eval" not in string:
            tb = os.path.join(args["output_path"], "tensorboard_testenv" + str(test_env))
            if not os.path.exists(tb):
                os.makedirs(tb, exist_ok=True)
            writer = SummaryWriter(tb)

    return output, writer, save_prefix

def count_gpu_process(num_gpu):
    """ count running gpu process """
    info = GPUInfo.get_info()
    gpu_process = [0] * num_gpu
    for k, v in info[0].items():
        for g in v: gpu_process[int(g)] += 1

    for i in range(num_gpu):
        if int(info[2][i]) < 20:
            gpu_process[i] = 0

    return gpu_process


def run_commands(commands, gpu_process_limits, conda=None, wait_seconds=3, change_time=None, gpu_process_limits2=None):
    """ run commands in que whenever required_gpu is available """
    num_gpu = len(gpu_process_limits)
    gpu_runs_buffer = [0] * num_gpu
    gpu_runs_info = []
    for c, (idx, file, command, required_gpu) in enumerate(commands):
        while 1:
            run, available_gpu, device = False, 0, ""
            if os.path.exists(file): 
                print("\t".join(["%4d" % c, idx, "log file already exists"]))
                break

            now = datetime.datetime.utcnow()+datetime.timedelta(hours=9)
            if change_time is not None and int(now.hour) < 12 and int(now.hour) > change_time:
                gpu_process_limits = gpu_process_limits2

            # check available_gpu
            gpu_process = count_gpu_process(num_gpu)
            for d in range(num_gpu):
                if gpu_process[d] + gpu_runs_buffer[d] < gpu_process_limits[d]:
                    if device == "": device += "%s"  % str(d)
                    else:            device += ",%s" % str(d)
                    available_gpu += 1
                    if available_gpu == required_gpu: break

            # run command if required_gpu is available
            if available_gpu == required_gpu:
                for d in device.split(","):
                    gpu_runs_buffer[int(d)] += 1
                gpu_runs_info.append([datetime.datetime.now(), device])

                print("\t".join(["%4d" % c, idx, "GPU%s" % device]))
                FILE = open("run.sh", "w")
                FILE.write("#!/bin/bash\n")
                if conda is not None: 
                    FILE.write("source %s\n" % conda)    
                FILE.write("CUDA_VISIBLE_DEVICES=%s " % device + command)
                FILE.close()
                subprocess.Popen(["./run.sh"], shell=True)
                time.sleep(3)
                break

            # check gpu_runs and sleep
            else:
                gpu_runs_info_new = []
                for run_time, device in gpu_runs_info:
                    # remove from the buffer if running_time is over than wait_seconds
                    if datetime.datetime.now() - run_time > datetime.timedelta(seconds=wait_seconds):
                        for d in device.split(","):
                            gpu_runs_buffer[int(d)] -= 1
                    else:
                        gpu_runs_info_new.append([run_time, device])
                gpu_runs_info = gpu_runs_info_new
                time.sleep(wait_seconds)

    print("DONE")


def get_training_results(file_idx, test_env, oracle=False, save_checkpoint=False):
    """ get training results (best validation) """
    if not os.path.exists(file_idx): return 0

    FILE = open(file_idx, "r")
    lines = FILE.readlines()
    FILE.close()

    best_step, best_val, best_test = 0, 0, 0
    offset = 2
    for line in lines:
        tokens = line.strip().split("\t")
        if len(tokens) > 1 and tokens[1] == "step":
            for token in tokens:
                if token.endswith("loss"):
                    offset += 1
        elif len(tokens) > 2 and offset != 2:
            step = int(tokens[1])
            if not oracle:
                val = np.average([float(tokens[offset + i * 2 + 0]) for i in range(4) if i != test_env])
            else:
                val = float(tokens[offset + test_env * 2 + 0])
            test = float(tokens[offset + test_env * 2 + 1])
            if val > best_val:
                best_step = step
                best_val = val
                best_test = test

    if save_checkpoint:
        path_idx, _ = os.path.split(file_idx)
        os.system("cp %s/checkpoints/%d.pt %s/checkpoints/best.pt" % (path_idx, best_step, path_idx))

    return best_test


def get_evaluation_results(file_idx, test_env):
    """ get evaluation results """
    if not os.path.exists(file_idx): return 0

    FILE = open(file_idx, "r")
    lines = FILE.readlines()
    FILE.close()

    skip = True
    offset = 2
    results_dict = OrderedDict()
    for line in lines:
        tokens = line.strip().split("\t")
        if len(tokens) > 1 and tokens[1] == "test":
            skip = False
        elif len(tokens) > 2 and not skip:
            results_dict[tokens[1]] = float(tokens[offset + test_env])

    return results_dict


class HTML:
    def __init__(self, title, file):
        self.title = title
        self.file = file
        self.doc = dominate.document(title=title)

    def add_header(self, text):
        with self.doc:
            h3(text)

    def add_text(self, text):
        with self.doc:
            p(text)

    def add_images(self, ims, txts, width=400):
        self.t = table(border=1, style="table-layout: fixed;")  # Insert a table
        self.doc.add(self.t)
        with self.t:
            with tr():
                for im, txt in zip(ims, txts):
                    with td(style="word-wrap: break-word;", halign="center", valign="top"):
                        with p():
                            with a(href=im):
                                img(style="width:%dpx" % width, src=im)
                            br()
                            p(txt)

    def save(self):
        f = open(self.file, 'wt')
        f.write(self.doc.render())
        f.close()


class ErrorReportBot():
    def __init__(self, args, task, error):
        with open("token.pkl", "rb") as file:
            token = pickle.load(file)
        
        self.token = token

        self.channel = '#kauai'
        self.server = 'kauai'
        self.model = args['model_config']
        self.test_env = args['test_env']
        self.task = task
        self.error = error

    def post_error(self):
        message = '### [ERROR]\n\nServer: %s\nModel: %s\nTestEnv: %s\nTask: %s\n\n\n%s\n###' %(self.server, self.model, str(self.test_env), self.task, self.error)

        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+self.token},
            data={"channel": self.channel,"text": message}
        )

        print(response)
    
    def post_success(self):
        message = '### [SUCCESS]\n\nServer: %s\nModel: %s\nTestEnv: %s\nTask: %s\n\n\nTraining Done\n###' %(self.server, self.model, str(self.test_env), self.task)

        response = requests.post("https://slack.com/api/chat.postMessage",
            headers={"Authorization": "Bearer "+self.token},
            data={"channel": self.channel,"text": message}
        )

        print(response)