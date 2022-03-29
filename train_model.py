import os
import sys
import argparse
os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch

import src.config as config
from src.data import get_datasets_and_iterators
from src.model import get_model
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, ErrorReportBot

import traceback
import preprocess_data_gnn



parser = argparse.ArgumentParser('Train a Domain Generalization Model for the CUB dataset')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--test-env', type=int, help='test environment')
parser.add_argument('--trial-seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')


def main():
    args = vars(parser.parse_args())
    model_cfg = config.ModelConfig(args["model_config"])
    # output, writer, save_prefix = set_output(args, "train_model_log")
    output, writer, save_prefix = set_output(args, "testenv%d_train_model_log" % args['test_env'], args['test_env'])

    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, [model_cfg], device, output)
    seed = set_seeds(args["model_config"], args["test_env"], args["trial_seed"])
    for i in range(torch.cuda.device_count()):
        torch.zeros((1)).to(torch.device("cuda:%d" % i))

    if args['test_env'] > 3 or args['test_env'] < 0:
        print("error test env")
        return
    print("test_env: "+ str(args['test_env']))

    text_flag = (args["model_config"].find('GVE') != -1) or (args["model_config"].find('GCN') != -1)
    print('text_flag: ' + model_cfg.text_feature)

    ## make description files
    if model_cfg.text_feature:
        preprocess_data_gnn.make_description(path='data/CUB-DG/', trial_seed=seed)

    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    datasets, iterators_train, iterators_eval, eval_names = get_datasets_and_iterators(text_flag)
    end = Print('end loading datasets', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    model = get_model(model_cfg, datasets[0].vocab)

    # Activate Multi-GPUs.
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    trainer = Trainer(model, device, data_parallel)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## train a model
    N_STEPS, CHECKPOINT_FREQ = 5000, 300
    start = Print('start training a model', output)
    trainer.headline("step", model.module.loss_names, eval_names, output)
    for step in range(N_STEPS):
        ### train
        minibatches = next(iterators_train)
        trainer.train(minibatches, args['test_env'])
        
        if (step + 1) % 10 == 0:
            print('# step [{}/{}]'.format(step + 1, N_STEPS), end='\r', file=sys.stderr)
        if ((step + 1) % CHECKPOINT_FREQ == 0) or (step + 1 == N_STEPS):
            for iterator_eval, eval_name in zip(iterators_eval, eval_names):
                for B, minibatch in enumerate(iterator_eval):
                    trainer.evaluate(minibatch, args['test_env'], eval_name, text_flag=text_flag)
                    if B % 5 == 0: print('# step [{}/{}] {} {:.1%}'.format(step + 1, N_STEPS, eval_name, B / len(iterator_eval)), end='\r', file=sys.stderr)
                print(' ' * 150, end='\r', file=sys.stderr)

            #trainer.save_model(step + 1, save_prefix, args["test_env"])
            trainer.log(step + 1, output, writer)

    # 마지막 하나만 저장
    trainer.save_model(step + 1, save_prefix, args["test_env"])

    end = Print('end training a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()


if __name__ == '__main__':
    args = vars(parser.parse_args())

    try:
        main()
        err_b = ErrorReportBot(args, 'Training', '')
        err_b.post_success()
    except Exception:
        err_m = traceback.format_exc()
        err_b = ErrorReportBot(args, 'Training', str(err_m))
        err_b.post_error()
    # main()