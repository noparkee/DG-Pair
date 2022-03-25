import os
import sys
import argparse
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import torch

import src.config as config
from src.data import get_datasets_and_iterators
from src.model import get_model
from src.train import Trainer
from src.utils import Print, set_seeds, set_output, ErrorReportBot

import traceback



parser = argparse.ArgumentParser('Evaluate a Domain Generalization Model for the CUB dataset')
parser.add_argument('--model-config', help='path for model configuration file')
parser.add_argument('--test-env', type=int, help='test environment')
parser.add_argument('--seed', type=int, default=0, help='random seed (default: 0)')
parser.add_argument('--checkpoint', default="5000", help='checkpoint to evaluate')
parser.add_argument('--output-path', help='path for outputs (default: stdout and without saving)')


def main():
    args = vars(parser.parse_args())
    model_cfg = config.ModelConfig(args["model_config"])
    output, writer, save_prefix = set_output(args, "testenv%d_evaluate_model_log" % args['test_env'], args['test_env'])
    
    device, data_parallel = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.cuda.device_count() > 1
    config.print_configs(args, [model_cfg], device, output)
    set_seeds(args["model_config"], args["test_env"], args["trial_seed"])
    for i in range(torch.cuda.device_count()):
        torch.zeros((1)).to(torch.device("cuda:%d" % i))

    if args['test_env'] > 3 or args['test_env'] < 0:
        print("error test env")
        return
    print("test_env: "+ str(args['test_env']))

    GVE_flag = (args["model_config"].find('GVE') != -1)
    print('GVE_flag:' + str(GVE_flag))
    
    ## Loading datasets
    start = Print(" ".join(['start loading datasets']), output)
    datasets, iterators_train, iterators_eval, eval_names = get_datasets_and_iterators(GVE_flag, eval_flag=True)
    end = Print('end loading datasets', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## setup trainer configurations
    start = Print('start setting trainer configurations', output)
    model = get_model(model_cfg, datasets[0].vocab)

    # Activate Multi-GPUs.
    if (device.type == 'cuda') and (torch.cuda.device_count() > 1):
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    trainer = Trainer(model, device, data_parallel)
    trainer.load_model(os.path.join(args["output_path"], "checkpoints/5000.pt"), output)
    end = Print('end setting trainer configurations', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)

    ## evaluate a model
    start = Print('start evaluating a model', output)
    trainer.headline("test", model.loss_names, eval_names, output)
    
    for iterator_eval, eval_name in zip(iterators_eval, eval_names):
        for B, minibatch in enumerate(iterator_eval):
            trainer.evaluate(minibatch, args["test_env"], eval_name, GVE_flag, eval_flag=True)
            if B % 5 == 0: print('# {} {:.1%}'.format(eval_name, B / len(iterator_eval)), end='\r', file=sys.stderr)
        print(' ' * 150, end='\r', file=sys.stderr)

    trainer.log("Accuracy", output, writer, save_prefix)
    end = Print('end evaluating a model', output)
    Print(" ".join(['elapsed time:', str(end - start)]), output, newline=True)
    if not output == sys.stdout: output.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        err_m = traceback.format_exc()
        err_b = ErrorReportBot(vars(parser.parse_args()), 'Evaluating', str(err_m))
        err_b.post_message()
