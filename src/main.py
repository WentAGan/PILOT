"""PILOT entry point.

PILOT is a prototype-guided one-step flow-matching recommender for
sequential recommendation.  The script trains and evaluates the model with
all-item ranking and HR/NDCG metrics.
"""

import os
import time
import random
import logging
import argparse
import pickle
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from utils import Data_Train, Data_Val, Data_Test
from model import create_model
from trainer import model_train


def get_args():
    p = argparse.ArgumentParser()
    # data / run
    p.add_argument('--dataset', default='amazon_beauty', help='amazon_beauty')
    p.add_argument('--log_file', default='log/')
    p.add_argument('--result_file', default='', help='optional JSONL path for appending final test metrics')
    p.add_argument('--random_seed', type=int, default=2026)
    p.add_argument('--max_len', type=int, default=50)
    p.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20])

    # backbone
    p.add_argument('--hidden_size', type=int, default=128)
    p.add_argument('--num_blocks', type=int, default=4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--emb_dropout', type=float, default=0.3)

    # optimisation
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--eval_interval', type=int, default=1)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--use_amp', type=int, default=1)
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['bf16', 'fp16'])
    p.add_argument('--ema_decay', type=float, default=0.0, help='EMA decay for weight averaging; 0 disables EMA')
    p.add_argument('--ema_warmup_epochs', type=int, default=0, help='minimum epoch before validation selection')
    p.add_argument('--sel_smooth_beta', type=float, default=0.0, help='validation-score smoothing beta; 0 disables smoothing')

    # flow / prior
    p.add_argument('--use_prototype', type=int, default=1, choices=[0, 1],
                   help='1: use the collaborative prototype read-out; 0: ablate it (x0 = history summary only)')
    p.add_argument('--num_prototypes', type=int, default=192, help='size of the collaborative prototype bank')
    p.add_argument('--prior_heads', type=int, default=4, help='attention heads for the prototype read-out')
    p.add_argument('--noise_std', type=float, default=0.1, help='Gaussian perturbation on interpolated states during training')
    p.add_argument('--eps', type=float, default=0.001, help='time clamp epsilon')
    p.add_argument('--s_modsamp', type=float, default=1.0, help='mode time-sampling strength')

    # loss weights (set any to 0 to ablate that component)
    p.add_argument('--flow_weight', type=float, default=1.0)
    p.add_argument('--ce_weight', type=float, default=0.5, help='next-item CE on the one-step prediction')
    p.add_argument('--prior_ce_weight', type=float, default=0.3, help='next-item CE on the collaborative prior path')
    return p.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    args = get_args()

    os.makedirs(args.log_file, exist_ok=True)
    os.makedirs(os.path.join(args.log_file, args.dataset), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(args.log_file, args.dataset,
                              time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.log'),
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
    )
    logger = logging.getLogger(__name__)
    logger.info(args)
    print(args)

    fix_seed(args.random_seed)

    path = os.path.join('..', 'datasets', 'data', args.dataset, 'dataset.pkl')
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap']) + 1

    tra = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    test = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args).get_pytorch_dataloaders()

    model = create_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[PILOT] item_num={args.item_num} params={n_params:,}")
    logger.info(f"item_num={args.item_num} params={n_params}")

    _, test_metrics, selection = model_train(tra, val, test, model, args, logger)

    if args.result_file:
        result_dir = os.path.dirname(os.path.abspath(args.result_file))
        if result_dir:
            os.makedirs(result_dir, exist_ok=True)
        row = {
            'model': 'PILOT',
            'dataset': args.dataset,
            'seed': int(args.random_seed),
            'patience': int(args.patience),
            'epochs': int(args.epochs),
            'selected_epoch': int(selection.get('selected_epoch', -1)),
            'valid_NDCG@20': float(selection.get('valid_NDCG@20', -1.0)),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
        }
        row.update({k: float(v) for k, v in test_metrics.items()})
        with open(args.result_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(f"[PILOT] appended result to {args.result_file}")
        logger.info(f"appended result to {args.result_file}: {row}")


if __name__ == '__main__':
    main()
