"""PILOT entry point for sequential recommendation."""

import argparse
import json
import logging
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn

from model import create_model
from trainer import model_train
from utils import Data_Test, Data_Train, Data_Val


def get_args():
    parser = argparse.ArgumentParser(description='Train and evaluate PILOT.')

    # data / run
    parser.add_argument('--dataset', default='amazon_beauty', choices=['amazon_beauty', 'steam', 'yelp'])
    parser.add_argument('--log_file', default='log/')
    parser.add_argument('--result_file', default='', help='optional JSONL path for final test metrics')
    parser.add_argument('--random_seed', type=int, default=2026)
    parser.add_argument('--max_len', type=int, default=50)
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 20])

    # backbone
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--emb_dropout', type=float, default=0.3)

    # optimisation
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--patience', type=int, default=25)

    # prior / flow
    parser.add_argument('--use_prototype', type=int, default=1, choices=[0, 1])
    parser.add_argument('--num_prototypes', type=int, default=192)
    parser.add_argument('--prior_heads', type=int, default=4)
    parser.add_argument('--noise_std', type=float, default=0.1)
    parser.add_argument('--eps', type=float, default=0.001)
    parser.add_argument('--s_modsamp', type=float, default=1.0)

    # loss weights
    parser.add_argument('--flow_weight', type=float, default=1.0)
    parser.add_argument('--ce_weight', type=float, default=0.5)
    parser.add_argument('--prior_ce_weight', type=float, default=0.3)
    return parser.parse_args()


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def resolve_device(device_name):
    if device_name == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    if device_name == 'cuda' and not torch.cuda.is_available():
        print('[PILOT] CUDA is unavailable; using CPU instead.')
    return 'cpu'


def dataset_path(dataset):
    root = Path(__file__).resolve().parents[1]
    return root / 'datasets' / 'data' / dataset / 'dataset.pkl'


def setup_logger(log_root, dataset):
    log_dir = Path(log_root) / dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) + '.log')
    logging.basicConfig(
        level=logging.INFO,
        filename=str(log_path),
        datefmt='%Y/%m/%d %H:%M:%S',
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
    )
    return logging.getLogger(__name__)


def append_result(path, args, test_metrics, selection):
    result_path = Path(path)
    if result_path.parent:
        result_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        'model': 'PILOT',
        'dataset': args.dataset,
        'seed': int(args.random_seed),
        'selected_epoch': int(selection.get('selected_epoch', -1)),
        'valid_NDCG@20': float(selection.get('valid_NDCG@20', -1.0)),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
    }
    row.update({k: float(v) for k, v in test_metrics.items()})
    with open(result_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    args = get_args()
    args.device = resolve_device(args.device)
    logger = setup_logger(args.log_file, args.dataset)
    fix_seed(args.random_seed)

    print(args)
    logger.info(args)

    path = dataset_path(args.dataset)
    with open(path, 'rb') as f:
        data_raw = pickle.load(f)
    args.item_num = len(data_raw['smap']) + 1

    train_loader = Data_Train(data_raw['train'], args).get_pytorch_dataloaders()
    val_loader = Data_Val(data_raw['train'], data_raw['val'], args).get_pytorch_dataloaders()
    test_loader = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], args).get_pytorch_dataloaders()

    model = create_model(args)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[PILOT] dataset={args.dataset} item_num={args.item_num} params={n_params:,}')
    logger.info(f'dataset={args.dataset} item_num={args.item_num} params={n_params}')

    _, test_metrics, selection = model_train(train_loader, val_loader, test_loader, model, args, logger)

    if args.result_file:
        append_result(args.result_file, args, test_metrics, selection)
        print(f'[PILOT] appended result to {args.result_file}')


if __name__ == '__main__':
    main()
