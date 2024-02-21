import argparse
import torch
import os

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--seed', type=int, default=1234)
    # parser.add_argument('--scenario', type=str, default='test_output_model')
    parser.add_argument('--scenario', type=str, default='train')

    parser.add_argument('--model-base', type=str, default='mbnv3')
    parser.add_argument('--pretrain_dir', type=str, default='SVTR_pretrained_large_2810')
    parser.add_argument('--ckpt_dir', type=str, default='SVTR_finetuned_vilage_short_1111')

    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epoch', type=str, default=100)
    parser.add_argument('--log_iter', type=int, default=10)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--num_worker', type=int, default=16)
    parser.add_argument('--shuffle', action='store_false')

    args = parser.parse_args()

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args