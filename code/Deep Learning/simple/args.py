import os
import argparse


def parse_args(mode='train'):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--run_name', default='pjh-simple_v4', type=str, help='wandb run name')
    parser.add_argument('--seed', default=42, type=int, help='seed')
    
    parser.add_argument('--device', default='cpu', type=str, help='cpu or gpu')
    
    parser.add_argument('--data_dir', default='/opt/ml/input/data/train_dataset', type=str, help='data directory')
    parser.add_argument('--asset_dir', default='simple/simple_asset/', type=str, help='data directory')
    
    parser.add_argument('--file_name', default='train_data.csv', type=str, help='train file name')
    
    parser.add_argument('--model_dir', default='simple/models/', type=str, help='model directory')
    parser.add_argument('--load_model_name', default='model.pt', type=str, help='model file name')
    parser.add_argument("--model_name", default="simple", type=str, help="Model Name")
    
    parser.add_argument('--output_dir', default='simple/output/', type=str, help='output directory')
    parser.add_argument('--output_name', default='output.csv', type=str, help='output directory')
    parser.add_argument('--test_file_name', default='test_data.csv', type=str, help='test file name')
    
    parser.add_argument('--max_seq_len', default=15, type=int, help='max sequence length 500 or 19')
    parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
    
    # 모델
    parser.add_argument('--hidden_dim', default=1024, type=int, help='hidden dimension size')
    parser.add_argument('--n_layers', default=3, type=int, help='number of layers')
    parser.add_argument('--n_heads', default=2, type=int, help='number of heads')
    parser.add_argument('--drop_out', default=0.1, type=float, help='drop out rate')
    
    # 훈련
    parser.add_argument('--n_epochs', default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=4096, type=int, help='batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='weight_decay')
    parser.add_argument('--clip_grad', default=10, type=int, help='clip grad')
    parser.add_argument('--patience', default=10, type=int, help='for early stopping')
    
    parser.add_argument('--log_steps', default=50, type=int, help='print log per n steps')
    
    ### 중요 ###
    parser.add_argument('--loss', default='bce', type=str, help='loss type')
    parser.add_argument('--model', default='simple', type=str, help='model type')
    parser.add_argument('--optimizer', default='adamW', type=str, help='optimizer type')
    parser.add_argument('--scheduler', default='cosine', type=str, help='scheduler type')
    
    args = parser.parse_args()
    
    return args