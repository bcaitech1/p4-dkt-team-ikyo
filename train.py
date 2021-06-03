import os
import time
from args import parse_args
from dkt.dataloader import Preprocess
from dkt import trainer
import torch
from dkt.utils import setSeeds
import wandb
def main(args):
    wandb.login()
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    start_time = time.time()
    preprocess = Preprocess(args)
    print('Data preprocessing start !!')
    
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    train_data, valid_data = preprocess.split_data(train_data)
    
    train_data = preprocess.augment_data(train_data)
    print('Data preprocessing finish !!')
    print(f'\t >>> elapsed time {((time.time()-start_time)/60):.2f} min')
    
    print('Training start !!')
    wandb.init(project='P4-DKT', config=vars(args), entity='team-ikyo', name=args.run_name)
    trainer.run(args, train_data, valid_data)
    print('Training finish !!')
    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)