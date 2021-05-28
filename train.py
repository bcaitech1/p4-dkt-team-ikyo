import os
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

    args.data_dir = os.environ.get('SM_CHANNEL_TRAIN', args.data_dir)
    args.model_dir = os.environ.get('SM_MODEL_DIR', args.model_dir)


    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()
    
    train_data, valid_data = preprocess.split_data(train_data)
    
    wandb.init(project='P4-DKT', entity='team-ikyo', name=args.run_name)
    trainer.run(args, train_data, valid_data)
    

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
