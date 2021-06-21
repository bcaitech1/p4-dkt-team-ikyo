import os
from .args import parse_args
from .dkt.dataloader import Preprocess
from .dkt import trainer
import torch
from .dkt.utils import setSeeds
import wandb

from sklearn.model_selection import KFold
from tqdm import tqdm

def main(args):
    wandb.login()
    
    setSeeds(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device

    preprocess = Preprocess(args)

    # train data preprocess
    preprocess.load_train_data(args.file_name)

    train_data = preprocess.get_train_data()

    kf = KFold(n_splits=5)  
    auc_list = []

    for fold, (train_idx, val_idx) in tqdm(enumerate(kf.split(train_data))):
        print(f"Fold: {fold}")
        train = train_data[train_idx]
        val = train_data[val_idx]
        
        #train = preprocess.data_augmentation(train, args.max_seq_len)
        #val = preprocess.data_augmentation(val, args.max_seq_len)

    
        wandb.init(project='P4-DKT', config=vars(args), entity="team-ikyo", name=f"bert_question-attention_newfeature2_fold{fold}")
        auc_list.append(trainer.run(args, train, val, fold))
        
        wandb.finish()
    print(auc_list)

if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)