import os
import time
import torch
import wandb
import random
import warnings
import pandas as pd
import numpy as np

from simple.args import parse_args
from datetime import datetime

from simple.dkt import trainer
from simple.dkt.utils import setSeeds
from simple.dkt.dataloader import Preprocess

warnings.filterwarnings('ignore')

def main(args):
    wandb.login()
    
    setSeeds(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    
    output_name = args.output_name

    start_time = time.time()
    preprocess = Preprocess(args)
    
    print('Data preprocessing start !!')
    preprocess.load_train_data(args.file_name)
    df = preprocess.get_train_data()

    preprocess.load_test_data(args.test_file_name)
    test_data = preprocess.get_test_data()
    print('Data preprocessing finish !!')
    print(f'\t >>> elapsed time {((time.time()-start_time)/60):.2f} min')
    
    # 유저별 5 Fold로 분리
    train_lst, test_lst = custom_train_test_split(df)

    for fold_num, (train, test) in enumerate(zip(train_lst, test_lst)):
        train_data = df2group(train)
        valid_data = df2group(test)
        
        is_augment = True
        if is_augment:
            train_data = preprocess.augment_data(train_data)
        
        print(f'fold-{fold_num+1} Training start !!')
        wandb.init(project='P4-DKT', config=vars(args), entity='team-ikyo', name=args.run_name+f'-{fold_num+1}')
        trainer.run(args, train_data, valid_data)
        
        args.output_name = f'{fold_num+1}-valid-'+output_name
        trainer.inference(args, valid_data)
        
        args.output_name = f'{fold_num+1}-test-'+output_name
        trainer.inference(args, test_data)
        wandb.finish()
        print(f'fold-{fold_num+1} Training finish !!')

def df2group(df):
    df = df.sort_values(by=['userID','Timestamp'], axis=0)
    columns = [
        'userID_KnowledgeTag_total_answer',
        'userID_KnowledgeTag_acc',
        'userID_question_class_total_answer',
        'userID_question_class_acc',
        
        'userID',
        'answerCode',
        'assessmentItemID',
        'KnowledgeTag',
        'question_class',\
    ]

    group = df[columns].groupby('userID').apply(lambda r: np.array([
                                                                    r['userID_KnowledgeTag_total_answer'].values,
                                                                    r['userID_KnowledgeTag_acc'].values,
                                                                    r['userID_question_class_total_answer'].values,
                                                                    r['userID_question_class_acc'].values,\

                                                                    r['answerCode'].values,
                                                                    r['assessmentItemID'].values,
                                                                    r['KnowledgeTag'].values,
                                                                    r['question_class'].values,\
                                                                    r['userID'].values,
    ]))
    data_group = group.values
    return data_group
    
def custom_train_test_split(df, ratio=0.2):
    random.seed(42)
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)

    train_lst = []
    test_lst = []

    max_train_data_len = int(ratio*len(df))
    sum_of_train_data = 0

    user_ids_lst = []
    user_ids = []
    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            user_ids_lst.append(user_ids)
            user_ids = []
            sum_of_train_data = 0
        else:
            user_ids.append(user_id)
    else:
        user_ids_lst.append(user_ids)

    for user_ids in user_ids_lst:
        train = df[df['userID'].isin(user_ids) == False]        
        test = df[df['userID'].isin(user_ids)]
        train = pd.concat([train, test[test['userID'] == test['userID'].shift(-1)]])
        train_lst.append(train)
        test_lst.append(test[test['userID'] != test['userID'].shift(-1)])

    return train_lst, test_lst
    
if __name__ == "__main__":
    args = parse_args(mode='train')
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)