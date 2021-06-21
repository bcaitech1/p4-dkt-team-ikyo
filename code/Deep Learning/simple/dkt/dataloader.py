import os
import time
import tqdm
import torch
import random
import warnings
import numpy as np
import pandas as pd

from tqdm import tqdm
from datetime import datetime
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from simple.dkt.feature_engineering import *

warnings.filterwarnings('ignore')

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def augment_data(self, data_group):
        augment_data_group = []
        for data in tqdm(data_group, desc="data augmentation"):
            _, seq_len = np.shape(data)
            n = seq_len
            for i in range(self.args.max_seq_len-1, n):
                augment_data_group.append(data[:,i:i+1])
        return augment_data_group
    
    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'question_class']

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            le = LabelEncoder()
            if is_train:
                #For UNKNOWN class
                a = df[col].unique().tolist() + ['unknown']
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col+'_classes.npy')
                le.classes_ = np.load(label_path)
                classes = set(le.classes_)
                
                df[col] = df[col].astype(str)
                df[col] = df[col].apply(lambda x: x if x in classes else 'unknown')
                
            #모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test
            
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        return df
    
    def __feature_engineering(self, df):
        df = question_class(df)
        print('\t question_class feature - finish')
        
        df = userID_KnowledgeTag_relative(df)
        print('\t userID_KnowledgeTag_relative feature - finish')
        
        df = userID_question_class_relative(df)
        print('\t userID_question_class_relative feature - finish')
        return df

    def load_cv_train_data_from_file(self, file_name, is_train=True):
        data_dir = '/opt/ml/input/data/train_dataset'
        train_csv_file_path = os.path.join(data_dir, 'train_data.csv')
        train_df = pd.read_csv(train_csv_file_path)
        train_df["next_userID"] = train_df['userID'].shift(-1)
        train_df["is_test_data"] = False
        
        test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
        test_df = pd.read_csv(test_csv_file_path)
        test_df["next_userID"] = test_df['userID'].shift(-1)
        test_df["is_test_data"] = test_df[["userID", "next_userID"]].apply(lambda data: False if data["userID"] == data["next_userID"] else True, axis=1)
        
        df = pd.concat([train_df, test_df])
        df = df[df["answerCode"] > -1]
        
        def random_answering(data):
            if data["userID"] != data["next_userID"]:
                return 1 if random.random() < 0.5 else 0
            else:
                return data["answerCode"]

        # test의 random 전략과 동일하게 맞춰주기 위해 진행
        df["answercode"] = df.apply(random_answering, axis=1)
        
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_class = len(np.load(os.path.join(self.args.asset_dir,'question_class_classes.npy')))
        return df

    def load_cv_test_data_from_file(self, file_name, is_train=True):
        data_dir = '/opt/ml/input/data/train_dataset'
        csv_file_path = os.path.join(data_dir, 'train_data.csv')
        df = pd.read_csv(csv_file_path)
        
        test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
        test_df = pd.read_csv(test_csv_file_path)
        
        test_df = pd.concat([df, test_df])

        not_test_df = test_df[test_df["answerCode"] != -1]
        not_test_df["is_test"] = False

        test_df = test_df[test_df["answerCode"] == -1]
        test_df["is_test"] = True
        
        df = pd.concat([not_test_df, test_df])
        df["next_userID"] = df["userID"].shift(-1)

        def random_answering(data):
            if data["is_test"]:
                return 1 if random.random() < 0.5 else 0
            else:
                return data["answerCode"]

        df["answercode"] = df.apply(random_answering, axis=1)
        
        df = self.__feature_engineering(df)
        
        # TEST DATA
        df = df[df["is_test"]]
        df.reset_index(drop=True, inplace=True)
        
        df = self.__preprocessing(df, is_train)
        
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
            'question_class',
        ]

        group = df[columns].groupby('userID').apply(lambda r: np.array([
                                                                        r['userID_KnowledgeTag_total_answer'].values,
                                                                        r['userID_KnowledgeTag_acc'].values,
                                                                        r['userID_question_class_total_answer'].values,
                                                                        r['userID_question_class_acc'].values,
                                                                        
                                                                        r['answerCode'].values,
                                                                        r['assessmentItemID'].values,
                                                                        r['KnowledgeTag'].values,
                                                                        r['question_class'].values,
                                                                        r['userID'].values,
        ]))
        data_group = group.values
        return data_group
    
    def load_train_data(self, file_name):
        self.train_data = self.load_cv_train_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_cv_test_data_from_file(file_name, is_train= False)

class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        
        cols = [i[-1] for i in row]
        
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols[:-1]):
            cols[i] = torch.tensor(col)
        
        return cols

    def __len__(self):
        return len(self.data)

def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            col_list[i].append(col)

    for i, _ in enumerate(col_list[:-1]):
        col_list[i] = torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   num_workers=args.num_workers,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   pin_memory=pin_memory,
                                                   collate_fn=collate,
                                                   drop_last=True)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset,
                                                   num_workers=args.num_workers,
                                                   shuffle=False,
                                                   batch_size=args.batch_size,
                                                   pin_memory=pin_memory,
                                                   collate_fn=collate)

    return train_loader, valid_loader