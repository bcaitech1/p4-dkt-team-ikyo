import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch
from collections import Counter
from tqdm import tqdm

class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.9, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed) # fix to default seed 0
            random.shuffle(data)
        
        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]
        
        return data_1, data_2
    
    def augment_data(self, data_group):
        augment_data_group = []
        for data in tqdm(data_group, desc="data augmentation"):
            _, seq_len = np.shape(data)
            n = seq_len//self.args.max_seq_len
            for i in range(1, n+1):
                augment_data_group.append(data[:,self.args.max_seq_len*(i-1):self.args.max_seq_len*i])
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
        df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
        
        def convert_time(s):
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)

        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
        
#         # non sequence - categorical feature
#         df["tmp"] = df.groupby(["userID", "testId"])['testId'].cumcount()
#         df['testid_experience'] = df['tmp'].apply(lambda x : 1 if x > 0 else 0)
        
#         df["tmp"] = df.groupby(["userID", "assessmentItemID"])['assessmentItemID'].cumcount()
#         df['assessmentItemID_experience'] = df['tmp'].apply(lambda x : 1 if x > 0 else 0)
#         print('\t no-sequence categorical feature finish')
        
        # non sequence - continuous feature
        df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=True)
#         df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
#         df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift(1))
#         df['user_correct_answer'].fillna(0, inplace=True)
#         df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]
#         df['user_acc'].fillna(0, inplace=True)
        
        df["tmp"] = df[["userID", "KnowledgeTag"]].apply(lambda data: str(data["userID"]) + "_" + str(data["KnowledgeTag"]), axis=1)
        df["knowledgetag_total_solving_num"] = df.groupby("tmp")["answerCode"].cumcount()
        df["knowledgetag_correct_solving_num"] = df.groupby("tmp")["answerCode"].transform(lambda x: x.cumsum().shift(1))
        df['knowledgetag_correct_solving_num'].fillna(0, inplace=True)
        df["knowledgetag_total_solving_rate"] = df["knowledgetag_correct_solving_num"] / df["knowledgetag_total_solving_num"]
        df['knowledgetag_total_solving_rate'].fillna(0, inplace=True)
        
        df["tmp"] = df[["userID", "question_class"]].apply(lambda data: str(data["userID"]) + "_" + data["question_class"], axis=1)
        df["question_class_total_solving_num"] = df.groupby("tmp")["answerCode"].cumcount()
        df["question_class_correct_solving_num"] = df.groupby("tmp")["answerCode"].transform(lambda x: x.cumsum().shift(1))
        df['question_class_correct_solving_num'].fillna(0, inplace=True)
        df["question_class_solving_rate"] = df["question_class_correct_solving_num"] / df["question_class_total_solving_num"]
        df['question_class_solving_rate'].fillna(0, inplace=True)
        print('\t no-sequence continuous feature finish')
        
#         # sequence - categorical feature
#         df = df.sort_values(by=['userID', 'Timestamp']).reset_index(drop=False)
#         df["tmp"] = df["user_total_answer"].apply(lambda x : x// self.args.max_seq_len)
        
#         last_seq_group = df.groupby(['userID', "tmp"])['index'].agg(["max"])
#         last_seq_group = last_seq_group.reset_index()
#         last_seq_group.columns = ['userID', 'tmp', 'last_seq']
#         df = pd.merge(df, last_seq_group, on=["userID", "tmp"], how="left")
        
#         def changed_time(x):
#             last_time = df['Timestamp'][x['last_seq']]
#             period = (last_time-x['Timestamp'])//(3600*24)

#             if period == 0:
#                 return 0
#             elif period < 14:
#                 return 1
#             else:
#                 return 2
        
#         df["new_time"] = df.apply(changed_time, axis=1)
#         print('\t sequence categorical feature finish')
        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)
        df = self.__feature_engineering(df)
        df = self.__preprocessing(df, is_train)
        
        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))
        self.args.n_class = len(np.load(os.path.join(self.args.asset_dir,'question_class_classes.npy')))
        
        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        columns = [
#             'testid_experience',
#             'assessmentItemID_experience',
#             'user_total_answer',
#             'user_correct_answer',
#             'user_acc',
            'knowledgetag_total_solving_num',
#             'knowledgetag_correct_solving_num',
            'knowledgetag_total_solving_rate',
            'question_class_total_solving_num',
#             'question_class_correct_solving_num',
            'question_class_solving_rate',
            'userID',
            'answerCode',
#             'testId',
            'assessmentItemID',
            'KnowledgeTag',
            'question_class',
#             'new_time'
        ]
        
        group = df[columns].groupby('userID').apply(lambda r: np.array([
#                                                                         r['testid_experience'].values, 
#                                                                         r['assessmentItemID_experience'].values,        # 1
#                                                                         r['user_total_answer'].values,
#                                                                         r['user_correct_answer'].values,
#                                                                         r['user_acc'].values,                           # 4
                                                                        r['knowledgetag_total_solving_num'].values,
#                                                                         r['knowledgetag_correct_solving_num'].values,
                                                                        r['knowledgetag_total_solving_rate'].values,    # 7
                                                                        r['question_class_total_solving_num'].values,
#                                                                         r['question_class_correct_solving_num'].values,
                                                                        r['question_class_solving_rate'].values,        # 10
                                                                        
                                                                        r['answerCode'].values,                         # 11
#                                                                         r['testId'].values, 
                                                                        r['assessmentItemID'].values,
                                                                        r['KnowledgeTag'].values,
                                                                        r['question_class'].values,
#                                                                         r['new_time'].values,                           # 16
        ]))
        data_group = group.values
        return data_group

    def load_train_data(self, file_name):
        self.train_data = self.load_data_from_file(file_name)

    def load_test_data(self, file_name):
        self.test_data = self.load_data_from_file(file_name, is_train= False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]
        non_seq_row, seq_row = row[:4], row[4:]
        
        tag_total_solving_num   = non_seq_row[0, -1]
        tag_total_solving_rate  = non_seq_row[1, -1]
        class_total_solving_num = non_seq_row[2, -1]
        class_solving_rate      = non_seq_row[3, -1]
        
        correct  = seq_row[0, -1]
        question = seq_row[1, -1]
        tag      = seq_row[2, -1]
        qclass   = seq_row[3, -1]
        
        cols = [
            tag_total_solving_num,
            tag_total_solving_rate,
            class_total_solving_num,
            class_solving_rate,
            correct,
            question,
            tag,
            qclass,
        ]
        
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols):
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

    for i, _ in enumerate(col_list):
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