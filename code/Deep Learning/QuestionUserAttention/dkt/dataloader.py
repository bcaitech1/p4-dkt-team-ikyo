import os
from datetime import datetime
import time
import tqdm
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
import numpy as np
import torch

from .feature_engineering import *

class Preprocess:
    def __init__(self,args):
        self.args = args
        self.train_data = None
        self.test_data = None
        
    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data
    
    def data_augmentation(self, train, max_seq):
        augmentation_data = []
        for train_data in train:
            train_data = np.array(list(np.array(element) for element in train_data))
            for start in range(0, len(train_data[0]), max_seq):
                if start + max_seq < len(train_data[0]):
                    augmentation_data.append(train_data[:, start:start + max_seq])
                else:
                    augmentation_data.append(train_data[:, start:])
            
#         print(len(augmentation_data))
        return augmentation_data
        

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        # 이렇게 분류하는게 맞는가?
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

    def __save_labels(self, encoder, name):
        # encode save
        le_path = os.path.join(self.args.asset_dir, name + '_classes.npy')
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, is_train = True):
        #cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag']
        cate_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', "userID_elapsed_cate", "question_class_acc", "KnowledgeTag_acc", "question_class", "userID_acc", "userID_assessmentItemID_experience", "userID_KnowledgeTag_total_answer", "userID_KnowledgeTag_acc"]

        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
            
        for col in cate_cols:
            if col in ['assessmentItemID', 'testId', 'KnowledgeTag', "userID_elapsed_cate", "question_class", "userID_assessmentItemID_experience"]:
                le = LabelEncoder()
                if is_train:
                    #For UNKNOWN class
                    a = df[col].unique().tolist() + ['unknown']
                    # 각 col의 라벨들을 encode
                    le.fit(a)
                    self.__save_labels(le, col)
                else:
                    label_path = os.path.join(self.args.asset_dir,col+'_classes.npy')
                    le.classes_ = np.load(label_path)
                    classes = set(le.classes_)
                    
                    df[col] = df[col].astype(str)
                    df[col] = df[col].apply(lambda x: x if x in classes else 'unknown')

                df[col]= df[col].astype(str)
                test = le.transform(df[col])
                # encode화 된 값으로 변경
                df[col] = test
                

        def convert_time(s):
            ## ??
            timestamp = time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M:%S').timetuple())
            return int(timestamp)
        # df의 Timestamp에 적용 되는것?
#        df['Timestamp'] = df['Timestamp'].apply(convert_time)
        
        if is_train:
            print(len(df))
            df = df[df["answerCode"] > -1]
            print(len(df))

        else:
            df = df[df["is_test_data"]]
            print(len(df))
        return df

    def __feature_engineering(self, df):
        #TODO
        ["userID_elapsed_cate", "IK_question_acc", "question_class", "IK_KnowledgeTag_acc", "userID_acc", "userID_assessmentItemID_experience", "user_question_class_solved", "solved_question", "userID_KnowledgeTag_total_answer", "userID_KnowledgeTag_acc"]
        # time_differ => userID_elapsed_cate
        df = userID_elapsed_cate(df)

        # question acc => question_class_acc => IK_question_acc
        df = IK_question_acc(df)

        # Large_cate => question_class
        df = question_class(df)

        # KnowledgeTag_acc => KnowledgeTag_acc => IK_KnowledgeTag_acc
        df = IK_KnowledgeTag_acc(df)

        # user_acc => userID_acc
        df = userID_relative(df)

        # same_question => userID_assessmentItemID_experience
        df = userID_assessmentItemID_experience(df)

        #user_question_class_solved
        df = user_question_class_solved(df)

        # solved_question
        df = solved_question(df)

        # user_cate_count => userID_KnowledgeTag_total_answer
        # userID_KnowledgeTag_acc
        df = userID_KnowledgeTag_relative(df)

        # userID_acc_rolling
        df = userID_acc_rolling(df)

        def cont_normalize(df, cont_names):
            for cont_name in cont_names:
                mean = df[cont_name].mean()
                std = df[cont_name].std()

                df[cont_name] = (df[cont_name] - mean) / std
            
            return df

        cont_names = ["IK_question_acc", "IK_KnowledgeTag_acc", "userID_acc", "user_question_class_solved", "solved_question", "userID_KnowledgeTag_total_answer", "userID_KnowledgeTag_acc", "userID_acc_rolling"]
        df = cont_normalize(df, cont_names)

        return df

    def load_data_from_file(self, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])#, nrows=100000)

        df = self.__feature_engineering(df)
        print(df.columns)
        df = self.__preprocessing(df, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

                
        self.args.n_questions = len(np.load(os.path.join(self.args.asset_dir,'assessmentItemID_classes.npy')))
        self.args.n_test = len(np.load(os.path.join(self.args.asset_dir,'testId_classes.npy')))
        self.args.n_tag = len(np.load(os.path.join(self.args.asset_dir,'KnowledgeTag_classes.npy')))

        # 추가 feature
        self.args.n_userID_elapsed_cate = len(np.load(os.path.join(self.args.asset_dir,'userID_elapsed_cate_classes.npy')))
        self.args.n_question_class = len(np.load(os.path.join(self.args.asset_dir,'question_class_classes.npy')))
        self.args.n_userID_assessmentItemID_experience = len(np.load(os.path.join(self.args.asset_dir,'userID_assessmentItemID_experience_classes.npy')))
        

        df = df.sort_values(by=['userID','Timestamp'], axis=0)
        #columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag']
        columns = ['userID', 'Timestamp','assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'userID_elapsed_cate', "IK_question_acc", "question_class", "IK_KnowledgeTag_acc","userID_acc",
                     "userID_assessmentItemID_experience", "user_question_class_solved", "solved_question", "userID_KnowledgeTag_total_answer", "userID_KnowledgeTag_acc", "userID_acc_rolling"]

        #userID 로 group화 ? 다른 feature를 사용하고 싶은경우 여기에다가??
        group = df[columns].groupby('userID').apply(
                lambda r: (
                    r["userID"].values,
                    r["Timestamp"].values,

                    r['testId'].values, 
                    r['assessmentItemID'].values,
                    r['KnowledgeTag'].values,
                    r['answerCode'].values,

                    r['userID_elapsed_cate'].values, 
                    r['IK_question_acc'].values,
                    r['question_class'].values,
                    r['IK_KnowledgeTag_acc'].values,
                    r['userID_acc'].values,
                    r['userID_assessmentItemID_experience'].values,
                    r['user_question_class_solved'].values,
                    r['solved_question'].values,
                    r['userID_KnowledgeTag_total_answer'].values,
                    r['userID_KnowledgeTag_acc'].values,
                    r['userID_acc_rolling'].values
            )
        )
        print(type(group))
        return group.values

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

        # 각 data의 sequence length
        seq_len = len(row[0])

        #test, question, tag, correct = row[0], row[1], row[2], row[3]
        #test, question, tag, correct, time_differ, question_acc, KnowledgeTag_acc, Large_category, user_acc = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
        userid, time, test, question, tag, correct, userID_elapsed_cate, IK_question_acc, question_class, IK_KnowledgeTag_acc, userID_acc, userID_assessmentItemID_experience, user_question_class_solved, solved_question, userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc, userID_acc_rolling = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16]
        # print(question_acc)

        #cate_cols = [test, question, tag, correct]
        cate_cols = [test, question, tag, correct, userID_elapsed_cate, IK_question_acc, question_class, IK_KnowledgeTag_acc, userID_acc, userID_assessmentItemID_experience, user_question_class_solved, solved_question, userID_KnowledgeTag_total_answer, userID_KnowledgeTag_acc, userID_acc_rolling]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len:]
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        # 좌측 패딩 mask : padding 
        else:
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)
        
        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            # print(i, col)
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)

# 데이터마다 length가 다를경우 collate를 이용하여 묶어준다.
def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

        
    # batch의 값들을 각 column끼리 그룹화
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col):] = col
            col_list[i].append(pre_padded)


    for i, _ in enumerate(col_list):
        col_list[i] =torch.stack(col_list[i])
    
    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = True
    train_loader, valid_loader = None, None
    
    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(trainset, num_workers=args.num_workers, shuffle=True,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(valset, num_workers=args.num_workers, shuffle=False,
                            batch_size=args.batch_size, pin_memory=pin_memory, collate_fn=collate)

    return train_loader, valid_loader