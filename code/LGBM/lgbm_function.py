import pandas as pd
import os
import random
import warnings
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
import random
from matplotlib import pylab as plt
warnings.filterwarnings('ignore')


def set_params():
    # 1순위 max_depth
    # 2순위 num_leaves
    # 3순위 min_data_in_leaf
    params = {}
    params["boosting_type"] = "dart" # gbdt, dart, goss
    params["learning_rate"] = 5e-2 # 1e-1, 5e-2, 1e-2, 5e-3, 1e-3
    params["objective"] = "binary"
    params["metric"] = "auc" # binary_logloss, rmse, huber, auc
    params["num_iterations"] = 100 # 100
    params["max_depth"] = -1 # -1
    params["num_leaves"] = 127 # 31 이상적으로 num_leaves값은 2 ^ (max_depth) 값보다 적거나 같아야 합니다.
    params["min_data_in_leaf"] = 100 # 20 100 ~ 1000 수백 또는 수천 개로 정하는 것
    params["max_bin"] = 256 # 256
#     params["scale_pos_weight"] = 0.9 # 1.1~1.5 data 불균형
#     params["tree_learner"] = "serial" # serial, feature, data, voting
#     params["early_stopping_rounds"] = 50
    params["bagging_fraction"] = 0.7 # 1.0
    params["feature_fraction"] = 0.7 # 1.0
    params["lambda_l1"] = 0.1 # 0.0
    params["lambda_l2"] = 0.1 # 0.0   
    
    print("="*30)
    print(params)
    print("="*30)
    print()
    
    return params


# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
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


def inference(FEATS, model, auc, acc, time, test=False):
    print("="*30)
    print("Start inference")
    print("="*30)
    print()
    
    # LOAD TRAIN DATA
    data_dir = '/opt/ml/input/data/train_dataset'
    csv_file_path = os.path.join(data_dir, 'train_data.csv')
    df = pd.read_csv(csv_file_path, parse_dates=['Timestamp'])

    # LOAD TEST DATA
    test_csv_file_path = os.path.join(data_dir, 'test_data.csv')
    test_df = pd.read_csv(test_csv_file_path, parse_dates=['Timestamp'])
    print("="*30)
    print("train, test shape")
    print(df.shape, test_df.shape)
    print("="*30)
    print()
    
    test_df = pd.concat([df, test_df])

    not_test_df = test_df[test_df["answerCode"] != -1]
    not_test_df["is_test"] = False

    test_df = test_df[test_df["answerCode"] == -1]
    test_df["is_test"] = True
    
    print("="*30)
    print("not test, test shape")
    print(not_test_df.shape, test_df.shape)
    print("="*30)
    print()
    
    not_test_df.sort_values(by=["userID", "Timestamp"], inplace=True)
    
    user_mean = not_test_df.groupby(["userID"])["answerCode"].agg(["mean"])
    user_mean.columns = ["user_mean"]
    
    data = pd.concat([not_test_df, test_df], join="inner")
    
    df = pd.merge(data, user_mean, on=["userID"], how="left")
    
    df["next_userID"] = df["userID"].shift(-1)
    
    def random_answering(data):
        if data["is_test"]:
            return 1 if random.random() < 0.5 else 0
        else:
            return data["answerCode"]

    df["answercode"] = df.apply(random_answering, axis=1)

    # FEATURE ENGINEERING
    df = feature_engineering(df)
    
    new_df = df.shift(1)
    new_df.columns = ["pred_" + i for i in new_df.columns]
    data = pd.concat([df, new_df], axis=1)

    # TEST DATA
    test_df = data[data["is_test"]]
    test_df.reset_index(drop=True, inplace=True)
    print("="*30)
    print("test shape check (744, 21)")
    print(test_df.shape)
    print("="*30)
    print()

    # DROP ANSWERCODE
    test_df = test_df.drop(["answerCode"], axis=1)

    # MAKE PREDICTION
    total_preds = model.predict(test_df[FEATS])
    
    # SAVE OUTPUT
    output_dir = 'output/'
    write_path = os.path.join(output_dir, f"output_{time}_AUC_{round(auc, 4)}_ACC_{round(acc, 4)}.csv")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)    
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))
            