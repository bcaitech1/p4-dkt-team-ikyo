import pandas as pd
import numpy as np


def find_time_difference(data):
    if data["userID"] == data["userID_shift"]:
        temp_time_difference = int(((data["Timestamp"] - data["next_timestamp"]) / pd.to_timedelta(1, unit='D')) * (60 * 60 * 24))
        if temp_time_difference > 600: # 10분 넘는 경우 # 변경 가능
            return 600
        elif temp_time_difference > 3600: # 1시간 넘는 경우 # 변경 가능:
            return 0
        return temp_time_difference
    else:
        return 0

    
def feature_engineering_sun(df):    
    # user별 timestamp 기준 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)
    
    # user 문제 푼 시간 측정
    df["next_timestamp"] = df["Timestamp"].shift(1)
    df["userID_shift"] = df["userID"].shift(1)
    # user별 시간 차이 계산
    df["time_difference"] = df[["userID", "userID_shift", "Timestamp", "next_timestamp"]].apply(find_time_difference, axis=1)
    
    # user의 문제 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["user_correct_answer"] = df.groupby("userID")["answerCode"].transform(lambda x: x.cumsum().shift(1))
    df["user_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
    df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]
    
    # testId 기준 mean, sum
    group_test = df.groupby(["testId"])["answerCode"].agg(["mean", "sum"])
    group_test.columns = ["test_mean", "test_sum"]
    # knowledge_tag 기준 mean, sum
    group_tag = df.groupby(["KnowledgeTag"])["answerCode"].agg(["mean", "sum"])
    group_tag.columns = ["tag_mean", "tag_sum"]
    # userID 기준 mean, sum
    group_user = df.groupby(["userID"])["answerCode"].agg(["mean", "sum"])
    group_user.columns = ["user_mean", "user_count"]
    # question 기준 mean, sum
    group_question = df.groupby(["assessmentItemID"])["answerCode"].agg(["mean", "sum"])
    group_question.columns = ["question_mean", "question_count"]
    # question class(assessmentItemID 두 번째 숫자) 기준 mean, sum
    group_question_class = df.groupby(["question_class"])["answerCode"].agg(["mean", "sum"])
    group_question_class.columns = ["question_class_mean", "question_class_count"]
    # time_difference 기준 mean, median
    group_time_difference = df.groupby(["userID"])["time_difference"].agg(["mean", "median"])
    group_time_difference.columns = ["time_difference_mean", "time_difference_median"]
    # userID_question_class 기준 mean, sum
    group_user_question_class = df.groupby(["userID_question_class"])["answerCode"].agg(["mean", "sum"])
    group_user_question_class.columns = ["user_question_class_mean", "user_question_class_count"]
    
    # merge
    df = pd.merge(df, group_test, on=["testId"], how="left")
    df = pd.merge(df, group_tag, on=["KnowledgeTag"], how="left")
    df = pd.merge(df, group_user, on=["userID"], how="left")
    df = pd.merge(df, group_question, on=["assessmentItemID"], how="left")
    df = pd.merge(df, group_question_class, on=["question_class"], how="left")
    df = pd.merge(df, group_time_difference, on=["userID"], how="left")
    df = pd.merge(df, group_user_question_class, on=["userID_question_class"], how="left")
    
    return df
