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
    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["KnowledgeTag", "Timestamp"], inplace=True)
    
    # KnowledgeTag 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["KnowledgeTag_correct_answer"] = df.groupby("KnowledgeTag")["answercode"].transform(lambda x: x.cumsum().shift(1))
    df["KnowledgeTag_total_answer"] = df.groupby("KnowledgeTag")["answercode"].cumcount()
    df["KnowledgeTag_acc"] = df["KnowledgeTag_correct_answer"] / df["KnowledgeTag_total_answer"]
    
    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["assessmentItemID", "Timestamp"], inplace=True)
    
    # assessmentItemID 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["question_correct_answer"] = df.groupby("assessmentItemID")["answercode"].transform(lambda x: x.cumsum().shift(1))
    df["question_total_answer"] = df.groupby("assessmentItemID")["answercode"].cumcount()
    df["question_acc"] = df["question_correct_answer"] / df["question_total_answer"]
    
    # question class
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
    # user_question_class
    df["userID_question_class"] = df[["userID", "question_class"]].apply(lambda data: str(data["userID"]) + "_" + data["question_class"], axis=1)
    
    # question_class, timestamp 기준 정렬
    df.sort_values(by=["question_class", "Timestamp"], inplace=True)
    
     # question_class 정답 수, 풀이 수, 정답률을 시간순으로 누적해서 계산
    df["question_class_correct_answer"] = df.groupby("question_class")["answercode"].transform(lambda x: x.cumsum().shift(1))
    df["question_class_total_answer"] = df.groupby("question_class")["answercode"].cumcount()
    df["question_class_acc"] = df["question_class_correct_answer"] / df["question_class_total_answer"]
    
    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["userID_question_class", "Timestamp"], inplace=True)
    
    # userID_question_class 정답 수, 풀이 수, 정답률을 시간순으로 누적해서 계산
    df["user_question_class_correct_answer"] = df.groupby("userID_question_class")["answercode"].transform(lambda x: x.cumsum().shift(1))
    df["user_question_class_total_answer"] = df.groupby("userID_question_class")["answercode"].cumcount()
    df["user_question_class_acc"] = df["user_question_class_correct_answer"] / df["user_question_class_total_answer"]
    
    # user별 timestamp 기준 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)
    
    # user 문제 푼 시간 측정
    df["next_timestamp"] = df["Timestamp"].shift(-1)
    df["userID_shift"] = df["userID"].shift(-1)
    # 3min 25s 소요..
    df["time_difference"] = df[["userID", "userID_shift", "Timestamp", "next_timestamp"]].apply(find_time_difference, axis=1)
    
    # question class
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
    
    # user의 문제 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["user_correct_answer"] = df.groupby("userID")["answercode"].transform(lambda x: x.cumsum().shift(1))
    df["user_total_answer"] = df.groupby("userID")["answercode"].cumcount()
    df["user_acc"] = df["user_correct_answer"] / df["user_total_answer"]
    
    # testId 기준 mean, sumanswercode
    group_test = df.groupby(["testId"])["answercode"].agg(["mean", "sum"])
    group_test.columns = ["test_mean", "test_sum"]
    # knowledge_tag 기준 mean, sum
    group_tag = df.groupby(["KnowledgeTag"])["answercode"].agg(["mean", "sum"])
    group_tag.columns = ["tag_mean", "tag_sum"]
    # userID 기준 mean, sum
    group_user = df.groupby(["userID"])["answercode"].agg(["sum"])
    group_user.columns = ["user_count"]
    # question 기준 mean, sum
    group_question = df.groupby(["assessmentItemID"])["answercode"].agg(["mean", "sum"])
    group_question.columns = ["question_mean", "question_count"]
    # question class(assessmentItemID 두 번째 숫자) 기준 mean, sum
    group_question_class = df.groupby(["question_class"])["answercode"].agg(["mean", "sum"])
    group_question_class.columns = ["question_class_mean", "question_class_count"]
    # time_difference 기준 mean, median
    group_time_difference = df.groupby(["userID"])["time_difference"].agg(["mean", "median"])
    group_time_difference.columns = ["time_difference_mean", "time_difference_median"]
    # userID_question_class 기준 mean, sum
    group_user_question_class = df.groupby(["userID_question_class"])["answercode"].agg(["mean", "sum"])
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
