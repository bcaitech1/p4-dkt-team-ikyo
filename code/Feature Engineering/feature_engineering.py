import time
import random
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA, KernelPCA, TruncatedSVD

tqdm.pandas()


def IK_question_acc(df):
    assessmentItemID_groupby = df.groupby("assessmentItemID").agg(
        {"answercode": "mean"}
    )

    df["IK_question_acc"] = assessmentItemID_groupby["answercode"][
        df["assessmentItemID"]
    ].values
    return df


def IK_KnowledgeTag_acc(df):
    KnowledgeTag_groupby = df.groupby("KnowledgeTag").agg({"answercode": "mean"})

    df["IK_KnowledgeTag_acc"] = KnowledgeTag_groupby["answercode"][
        df["KnowledgeTag"]
    ].values

    return df


def solved_question(df):
    df["solved_question"] = df.groupby(["userID"]).cumcount()

    return df


def user_question_class_solved(df):
    if "question_class" not in df.columns:
        df = question_class(df)

    df["user_question_class_solved"] = df.groupby(
        ["userID", "question_class"]
    ).cumcount()
    return df


def userID_elapsed_cate(df, max_time=600):
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # sample별 elapsed time
    diff = df.loc[:, ["userID", "Timestamp"]].groupby("userID").diff().shift(-1)
    elapsed = diff["Timestamp"].apply(
        lambda x: int(x.total_seconds() // 10 * 10)
        if max_time > x.total_seconds()
        else 1
    )
    df["userID_elapsed_cate"] = elapsed

    return df


def userID_testid_experience(df):
    # userID별 시간 순으로 정렬
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

    # userID 별로 testid를 풀어본 적 있는지
    df["userID_testid_experience"] = df.groupby(["userID", "testId"])[
        "testId"
    ].cumcount()
    df["userID_testid_experience"] = df["userID_testid_experience"].apply(
        lambda x: 1 if x > 0 else 0
    )
    return df


def userID_assessmentItemID_experience(df):
    # userID별 시간 순으로 정렬
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

    # userID 별로 assessmentItemID를 풀어본 적 있는지
    df["userID_assessmentItemID_experience"] = df.groupby(
        ["userID", "assessmentItemID"]
    )["assessmentItemID"].cumcount()
    df["userID_assessmentItemID_experience"] = df[
        "userID_assessmentItemID_experience"
    ].apply(lambda x: 1 if x > 0 else 0)
    return df


def userID_time_diff_from_last(df):
    def convert_time(s):
        timestamp = time.mktime(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple())
        return int(timestamp)

    # 초 단위 시간
    df["sec"] = df["Timestamp"].apply(convert_time)

    # userID별 시간 순으로 정렬 + index column 생성
    df = df.sort_values(by=["userID", "sec"]).reset_index(drop=False)

    # userID별 마지막 index 값
    last_idx_group = df.groupby(["userID"])["index"].agg(["max"])
    last_idx_group = last_idx_group.reset_index()
    last_idx_group.columns = ["userID", "last_index"]
    df = pd.merge(df, last_idx_group, on=["userID"], how="left")

    def changed_time(x):
        last_time = df["sec"][x["last_index"]]
        period = last_time - x["sec"]
        return period

    # userID별 마지막 index의 시간과의 차이 계산
    df["userID_time_diff_from_last"] = df.apply(changed_time, axis=1)

    df.drop("sec", axis=1, inplace=True)
    df.drop("index", axis=1, inplace=True)
    return df


def userID_KnowledgeTag_relative(df):
    # userID별 시간 순으로 정렬
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    # userID, KnowledgeTag 키값 생성(temp)
    df["tmp"] = df[["userID", "KnowledgeTag"]].apply(
        lambda data: str(data["userID"]) + "_" + str(data["KnowledgeTag"]), axis=1
    )
    # userID, KnowledgeTag별 누적 풀이 수, 정답 수, 정답률
    df["userID_KnowledgeTag_total_answer"] = df.groupby("tmp")["answercode"].cumcount()
    df["userID_KnowledgeTag_correct_answer"] = df.groupby("tmp")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["userID_KnowledgeTag_correct_answer"].fillna(0, inplace=True)
    df["userID_KnowledgeTag_acc"] = (
        df["userID_KnowledgeTag_correct_answer"]
        / df["userID_KnowledgeTag_total_answer"]
    )
    df["userID_KnowledgeTag_acc"].fillna(0, inplace=True)
    df.drop("tmp", axis=1, inplace=True)
    return df


def userID_question_num_relative(df):
    # question_num이 있어야 계산 가능
    if "question_num" not in df.columns:
        df = question_num(df)
    # userID별 시간 순으로 정렬
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    # userID_question_class 키값 생성(temp)
    df["tmp"] = df[["userID", "question_num"]].apply(
        lambda data: str(data["userID"]) + "_" + str(data["question_num"]), axis=1
    )
    # userID, question_num별 누적 풀이 수, 정답 수, 정답률
    df["userID_question_num_total_answer"] = df.groupby("tmp")["answercode"].cumcount()
    df["userID_question_num_correct_answer"] = df.groupby("tmp")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["userID_question_num_correct_answer"].fillna(0, inplace=True)
    df["userID_question_num_acc"] = (
        df["userID_question_num_correct_answer"]
        / df["userID_question_num_total_answer"]
    )
    df["userID_question_num_acc"].fillna(0, inplace=True)
    df.drop("tmp", axis=1, inplace=True)
    return df


def userID_elapsed_median(df, max_time=600):
    # 약 1m 50s 소요(Progress bar 2개 생김)
    # userID별 시간 순으로 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # sample별 elapsed time
    diff = df.loc[:, ["userID", "Timestamp"]].groupby("userID").diff().shift(-1)
    elapsed = diff["Timestamp"].progress_apply(
        lambda x: x.total_seconds() if max_time > x.total_seconds() else None
    )
    df["userID_elapsed_median"] = elapsed

    # userID별 마지막 문제의 풀이 시간(데이터에서 알 수 없는)을
    # userID별 문제 풀이 시간의 "중앙값"으로 반환하기 위한 Aggregation
    user_median = df.groupby("userID")["userID_elapsed_median"].median()
    df = pd.merge(df, user_median, on=["userID"], how="left")

    # 결측치 중앙값 변환 및 임시 열 삭제
    df["userID_elapsed_median_x"] = df["userID_elapsed_median_x"].fillna("missing")

    def changed_elapsed(data):
        return (
            data["userID_elapsed_median_x"]
            if data["userID_elapsed_median_x"] != "missing"
            else data["userID_elapsed_median_y"]
        )

    df["userID_elapsed_median"] = df.progress_apply(changed_elapsed, axis=1)
    df.drop("userID_elapsed_median_x", axis=1, inplace=True)
    df.drop("userID_elapsed_median_y", axis=1, inplace=True)
    return df


def userID_elapsed_median_rolling(df, window=10):
    # userID_elapsed_median이 있어야 이동평균 계산 가능
    if "userID_elapsed_median" not in df.columns:
        df = userID_elapsed_median(df)
    # userID별 시간 순으로 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # userID별 문제 풀이 시간의 이동평균
    df["userID_elapsed_median_rolling"] = (
        df.groupby(["userID"])["userID_elapsed_median"].rolling(window).mean().values
    )
    # 유저별 window-1만큼 N/A data가 생김(rolling의 특성상 앞데이터에 생김)
    # 유저별 userID_elapsed_median_rolling의 중앙값으로 대체
    def changed_mean_time(data):
        return (
            data["userID_elapsed_median_rolling_x"]
            if data["userID_elapsed_median_rolling_x"] != "missing"
            else data["userID_elapsed_median_rolling_y"]
        )

    user_median = df.groupby("userID")["userID_elapsed_median_rolling"].median()
    df = pd.merge(df, user_median, on=["userID"], how="left")

    # 결측치 중앙값 변환 및 임시 열 삭제
    df["userID_elapsed_median_rolling_x"] = df[
        "userID_elapsed_median_rolling_x"
    ].fillna("missing")
    df["userID_elapsed_median_rolling_" + str(window)] = df.progress_apply(
        changed_mean_time, axis=1
    )
    df.drop("userID_elapsed_median_rolling_x", axis=1, inplace=True)
    df.drop("userID_elapsed_median_rolling_y", axis=1, inplace=True)
    return df


def question_num(df):
    # 문제지 안 문제 번호
    df["question_num"] = df["assessmentItemID"].apply(lambda x: x[-3:])
    return df


def question_class(df):
    # 문제지 안 문제 번호
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
    return df


def KnowledgeTag_relative(df):
    # KnowledgeTag별 누적 풀이 수, 정답 수, 정답률
    df_KnowledgeTag = df.sort_values(by=["KnowledgeTag", "Timestamp"])
    df["KnowledgeTag_total_answer"] = df_KnowledgeTag.groupby("KnowledgeTag")[
        "answerCode"
    ].cumcount()
    df["KnowledgeTag_correct_answer"] = (
        df_KnowledgeTag.groupby("KnowledgeTag")["answerCode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["KnowledgeTag_acc"] = (
        df["KnowledgeTag_correct_answer"] / df["KnowledgeTag_total_answer"]
    ).fillna(0)
    return df


def assessmentItemID_relative(df):
    # assessmentItemID별 누적 풀이 수, 정답 수, 정답률
    df_assessmentItemID = df.sort_values(by=["assessmentItemID", "Timestamp"])
    df["assessmentItemID_total_answer"] = df_assessmentItemID.groupby(
        "assessmentItemID"
    )["answerCode"].cumcount()
    df["assessmentItemID_correct_answer"] = (
        df_assessmentItemID.groupby("assessmentItemID")["answerCode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["assessmentItemID_acc"] = (
        df["assessmentItemID_correct_answer"] / df["assessmentItemID_total_answer"]
    ).fillna(0)
    return df


def question_class_relative(df):
    if "question_class" not in df.columns:
        df = question_class(df)
    # Question Class 별 누적 풀이 수, 정답 수, 정답률
    df.sort_values(by=["question_class", "Timestamp"], inplace=True)
    df["question_class_correct_answer"] = (
        df.groupby("question_class")["answerCode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["question_class_total_answer"] = df.groupby("question_class")[
        "answerCode"
    ].cumcount()
    df["question_class_acc"] = (
        df["question_class_correct_answer"] / df["question_class_total_answer"]
    ).fillna(0)
    return df


def question_num_relative(df):
    if "question_num" not in df.columns:
        df = question_class(df)
    # Question Class 별 누적 풀이 수, 정답 수, 정답률
    df.sort_values(by=["question_num", "Timestamp"], inplace=True)
    df["question_num_correct_answer"] = (
        df.groupby("question_num")["answercode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["question_num_total_answer"] = df.groupby("question_num")[
        "answercode"
    ].cumcount()
    df["question_num_acc"] = (
        df["question_num_correct_answer"] / df["question_num_total_answer"]
    ).fillna(0)
    return df


def userID_question_class_relative(df):
    # question_class 있어야 계산 가능
    if "question_class" not in df.columns:
        df = question_class(df)

    # userID별 시간 순으로 정렬
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    # userID_question_class 키값 생성(temp)
    df["tmp"] = df[["userID", "question_class"]].apply(
        lambda data: str(data["userID"]) + "_" + data["question_class"], axis=1
    )
    # userID_question_class 별 누적 풀이 수, 정답 수, 정답률
    df["userID_question_class_total_answer"] = df.groupby("tmp")[
        "answercode"
    ].cumcount()
    df["userID_question_class_correct_answer"] = df.groupby("tmp")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["userID_question_class_correct_answer"].fillna(0, inplace=True)
    df["userID_question_class_acc"] = (
        df["userID_question_class_correct_answer"]
        / df["userID_question_class_total_answer"]
    )
    df["userID_question_class_acc"].fillna(0, inplace=True)
    df.drop("tmp", axis=1, inplace=True)
    return df


def userID_relative(df):
    # userID별 시간 순으로 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)
    # user 별 누적 풀이 수, 정답 수, 정답률
    df["userID_correct_answer"] = (
        df.groupby("userID")["answerCode"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["userID_total_answer"] = df.groupby("userID")["answerCode"].cumcount()
    df["userID_acc"] = (df["userID_correct_answer"] / df["userID_total_answer"]).fillna(
        0
    )
    return df


def userID_acc_rolling(df, window=5):
    # user_acc 있어야 이동평균 계산 가능
    if "userID_acc" not in df.columns:
        df = userID_relative(df)
    # userID별 시간 순으로 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # userID별 정답률(user_acc)의 이동 평균
    df["userID_acc_rolling"] = (
        df.groupby(["userID"])["userID_acc"].rolling(window).mean().values
    )
    # userID별 window-1만큼 N/A data가 생김(rolling의 특성상 앞데이터에 생김)
    # userID별 user_acc_rolling의 중앙값으로 대체
    def changed_user_acc_rolling(data):
        return (
            data["userID_acc_rolling_x"]
            if data["userID_acc_rolling_x"] != "missing"
            else data["userID_acc_rolling_y"]
        )

    user_median = df.groupby("userID")["userID_acc_rolling"].median()
    df = pd.merge(df, user_median, on=["userID"], how="left")
    # 결측치 중앙값 변환 및 임시 열 삭제
    df["userID_acc_rolling_x"] = df["userID_acc_rolling_x"].fillna("missing")
    df["userID_acc_rolling_" + str(window)] = df.progress_apply(
        changed_user_acc_rolling, axis=1
    )
    df.drop("userID_acc_rolling_x", axis=1, inplace=True)
    df.drop("userID_acc_rolling_y", axis=1, inplace=True)

    return df


def assessmentItemID_time_relative(df):
    # 문제별 풀이 시간의 중앙값&평균값
    # userID_elapsed_median 있어야 assessmentItemID_time 계산 가능
    if "userID_elapsed_median" not in df.columns:
        df = userID_elapsed_median(df)
    # assessmentItemID별 풀이 시간의 중앙값&평균값
    df_total_agg = df.copy()
    agg_df = df_total_agg.groupby("assessmentItemID")["userID_elapsed_median"].agg(
        ["median", "mean"]
    )
    # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
    agg_dict = agg_df.to_dict()
    # 구한 통계량을 각 사용자에게 mapping
    df["assessmentItemID_time_median"] = df_total_agg["assessmentItemID"].map(
        agg_dict["median"]
    )
    df["assessmentItemID_time_mean"] = df_total_agg["assessmentItemID"].map(
        agg_dict["mean"]
    )
    return df


def userID_time_relative(df):
    # 유저별 풀이 시간의 중앙값&평균값
    # userID_elapsed_median 있어야 userID_time_relative 계산 가능
    if "userID_elapsed_median" not in df.columns:
        df = userID_elapsed_median(df)
    # assessmentItemID별 풀이 시간의 중앙값&평균값
    df_total_agg = df.copy()
    agg_df = df_total_agg.groupby("userID")["userID_elapsed_median"].agg(
        ["median", "mean"]
    )
    # mapping을 위해 pandas DataFrame을 dictionary형태로 변환
    agg_dict = agg_df.to_dict()
    # 구한 통계량을 각 사용자에게 mapping
    df["userID_time_median"] = df_total_agg["userID"].map(agg_dict["median"])
    df["userID_time_mean"] = df_total_agg["userID"].map(agg_dict["mean"])
    return df


def userID_elapsed_normalize(df):
    # userID_elapsed_normalize 있어야 userID_elapsed_normalize 계산 가능
    if "userID_elapsed_normalize" not in df.columns:
        df = userID_elapsed_median(df)
    df_total_norm = df.copy()
    df["userID_elapsed_normalize"] = df_total_norm.groupby("userID")[
        "userID_elapsed_median"
    ].transform(lambda x: (x - x.mean()) / x.std())
    return df


def feature_dimension_reduction(df, kind="lda"):
    if "assessmentItemID_total_answer" not in df.columns:
        df = assessmentItemID_relative(df)
    if "KnowledgeTag_total_answer" not in df.columns:
        df = KnowledgeTag_relative(df)
    if "question_class_correct_answer" not in df.columns:
        df = question_class_relative(df)
    if "userID_question_class_correct_answer" not in df.columns:
        df = userID_question_class_relative(df)

    if kind == "lda":
        model = LDA(n_components=1)
    elif kind == "pca":
        model = PCA(n_components=1)
    elif kind == "kpca":
        model = KernelPCA(n_components=1)
    elif kind == "kpca_rbf":
        model = KernelPCA(n_components=1, kernel="rbf")
    elif kind == "kpca_poly":
        model = KernelPCA(n_components=1, kernel="poly")
    elif kind == "svd":
        model = TruncatedSVD(n_components=1)
    else:
        return df

    y = df["answercode"]

    # KnowledgeTag_dimension_reduction
    X = df[
        ["KnowledgeTag_total_answer", "KnowledgeTag_correct_answer", "KnowledgeTag_acc"]
    ].fillna(0)
    df["KnowledgeTag_" + kind] = model.fit_transform(X, y)
    # userID_KnowledgeTag_dimension_reduction
    X = df[
        [
            "userID_KnowledgeTag_total_answer",
            "userID_KnowledgeTag_correct_answer",
            "userID_KnowledgeTag_acc",
        ]
    ].fillna(0)
    df["userID_KnowledgeTag_" + kind] = model.fit_transform(X, y)
    # assessmentItemID_dimension_reduction
    X = df[
        [
            "assessmentItemID_total_answer",
            "assessmentItemID_correct_answer",
            "assessmentItemID_acc",
        ]
    ].fillna(0)
    df["assessmentItemID_" + kind] = model.fit_transform(X, y)
    # question_class_dimension_reduction
    X = df[
        [
            "question_class_correct_answer",
            "question_class_total_answer",
            "question_class_acc",
        ]
    ].fillna(0)
    df["question_class_" + kind] = model.fit_transform(X, y)
    # user_question_class_dimension_reductio
    X = df[
        [
            "userID_question_class_correct_answer",
            "userID_question_class_total_answer",
            "userID_question_class_acc",
        ]
    ].fillna(0)
    df["userID_question_class_" + kind] = model.fit_transform(X, y)
    # question_num_dimension_reduction
    X = df[
        ["question_num_correct_answer", "question_num_total_answer", "question_num_acc"]
    ].fillna(0)
    df["question_num_" + kind] = model.fit_transform(X, y)
    # user_question_num_dimension_reduction
    X = df[
        [
            "userID_question_num_correct_answer",
            "userID_question_num_total_answer",
            "userID_question_num_acc",
        ]
    ].fillna(0)
    df["userID_question_num_" + kind] = model.fit_transform(X, y)
    # userID_dimension_reduction
    X = df[["userID_correct_answer", "userID_total_answer", "userID_acc"]].fillna(0)
    df["userID_" + kind] = model.fit_transform(X, y)
    # all_data_dimension_reduction
    X = df.iloc[:, -8:]
    df["all_data_" + kind] = model.fit_transform(X, y)

    return df


def find_time_difference(data):
    if data["userID"] == data["userID_shift"]:
        temp_time_difference = int(
            (
                (data["Timestamp"] - data["next_timestamp"])
                / pd.to_timedelta(1, unit="D")
            )
            * (60 * 60 * 24)
        )
        if temp_time_difference > 600:  # 10분 넘는 경우 # 변경 가능
            return 600
        elif temp_time_difference > 3600:  # 1시간 넘는 경우 # 변경 가능:
            return 0
        return temp_time_difference
    else:
        return 0


def feature_engineering_sun(df):
    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["KnowledgeTag", "Timestamp"], inplace=True)

    # KnowledgeTag 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["KnowledgeTag_correct_answer"] = df.groupby("KnowledgeTag")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["KnowledgeTag_total_answer"] = df.groupby("KnowledgeTag")[
        "answercode"
    ].cumcount()
    df["KnowledgeTag_acc"] = (
        df["KnowledgeTag_correct_answer"] / df["KnowledgeTag_total_answer"]
    )

    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["assessmentItemID", "Timestamp"], inplace=True)

    # assessmentItemID 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["question_correct_answer"] = df.groupby("assessmentItemID")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["question_total_answer"] = df.groupby("assessmentItemID")[
        "answercode"
    ].cumcount()
    df["question_acc"] = df["question_correct_answer"] / df["question_total_answer"]

    # question class
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
    # user_question_class
    df["userID_question_class"] = df[["userID", "question_class"]].apply(
        lambda data: str(data["userID"]) + "_" + data["question_class"], axis=1
    )

    # question_class, timestamp 기준 정렬
    df.sort_values(by=["question_class", "Timestamp"], inplace=True)

    # question_class 정답 수, 풀이 수, 정답률을 시간순으로 누적해서 계산
    df["question_class_correct_answer"] = df.groupby("question_class")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["question_class_total_answer"] = df.groupby("question_class")[
        "answercode"
    ].cumcount()
    df["question_class_acc"] = (
        df["question_class_correct_answer"] / df["question_class_total_answer"]
    )

    # assessmentItemID, timestamp 기준 정렬
    df.sort_values(by=["userID_question_class", "Timestamp"], inplace=True)

    # userID_question_class 정답 수, 풀이 수, 정답률을 시간순으로 누적해서 계산
    df["user_question_class_correct_answer"] = df.groupby("userID_question_class")[
        "answercode"
    ].transform(lambda x: x.cumsum().shift(1))
    df["user_question_class_total_answer"] = df.groupby("userID_question_class")[
        "answercode"
    ].cumcount()
    df["user_question_class_acc"] = (
        df["user_question_class_correct_answer"]
        / df["user_question_class_total_answer"]
    )

    # user별 timestamp 기준 정렬
    df.sort_values(by=["userID", "Timestamp"], inplace=True)

    # user 문제 푼 시간 측정
    df["next_timestamp"] = df["Timestamp"].shift(-1)
    df["userID_shift"] = df["userID"].shift(-1)
    # 3min 25s 소요..
    df["time_difference"] = df[
        ["userID", "userID_shift", "Timestamp", "next_timestamp"]
    ].apply(find_time_difference, axis=1)

    # question class
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])

    # user의 문제 풀이 수, 정답 수, 정답률을 시간순으로 누적해서 계산
    df["user_correct_answer"] = df.groupby("userID")["answercode"].transform(
        lambda x: x.cumsum().shift(1)
    )
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
    group_question_class = df.groupby(["question_class"])["answercode"].agg(
        ["mean", "sum"]
    )
    group_question_class.columns = ["question_class_mean", "question_class_count"]
    # time_difference 기준 mean, median
    group_time_difference = df.groupby(["userID"])["time_difference"].agg(
        ["mean", "median"]
    )
    group_time_difference.columns = ["time_difference_mean", "time_difference_median"]
    # userID_question_class 기준 mean, sum
    group_user_question_class = df.groupby(["userID_question_class"])["answercode"].agg(
        ["mean", "sum"]
    )
    group_user_question_class.columns = [
        "user_question_class_mean",
        "user_question_class_count",
    ]

    # merge
    df = pd.merge(df, group_test, on=["testId"], how="left")
    df = pd.merge(df, group_tag, on=["KnowledgeTag"], how="left")
    df = pd.merge(df, group_user, on=["userID"], how="left")
    df = pd.merge(df, group_question, on=["assessmentItemID"], how="left")
    df = pd.merge(df, group_question_class, on=["question_class"], how="left")
    df = pd.merge(df, group_time_difference, on=["userID"], how="left")
    df = pd.merge(
        df, group_user_question_class, on=["userID_question_class"], how="left"
    )

    return df


# ======================================================================================================= #

# ACC 같은 민감한 정보를 Categorical로 바꾸는 함수
def make_grade(data):
    if data < 0.05:
        return 0
    elif data < 0.10:
        return 1
    elif data < 0.15:
        return 2
    elif data < 0.20:
        return 3
    elif data < 0.25:
        return 4
    elif data < 0.30:
        return 5
    elif data < 0.35:
        return 6
    elif data < 0.40:
        return 7
    elif data < 0.45:
        return 8
    elif data < 0.50:
        return 9
    elif data < 0.55:
        return 10
    elif data < 0.60:
        return 11
    elif data < 0.65:
        return 12
    elif data < 0.70:
        return 13
    elif data < 0.75:
        return 14
    elif data < 0.80:
        return 15
    elif data < 0.85:
        return 16
    elif data < 0.90:
        return 17
    elif data < 0.95:
        return 18
    else:
        return 19


# dataframe을 만들고 기본 세팅하는 함수 (1분 소요)
def get_df():
    dtype = {"userID": "int16", "answerCode": "int8", "KnowledgeTag": "int16"}

    DATA_PATH = "/opt/ml/input/data/train_dataset/train_data.csv"
    TEST_DATA_PATH = "/opt/ml/input/data/train_dataset/test_data.csv"

    train_df = pd.read_csv(DATA_PATH, dtype=dtype, parse_dates=["Timestamp"])
    train_df = train_df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    test_df = pd.read_csv(TEST_DATA_PATH, dtype=dtype, parse_dates=["Timestamp"])
    test_df = test_df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)

    train_df["is_test"] = False
    test_df["is_test"] = True

    df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    df["next_userID"] = df["userID"].shift(-1).fillna(9999)

    def answer_masking(df):
        if df["userID"] != df["next_userID"]:
            return 1 if random.random() < 0.5 else 0
        else:
            return df["answerCode"]

    df["masked_answer"] = df.apply(answer_masking, axis=1)

    return df


# 문제의 난이도 Feature
def get_question_grade(df):
    tmp_df = df.groupby("assessmentItemID")["masked_answer"].mean().reset_index()
    tmp_df.columns = ["assessmentItemID", "question_grade"]
    tmp_df["question_grade"] = tmp_df["question_grade"].apply(make_grade)
    df = pd.merge(left=df, right=tmp_df, on=["assessmentItemID"], how="left")
    return df


# 문제의 번호 Feature
def get_question_order(df):
    df["question_order"] = df["assessmentItemID"].apply(lambda x: int(x[-3:]))
    return df


# 문제의 대분류 Feature
def get_question_large_cate(df):
    df["question_large_cate"] = df.apply(lambda x: int(x["testId"][2]), axis=1)
    return df


# User가 해당 대분류의 문제를 몇번 풀었는지 Feature
def get_userID_cnt_item_in_largeCate(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    if "question_large_cate" not in df.columns:
        df = get_question_large_cate(df)
    tmp_df = df.copy()
    tmp_df["tmp"] = tmp_df[["userID", "question_large_cate"]].apply(
        lambda data: str(data["userID"]) + "_" + str(data["question_large_cate"]),
        axis=1,
    )
    df["userID_cnt_item_in_largeCate"] = tmp_df.groupby("tmp")[
        "assessmentItemID"
    ].cumcount()
    return df


# User의 해당 대분류에 대한 정답률 Grade Feature
def get_userID_answerRate_in_largeCate(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    if "question_large_cate" not in df.columns:
        df = get_question_large_cate(df)
    tmp_df = df.copy()
    tmp_df["tmp"] = tmp_df.apply(
        lambda data: str(data["userID"]) + "_" + str(data["question_large_cate"]),
        axis=1,
    )
    tmp_df["answers"] = (
        tmp_df.groupby("tmp")["masked_answer"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    df["userID_answerRate_in_largeCate"] = (
        tmp_df["answers"] / tmp_df["userID_cnt_item_in_largeCate"]
    ).fillna(1)
    df["userID_answerRate_in_largeCate"] = df["userID_answerRate_in_largeCate"].apply(
        make_grade
    )
    return df


# User의 해당 Tag에 대한 정답률 Grade Feature
def get_userID_answerRate_in_tag(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    tmp_df = df.copy()
    tmp_df["tmp"] = tmp_df.apply(
        lambda data: str(data["userID"]) + "_" + str(data["KnowledgeTag"]), axis=1
    )
    tmp_df["answers"] = (
        tmp_df.groupby("tmp")["masked_answer"]
        .transform(lambda x: x.cumsum().shift(1))
        .fillna(0)
    )
    tmp_df["userID_cnt_item_in_tag"] = tmp_df.groupby("tmp")[
        "assessmentItemID"
    ].cumcount()
    df["userID_answerRate_in_tag"] = (
        tmp_df["answers"] / tmp_df["userID_cnt_item_in_tag"]
    ).fillna(1)
    df["userID_answerRate_in_tag"] = df["userID_answerRate_in_tag"].apply(make_grade)
    return df


# User가 해당 문제를 풀어본 경험 Feature
def get_userID_question_experience(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    df["userID_question_experience"] = df.groupby(["userID", "assessmentItemID"])[
        "assessmentItemID"
    ].cumcount()
    return df


# User가 문제를 순서대로 접근하는지 Feature
def get_question_solve_order(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    if "question_order" not in df.columns:
        df = get_question_order(df)
    tmp_df = (
        df.loc[:, ["userID", "testId", "question_order"]]
        .groupby(["userID", "testId"])
        .diff()
        .fillna(1)
    )
    df["question_solve_order"] = tmp_df["question_order"].apply(
        lambda x: 1 if x == 1 else 0
    )
    return df


# 문제를 푸는데 걸리는 시간 Feature (UserID와 TestID 기준으로 구하고, Max 및 nan 값은 125로 사용)
def get_userID_elapsed_by_test_125(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    diff = (
        df.loc[:, ["userID", "testId", "Timestamp"]]
        .groupby(["userID", "testId"])
        .diff()
        .shift(-1)
        .fillna(pd.Timedelta(seconds=0))
    )
    diff = diff.fillna(pd.Timedelta(seconds=0))
    diff = diff["Timestamp"].apply(lambda x: x.total_seconds())
    df["elapsed_solving"] = diff
    df["elapsed_solving"] = df["elapsed_solving"].apply(
        lambda x: 125 if x == 0 or x > 125 else x
    )
    return df


# User가 문제를 풀때 걸리는 시간의 중앙값 Feature
def get_userID_elapsed_median(df):
    if "elapsed_solving" not in df.columns:
        df = get_userID_elapsed_by_test_125(df)
    tmp_df = df.groupby("userID")["elapsed_solving"].median()
    tmp_df.name = "userID_elapsed_median"
    tmp_df = tmp_df.reset_index()
    df = pd.merge(left=df, right=tmp_df, on="userID", how="left")
    return df


# 문제를 맞춘 사람과 못맞춘 사람이 걸리는 시간의 중앙값 Feature
def get_question_elapsed_median(df):
    tmp_df = df.groupby(["assessmentItemID", "masked_answer"]).agg(
        {"elapsed_solving": "median"}
    )
    tmp_df = tmp_df.reset_index()
    tmp_df_correct = tmp_df[tmp_df["masked_answer"] == 1]
    tmp_df_incorrect = tmp_df[tmp_df["masked_answer"] == 0]
    tmp_df_correct.columns = [
        "assessmentItemID",
        "masked_answer",
        "question_correct_elapsed_median",
    ]
    tmp_df_incorrect.columns = [
        "assessmentItemID",
        "masked_answer",
        "question_incorrect_elapsed_median",
    ]
    tmp_df_correct = tmp_df_correct.drop("masked_answer", axis=1)
    tmp_df_incorrect = tmp_df_incorrect.drop("masked_answer", axis=1)
    df = pd.merge(left=df, right=tmp_df_correct, on=["assessmentItemID"], how="left")
    df = pd.merge(left=df, right=tmp_df_incorrect, on=["assessmentItemID"], how="left")
    return df


# 문제를 접근한 날짜 Feature (월단위)
def get_question_solve_month(df):
    df["question_solve_month"] = df["Timestamp"].apply(lambda x: x.month)
    return df


# User가 이전에 몇 문제를 풀었는지 Feature
def get_userID_cnt_items(df):
    df = df.sort_values(by=["userID", "Timestamp"]).reset_index(drop=True)
    df["userID_cnt_items"] = df.groupby("userID")["assessmentItemID"].cumcount()
    return df


# User가 이전에 몇개의 시험지를 풀었는지 Feature
def get_userID_cnt_tests(df):
    tmp_df = df[["userID", "testId"]]
    tmp_df = tmp_df.drop_duplicates()
    tmp_df["userID_cnt_tests"] = tmp_df.groupby("userID")["testId"].cumcount()
    df = pd.merge(left=df, right=tmp_df, on=["userID", "testId"], how="left")
    return df


# User가 이전에 몇개의 Tag를 풀었는지 Feature
def get_userID_cnt_tags(df):
    tmp_df = df[["userID", "KnowledgeTag"]]
    tmp_df = tmp_df.drop_duplicates()
    tmp_df["userID_cnt_tags"] = tmp_df.groupby("userID")["KnowledgeTag"].cumcount()
    df = pd.merge(left=df, right=tmp_df, on=["userID", "KnowledgeTag"], how="left")
    return df


# Dataframe부터, 전체 Feature Engineering까지 수행되는 Code
def get_all():
    df = get_df()
    df = get_question_grade(df)
    df = get_question_order(df)
    df = get_question_large_cate(df)
    df = get_userID_cnt_item_in_largeCate(df)
    df = get_userID_answerRate_in_largeCate(df)
    df = get_userID_answerRate_in_tag(df)
    df = get_userID_question_experience(df)
    df = get_question_solve_order(df)
    df = get_userID_elapsed_by_test_125(df)
    df = get_userID_elapsed_median(df)
    df = get_question_elapsed_median(df)
    df = get_question_solve_month(df)
    df = get_userID_cnt_items(df)
    df = get_userID_cnt_tests(df)
    df = get_userID_cnt_tags(df)
    return df


def timestamp(df):
    # year, month
    df["year"] = df["Timestamp"].apply(lambda x: x.year)
    df["month"] = df["Timestamp"].apply(lambda x: x.month)
    df["year"] = df["year"].astype("category")
    df["month"] = df["month"].astype("category")
    return df


def assessmentItemID(df):
    df["assessmentItemID"] = df["assessmentItemID"].astype("category")
    df["question_num"] = df["assessmentItemID"].apply(lambda x: int(x[-2:]))
    df["question_class"] = df["assessmentItemID"].apply(lambda x: x[2])
    return df


def feature_engineering_sun(df):
    print("Timestamp 관련 feature")
    df = timestamp(df)

    print("assessmentItemID 관련 feature")
    df = assessmentItemID(df)

    print("KnowledgeTag별 누적 풀이 수, 정답 수, 정답률")
    df = KnowledgeTag_relative(df)

    print("userID, KnowledgeTag별 누적 풀이 수, 정답 수, 정답률")
    df = userID_KnowledgeTag_relative(df)

    print("assessmentItemID별 누적 풀이 수, 정답 수, 정답률")
    df = assessmentItemID_relative(df)

    print("question class별 누적 풀이 수, 정답 수, 정답률")
    df = question_class_relative(df)

    print("userID_question_class별 누적 풀이 수, 정답 수, 정답률")
    df = userID_question_class_relative(df)

    print("question num별 누적 풀이 수, 정답 수, 정답률")
    df = question_num_relative(df)

    print("userID_question_num별 누적 풀이 수, 정답 수, 정답률")
    df = userID_question_num_relative(df)

    print("user 별 누적 풀이 수, 정답 수, 정답률")
    df = userID_relative(df)

    print("userID별 timestamp 중앙값")
    df = userID_elapsed_median(df, max_time=600)

    print("문제별 풀이 시간의 중앙값&평균값")
    df = assessmentItemID_time_relative(df)

    print("userID별 정답률(user_acc)의 이동 평균 및 중앙값")
    window_list = [5, 10, 15, 30]
    for window in window_list:
        print(window)
        df = userID_acc_rolling(df, window=window)
        df = userID_elapsed_median_rolling(df, window=window)

    print("feature_dimension_reduction")
    #     dimension_reduction_type = ["pca", "kpca", "kpca_rbf", "kpca_poly", "svd", "lda"]
    dimension_reduction_type = ["lda", "svd"]
    for kind in dimension_reduction_type:
        print(kind)
        df = feature_dimension_reduction(df, kind=kind)

    print("User가 해당 문제를 풀어본 경험 Feature")
    df = userID_assessmentItemID_experience(df)
    print("User가 해당 test를 풀어본 경험 Feature")
    df = userID_testid_experience(df)

    df["userID"] = df["userID"].astype("category")
    df["testId"] = df["testId"].astype("category")
    df["question_class"] = df["question_class"].astype("category")
    df["KnowledgeTag"] = df["KnowledgeTag"].astype("category")
    return df
