"""
pk_dict.pkl을 활용한 유틸리티 함수
"""
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm


def load_pk_dict(pkl_path='pk_dict.pkl'):
    """pk_dict.pkl 로드"""
    with open(pkl_path, 'rb') as f:
        pk_dict = pickle.load(f)
    print(f"✅ pk_dict loaded: {len(pk_dict)} keys")
    return pk_dict


def pp_concat_user_features(dfB, pk_dict_loaded):
    """
    유저(PrimaryKey) 기준 시계열 피처 통합 생성 함수
    --------------------------------------------------
    입력:
        - dfB : 병합 대상 DataFrame (PrimaryKey, TestDate 포함)
        - pk_dict_loaded : {PrimaryKey: [(TestDate, Label), ...]} 형식
    --------------------------------------------------
    생성되는 피처:
        기본:
            A_SuccessCount, A_FailCount, A_AttemptCount
            A_SuccessRate, A_RecentSuccessRate
            A_LastSuccessGap, A_LastFailGap
            A_Streak, A_SuccessTrend
        추가:
            A_LastOutcome, A_AvgInterval, A_LastInterval
            A_RecentFailRate, A_FirstToLastGap
            A_RecentStreakLen, A_SuccessMomentum, A_SuccessVar
    --------------------------------------------------
    """
    # 결과 리스트 초기화
    success_results, fail_results = [], []
    recent_success_rates, last_success_gaps, last_fail_gaps, streaks = [], [], [], []
    success_trends, last_outcomes = [], []
    avg_intervals, last_intervals = [], []
    recent_fail_rates, first_to_last_gaps = [], []
    recent_streaks, success_momentums, success_vars = [], [], []

    for _, row in tqdm(dfB.iterrows(), total=len(dfB), desc="🔄 병합 진행중"):
        pk = row["PrimaryKey"]
        t = pd.to_datetime(row["TestDate"])

        # 초기값
        success_cnt = fail_cnt = 0
        recent_success_rate = np.nan
        last_success_gap = last_fail_gap = np.nan
        streak = trend = 0
        last_outcome = np.nan
        avg_interval = last_interval = np.nan
        recent_fail_rate = np.nan
        first_to_last_gap = np.nan
        recent_streak = 0
        success_momentum = success_var = 0

        if pk in pk_dict_loaded:
            records = pk_dict_loaded[pk]
            df_pk = pd.DataFrame(
                records,
                columns=["TestDate", "MaybeLabel", "Label"]
                if len(records[0]) == 3 else ["TestDate", "Label"]
            )
            df_pk["TestDate"] = pd.to_datetime(df_pk["TestDate"])
            if "MaybeLabel" in df_pk.columns:
                df_pk = df_pk[["TestDate", "Label"]]
            df_pk = df_pk[df_pk["TestDate"] < t].sort_values("TestDate")

            if not df_pk.empty:
                # 기본 카운트
                success_cnt = (df_pk["Label"] == 1).sum()
                fail_cnt = (df_pk["Label"] == 0).sum()

                # 최근 3회 성공률
                recent_success_rate = (
                    (df_pk["Label"].tail(3) == 1).mean()
                    if len(df_pk) >= 3 else (df_pk["Label"] == 1).mean()
                )

                # 마지막 성공/실패 경과일
                last_success = df_pk[df_pk["Label"] == 1]["TestDate"].max()
                last_fail = df_pk[df_pk["Label"] == 0]["TestDate"].max()
                if pd.notna(last_success):
                    last_success_gap = (t - last_success).days
                if pd.notna(last_fail):
                    last_fail_gap = (t - last_fail).days

                # streak 계산
                seq = df_pk["Label"].tolist()
                streak = 0
                for val in reversed(seq):
                    if val == 1:
                        streak = streak + 1 if streak >= 0 else 1
                    else:
                        streak = streak - 1 if streak <= 0 else -1
                    if (val == 1 and streak < 0) or (val == 0 and streak > 0):
                        break

                # 성공률 추세 (polyfit)
                if len(df_pk) >= 3:
                    y = df_pk["Label"].tail(5).to_numpy()
                    x = np.arange(len(y))
                    trend = np.polyfit(x, y, 1)[0]

                last_outcome = df_pk["Label"].iloc[-1]
                first_date = df_pk["TestDate"].min()
                first_to_last_gap = (t - first_date).days

                if len(df_pk) > 1:
                    intervals = df_pk["TestDate"].diff().dt.days.dropna()
                    avg_interval = intervals.mean()
                    last_interval = (t - df_pk["TestDate"].iloc[-1]).days
                    success_var = df_pk["Label"].rolling(5).mean().var()

                if len(df_pk) >= 3:
                    recent_fail_rate = (df_pk["Label"].tail(3) == 0).mean()
                    recent_streak = sum(
                        1 for v in reversed(df_pk["Label"])
                        if v == df_pk["Label"].iloc[-1]
                    )

                if len(df_pk) >= 6:
                    rates = df_pk["Label"].rolling(3).mean().dropna().values
                    if len(rates) > 1:
                        success_momentum = rates[-1] - rates[-2]

        # 결과 저장
        success_results.append(success_cnt)
        fail_results.append(fail_cnt)
        recent_success_rates.append(recent_success_rate)
        last_success_gaps.append(last_success_gap)
        last_fail_gaps.append(last_fail_gap)
        streaks.append(streak)
        success_trends.append(trend)
        last_outcomes.append(last_outcome)
        avg_intervals.append(avg_interval)
        last_intervals.append(last_interval)
        recent_fail_rates.append(recent_fail_rate)
        first_to_last_gaps.append(first_to_last_gap)
        recent_streaks.append(recent_streak)
        success_momentums.append(success_momentum)
        success_vars.append(success_var)

    dfB["A_SuccessCount"] = success_results
    dfB["A_FailCount"] = fail_results
    dfB["A_AttemptCount"] = dfB["A_SuccessCount"] + dfB["A_FailCount"]
    dfB["A_SuccessRate"] = dfB["A_SuccessCount"] / (dfB["A_AttemptCount"] + 1e-6)
    dfB["A_RecentSuccessRate"] = recent_success_rates
    dfB["A_LastSuccessGap"] = last_success_gaps
    dfB["A_LastFailGap"] = last_fail_gaps
    dfB["A_Streak"] = streaks
    dfB["A_SuccessTrend"] = success_trends
    dfB["A_LastOutcome"] = last_outcomes
    dfB["A_AvgInterval"] = avg_intervals
    dfB["A_LastInterval"] = last_intervals
    dfB["A_RecentFailRate"] = recent_fail_rates
    dfB["A_FirstToLastGap"] = first_to_last_gaps
    dfB["A_RecentStreakLen"] = recent_streaks
    dfB["A_SuccessMomentum"] = success_momentums
    dfB["A_SuccessVar"] = success_vars

    dfB.fillna({
        "A_SuccessCount": 0,
        "A_FailCount": 0,
        "A_AttemptCount": 0,
        "A_SuccessRate": 0,
        "A_RecentSuccessRate": 0,
        "A_LastSuccessGap": -1,
        "A_LastFailGap": -1,
        "A_Streak": 0,
        "A_SuccessTrend": 0,
        "A_LastOutcome": 0,
        "A_AvgInterval": -1,
        "A_LastInterval": -1,
        "A_RecentFailRate": 0,
        "A_FirstToLastGap": -1,
        "A_RecentStreakLen": 0,
        "A_SuccessMomentum": 0,
        "A_SuccessVar": 0,
    }, inplace=True)

    # AGE 컬럼이 있는 경우에만 Age 정규화 피처 추가
    if "AGE" in dfB.columns:
        dfB["Age_SuccessGapNorm"] = dfB["A_LastSuccessGap"] / (dfB["AGE"] + 1)
        dfB["Age_FailGapNorm"] = dfB["A_LastFailGap"] / (dfB["AGE"] + 1)
    else:
        dfB["Age_SuccessGapNorm"] = -1
        dfB["Age_FailGapNorm"] = -1

    return dfB


def pp_concat_user_core_features(dfB, pk_dict_or_path):
    """
    pp_concat_user_features의 래퍼 함수 (사용 편의성)
    """
    if isinstance(pk_dict_or_path, str):
        pk_dict = load_pk_dict(pk_dict_or_path)
    else:
        pk_dict = pk_dict_or_path
    
    return pp_concat_user_features(dfB, pk_dict)

