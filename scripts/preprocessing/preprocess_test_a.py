import os
import json
import numpy as np
import pandas as pd
from functools import lru_cache




# =========================================================
# 🔸 최적화된 Helper 함수들 (벡터화)
# =========================================================

@lru_cache(maxsize=10000)
def parse_seq_cached(seq_str):
    """캐싱을 활용한 시퀀스 파싱 (반복되는 패턴 빠르게 처리)"""
    if not seq_str or seq_str == 'nan' or seq_str == '':
        return []
    try:
        return [float(x.strip()) for x in seq_str.split(",") if x.strip()]
    except:
        return []


def vectorized_count_values(series, condition_func):
    """벡터화된 값 카운트 (조건 함수 적용)"""
    # 문자열을 먼저 처리
    series_str = series.fillna("").astype(str)
    
    result = np.zeros(len(series_str), dtype=int)
    for idx, val in enumerate(series_str):
        if val:
            try:
                values = [float(x.strip()) for x in val.split(",") if x.strip()]
                result[idx] = sum(1 for v in values if condition_func(v))
            except:
                result[idx] = 0
    
    return result


def fast_parse_and_abs_mean(series):
    """빠른 절대값 평균 계산"""
    result = np.full(len(series), np.nan)
    series_str = series.fillna("").astype(str)
    
    for idx, val in enumerate(series_str):
        if val and val != '':
            try:
                values = [float(x.strip()) for x in val.split(",") if x.strip()]
                if values:
                    result[idx] = np.mean(np.abs(values))
            except:
                pass
    
    return result


def fast_reaction_direction(series):
    """빠른 반응 방향 계산"""
    result = np.full(len(series), 0.5)
    series_str = series.fillna("").astype(str)
    
    for idx, val in enumerate(series_str):
        if val and val != '':
            try:
                values = [float(x.strip()) for x in val.split(",") if x.strip()]
                if values:
                    pos_count = sum(1 for v in values if v > 0)
                    neg_count = sum(1 for v in values if v < 0)
                    if pos_count > neg_count:
                        result[idx] = 1
                    elif neg_count > pos_count:
                        result[idx] = 0
            except:
                pass
    
    return result


def fast_diff_resp_match(df, diff_col, resp_col, name_prefix):
    """최적화된 난이도-응답 매칭"""
    if (diff_col not in df.columns) or (resp_col not in df.columns):
        return
    
    speed1_ones = np.zeros(len(df), dtype=int)
    speed2_ones = np.zeros(len(df), dtype=int)
    speed3_ones = np.zeros(len(df), dtype=int)
    
    diff_str = df[diff_col].fillna("").astype(str)
    resp_str = df[resp_col].fillna("").astype(str)
    
    for idx in range(len(df)):
        try:
            diffs = [int(float(x.strip())) for x in diff_str.iloc[idx].split(",") if x.strip()]
            resps = [int(float(x.strip())) for x in resp_str.iloc[idx].split(",") if x.strip()]
            L = min(len(diffs), len(resps))
            
            for i in range(L):
                if resps[i] == 1:
                    if diffs[i] == 1:
                        speed1_ones[idx] += 1
                    elif diffs[i] == 2:
                        speed2_ones[idx] += 1
                    elif diffs[i] == 3:
                        speed3_ones[idx] += 1
        except:
            pass
    
    df[f"{name_prefix}_speed1_correct_cnt"] = speed1_ones
    df[f"{name_prefix}_speed2_correct_cnt"] = speed2_ones
    df[f"{name_prefix}_speed3_correct_cnt"] = speed3_ones


# =========================================================
# 🔸 새로운 파생변수용 헬퍼 함수들
# =========================================================

def parse_list_string(list_str):
    """리스트 문자열을 파싱하여 정수 리스트로 변환"""
    if pd.isna(list_str) or list_str == "" or list_str == "nan":
        return []
    try:
        return [int(float(x.strip())) for x in str(list_str).split(",") if x.strip()]
    except:
        return []


def count_diff_indices(list1_str, list2_str):
    """두 리스트에서 값이 다른 인덱스의 개수"""
    list1 = parse_list_string(list1_str)
    list2 = parse_list_string(list2_str)
    
    if not list1 or not list2:
        return 0
    
    min_len = min(len(list1), len(list2))
    diff_count = 0
    
    for i in range(min_len):
        if list1[i] != list2[i]:
            diff_count += 1
    
    return diff_count


def count_diff_indices_with_condition(list1_str, list2_str, condition_str, condition_value):
    """두 리스트에서 값이 다른 인덱스 중에서 condition에서 특정 값을 가진 인덱스의 개수"""
    list1 = parse_list_string(list1_str)
    list2 = parse_list_string(list2_str)
    condition = parse_list_string(condition_str)
    
    if not list1 or not list2 or not condition:
        return 0
    
    min_len = min(len(list1), len(list2), len(condition))
    count = 0
    
    for i in range(min_len):
        if list1[i] != list2[i] and condition[i] == condition_value:
            count += 1
    
    return count


def count_diff_indices_with_conditions(list1_str, list2_str, condition1_str, condition2_str, condition1_value, condition2_value):
    """두 리스트에서 값이 다른 인덱스 중에서 두 조건을 모두 만족하는 인덱스의 개수"""
    list1 = parse_list_string(list1_str)
    list2 = parse_list_string(list2_str)
    condition1 = parse_list_string(condition1_str)
    condition2 = parse_list_string(condition2_str)
    
    if not list1 or not list2 or not condition1 or not condition2:
        return 0
    
    min_len = min(len(list1), len(list2), len(condition1), len(condition2))
    count = 0
    
    for i in range(min_len):
        if list1[i] != list2[i] and condition1[i] == condition1_value and condition2[i] == condition2_value:
            count += 1
    
    return count


def count_consecutive_diff_indices(list1_str, list2_str):
    """두 리스트에서 값이 다른 연속된 인덱스 그룹의 개수"""
    list1 = parse_list_string(list1_str)
    list2 = parse_list_string(list2_str)
    
    if not list1 or not list2:
        return 0
    
    min_len = min(len(list1), len(list2))
    consecutive_groups = 0
    in_consecutive = False
    
    for i in range(min_len):
        if list1[i] != list2[i]:
            if not in_consecutive:
                consecutive_groups += 1
                in_consecutive = True
        else:
            in_consecutive = False
    
    return consecutive_groups


def count_condition_response(cond1_str, cond2_str, response_str, cond1_value, cond2_value, response_value):
    """조건 1, 조건 2가 특정 값일 때 응답이 특정 값인 경우의 개수"""
    cond1 = parse_list_string(cond1_str)
    cond2 = parse_list_string(cond2_str)
    response = parse_list_string(response_str)
    
    if not cond1 or not cond2 or not response:
        return 0
    
    min_len = min(len(cond1), len(cond2), len(response))
    count = 0
    
    for i in range(min_len):
        if cond1[i] == cond1_value and cond2[i] == cond2_value and response[i] == response_value:
            count += 1
    
    return count


def count_condition_values(condition_str, response_str, condition_values, response_value):
    """조건이 특정 값들 중 하나일 때 응답이 특정 값인 경우의 개수"""
    condition = parse_list_string(condition_str)
    response = parse_list_string(response_str)
    
    if not condition or not response:
        return 0
    
    min_len = min(len(condition), len(response))
    count = 0
    
    for i in range(min_len):
        if condition[i] in condition_values and response[i] == response_value:
            count += 1
    
    return count


# (제거) CNN 특징 추출 관련 함수 전부 삭제


# =========================================================
# 🔸 PrimaryKey 기준 TestDate 딕셔너리 생성 및 사용
# =========================================================

def build_primarykey_testdate_dict(df, save_path="./model/primarykey_testdate_dict.json"):
    """
    PrimaryKey 기준 TestDate 딕셔너리 생성
    - Key: PrimaryKey
    - Value: 해당 PrimaryKey의 모든 TestDate 리스트
    """
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        print("⚠️ PrimaryKey or TestDate not found")
        return {}
    
    pk_date_dict = {}
    for pk, group in df.groupby('PrimaryKey'):
        dates = group['TestDate'].tolist()
        pk_date_dict[str(pk)] = dates
    
    print(f"✅ PrimaryKey-TestDate dictionary built: {len(pk_date_dict):,} keys")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(pk_date_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {save_path}")
    
    return pk_date_dict


def load_primarykey_testdate_dict(load_path="./model/primarykey_testdate_dict.json"):
    """PrimaryKey-TestDate 딕셔너리 로드"""
    if not os.path.exists(load_path):
        print(f"⚠️ PrimaryKey-TestDate dictionary not found")
        return {}
    
    with open(load_path, 'r', encoding='utf-8') as f:
        pk_date_dict = json.load(f)
    
    print(f"✅ PrimaryKey-TestDate dictionary loaded: {len(pk_date_dict):,} keys")
    return pk_date_dict


def calculate_prev_month_diff(row, pk_date_dict):
    """
    현재 시점보다 이전 시점과의 월 차이 계산
    - PrimaryKey의 TestDate 리스트에서 현재 시점보다 이전인 날짜들만 찾기
    - 이전 시점이 있으면: (현재 시점 - 최근 이전 시점) 의 월 차이
    - 이전 시점이 없으면: 0 (첫 번째 시점)
    - PrimaryKey가 없거나 딕셔너리에 없으면: 0
    """
    pk = str(row.get('PrimaryKey', ''))
    current_date = row.get('TestDate')
    
    # PrimaryKey가 없거나 딕셔너리에 없거나 TestDate가 없으면 0 반환
    if not pk or pk == 'nan' or pk not in pk_date_dict or not current_date:
        return 0
    
    # PrimaryKey의 모든 TestDate 리스트
    all_dates = pk_date_dict[pk]
    
    # YYYYMM 형식을 datetime으로 변환
    def parse_date(date_str):
        try:
            date_str = str(date_str)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            return pd.Timestamp(year=year, month=month, day=1)
        except:
            return None
    
    current_dt = parse_date(current_date)
    if current_dt is None:
        return -1
    
    # 이전 시점들만 필터링
    prev_dates = []
    for d in all_dates:
        d_dt = parse_date(d)
        if d_dt and d_dt < current_dt:
            prev_dates.append(d_dt)
    
    # 이전 시점이 없으면 0 (첫 번째 시점)
    if not prev_dates:
        return 0
    
    # 가장 가까운 이전 시점과의 차이
    most_recent_prev = max(prev_dates)
    month_diff = (current_dt.year - most_recent_prev.year) * 12 + (current_dt.month - most_recent_prev.month)
    
    return month_diff


def calculate_avg_prev_month_diff(row, pk_date_dict):
    """
    이전 시점들과의 평균 월 차이 계산
    - PrimaryKey가 없거나 딕셔너리에 없으면: 0
    """
    pk = str(row.get('PrimaryKey', ''))
    current_date = row.get('TestDate')
    
    # PrimaryKey가 없거나 딕셔너리에 없거나 TestDate가 없으면 0 반환
    if not pk or pk == 'nan' or pk not in pk_date_dict or not current_date:
        return 0
    
    all_dates = pk_date_dict[pk]
    
    def parse_date(date_str):
        try:
            date_str = str(date_str)
            year = int(date_str[:4])
            month = int(date_str[4:6])
            return pd.Timestamp(year=year, month=month, day=1)
        except:
            return None
    
    current_dt = parse_date(current_date)
    if current_dt is None:
        return 0
    
    # 이전 시점들만 필터링
    prev_dates = []
    for d in all_dates:
        d_dt = parse_date(d)
        if d_dt and d_dt < current_dt:
            prev_dates.append(d_dt)
    
    # 이전 시점이 없으면 0 (첫 번째 시점)
    if not prev_dates:
        return 0
    
    # 모든 이전 시점과의 평균 월 차이
    month_diffs = []
    for prev_dt in prev_dates:
        month_diff = (current_dt.year - prev_dt.year) * 12 + (current_dt.month - prev_dt.month)
        month_diffs.append(month_diff)
    
    return np.mean(month_diffs)


def add_primarykey_month_diff_features(df, pk_date_dict):
    """
    PrimaryKey 기준 이전 시점과의 월 차이 파생변수 추가
    """
    print("🔧 Adding PrimaryKey month difference features...")
    
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        print("⚠️ PrimaryKey or TestDate not found, skipping...")
        return df
    
    # 최근 이전 시점과의 월 차이
    df['PK_prev_month_diff'] = df.apply(lambda row: calculate_prev_month_diff(row, pk_date_dict), axis=1)
    
    # 평균 이전 시점과의 월 차이
    df['PK_avg_prev_month_diff'] = df.apply(lambda row: calculate_avg_prev_month_diff(row, pk_date_dict), axis=1)
    
    print(f"✅ Added 2 PrimaryKey month difference features")
    
    return df


# =========================================================
# 🔸 최적화된 메인 전처리 함수
# =========================================================

def preprocess_a_features(df, use_cnn=True, rt_stats=None, use_a3_advanced=True, use_a4_advanced=True, use_a5_advanced=True, use_a9_advanced=True, pk_date_dict=None):
    """
    최적화된 A 검사 전처리 함수 (벡터화 및 캐싱 활용)
    """
    import time
    
    df = df.copy()
    
    print("\n⚡ Fast preprocessing started...")
    t_start = time.time()
    
    # Age 변환 (벡터화 가능)
    if 'Age' in df.columns:
        def convert_age_vec(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            if not s:
                return np.nan
            try:
                base = int(s[:-1])
                return base if s[-1] == "a" else base + 5
            except:
                try:
                    return int(s)
                except:
                    return np.nan
        
        df['Age_num'] = df['Age'].apply(convert_age_vec)
    
    # 음수 카운트 (벡터화)
    minus_columns = ["A1-4", "A2-4"]
    for col in minus_columns:
        if col in df.columns:
            new_name = f"{col.replace('-', '_')}_neg_count"
            df[new_name] = vectorized_count_values(df[col], lambda v: v < 0)
    
    # 응답 1 카운트 (벡터화)
    ones_columns = ["A1-3", "A2-3", "A3-6", "A3-5", "A4-3", "A4-4", "A5-2", "A5-3"]
    for col in ones_columns:
        if col in df.columns:
            new_name = f"{col.replace('-', '_')}_one_count"
            df[new_name] = vectorized_count_values(df[col], lambda v: v == 1)
    
    # A1 / A2 난이도-응답 매칭 (최적화)
    fast_diff_resp_match(df, "A1-2", "A1-3", "A1_2")
    fast_diff_resp_match(df, "A2-1", "A2-3", "A2_1")
    fast_diff_resp_match(df, "A2-2", "A2-3", "A2_2")
    
    # 반응속도 관련 파생변수 (최적화)
    if rt_stats is not None:
        rt_columns = ["A1-4", "A2-4", "A3-7", "A4-5"]
        created_count = 0
        
        for col in rt_columns:
            if col in df.columns:
                train_abs_mean = rt_stats.get(col, 0.0)
                if train_abs_mean != 0.0:
                    # 절대값 평균 (최적화)
                    row_abs_means = fast_parse_and_abs_mean(df[col])
                    ratio_col = f"{col.replace('-', '_')}_abs_mean_ratio"
                    dir_col = f"{col.replace('-', '_')}_reaction_dir"
                    df[ratio_col] = (row_abs_means / train_abs_mean).astype(float)
                    df[ratio_col] = df[ratio_col].fillna(1.0)
                    df[dir_col] = fast_reaction_direction(df[col])
                    
                    created_count += 1
        
        print(f"✅ Reaction time features: {created_count} features")
    
 
    
    # PrimaryKey 기준 월 차이 파생변수 추가
    if pk_date_dict is not None:
        t_pk = time.time()
        df = add_primarykey_month_diff_features(df, pk_date_dict)
        print(f"⏱️ PK month diff time: {time.time()-t_pk:.2f}s")
    else:
        print("ℹ️ PrimaryKey month difference features disabled")
    
    print(f"⏱️ Total preprocessing time: {time.time()-t_start:.2f}s")
    
    return df


# =========================================================
# 🔸 반응속도 통계 (최적화)
# =========================================================

def build_reaction_time_stats_fast(df, save_path="./model/reaction_time_stats.json"):
    """최적화된 반응속도 통계 생성"""
    rt_columns = ["A1-4", "A2-4", "A3-7", "A4-5"]
    rt_stats = {}
    
    for col in rt_columns:
        if col in df.columns:
            # 벡터화된 절대값 평균 계산
            abs_means = fast_parse_and_abs_mean(df[col])
            overall_mean = np.nanmean(abs_means)
            rt_stats[col] = float(overall_mean) if not np.isnan(overall_mean) else 0.0
            print(f"   {col}: abs_mean = {rt_stats[col]:.4f}")
        else:
            rt_stats[col] = 0.0
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(rt_stats, f, ensure_ascii=False, indent=2)
        print(f"✅ Reaction time stats saved")
    
    return rt_stats


def load_reaction_time_stats(load_path="./model/reaction_time_stats.json"):
    if not os.path.exists(load_path):
        print(f"⚠️ Reaction time stats not found")
        return {"A1-4": 0.0, "A2-4": 0.0, "A3-7": 0.0, "A4-5": 0.0}
    
    with open(load_path, 'r', encoding='utf-8') as f:
        rt_stats = json.load(f)
    
    print(f"✅ Reaction time stats loaded")
    return rt_stats


# === PrimaryKey 과거 라벨 히스토리 ===
import os, json
import pandas as pd

def _parse_yyyymm(date_str):
    try:
        s = str(date_str)
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=1)
    except:
        return None

def build_primary_label_history(df, save_path: str | None = None):
    """PrimaryKey별 최대 라벨 딕셔너리 생성 (기존 방식: 1이 있으면 1, 없으면 0)"""
    if 'PrimaryKey' not in df.columns or 'Label' not in df.columns:
        return {}
    hist = df.groupby('PrimaryKey')['Label'].max().astype(int).to_dict()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): int(v) for k, v in hist.items()}, f, ensure_ascii=False, indent=2)
    return hist

def add_primary_history_features(df: pd.DataFrame, primary_label_history: dict, out_col: str = 'primary_past_label') -> pd.DataFrame:
    """PrimaryKey별 최대 라벨 부여 (기존 방식)"""
    df = df.copy()
    if 'PrimaryKey' not in df.columns:
        df[out_col] = -1
        return df
    df[out_col] = df['PrimaryKey'].map(lambda pk: primary_label_history.get(pk, primary_label_history.get(str(pk), -1))).fillna(-1).astype(int)
    return df

def build_primary_label_history_with_date(df, save_path: str | None = None):
    """PrimaryKey별로 TestDate 순 정렬된 (TestDate, Label) 리스트 딕셔너리 생성 (날짜 기반)"""
    if 'PrimaryKey' not in df.columns or 'Label' not in df.columns or 'TestDate' not in df.columns:
        return {}
    hist = {}
    for pk, group in df.groupby('PrimaryKey'):
        # PrimaryKey별로 TestDate 순 정렬된 (TestDate, Label) 리스트
        pk_list = [(int(row['TestDate']), int(row['Label'])) for _, row in group.iterrows()]
        pk_list.sort(key=lambda x: x[0])  # TestDate 순 정렬
        hist[str(pk)] = pk_list
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    return hist

def add_primary_history_features_with_date(df: pd.DataFrame, primary_label_history: dict, out_col: str = 'primary_past_label') -> pd.DataFrame:
    """각 행의 TestDate보다 이전 중 최신 TestDate의 라벨 가져오기 (날짜 기반, 딕셔너리 기반 빠른 조회)"""
    df = df.copy()
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        df[out_col] = -1
        return df
    
    # PrimaryKey별 조회용 딕셔너리 준비
    pk_to_past_label = {}
    for pk_str, pk_list in primary_label_history.items():
        pk_to_past_label[pk_str] = pk_list  # 정렬된 (TestDate, Label) 리스트
    
    def get_prev_label_fast(row):
        pk = str(row.get('PrimaryKey', ''))
        current_date = int(row.get('TestDate', 0))
        if not pk or pk not in pk_to_past_label:
            return -1
        pk_list = pk_to_past_label[pk]
        # 정렬된 리스트에서 가장 최근 이전 날짜 찾기
        prev_label = -1
        for td, label in pk_list:
            if td < current_date:
                prev_label = label
            else:
                break  # 정렬되어 있으므로 이후는 모두 current_date 이상
        return prev_label
    
    df[out_col] = df.apply(get_prev_label_fast, axis=1).astype(int)
    return df


def build_b_previous_label_dict_for_a(df_b, save_path: str | None = None):
    """
    B 데이터에서 PrimaryKey별 최대 라벨 딕셔너리 생성 (any 방식: 1이 있으면 1, 없으면 0)
    A 데이터에서 B의 라벨을 찾기 위해 사용
    
    Args:
        df_b: B 데이터 DataFrame (PrimaryKey, Label 컬럼 필요)
        save_path: 저장할 경로 (None이면 저장하지 않음)
    
    Returns:
        dict: {PrimaryKey: max_label} 형식의 딕셔너리 (any 방식)
    """
    if 'PrimaryKey' not in df_b.columns or 'Label' not in df_b.columns:
        return {}
    
    # PrimaryKey별 최대 라벨 (1이 있으면 1, 없으면 0)
    hist = df_b.groupby('PrimaryKey')['Label'].max().astype(int).to_dict()
    hist = {str(k): int(v) for k, v in hist.items()}
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    
    return hist


def add_b_previous_label_to_a(df_a: pd.DataFrame, b_label_history: dict, out_col: str = 'b_previous_label') -> pd.DataFrame:
    """
    A 데이터에 B의 라벨 추가 (any 방식)
    - PrimaryKey가 A, B 둘 다에 있는 경우
    - B에서 라벨 1이 한 번이라도 있으면 1, 없으면 0 (TestDate 조건 없음)
    
    Args:
        df_a: A 데이터 DataFrame (PrimaryKey 컬럼 필요)
        b_label_history: build_b_previous_label_dict_for_a로 생성한 딕셔너리 (any 방식)
        out_col: 출력 컬럼명
    
    Returns:
        DataFrame: out_col 컬럼이 추가된 DataFrame
    """
    df = df_a.copy()
    if 'PrimaryKey' not in df.columns:
        df[out_col] = -1
        return df
    
    # PrimaryKey별로 B의 최대 라벨 매핑 (any 방식)
    df[out_col] = df['PrimaryKey'].map(
        lambda pk: b_label_history.get(str(pk), b_label_history.get(pk, -1))
    ).fillna(-1).astype(int)
    
    return df


def add_month_prev_label_adjustment_feature(df_a: pd.DataFrame, primary_label_history: dict, 
                                             month_threshold=9, 
                                             short_interval_prob=0.9, 
                                             long_interval_prob=0.66,
                                             out_col: str = 'month_prev_label_adj') -> pd.DataFrame:
    """
    A 데이터에 개월 수와 이전 라벨 기반 확률 값 부여 (후처리 대신 변수로 생성)
    - 각 행의 PrimaryKey와 TestDate로 바로 이전 시점 조회
    - 개월 수가 9 이전이면 short_interval_prob (0.9)
    - 개월 수가 9 이상이면 long_interval_prob (0.66)
    - 매칭 안되는 경우 -1
    
    Args:
        df_a: A 데이터 DataFrame (PrimaryKey, TestDate 컬럼 필요)
        primary_label_history: build_primary_label_history_with_date로 생성한 딕셔너리
                              {PrimaryKey: [(TestDate, Label), ...]} 형식
        month_threshold: 개월 수 기준값 (기본 9)
        short_interval_prob: 짧은 간격일 때 확률 값 (기본 0.9)
        long_interval_prob: 긴 간격일 때 확률 값 (기본 0.66)
        out_col: 출력 컬럼명
    
    Returns:
        DataFrame: out_col 컬럼이 추가된 DataFrame
    """
    df = df_a.copy()
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        df[out_col] = -1.0
        return df
    
    def yyyymm_to_months(yyyymm):
        """YYYYMM 정수를 월 단위 정수로 변환"""
        try:
            year = int(yyyymm) // 100
            month = int(yyyymm) % 100
            return year * 12 + month
        except:
            return None
    
    def month_diff_yyyymm(a, b):
        """두 YYYYMM(정수) 사이의 월 차이 | a - b | 반환"""
        months_a = yyyymm_to_months(a)
        months_b = yyyymm_to_months(b)
        if months_a is None or months_b is None:
            return None
        return abs(months_a - months_b)
    
    def get_adjustment_value(row):
        pk = str(row.get('PrimaryKey', ''))
        current_date = int(row.get('TestDate', 0))
        
        if not pk or pk not in primary_label_history:
            return -1.0
        
        pk_list = primary_label_history[pk]
        
        # 정렬된 리스트에서 바로 이전 시점 찾기
        prev_date = None
        prev_label = None
        for td, label in pk_list:
            if td < current_date:
                prev_date = td
                prev_label = label
            else:
                break
        
        # 이전 시점이 없으면 -1
        if prev_date is None or prev_label is None:
            return -1.0
        
        # 이전 라벨이 0이면 -1
        if prev_label == 0:
            return -1.0
        
        # 이전 라벨이 1일 때만 확률 값 부여
        # 개월 수 차이 계산
        month_diff = month_diff_yyyymm(current_date, prev_date)
        if month_diff is None:
            return -1.0
        
        # 9개월 이전이면 0.9, 9개월 이상이면 0.66
        if month_diff < month_threshold:
            return float(short_interval_prob)
        else:
            return float(long_interval_prob)
    
    df[out_col] = df.apply(get_adjustment_value, axis=1).astype(float)
    return df


def _ensure_testdate_int(value):
    """Convert TestDate value to integer YYYYMM format if possible."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return int(value.strftime("%Y%m"))
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return int(value)
    except (ValueError, TypeError):
        try:
            ts = pd.to_datetime(value)
            return int(ts.strftime("%Y%m"))
        except Exception:
            return None


def build_label_pattern_history(df, save_path: str | None = None):
    """
    전체 train 데이터를 순회하여 PrimaryKey별로 TestDate 순서대로 Label 시퀀스를 딕셔너리로 저장
    
    Args:
        df: train DataFrame (PrimaryKey, TestDate, Label 컬럼 필요)
        save_path: 저장할 경로 (None이면 저장하지 않음)
    
    Returns:
        dict: {PrimaryKey: [(TestDate, Label), ...]} 형식의 딕셔너리
    """
    if 'PrimaryKey' not in df.columns or 'Label' not in df.columns or 'TestDate' not in df.columns:
        return {}
    
    hist = {}
    for pk, group in df.groupby('PrimaryKey'):
        pk_list = []
        for _, row in group.iterrows():
            td_int = _ensure_testdate_int(row['TestDate'])
            if td_int is None:
                continue
            pk_list.append((td_int, int(row['Label'])))
        pk_list.sort(key=lambda x: x[0])  # TestDate 순 정렬
        hist[str(pk)] = pk_list
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)
    
    return hist


def add_label_pattern_features(df: pd.DataFrame, label_pattern_history: dict, 
                                out_col_prefix: str = 'pattern') -> pd.DataFrame:
    """
    각 행에 대해 TestDate 이전 시점까지의 4가지 패턴 출현 횟수를 계산하여 파생변수 생성
    
    패턴:
    - pattern_1to1: 이전 라벨 1 → 현재 라벨 1
    - pattern_1to0: 이전 라벨 1 → 현재 라벨 0
    - pattern_0to1: 이전 라벨 0 → 현재 라벨 1
    - pattern_0to0: 이전 라벨 0 → 현재 라벨 0
    
    Args:
        df: DataFrame (PrimaryKey, TestDate 컬럼 필요)
        label_pattern_history: build_label_pattern_history로 생성한 딕셔너리
        out_col_prefix: 출력 컬럼명 prefix (기본 'pattern')
    
    Returns:
        DataFrame: 4가지 패턴 컬럼이 추가된 DataFrame
    """
    df = df.copy()
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        df[f'{out_col_prefix}_1to1'] = 0
        df[f'{out_col_prefix}_1to0'] = 0
        df[f'{out_col_prefix}_0to1'] = 0
        df[f'{out_col_prefix}_0to0'] = 0
        return df
    
    def count_patterns(row):
        pk = str(row.get('PrimaryKey', ''))
        current_date = _ensure_testdate_int(row.get('TestDate', None))
        
        if not pk or pk not in label_pattern_history or current_date is None:
            return 0, 0, 0, 0
        
        pk_list = label_pattern_history[pk]
        
        # 해당 TestDate 이전 시점만 필터링
        prev_list = [(td, label) for td, label in pk_list if td < current_date]
        
        # 과거 label 이력이 2개 미만이면 모든 패턴 0
        if len(prev_list) < 2:
            return 0, 0, 0, 0
        
        # 패턴 카운트
        pattern_1to1 = 0
        pattern_1to0 = 0
        pattern_0to1 = 0
        pattern_0to0 = 0
        
        for i in range(1, len(prev_list)):
            prev_label = prev_list[i-1][1]
            curr_label = prev_list[i][1]
            
            if prev_label == 1 and curr_label == 1:
                pattern_1to1 += 1
            elif prev_label == 1 and curr_label == 0:
                pattern_1to0 += 1
            elif prev_label == 0 and curr_label == 1:
                pattern_0to1 += 1
            elif prev_label == 0 and curr_label == 0:
                pattern_0to0 += 1
        
        return pattern_1to1, pattern_1to0, pattern_0to1, pattern_0to0
    
    pattern_series = df.apply(count_patterns, axis=1)
    columns = [
        f'{out_col_prefix}_1to1',
        f'{out_col_prefix}_1to0',
        f'{out_col_prefix}_0to1',
        f'{out_col_prefix}_0to0',
    ]
    if pattern_series.empty:
        pattern_df = pd.DataFrame([], index=df.index, columns=columns)
    else:
        pattern_df = pd.DataFrame(list(pattern_series), index=df.index, columns=columns)

    for col in columns:
        df[col] = pattern_df.get(col, pd.Series(0, index=df.index)).fillna(0).astype(int)
    
    return df


def add_is_first_test_feature(df: pd.DataFrame, label_history_dict: dict = None, out_col: str = 'is_first_test') -> pd.DataFrame:
    """
    각 행에 대해 PrimaryKey 기준으로 현재 TestDate 이전에 과거 이력이 있는지 확인
    - 이력 없음 (첫 시험) = 1
    - 이력 있음 = 0
    
    Parameters:
    -----------
    df : pd.DataFrame
        PrimaryKey, TestDate 컬럼을 포함한 데이터프레임
    label_history_dict : dict, optional
        {PrimaryKey: [(TestDate, Label), ...]} 형태의 딕셔너리
        없으면 df로부터 자동 생성
    out_col : str
        출력 컬럼명 (기본: 'is_first_test')
    
    Returns:
    --------
    pd.DataFrame
        is_first_test 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        df[out_col] = -1
        return df
    
    # label_history_dict가 없으면 현재 df로부터 생성
    if label_history_dict is None:
        print(f"[INFO] label_history_dict가 없어서 현재 데이터로부터 생성합니다.")
        label_history_dict = {}
        
        # Label 컬럼이 있으면 사용, 없으면 0으로 대체
        if 'Label' in df.columns:
            temp_df = df[['PrimaryKey', 'TestDate', 'Label']].copy()
        else:
            temp_df = df[['PrimaryKey', 'TestDate']].copy()
            temp_df['Label'] = 0
        
        temp_df['TestDate'] = pd.to_numeric(temp_df['TestDate'], errors='coerce').fillna(0).astype(int)
        temp_df = temp_df.sort_values(['PrimaryKey', 'TestDate'])
        
        for pk, group in temp_df.groupby('PrimaryKey'):
            pk_str = str(pk)
            label_history_dict[pk_str] = [
                (int(row['TestDate']), int(row.get('Label', 0))) 
                for _, row in group.iterrows()
            ]
    
    # PrimaryKey별 조회용 딕셔너리 준비
    pk_to_history = {}
    for pk_str, pk_list in label_history_dict.items():
        pk_to_history[pk_str] = pk_list  # 정렬된 (TestDate, Label) 리스트
    
    def check_is_first_test(row):
        """현재 TestDate 이전에 이력이 있는지 확인"""
        pk = str(row.get('PrimaryKey', ''))
        current_date = int(row.get('TestDate', 0))
        
        if not pk or pk not in pk_to_history:
            return 1  # 이력 없음 = 첫 시험
        
        pk_list = pk_to_history[pk]
        
        # 정렬된 리스트에서 현재 날짜 이전에 기록이 있는지 확인
        has_previous_record = False
        for td, label in pk_list:
            if td < current_date:
                has_previous_record = True
                break  # 하나라도 있으면 첫 시험이 아님
        
        return 0 if has_previous_record else 1  # 이력 있음=0, 없음=1
    
    df[out_col] = df.apply(check_is_first_test, axis=1).astype(int)
    
    return df


def add_holiday_season_feature(df: pd.DataFrame, out_col: str = 'has_holiday_season', months_ahead: int = 6) -> pd.DataFrame:
    """
    현재 TestDate 기준으로 6개월 이내에 9월 또는 10월이 있는지 확인
    - 9월/10월 포함 = 1 (연휴 있음)
    - 9월/10월 없음 = 0 (연휴 없음)
    
    Parameters:
    -----------
    df : pd.DataFrame
        TestDate 컬럼을 포함한 데이터프레임 (YYYYMM 형식)
    out_col : str
        출력 컬럼명 (기본: 'has_holiday_season')
    months_ahead : int
        확인할 미래 월 수 (기본: 6개월)
    
    Returns:
    --------
    pd.DataFrame
        has_holiday_season 컬럼이 추가된 데이터프레임
    
    Examples:
    ---------
    TestDate=202007 → 6개월 이내 범위: 202007~202101 → 202009, 202010 포함 → 1
    TestDate=202011 → 6개월 이내 범위: 202011~202105 → 범위 내 없음 → 0
    TestDate=202003 → 6개월 이내 범위: 202003~202009 → 202009 포함 → 1
    """
    df = df.copy()
    
    if 'TestDate' not in df.columns:
        df[out_col] = -1
        return df
    
    def has_sep_or_oct_in_range(test_date):
        """주어진 TestDate로부터 months_ahead 이내에 9월 또는 10월이 있는지 확인"""
        try:
            test_date_int = int(test_date)
            if test_date_int <= 0:
                return -1
            
            # YYYYMM 파싱
            year = test_date_int // 100
            month = test_date_int % 100
            
            if month < 1 or month > 12:
                return -1
            
            # 현재 월부터 months_ahead 개월까지 확인
            for i in range(months_ahead + 1):
                check_month = month + i
                check_year = year
                
                # 월이 12를 넘으면 년도 증가
                while check_month > 12:
                    check_month -= 12
                    check_year += 1
                
                # 9월 또는 10월인지 확인
                if check_month == 9 or check_month == 10:
                    return 1
            
            return 0
        except:
            return -1
    
    df[out_col] = df['TestDate'].apply(has_sep_or_oct_in_range).astype(int)
    
    return df


def add_a2_condition_reaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A2 컬럼들의 조건별, 부호별 반응속도 평균 계산
    
    A2-1: condition 1 (1=SLOW, 2=NORMAL, 3=FAST)
    A2-2: condition 2 (1=SLOW, 2=NORMAL, 3=FAST)
    A2-3: response (0=No, 1=Yes)
    A2-4: response time (+ / -)
    
    A2-3 == 1인 trial만 선택하여:
    - A2-1, A2-2 각각의 조건별 (SLOW/NORMAL/FAST)
    - A2-4의 부호별 (양수/음수)
    - 반응속도 평균 계산
    
    생성되는 변수:
    - A2-1_SLOW_pos_mean, A2-1_SLOW_neg_mean
    - A2-1_NORMAL_pos_mean, A2-1_NORMAL_neg_mean
    - A2-1_FAST_pos_mean, A2-1_FAST_neg_mean
    - A2-2_SLOW_pos_mean, A2-2_SLOW_neg_mean
    - A2-2_NORMAL_pos_mean, A2-2_NORMAL_neg_mean
    - A2-2_FAST_pos_mean, A2-2_FAST_neg_mean
    
    Parameters:
    -----------
    df : pd.DataFrame
        A2-1, A2-2, A2-3, A2-4 컬럼을 포함한 데이터프레임
    
    Returns:
    --------
    pd.DataFrame
        12개의 파생변수가 추가된 데이터프레임
    """
    df = df.copy()
    
    # 필요한 컬럼 체크
    required_cols = ['A2-1', 'A2-2', 'A2-3', 'A2-4']
    if not all(col in df.columns for col in required_cols):
        # 컬럼이 없으면 모든 파생변수를 -1로 설정
        for cond_col in ['A2-1', 'A2-2']:
            for condition in ['SLOW', 'NORMAL', 'FAST']:
                for sign in ['pos', 'neg']:
                    df[f'{cond_col}_{condition}_{sign}_mean'] = -1
        return df
    
    # 조건값 매핑
    condition_map = {1: 'SLOW', 2: 'NORMAL', 3: 'FAST'}
    
    def calculate_condition_reaction_stats(row):
        """각 행에 대해 조건별, 부호별 반응속도 평균 계산"""
        result = {}
        
        # 초기값 설정 (데이터 없을 때 -1)
        for cond_col in ['A2-1', 'A2-2']:
            for condition in ['SLOW', 'NORMAL', 'FAST']:
                for sign in ['pos', 'neg']:
                    result[f'{cond_col}_{condition}_{sign}_mean'] = -1
        
        try:
            # 각 컬럼 파싱
            a2_1_vals = parse_seq_cached(str(row.get('A2-1', '')))
            a2_2_vals = parse_seq_cached(str(row.get('A2-2', '')))
            a2_3_vals = parse_seq_cached(str(row.get('A2-3', '')))
            a2_4_vals = parse_seq_cached(str(row.get('A2-4', '')))
            
            # 길이 체크
            if not a2_1_vals or not a2_2_vals or not a2_3_vals or not a2_4_vals:
                return result
            
            min_len = min(len(a2_1_vals), len(a2_2_vals), len(a2_3_vals), len(a2_4_vals))
            
            # A2-3 == 1인 trial만 선택
            selected_indices = [i for i in range(min_len) if a2_3_vals[i] == 1]
            
            if not selected_indices:
                return result
            
            # A2-1, A2-2 각각에 대해 처리
            for cond_col_name, cond_vals in [('A2-1', a2_1_vals), ('A2-2', a2_2_vals)]:
                # 조건별, 부호별 그룹화
                groups = {
                    'SLOW': {'pos': [], 'neg': []},
                    'NORMAL': {'pos': [], 'neg': []},
                    'FAST': {'pos': [], 'neg': []}
                }
                
                for idx in selected_indices:
                    # 조건값 가져오기
                    cond_val = int(cond_vals[idx])
                    reaction_time = a2_4_vals[idx]
                    
                    # 조건 매핑
                    condition_name = condition_map.get(cond_val)
                    if condition_name is None:
                        continue
                    
                    # 부호별 분류
                    if reaction_time > 0:
                        groups[condition_name]['pos'].append(abs(reaction_time))
                    elif reaction_time < 0:
                        groups[condition_name]['neg'].append(abs(reaction_time))
                
                # 평균 계산
                for condition in ['SLOW', 'NORMAL', 'FAST']:
                    for sign in ['pos', 'neg']:
                        values = groups[condition][sign]
                        if values:
                            result[f'{cond_col_name}_{condition}_{sign}_mean'] = np.mean(values)
                        # else: 이미 -1로 초기화됨
            
        except Exception:
            pass  # 에러 시 -1 유지
        
        return result
    
    # 각 행에 대해 계산
    stats_df = df.apply(calculate_condition_reaction_stats, axis=1, result_type='expand')
    
    # 결과 병합
    for col in stats_df.columns:
        df[col] = stats_df[col].fillna(-1)
    
    return df


def add_a1_condition_reaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A1 컬럼들의 조건별, 부호별 반응속도 평균 계산
    
    A1-1: condition 1 (1=LEFT, 2=RIGHT)
    A1-2: condition 2 (1=SLOW, 2=NORMAL, 3=FAST)
    A1-3: response (0=No, 1=Yes)
    A1-4: response time (+ / -)
    
    A1-3 == 1인 trial만 선택하여:
    - A1-1 조건별 (LEFT/RIGHT)
    - A1-2 조건별 (SLOW/NORMAL/FAST)
    - A1-4의 부호별 (양수/음수)
    - 반응속도 평균 계산
    
    생성되는 변수:
    - A1-1_LEFT_pos_mean, A1-1_LEFT_neg_mean
    - A1-1_RIGHT_pos_mean, A1-1_RIGHT_neg_mean
    - A1-2_SLOW_pos_mean, A1-2_SLOW_neg_mean
    - A1-2_NORMAL_pos_mean, A1-2_NORMAL_neg_mean
    - A1-2_FAST_pos_mean, A1-2_FAST_neg_mean
    
    Parameters:
    -----------
    df : pd.DataFrame
        A1-1, A1-2, A1-3, A1-4 컬럼을 포함한 데이터프레임
    
    Returns:
    --------
    pd.DataFrame
        10개의 파생변수가 추가된 데이터프레임
    """
    df = df.copy()
    
    # 필요한 컬럼 체크
    required_cols = ['A1-1', 'A1-2', 'A1-3', 'A1-4']
    if not all(col in df.columns for col in required_cols):
        # 컬럼이 없으면 모든 파생변수를 -1로 설정
        for condition in ['LEFT', 'RIGHT']:
            for sign in ['pos', 'neg']:
                df[f'A1-1_{condition}_{sign}_mean'] = -1
        for condition in ['SLOW', 'NORMAL', 'FAST']:
            for sign in ['pos', 'neg']:
                df[f'A1-2_{condition}_{sign}_mean'] = -1
        return df
    
    # 조건값 매핑
    a1_1_map = {1: 'LEFT', 2: 'RIGHT'}
    a1_2_map = {1: 'SLOW', 2: 'NORMAL', 3: 'FAST'}
    
    def calculate_a1_condition_reaction_stats(row):
        """각 행에 대해 A1 조건별, 부호별 반응속도 평균 계산"""
        result = {}
        
        # 초기값 설정 (데이터 없을 때 -1)
        for condition in ['LEFT', 'RIGHT']:
            for sign in ['pos', 'neg']:
                result[f'A1-1_{condition}_{sign}_mean'] = -1
        for condition in ['SLOW', 'NORMAL', 'FAST']:
            for sign in ['pos', 'neg']:
                result[f'A1-2_{condition}_{sign}_mean'] = -1
        
        try:
            # 각 컬럼 파싱
            a1_1_vals = parse_seq_cached(str(row.get('A1-1', '')))
            a1_2_vals = parse_seq_cached(str(row.get('A1-2', '')))
            a1_3_vals = parse_seq_cached(str(row.get('A1-3', '')))
            a1_4_vals = parse_seq_cached(str(row.get('A1-4', '')))
            
            # 길이 체크
            if not a1_1_vals or not a1_2_vals or not a1_3_vals or not a1_4_vals:
                return result
            
            min_len = min(len(a1_1_vals), len(a1_2_vals), len(a1_3_vals), len(a1_4_vals))
            
            # A1-3 == 1인 trial만 선택
            selected_indices = [i for i in range(min_len) if a1_3_vals[i] == 1]
            
            if not selected_indices:
                return result
            
            # A1-1 처리 (LEFT/RIGHT)
            groups_a1_1 = {
                'LEFT': {'pos': [], 'neg': []},
                'RIGHT': {'pos': [], 'neg': []}
            }
            
            for idx in selected_indices:
                cond_val = int(a1_1_vals[idx])
                reaction_time = a1_4_vals[idx]
                
                condition_name = a1_1_map.get(cond_val)
                if condition_name is None:
                    continue
                
                # 부호별 분류 (0 제외)
                if reaction_time > 0:
                    groups_a1_1[condition_name]['pos'].append(abs(reaction_time))
                elif reaction_time < 0:
                    groups_a1_1[condition_name]['neg'].append(abs(reaction_time))
            
            # A1-1 평균 계산
            for condition in ['LEFT', 'RIGHT']:
                for sign in ['pos', 'neg']:
                    values = groups_a1_1[condition][sign]
                    if values:
                        result[f'A1-1_{condition}_{sign}_mean'] = np.mean(values)
            
            # A1-2 처리 (SLOW/NORMAL/FAST)
            groups_a1_2 = {
                'SLOW': {'pos': [], 'neg': []},
                'NORMAL': {'pos': [], 'neg': []},
                'FAST': {'pos': [], 'neg': []}
            }
            
            for idx in selected_indices:
                cond_val = int(a1_2_vals[idx])
                reaction_time = a1_4_vals[idx]
                
                condition_name = a1_2_map.get(cond_val)
                if condition_name is None:
                    continue
                
                # 부호별 분류 (0 제외)
                if reaction_time > 0:
                    groups_a1_2[condition_name]['pos'].append(abs(reaction_time))
                elif reaction_time < 0:
                    groups_a1_2[condition_name]['neg'].append(abs(reaction_time))
            
            # A1-2 평균 계산
            for condition in ['SLOW', 'NORMAL', 'FAST']:
                for sign in ['pos', 'neg']:
                    values = groups_a1_2[condition][sign]
                    if values:
                        result[f'A1-2_{condition}_{sign}_mean'] = np.mean(values)
            
        except Exception:
            pass  # 에러 시 -1 유지
        
        return result
    
    # 각 행에 대해 계산
    stats_df = df.apply(calculate_a1_condition_reaction_stats, axis=1, result_type='expand')
    
    # 결과 병합
    for col in stats_df.columns:
        df[col] = stats_df[col].fillna(-1)
    
    return df


def add_a3_condition_reaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A3 컬럼들의 조건별, 방향별, 위치별 반응속도 파생변수 계산
    
    A3-1: Condition1 (1=small, 2=big)
    A3-2: Condition2 (1~8, 시계방향 위치)
    A3-3: Condition3 (1=left, 2=right)
    A3-4: Condition4 (1~8, 보조 위치)
    A3-5: Response1 (1=valid correct, 2=valid incorrect, 3=invalid correct, 4=invalid incorrect)
    A3-6: Response2 (0=No, 1=Yes)
    A3-7: ResponseTime
    
    A3-6 == 1인 trial만 선택하여:
    
    2단계 - Condition1 × Response1 (16개):
    - A3_small_valid_correct_mean, A3_small_valid_correct_count
    - A3_small_valid_incorrect_mean, A3_small_valid_incorrect_count
    - A3_small_invalid_correct_mean, A3_small_invalid_correct_count
    - A3_small_invalid_incorrect_mean, A3_small_invalid_incorrect_count
    - A3_big_valid_correct_mean, A3_big_valid_correct_count
    - A3_big_valid_incorrect_mean, A3_big_valid_incorrect_count
    - A3_big_invalid_correct_mean, A3_big_invalid_correct_count
    - A3_big_invalid_incorrect_mean, A3_big_invalid_incorrect_count
    
    3단계 - 방향·위치 기반 (24개):
    좌/우:
    - A3_left_rt_mean, A3_left_rt_count
    - A3_right_rt_mean, A3_right_rt_count
    - A3_left_right_rt_diff
    
    위치별:
    - A3_pos1_rt_mean ~ A3_pos8_rt_mean
    - A3_pos1_rt_count ~ A3_pos8_rt_count
    
    요약:
    - A3_pos_mean_std
    - A3_pos_with_max_mean
    - A3_pos_with_min_mean
    
    Returns:
    --------
    pd.DataFrame
        40개의 파생변수가 추가된 데이터프레임
    """
    df = df.copy()
    
    # 필요한 컬럼 체크
    required_cols = ['A3-1', 'A3-2', 'A3-3', 'A3-5', 'A3-6', 'A3-7']
    if not all(col in df.columns for col in required_cols):
        # 컬럼이 없으면 모든 파생변수를 -1로 설정
        # 2단계 변수 초기화
        for size in ['small', 'big']:
            for resp in ['valid_correct', 'valid_incorrect', 'invalid_correct', 'invalid_incorrect']:
                df[f'A3_{size}_{resp}_mean'] = -1
                df[f'A3_{size}_{resp}_count'] = 0
        
        # 3단계 변수 초기화
        df['A3_left_rt_mean'] = -1
        df['A3_left_rt_count'] = 0
        df['A3_right_rt_mean'] = -1
        df['A3_right_rt_count'] = 0
        df['A3_left_right_rt_diff'] = -1
        
        for pos in range(1, 9):
            df[f'A3_pos{pos}_rt_mean'] = -1
            df[f'A3_pos{pos}_rt_count'] = 0
        
        df['A3_pos_mean_std'] = -1
        df['A3_pos_with_max_mean'] = -1
        df['A3_pos_with_min_mean'] = -1
        
        return df
    
    # 조건값 매핑
    condition1_map = {1: 'small', 2: 'big'}
    response1_map = {1: 'valid_correct', 2: 'valid_incorrect', 3: 'invalid_correct', 4: 'invalid_incorrect'}
    
    def calculate_a3_stats(row):
        """각 행에 대해 A3 관련 모든 통계 계산"""
        result = {}
        
        # 초기값 설정
        # 2단계 초기화
        for size in ['small', 'big']:
            for resp in ['valid_correct', 'valid_incorrect', 'invalid_correct', 'invalid_incorrect']:
                result[f'A3_{size}_{resp}_mean'] = -1
                result[f'A3_{size}_{resp}_count'] = 0
        
        # 3단계 초기화
        result['A3_left_rt_mean'] = -1
        result['A3_left_rt_count'] = 0
        result['A3_right_rt_mean'] = -1
        result['A3_right_rt_count'] = 0
        result['A3_left_right_rt_diff'] = -1
        
        for pos in range(1, 9):
            result[f'A3_pos{pos}_rt_mean'] = -1
            result[f'A3_pos{pos}_rt_count'] = 0
        
        result['A3_pos_mean_std'] = -1
        result['A3_pos_with_max_mean'] = -1
        result['A3_pos_with_min_mean'] = -1
        
        try:
            # 각 컬럼 파싱
            a3_1_vals = parse_seq_cached(str(row.get('A3-1', '')))
            a3_2_vals = parse_seq_cached(str(row.get('A3-2', '')))
            a3_3_vals = parse_seq_cached(str(row.get('A3-3', '')))
            a3_5_vals = parse_seq_cached(str(row.get('A3-5', '')))
            a3_6_vals = parse_seq_cached(str(row.get('A3-6', '')))
            a3_7_vals = parse_seq_cached(str(row.get('A3-7', '')))
            
            # 길이 체크 (시퀀스 불일치 대비)
            if not all([a3_1_vals, a3_2_vals, a3_3_vals, a3_5_vals, a3_6_vals, a3_7_vals]):
                return result
            
            min_len = min(len(a3_1_vals), len(a3_2_vals), len(a3_3_vals), 
                         len(a3_5_vals), len(a3_6_vals), len(a3_7_vals))
            
            if min_len == 0:
                return result
            
            # A3-6 == 1인 trial만 선택
            selected_indices = [i for i in range(min_len) if i < len(a3_6_vals) and a3_6_vals[i] == 1]
            
            if not selected_indices:
                return result
            
            # ==============================================
            # 2단계: Condition1 × Response1
            # ==============================================
            cond1_resp1_groups = {}
            for size in ['small', 'big']:
                for resp in ['valid_correct', 'valid_incorrect', 'invalid_correct', 'invalid_incorrect']:
                    cond1_resp1_groups[f'{size}_{resp}'] = []
            
            for idx in selected_indices:
                try:
                    cond1_val = int(a3_1_vals[idx])
                    resp1_val = int(a3_5_vals[idx])
                    rt_val = a3_7_vals[idx]
                    
                    size_name = condition1_map.get(cond1_val)
                    resp_name = response1_map.get(resp1_val)
                    
                    if size_name and resp_name:
                        key = f'{size_name}_{resp_name}'
                        cond1_resp1_groups[key].append(rt_val)
                except (IndexError, ValueError):
                    continue
            
            # 2단계 통계 계산
            for key, values in cond1_resp1_groups.items():
                result[f'A3_{key}_count'] = len(values)
                if values:
                    result[f'A3_{key}_mean'] = np.mean(values)
            
            # ==============================================
            # 3단계: 방향·위치 기반
            # ==============================================
            
            # 좌/우 그룹
            left_rts = []
            right_rts = []
            
            for idx in selected_indices:
                try:
                    direction = int(a3_3_vals[idx])
                    rt_val = a3_7_vals[idx]
                    
                    if direction == 1:  # left
                        left_rts.append(rt_val)
                    elif direction == 2:  # right
                        right_rts.append(rt_val)
                except (IndexError, ValueError):
                    continue
            
            # 좌/우 통계
            if left_rts:
                result['A3_left_rt_mean'] = np.mean(left_rts)
                result['A3_left_rt_count'] = len(left_rts)
            
            if right_rts:
                result['A3_right_rt_mean'] = np.mean(right_rts)
                result['A3_right_rt_count'] = len(right_rts)
            
            # 좌우 차이
            if left_rts and right_rts:
                result['A3_left_right_rt_diff'] = result['A3_left_rt_mean'] - result['A3_right_rt_mean']
            
            # 위치별 그룹
            position_groups = {pos: [] for pos in range(1, 9)}
            
            for idx in selected_indices:
                try:
                    position = int(a3_2_vals[idx])
                    rt_val = a3_7_vals[idx]
                    
                    if 1 <= position <= 8:
                        position_groups[position].append(rt_val)
                except (IndexError, ValueError):
                    continue
            
            # 위치별 통계
            position_means = []
            for pos in range(1, 9):
                values = position_groups[pos]
                result[f'A3_pos{pos}_rt_count'] = len(values)
                if values:
                    mean_val = np.mean(values)
                    result[f'A3_pos{pos}_rt_mean'] = mean_val
                    position_means.append(mean_val)
            
            # 위치 기반 요약 지표
            if position_means and len(position_means) >= 2:
                result['A3_pos_mean_std'] = np.std(position_means)
                
                # 최대/최소 평균을 가진 위치 찾기
                valid_positions = [(pos, result[f'A3_pos{pos}_rt_mean']) 
                                  for pos in range(1, 9) 
                                  if result[f'A3_pos{pos}_rt_mean'] != -1]
                
                if valid_positions:
                    max_pos = min([pos for pos, val in valid_positions 
                                  if val == max(v for _, v in valid_positions)])
                    min_pos = min([pos for pos, val in valid_positions 
                                  if val == min(v for _, v in valid_positions)])
                    
                    result['A3_pos_with_max_mean'] = max_pos
                    result['A3_pos_with_min_mean'] = min_pos
            
        except Exception:
            pass  # 에러 시 초기값 유지
        
        return result
    
    # 각 행에 대해 계산
    stats_df = df.apply(calculate_a3_stats, axis=1, result_type='expand')
    
    # 결과 병합
    for col in stats_df.columns:
        if col not in df.columns:
            df[col] = stats_df[col].fillna(-1 if 'mean' in col or 'diff' in col or 'std' in col or 'with' in col else 0)
    
    return df


def add_a4_condition_reaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    A4 컬럼들의 조건별 반응속도 평균 계산
    
    A4-1: Condition1 (1=con, 2=incon)
    A4-2: Condition2 (1=red, 2=green)
    A4-3: Response (0=오답, 1=정답)
    A4-4: (사용 안 함)
    A4-5: ResponseTime
    
    A4-3 == 1인 trial(정답 trial)만 선택하여:
    - A4-1 × A4-2 조합별 A4-5 평균 계산
    
    생성되는 변수:
    - A4_con_red_rt_mean        (A4-1=1, A4-2=1)
    - A4_con_green_rt_mean      (A4-1=1, A4-2=2)
    - A4_incon_red_rt_mean      (A4-1=2, A4-2=1)
    - A4_incon_green_rt_mean    (A4-1=2, A4-2=2)
    
    Parameters:
    -----------
    df : pd.DataFrame
        A4-1, A4-2, A4-3, A4-5 컬럼을 포함한 데이터프레임
    
    Returns:
    --------
    pd.DataFrame
        4개의 파생변수가 추가된 데이터프레임
    """
    df = df.copy()
    
    # 필요한 컬럼 체크
    required_cols = ['A4-1', 'A4-2', 'A4-3', 'A4-5']
    if not all(col in df.columns for col in required_cols):
        # 컬럼이 없으면 모든 파생변수를 -1로 설정
        df['A4_con_red_rt_mean'] = -1
        df['A4_con_green_rt_mean'] = -1
        df['A4_incon_red_rt_mean'] = -1
        df['A4_incon_green_rt_mean'] = -1
        return df
    
    # 조건값 매핑
    a4_1_map = {1: 'con', 2: 'incon'}
    a4_2_map = {1: 'red', 2: 'green'}
    
    def calculate_a4_stats(row):
        """각 행에 대해 A4 조합별 평균 계산"""
        result = {
            'A4_con_red_rt_mean': -1,
            'A4_con_green_rt_mean': -1,
            'A4_incon_red_rt_mean': -1,
            'A4_incon_green_rt_mean': -1,
        }
        
        try:
            # 각 컬럼 파싱
            a4_1_vals = parse_seq_cached(str(row.get('A4-1', '')))
            a4_2_vals = parse_seq_cached(str(row.get('A4-2', '')))
            a4_3_vals = parse_seq_cached(str(row.get('A4-3', '')))
            a4_5_vals = parse_seq_cached(str(row.get('A4-5', '')))
            
            # 길이 체크 (시퀀스 불일치 대비)
            if not all([a4_1_vals, a4_2_vals, a4_3_vals, a4_5_vals]):
                return result
            
            min_len = min(len(a4_1_vals), len(a4_2_vals), len(a4_3_vals), len(a4_5_vals))
            
            if min_len == 0:
                return result
            
            # A4-3 == 1인 trial만 선택 (정답만)
            selected_indices = [i for i in range(min_len) if i < len(a4_3_vals) and a4_3_vals[i] == 1]
            
            if not selected_indices:
                return result
            
            # 조합별 그룹화
            groups = {
                'con_red': [],
                'con_green': [],
                'incon_red': [],
                'incon_green': [],
            }
            
            for idx in selected_indices:
                try:
                    cond1_val = int(a4_1_vals[idx])
                    cond2_val = int(a4_2_vals[idx])
                    rt_val = a4_5_vals[idx]
                    
                    cond1_name = a4_1_map.get(cond1_val)
                    cond2_name = a4_2_map.get(cond2_val)
                    
                    if cond1_name and cond2_name:
                        key = f'{cond1_name}_{cond2_name}'
                        groups[key].append(rt_val)
                except (IndexError, ValueError):
                    continue
            
            # 평균 계산
            for key, values in groups.items():
                if values:
                    result[f'A4_{key}_rt_mean'] = np.mean(values)
            
        except Exception:
            pass  # 에러 시 -1 유지
        
        return result
    
    # 각 행에 대해 계산
    stats_df = df.apply(calculate_a4_stats, axis=1, result_type='expand')
    
    # 결과 병합
    for col in stats_df.columns:
        df[col] = stats_df[col].fillna(-1)
    
    return df



