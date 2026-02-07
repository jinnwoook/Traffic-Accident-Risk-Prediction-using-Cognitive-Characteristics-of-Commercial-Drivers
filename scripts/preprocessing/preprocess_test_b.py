# pre_b utilities
import os, json
import numpy as np
import pandas as pd
from scipy import stats

def preprocess_b_features(df):
    
    # ------------------ 유틸 함수들 ------------------
    def convert_age_features(df, col="Age"):
        def _convert(val):
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
        df = df.copy()
        if col in df.columns:
            df[f"{col}_num"] = df[col].apply(_convert)
        return df

    def count_minus_values(series):
        def _count_minus(val):
            if pd.isna(val) or val == "":
                return 0
            try:
                values = [float(x.strip()) for x in str(val).split(",") if x.strip()]
                return sum(1 for v in values if v < 0)
            except:
                return 0
        return series.fillna("").apply(_count_minus)

    def count_ones(series):
        def _count_ones(val):
            if pd.isna(val) or val == "":
                return 0
            try:
                values = [int(float(x.strip())) for x in str(val).split(",") if x.strip()]
                return sum(1 for v in values if v == 1)
            except:
                return 0
        return series.fillna("").apply(_count_ones)

    def count_zeros(series):
        def _count_zeros(val):
            if pd.isna(val) or val == "":
                return 0
            try:
                values = [float(x.strip()) for x in str(val).split(",") if x.strip()]
                return sum(1 for v in values if v == 0.0)
            except:
                return 0
        return series.fillna("").apply(_count_zeros)

    def count_up_one_or_zero(series):
        def _count_condition(val):
            if pd.isna(val) or val == "":
                return 0
            try:
                values = [float(x.strip()) for x in str(val).split(",") if x.strip()]
                return sum(1 for v in values if (v >= 1.0) or (v == 0.0))
            except:
                return 0
        return series.fillna("").apply(_count_condition)

    def parse_list_column(col):
        """문자열 -> 리스트(숫자로 변환 시도). NaN -> []"""
        if pd.isna(col):
            return []
        s = str(col).strip()
        if s == "":
            return []
        parts = s.split(',')
        out = []
        for p in parts:
            p = p.strip()
            if p == "":
                continue
            try:
                if '.' in p:
                    out.append(float(p))
                else:
                    out.append(int(p))
            except:
                try:
                    out.append(float(p))
                except:
                    out.append(p)
        return out

    def safe_mean(lst):
        if not lst:
            return np.nan
        try:
            nums = [x for x in lst if isinstance(x, (int, float, np.number))]
            if not nums:
                return np.nan
            return float(np.mean(nums))
        except:
            return np.nan

    def safe_std(lst):
        if not lst or len(lst) < 2:
            return np.nan
        try:
            nums = [x for x in lst if isinstance(x, (int, float, np.number))]
            if len(nums) < 2:
                return np.nan
            return float(np.std(nums))
        except:
            return np.nan

    def mean_rt_by_codes(resp_list, rt_list, code_set):
        if not isinstance(resp_list, (list, tuple)) or not isinstance(rt_list, (list, tuple)):
            return np.nan
        L = min(len(resp_list), len(rt_list))
        selected = [rt_list[i] for i in range(L) if resp_list[i] in code_set and isinstance(rt_list[i], (int, float, np.number))]
        return safe_mean(selected)

    def rt_diff_correct_incorrect(rt_list, resp_list):
        """정답과 오답의 반응시간 차이"""
        correct_rt = [rt for rt, resp in zip(rt_list, resp_list) if resp in [1,3,5]]
        incorrect_rt = [rt for rt, resp in zip(rt_list, resp_list) if resp in [2,4,6]]
        if correct_rt and incorrect_rt:
            return np.mean(incorrect_rt) - np.mean(correct_rt)
        return np.nan

    # ------------------ 시작 본문 ------------------
    df = df.copy()
    print("\n=== 전처리 시작 ===")

    # Age 변환
    print("Age 변환 중...")
    df = convert_age_features(df, col="Age")

    

    # ==================== B 관련 파생변수 (B.py + B2.py 통합) ====================
    
    # B1 검사
    print("\nB1 검사 특징 추출 중...")
    if any(c in df.columns for c in ["B1-1", "B1-2", "B1-3"]):
        df['B1_1_list'] = df['B1-1'].apply(parse_list_column) if 'B1-1' in df.columns else [[] for _ in range(len(df))]
        df['B1_2_list'] = df['B1-2'].apply(parse_list_column) if 'B1-2' in df.columns else [[] for _ in range(len(df))]
        df['B1_3_list'] = df['B1-3'].apply(parse_list_column) if 'B1-3' in df.columns else [[] for _ in range(len(df))]

        df['b1_acc'] = df['B1_1_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)
        df['b1_rt_mean'] = df['B1_2_list'].apply(safe_mean)
        df['b1_rt_std'] = df['B1_2_list'].apply(safe_std)

        df['b1_change_correct_cnt'] = df['B1_3_list'].apply(lambda x: sum([1 for i in x if i == 1]))
        df['b1_change_incorrect_cnt'] = df['B1_3_list'].apply(lambda x: sum([1 for i in x if i == 2]))
        df['b1_nonchange_correct_cnt'] = df['B1_3_list'].apply(lambda x: sum([1 for i in x if i == 3]))
        df['b1_nonchange_incorrect_cnt'] = df['B1_3_list'].apply(lambda x: sum([1 for i in x if i == 4]))

    # B2 검사
    print("B2 검사 특징 추출 중...")
    if any(c in df.columns for c in ["B2-1", "B2-2", "B2-3"]):
        df['B2_1_list'] = df['B2-1'].apply(parse_list_column) if 'B2-1' in df.columns else [[] for _ in range(len(df))]
        df['B2_2_list'] = df['B2-2'].apply(parse_list_column) if 'B2-2' in df.columns else [[] for _ in range(len(df))]
        df['B2_3_list'] = df['B2-3'].apply(parse_list_column) if 'B2-3' in df.columns else [[] for _ in range(len(df))]

        df['b2_acc'] = df['B2_1_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)
        df['b2_rt_mean'] = df['B2_2_list'].apply(safe_mean)
        df['b2_rt_std'] = df['B2_2_list'].apply(safe_std)

        df['b2_change_correct_cnt'] = df['B2_3_list'].apply(lambda x: sum([1 for i in x if i == 1]))
        df['b2_change_incorrect_cnt'] = df['B2_3_list'].apply(lambda x: sum([1 for i in x if i == 2]))
        df['b2_nonchange_correct_cnt'] = df['B2_3_list'].apply(lambda x: sum([1 for i in x if i == 3]))
        df['b2_nonchange_incorrect_cnt'] = df['B2_3_list'].apply(lambda x: sum([1 for i in x if i == 4]))

    # # B3 검사
    print("B3 검사 특징 추출 중...")
    if any(c in df.columns for c in ["B3-1", "B3-2"]):
        df['B3_1_list'] = df['B3-1'].apply(parse_list_column) if 'B3-1' in df.columns else [[] for _ in range(len(df))]
        df['B3_2_list'] = df['B3-2'].apply(parse_list_column) if 'B3-2' in df.columns else [[] for _ in range(len(df))]

        df['b3_acc'] = df['B3_1_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)
        df['b3_rt_mean'] = df['B3_2_list'].apply(safe_mean)
        df['b3_rt_std'] = df['B3_2_list'].apply(safe_std)

    # B4 검사
    print("B4 검사 특징 추출 중...")
    if any(c in df.columns for c in ["B4-1", "B4-2"]):
        df['B4_1_list'] = df['B4-1'].apply(parse_list_column) if 'B4-1' in df.columns else [[] for _ in range(len(df))]
        df['B4_2_list'] = df['B4-2'].apply(parse_list_column) if 'B4-2' in df.columns else [[] for _ in range(len(df))]

        df['b4_congruent_correct_cnt'] = df['B4_1_list'].apply(lambda x: sum([1 for i in x if i == 1]))
        df['b4_congruent_incorrect_cnt'] = df['B4_1_list'].apply(lambda x: sum([1 for i in x if i == 2]))
        df['b4_incongruent_correct_cnt'] = df['B4_1_list'].apply(lambda x: sum([1 for i in x if i in [3, 5]]))
        df['b4_incongruent_incorrect_cnt'] = df['B4_1_list'].apply(lambda x: sum([1 for i in x if i in [4, 6]]))
        df['b4_acc'] = df['B4_1_list'].apply(lambda x: (sum([1 for i in x if i in [1, 3, 5]]) / len(x)) if len(x) > 0 else np.nan)
        df['b4_rt_mean'] = df['B4_2_list'].apply(safe_mean)
        df['b4_rt_std'] = df['B4_2_list'].apply(safe_std)

        # 반응시간 차이 (정답 vs 오답)
        df['b4_rt_diff_incorrect_minus_correct'] = df.apply(
            lambda row: rt_diff_correct_incorrect(row['B4_2_list'], row['B4_1_list']), axis=1
        )

        # 일치/불일치 조건별 반응시간
        df['b4_rt_congruent_mean'] = df.apply(
            lambda r: mean_rt_by_codes(r['B4_1_list'], r['B4_2_list'], {1,2}), axis=1
        )
        df['b4_rt_incongruent_mean'] = df.apply(
            lambda r: mean_rt_by_codes(r['B4_1_list'], r['B4_2_list'], {3,4,5,6}), axis=1
        )

    # B5 검사
    print("B5 검사 특징 추출 중...")
    if any(c in df.columns for c in ["B5-1", "B5-2"]):
        df['B5_1_list'] = df['B5-1'].apply(parse_list_column) if 'B5-1' in df.columns else [[] for _ in range(len(df))]
        df['B5_2_list'] = df['B5-2'].apply(parse_list_column) if 'B5-2' in df.columns else [[] for _ in range(len(df))]

        df['b5_acc'] = df['B5_1_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)
        df['B5_mean_rt'] = df['B5_2_list'].apply(safe_mean)
        df['B5_std_rt'] = df['B5_2_list'].apply(safe_std)

    # # B6 검사
    print("B6 검사 특징 추출 중...")
    if 'B6' in df.columns:
        df['B6_list'] = df['B6'].apply(parse_list_column)
        df['b6_acc'] = df['B6_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)

    # B7 검사
    print("B7 검사 특징 추출 중...")
    if 'B7' in df.columns:
        df['B7_list'] = df['B7'].apply(parse_list_column)
        df['b7_acc'] = df['B7_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)

    # B8 검사
    print("B8 검사 특징 추출 중...")
    if 'B8' in df.columns:
        df['B8_list'] = df['B8'].apply(parse_list_column)
        df['b8_acc'] = df['B8_list'].apply(lambda x: sum([1 for i in x if i == 1]) / len(x) if len(x) > 0 else np.nan)

    # B9 검사 (Signal Detection Theory)
    print("B9 검사 특징 추출 중...")
    if all(c in df.columns for c in ['B9-1', 'B9-2', 'B9-3', 'B9-4', 'B9-5']):
        df['b9_aud_sensitivity'] = np.where(
            ((df['B9-1'] + df['B9-2']) > 0) & ((df['B9-3'] + df['B9-4']) > 0),
            stats.norm.ppf((df['B9-1'] + 0.5) / (df['B9-1'] + df['B9-2'] + 1)) - 
            stats.norm.ppf((df['B9-3'] + 0.5) / (df['B9-3'] + df['B9-4'] + 1)),
            np.nan
        )
        df['b9_aud_bias'] = np.where(
            ((df['B9-1'] + df['B9-2']) > 0) & ((df['B9-3'] + df['B9-4']) > 0),
            -0.5 * (stats.norm.ppf((df['B9-1'] + 0.5) / (df['B9-1'] + df['B9-2'] + 1)) + 
                    stats.norm.ppf((df['B9-3'] + 0.5) / (df['B9-3'] + df['B9-4'] + 1))),
            np.nan
        )
        df['b9_aud_hit_rate'] = df['B9-1'] / (df['B9-1'] + df['B9-2'])
        df['b9_aud_false_alarm_rate'] = df['B9-3'] / (df['B9-3'] + df['B9-4'])
        df['b9_vis_error_rate'] = df['B9-5'] / 32

    # B10 검사 (Signal Detection Theory)
    print("B10 검사 특징 추출 중...")
    if all(c in df.columns for c in ['B10-1', 'B10-2', 'B10-3', 'B10-4', 'B10-5', 'B10-6']):
        df['b10_aud_sensitivity'] = np.where(
            ((df['B10-1'] + df['B10-2']) > 0) & ((df['B10-3'] + df['B10-4']) > 0),
            stats.norm.ppf((df['B10-1'] + 0.5) / (df['B10-1'] + df['B10-2'] + 1)) - 
            stats.norm.ppf((df['B10-3'] + 0.5) / (df['B10-3'] + df['B10-4'] + 1)),
            np.nan
        )
        df['b10_aud_bias'] = np.where(
            ((df['B10-1'] + df['B10-2']) > 0) & ((df['B10-3'] + df['B10-4']) > 0),
            -0.5 * (stats.norm.ppf((df['B10-1'] + 0.5) / (df['B10-1'] + df['B10-2'] + 1)) + 
                    stats.norm.ppf((df['B10-3'] + 0.5) / (df['B10-3'] + df['B10-4'] + 1))),
            np.nan
        )
        df['b10_aud_hit_rate'] = df['B10-1'] / (df['B10-1'] + df['B10-2'])
        df['b10_aud_false_alarm_rate'] = df['B10-3'] / (df['B10-3'] + df['B10-4'])
        df['b10_vis1_error_rate'] = df['B10-5'] / 52
        df['b10_vis2_accuracy'] = df['B10-6'] / 20

    # ==================== B 점수 파생변수 (B2.py 로직) ====================
    print("\n점수 파생변수 생성 중...")

    # 1. 시야각검사 점수 (B1+B2)
    if all(c in df.columns for c in ['B1_change_correct', 'B1_nonchange_correct', 
                                       'B2_change_correct', 'B2_nonchange_correct',
                                       'B1_3_list', 'B2_3_list']):
        df['b1b2_score'] = (
            df['B1_change_correct'] + df['B1_nonchange_correct'] +
            df['B2_change_correct'] + df['B2_nonchange_correct'] +
            df['B1_3_list'].apply(lambda x: len([i for i in x if i in [1, 3]])) +
            df['B2_3_list'].apply(lambda x: len([i for i in x if i in [1, 3]]))
        )

    # 2. 신호등검사 점수 (B3) - 정답시 반응속도 평균
    if 'B3_1_list' in df.columns and 'B3_2_list' in df.columns:
        def get_correct_rt_mean(idx):
            correct_list = df.loc[idx, 'B3_1_list']
            rt_list = df.loc[idx, 'B3_2_list']
            if len(correct_list) == 0 or len(rt_list) == 0:
                return np.nan
            correct_rts = [rt_list[i] for i in range(min(len(correct_list), len(rt_list))) 
                          if correct_list[i] == 1]
            return np.mean(correct_rts) if len(correct_rts) > 0 else np.nan
        df['b3_score'] = df.index.map(get_correct_rt_mean)

    # 3. 화살표검사 점수 (B4)
    if 'B4_1_list' in df.columns:
        df['b4_congruent_total'] = df['b4_congruent_correct_cnt'] + df['b4_congruent_incorrect_cnt']
        df['b4_incongruent_total'] = df['b4_incongruent_correct_cnt'] + df['b4_incongruent_incorrect_cnt']
        df['b4_accuracy_diff'] = df['b4_congruent_correct_cnt'] - df['b4_incongruent_correct_cnt']
        
        if 'B4_incongruent_rt' in df.columns and 'B4_congruent_rt' in df.columns:
            df['b4_rt_diff'] = df['b4_rt_incongruent_mean'] - df['b4_rt_congruent_mean']

    # 4. 도로찾기검사 점수 (B5)
    if 'B5_1_list' in df.columns:
        df['b5_score'] = df['B5_1_list'].apply(lambda x: sum([1 for i in x if i == 1]))

    # 5. 표지판검사 점수 (B6, B7)
    if 'B6_list' in df.columns:
        df['b6_correct_cnt'] = df['B6_list'].apply(lambda x: sum([1 for i in x if i == 1]))
    if 'B7_list' in df.columns:
        df['b7_correct_cnt'] = df['B7_list'].apply(lambda x: sum([1 for i in x if i == 1]))

    # 6. 추적검사 점수 (B8)
    if 'B8_list' in df.columns:
        df['b8_score'] = df['B8_list'].apply(lambda x: sum([1 for i in x if i == 1]))

    # 7. 복합기능검사A 점수 (B9)
    if all(c in df.columns for c in ['B9-1', 'B9-5']):
        df['b9_aud_correct_cnt'] = df['B9-1']
        df['b9_vis_correct_cnt'] = 32 - df['B9-5']

    # 8. 복합기능검사B 점수 (B10)
    if all(c in df.columns for c in ['B10-1', 'B10-5', 'B10-6']):
        df['b10_aud_correct_cnt'] = df['B10-1']
        df['b10_vis1_correct_cnt'] = 52 - df['B10-5']
        df['b10_vis2_correct_cnt'] = df['B10-6']

    

    print("\n=== 전처리 완료! ===")
    print("\n생성된 주요 변수:")
    print("- B 검사 기본 통계량 (accuracy, mean_rt, std_rt)")
    print("- B 검사 세부 카운트 (correct/incorrect)")
    print("- B 검사 점수 파생변수 (score_B1_B2, score_B3, score_B5, score_B8 등)")

    return df


# (정리) 사용되지 않는 보조 함수 제거


# =========================================================
# 🔸 PrimaryKey 기준 TestDate 딕셔너리 + 월 차이 파생변수 (A와 동일 규칙)
# =========================================================

def build_primarykey_testdate_dict(df, save_path="./model/primarykey_testdate_dict_b.json"):
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        print("⚠️ PrimaryKey or TestDate not found")
        return {}
    pk_date_dict = {}
    for pk, group in df.groupby('PrimaryKey'):
        pk_date_dict[str(pk)] = group['TestDate'].tolist()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(pk_date_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved: {save_path}")
    return pk_date_dict


def _parse_yyyymm(date_str):
    try:
        s = str(date_str)
        return pd.Timestamp(year=int(s[:4]), month=int(s[4:6]), day=1)
    except:
        return None


def _prev_month_diff(current_dt, prev_dt):
    return (current_dt.year - prev_dt.year) * 12 + (current_dt.month - prev_dt.month)


def calculate_prev_month_diff(row, pk_date_dict):
    pk = str(row.get('PrimaryKey', ''))
    current_date = row.get('TestDate')
    if not pk or pk not in pk_date_dict or not current_date:
        return 0
    current_dt = _parse_yyyymm(current_date)
    if current_dt is None:
        return 0
    prev_dates = []
    for d in pk_date_dict[pk]:
        dt = _parse_yyyymm(d)
        if dt and dt < current_dt:
            prev_dates.append(dt)
    if not prev_dates:
        return 0
    most_recent_prev = max(prev_dates)
    return _prev_month_diff(current_dt, most_recent_prev)


def calculate_avg_prev_month_diff(row, pk_date_dict):
    pk = str(row.get('PrimaryKey', ''))
    current_date = row.get('TestDate')
    if not pk or pk not in pk_date_dict or not current_date:
        return 0
    current_dt = _parse_yyyymm(current_date)
    if current_dt is None:
        return 0
    prev_dates = []
    for d in pk_date_dict[pk]:
        dt = _parse_yyyymm(d)
        if dt and dt < current_dt:
            prev_dates.append(dt)
    if not prev_dates:
        return 0
    diffs = [_prev_month_diff(current_dt, p) for p in prev_dates]
    return float(np.mean(diffs)) if diffs else 0


def add_primarykey_month_diff_features(df, pk_date_dict):
    if 'PrimaryKey' not in df.columns or 'TestDate' not in df.columns:
        print("⚠️ PrimaryKey or TestDate not found, skipping...")
        return df
    df = df.copy()
    df['PK_prev_month_diff'] = df.apply(lambda row: calculate_prev_month_diff(row, pk_date_dict), axis=1)
    df['PK_avg_prev_month_diff'] = df.apply(lambda row: calculate_avg_prev_month_diff(row, pk_date_dict), axis=1)
    print("✅ Added B: PK_prev_month_diff, PK_avg_prev_month_diff")
    return df

# === PrimaryKey 과거 라벨 히스토리 (A와 동일 시그니처 제공) ===
import os, json
import pandas as pd

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
        # PrimaryKey별로 TestDate 순 정렬된 (TestDate, Label) 리스트
        pk_list = [(int(row['TestDate']), int(row['Label'])) for _, row in group.iterrows()]
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
        current_date = int(row.get('TestDate', 0))
        
        if not pk or pk not in label_pattern_history:
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
    
    patterns = df.apply(count_patterns, axis=1, result_type='expand')
    df[f'{out_col_prefix}_1to1'] = patterns[0].astype(int)
    df[f'{out_col_prefix}_1to0'] = patterns[1].astype(int)
    df[f'{out_col_prefix}_0to1'] = patterns[2].astype(int)
    df[f'{out_col_prefix}_0to0'] = patterns[3].astype(int)
    
    return df


def add_is_first_test_feature_b(df: pd.DataFrame, label_history_dict: dict = None, out_col: str = 'is_first_test') -> pd.DataFrame:
    """
    각 행에 대해 PrimaryKey 기준으로 현재 TestDate 이전에 과거 이력이 있는지 확인 (B용)
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