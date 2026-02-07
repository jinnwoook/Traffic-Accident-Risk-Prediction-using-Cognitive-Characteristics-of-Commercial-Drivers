import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve
import json
import sys
from catboost import CatBoostClassifier, Pool
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# optimal_preprocess_full 모듈 경로 추가
from optimal_preprocess_full import BTestPreprocessor
from pre_b import (
    build_primarykey_testdate_dict,
    add_primarykey_month_diff_features,
)
from generate_retention_features_fast import (
    build_gap_label_same_statistics,
    build_consecutive_00_statistics,
    add_retention_features_fast,
)


def _normalize_test_date(value):
    """Convert TestDate value to integer timestamp (seconds)"""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None

    ts = None

    # 이미 Timestamp/Datetime 객체인 경우
    if isinstance(value, (pd.Timestamp, datetime)):
        ts = value
    else:
        # 숫자형인 경우 유형별로 처리
        if isinstance(value, (np.integer, int, np.floating, float)):
            # 정수/실수 → 우선 정수로 변환
            try:
                numeric_val = int(value)
            except (TypeError, ValueError, OverflowError):
                numeric_val = None

            if numeric_val is None:
                return None

            # 1970년 이후 초 단위 타임스탬프 추정 (약 2001년 이후 값)
            if numeric_val >= 10**9:
                ts = pd.to_datetime(numeric_val, unit='s', errors='coerce')
            else:
                value_str = str(numeric_val)
                # 길이에 따라 YYYYMM / YYYYMMDD / YYYYMMDDHHMM 등 추정
                parse_formats = ['%Y%m', '%Y%m%d', '%Y%m%d%H%M', '%Y%m%d%H%M%S']
                for fmt in parse_formats:
                    ts = pd.to_datetime(value_str, format=fmt, errors='coerce')
                    if not pd.isna(ts):
                        break
        else:
            # 문자열 등은 to_datetime으로 직접 파싱
            ts = pd.to_datetime(value, errors='coerce')

    if ts is None or pd.isna(ts):
        return None

    return int(ts.value // 10**9)


def build_label_pattern_history(df, save_path: str | None = None):
    """
    전체 train 데이터를 순회하여 PrimaryKey별로 TestDate 순서대로 Label 시퀀스를 딕셔너리로 저장
    """
    required_cols = {'PrimaryKey', 'TestDate', 'Label'}
    if not required_cols.issubset(df.columns):
        return {}

    hist = {}

    df_work = df[list(required_cols)].copy()
    df_work = df_work.dropna(subset=['PrimaryKey', 'Label'])

    for pk, group in df_work.groupby('PrimaryKey'):
        pk_list = []
        for _, row in group.iterrows():
            test_date = _normalize_test_date(row['TestDate'])
            if test_date is None:
                continue

            try:
                label_val = int(row['Label'])
            except (TypeError, ValueError):
                continue

            pk_list.append((test_date, label_val))

        if len(pk_list) == 0:
            continue

        pk_list.sort(key=lambda x: x[0])
        hist[str(pk)] = pk_list

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(hist, f, ensure_ascii=False, indent=2)

    return hist


def add_label_pattern_features(df: pd.DataFrame, label_pattern_history: dict,
                               out_col_prefix: str = 'pattern') -> pd.DataFrame:
    """
    각 행에 대해 TestDate 이전 시점까지의 4가지 패턴 출현 횟수를 계산하여 파생변수 생성
    """
    df_copy = df.copy()

    pattern_cols = [
        f'{out_col_prefix}_1to1',
        f'{out_col_prefix}_1to0',
        f'{out_col_prefix}_0to1',
        f'{out_col_prefix}_0to0',
    ]

    if 'PrimaryKey' not in df_copy.columns or 'TestDate' not in df_copy.columns:
        for col in pattern_cols:
            df_copy[col] = 0
        return df_copy

    def count_patterns(row):
        pk_val = row.get('PrimaryKey')
        if pd.isna(pk_val):
            return 0, 0, 0, 0

        pk = str(pk_val)
        current_date = _normalize_test_date(row.get('TestDate'))
        if current_date is None or pk not in label_pattern_history:
            return 0, 0, 0, 0

        pk_list = label_pattern_history[pk]
        prev_list = [(td, label) for td, label in pk_list if td < current_date]

        if len(prev_list) < 2:
            return 0, 0, 0, 0

        pattern_1to1 = pattern_1to0 = pattern_0to1 = pattern_0to0 = 0

        for i in range(1, len(prev_list)):
            prev_label = prev_list[i - 1][1]
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

    patterns = df_copy.apply(count_patterns, axis=1, result_type='expand')
    for idx, col in enumerate(pattern_cols):
        df_copy[col] = patterns[idx].fillna(0).astype(int)

    return df_copy


def add_label_count_features(df: pd.DataFrame, 
                             label_pattern_history_b: dict,
                             label_pattern_history_a: dict = None) -> pd.DataFrame:
    """현재 TestDate 이전까지 B와 A 각각의 총 기록 수, A 최신 기록과의 월 차이, A 최신 라벨과 B 최초 라벨 일치 여부를 계산"""
    df_copy = df.copy()

    b_total_col = 'b_past_label_total_count'
    a_total_col = 'a_past_label_total_count'
    a_last_month_diff_col = 'a_last_testdate_month_diff'
    label_match_col = 'a_last_b_first_label_match'

    df_copy[b_total_col] = 0
    df_copy[a_total_col] = 0
    df_copy[a_last_month_diff_col] = -1
    df_copy[label_match_col] = -1

    if 'PrimaryKey' not in df_copy.columns or 'TestDate' not in df_copy.columns:
        return df_copy

    def count_prev_records(row):
        pk = str(row.get('PrimaryKey', ''))
        if not pk:
            return 0, 0, -1, -1

        current_ts = _normalize_test_date(row.get('TestDate'))
        if current_ts is None:
            return 0, 0, -1, -1

        # B 기록 수 계산
        b_count = 0
        b_first_label = None
        if pk in label_pattern_history_b:
            pk_list_b = label_pattern_history_b[pk]
            prev_b = [(ts, label) for ts, label in pk_list_b if ts < current_ts]
            b_count = len(prev_b)
            if prev_b:
                b_first_label = prev_b[0][1]  # 가장 오래된 B 라벨

        # A 기록 수 계산 및 A 최신 TestDate와의 월 차이 계산
        a_count = 0
        a_month_diff = -1
        a_last_label = None
        if label_pattern_history_a and pk in label_pattern_history_a:
            pk_list_a = label_pattern_history_a[pk]
            prev_a = [(ts, label) for ts, label in pk_list_a if ts < current_ts]
            a_count = len(prev_a)
            
            if prev_a:
                # 가장 최근 A 기록의 TestDate와 Label
                last_a_ts = prev_a[-1][0]
                a_last_label = prev_a[-1][1]
                try:
                    # 타임스탬프를 날짜로 변환
                    from datetime import datetime
                    current_dt = datetime.fromtimestamp(current_ts)
                    last_a_dt = datetime.fromtimestamp(last_a_ts)
                    # 월 차이 계산
                    a_month_diff = (current_dt.year - last_a_dt.year) * 12 + (current_dt.month - last_a_dt.month)
                except:
                    a_month_diff = -1

        # A 최신 라벨과 B 최초 라벨 일치 여부 (둘 다 있을 때만)
        label_match = -1
        if a_last_label is not None and b_first_label is not None:
            label_match = 1 if a_last_label == b_first_label else 0

        return b_count, a_count, a_month_diff, label_match

    counts = df_copy.apply(count_prev_records, axis=1, result_type='expand')
    df_copy[b_total_col] = counts[0].fillna(0).astype(int)
    df_copy[a_total_col] = counts[1].fillna(0).astype(int)
    df_copy[a_last_month_diff_col] = counts[2].fillna(-1).astype(int)
    df_copy[label_match_col] = counts[3].fillna(-1).astype(int)
    return df_copy


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    ece = np.sum(bin_weights * np.abs(prob_true - prob_pred))
    return ece

def competition_metric(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece
    return score, auc, brier, ece

if __name__ == "__main__":
    data_dir = DATA_DIR
    train_label = pd.read_csv(data_dir+"/train.csv")
    print(f"✅ train_label loaded: {train_label.shape}")

    # =========================================================
    # 🔸 A 데이터 로드 및 딕셔너리 생성
    # =========================================================
    print("\n📚 A 데이터 로드 및 dict_A 생성...")
    A_feat = pd.read_csv(data_dir+"/train/A.csv")
    dfA = train_label[train_label["Test"] == "A"].merge(A_feat, on=["Test_id", "Test"], how="left")
    print(f"✅ A merged shape: {dfA.shape}")
    
    dict_A = dfA.groupby('PrimaryKey')['Label'].max().to_dict()
    print(f"✅ dict_A 생성 완료: {len(dict_A)} keys")
    
    # =========================================================
    # 🔸 B 데이터 로드 및 전처리 (optimal_preprocess_full 사용)
    # =========================================================
    print("\n📊 B 데이터 로드 및 전처리 중 (optimal_preprocess_full)...")
    print("⚡ 최적화된 과거 이력 계산 (벡터화) 적용!")
    B_feat = pd.read_csv(data_dir+"/train/B.csv")
    dfB = train_label[train_label["Test"] == "B"].merge(B_feat, on=["Test_id", "Test"], how="left")
    print(f"✅ B merged shape: {dfB.shape}")

    # Label pattern history 생성 및 저장 (B와 A 각각)
    print("\n🧠 B Label pattern history 생성 중...")
    label_pattern_history_b_path = "models/label_pattern_history_b.json"
    label_pattern_history_b = build_label_pattern_history(
        dfB[["PrimaryKey", "TestDate", "Label"]].copy(),
        save_path=label_pattern_history_b_path
    )
    print(f"✅ B Label pattern history 생성 완료: {len(label_pattern_history_b)} keys → {label_pattern_history_b_path}")

    print("🧠 A Label pattern history 생성 중...")
    label_pattern_history_a_path = "models/label_pattern_history_a.json"
    label_pattern_history_a = build_label_pattern_history(
        dfA[["PrimaryKey", "TestDate", "Label"]].copy(),
        save_path=label_pattern_history_a_path
    )
    print(f"✅ A Label pattern history 생성 완료: {len(label_pattern_history_a)} keys → {label_pattern_history_a_path}")

    # PrimaryKey-TestDate dictionary & month diff features
    print("\n🗓 PrimaryKey-TestDate dict 생성 중...")
    pk_testdate_dict_path = "models/primarykey_testdate_dict_b.json"
    pk_testdate_dict = build_primarykey_testdate_dict(
        dfB[["PrimaryKey", "TestDate"]].copy(),
        save_path=pk_testdate_dict_path
    )
    print(f"✅ PrimaryKey-TestDate dict 생성 완료: {len(pk_testdate_dict)} keys → {pk_testdate_dict_path}")

    print("➕ 월간 간격 파생변수 추가 중...")
    dfB = add_primarykey_month_diff_features(dfB, pk_testdate_dict)

    print("➕ 과거 B/A 기록 수 계산 중...")
    dfB = add_label_count_features(dfB, label_pattern_history_b, label_pattern_history_a)
    
    # Gap별 Label Same Rate 통계 계산 (A→B 페어만)
    print("\n🧮 Gap별 Label Same Rate 통계 계산 중...")
    gap_stats_month, gap_stats_bin = build_gap_label_same_statistics(
        dfA[["PrimaryKey", "TestDate", "Label"]].copy(),
        dfB[["PrimaryKey", "TestDate", "Label"]].copy(),
        save_dir="models"
    )
    print(f"✅ Gap 통계 계산 완료!")
    
    # Consecutive 0-0 유지율 통계 계산
    print("\n🧮 Consecutive 0-0 유지율 통계 계산 중...")
    consecutive_00_stats_month, consecutive_00_stats_bin = build_consecutive_00_statistics(
        dfB[["PrimaryKey", "TestDate", "Label"]].copy(),
        save_dir="models"
    )
    print(f"✅ Consecutive 0-0 통계 계산 완료!")
    
    # 유지율 파생변수 추가 (LabelSameRate_gap + Consecutive00_RetentionRate)
    print("\n➕ 유지율 파생변수 추가 중...")
    dfB = add_retention_features_fast(
        dfB,
        dfA[["PrimaryKey", "TestDate", "Label"]].copy(),
        gap_stats_bin,
        gap_stats_month,
        consecutive_00_stats_bin,
        consecutive_00_stats_month
    )
    print(f"✅ 유지율 파생변수 추가 완료!")
    
    # BTestPreprocessor 초기화 및 학습
    print("\n🔧 BTestPreprocessor fit...")
    preprocessor = BTestPreprocessor(dict_A=dict_A)
    preprocessor.fit(dfB)
    print(f"✅ Preprocessor fitted!")
    
    # 전체 B 데이터 전처리 (56 columns: 51 PCs + 5 metadata)
    print("\n🔄 B 데이터 전처리 (transform)...")
    dfB_processed = preprocessor.transform(dfB)
    dfB_processed = add_label_pattern_features(dfB_processed, label_pattern_history_b, out_col_prefix='pattern')
    dfB_processed['PK_prev_month_diff'] = dfB['PK_prev_month_diff'].values
    dfB_processed['PK_avg_prev_month_diff'] = dfB['PK_avg_prev_month_diff'].values
    # dfB_processed['b_past_label_total_count'] = dfB['b_past_label_total_count'].values
    # dfB_processed['a_past_label_total_count'] = dfB['a_past_label_total_count'].values
    # dfB_processed['a_last_testdate_month_diff'] = dfB['a_last_testdate_month_diff'].values
    # dfB_processed['a_last_b_first_label_match'] = dfB['a_last_b_first_label_match'].values
    dfB_processed['LabelSameRate_gap'] = dfB['LabelSameRate_gap'].values
    dfB_processed['Consecutive00_RetentionRate'] = dfB['Consecutive00_RetentionRate'].values
    print(f"✅ B 전처리 완료! Shape: {dfB_processed.shape}")
    print(f"   Columns: {list(dfB_processed.columns)[:10]}...")
    
    # Label과 Test_id 추가
    dfB_processed['Label'] = dfB['Label'].values
    dfB_processed['Test_id'] = dfB['Test_id'].values
    
    # =========================================================
    # 🔸 Train/Val Split
    # =========================================================
    print("\n✂️ Train/Val Split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        dfB_processed,
        dfB_processed["Label"],
        test_size=0.15,
        stratify=dfB_processed["Label"],
        random_state=42
    )
    print(f"Train shape: {X_tr.shape}, Val shape: {X_val.shape}")

    # ================================Train 준비=============================================
    print("\n📋 Train 데이터 준비 중...")
    train_Y = X_tr["Label"].values
    train_X = X_tr.drop(columns=["Label", "Test_id"], errors="ignore")

    categorical_columns = ['past_A_history']

    for col in categorical_columns:
        if col in train_X.columns:
            train_X[col] = train_X[col].astype(str)

    if 'TestDate' in train_X.columns:
        train_X['TestDate'] = pd.to_datetime(train_X['TestDate']).astype(np.int64) // 10**9

    drop_feature_cols = ['PrimaryKey', 'past_label_prev_label', 'past_label_label1_flag', 'past_label_label1_count', 
                          'b_past_label_total_count', 'a_past_label_total_count', 'a_last_testdate_month_diff',
                          'a_last_b_first_label_match']
    train_X = train_X.drop(columns=drop_feature_cols, errors='ignore')

    numeric_cols = train_X.select_dtypes(include=[np.number]).columns
    train_X[numeric_cols] = train_X[numeric_cols].fillna(-999)

    meta_data = train_X.columns.tolist()
    print(f"✅ Train features: {len(meta_data)} columns")
    print(f"   Categorical: {categorical_columns}")

    cat_feature_indices = [
        i for i, col in enumerate(train_X.columns)
        if col in categorical_columns
    ]

    # ==========================================================================================
    # 🔸 Validation 준비
    # ==========================================================================================
    print("\n📋 Validation 데이터 준비 중...")
    val_Y = y_val.values
    val_X = X_val.drop(columns=["Label", "Test_id"], errors="ignore")

    for col in categorical_columns:
        if col in val_X.columns:
            val_X[col] = val_X[col].astype(str)

    if 'TestDate' in val_X.columns:
        val_X['TestDate'] = pd.to_datetime(val_X['TestDate']).astype(np.int64) // 10**9

    val_X = val_X.drop(columns=drop_feature_cols, errors='ignore')

    numeric_cols = val_X.select_dtypes(include=[np.number]).columns
    val_X[numeric_cols] = val_X[numeric_cols].fillna(-999)

    print(f"✅ Validation features: {val_X.shape[1]} columns")
    print(f"📊 최종 shape: Train X={train_X.shape}, y={train_Y.shape} | Val X={val_X.shape}, y={val_Y.shape}")
    
    # =========================================================
    # 🔸 CatBoost 모델 학습
    # =========================================================
    print("\n🚀 CatBoost 모델 학습 시작...")
    
    b_param = {
        "iterations": 3000,
        "learning_rate": 0.005,
        "depth": 9,
        "l2_leaf_reg": 10,
        "bootstrap_type": "Bernoulli",
        "eval_metric": "AUC",
        "subsample": 1,
        "task_type": "CPU",
        "verbose": 100,
    }

    # CatBoost Pool 생성
    train_pool = Pool(train_X, train_Y, cat_features=cat_feature_indices)
    val_pool = Pool(val_X, val_Y, cat_features=cat_feature_indices)

    # Model training with validation
    model = CatBoostClassifier(**b_param)
    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=200,
        use_best_model=True,
    )

    # Validation performance
    prob_val = model.predict_proba(val_X)[:, 1]
    score, auc, brier, ece = competition_metric(val_Y, prob_val)
    print(f"\n{'='*60}")
    print(f"🎯 Validation 성능")
    print(f"{'='*60}")
    print(f"AUC   = {auc:.4f}")
    print(f"Brier = {brier:.4f}")
    print(f"ECE   = {ece:.4f}")
    print(f"Score = {score:.6f}")
    print(f"{'='*60}\n")
    
    out_path = "models"
    # Save model
    out_path = out_path + f"/cat_{score:.6f}.cbm"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_model(out_path)
    print(f"✅ CatBoost model saved → {out_path}")
    
    # =========================================================
    # 🔸 dict_A 저장 (추론 시 사용)
    # =========================================================
    dict_A_path = "models/dict_A.json"
    with open(dict_A_path, 'w', encoding='utf-8') as f:
        json.dump({str(k): int(v) for k, v in dict_A.items()}, f, ensure_ascii=False, indent=2)
    print(f"✅ dict_A saved → {dict_A_path}")
    
    # =========================================================
    # 🔸 Preprocessor 저장 (Test 추론 시 필수!)
    # =========================================================
    import pickle
    preprocessor_path = "models/preprocessor.pkl"
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"✅ Preprocessor (모든 PCA 포함) saved → {preprocessor_path}")
    print(f"   - 이 파일은 Test 데이터 전처리 시 필수입니다!")

    # =========================================================
    # 🔸 Feature 이름 저장
    # =========================================================
    feature_names_path = "models/feature_names.json"
    feature_payload = {
        "features": meta_data,
        "categorical_features": [col for col in categorical_columns if col in meta_data],
        "cat_feature_indices": cat_feature_indices,
    }
    with open(feature_names_path, 'w', encoding='utf-8') as f:
        json.dump(feature_payload, f, ensure_ascii=False, indent=2)
    print(f"✅ Feature 이름 saved → {feature_names_path}")
    
    # =========================================================
    # 🔸 최종 B DataFrame 저장 (LabelSameRate_gap 포함)
    # =========================================================
    b_with_label_same_rate_path = "models/b_with_label_same_rate.csv"
    dfB.to_csv(b_with_label_same_rate_path, index=False, encoding='utf-8')
    print(f"✅ B with LabelSameRate_gap saved → {b_with_label_same_rate_path}")
    
    # =========================================================
    # 📋 필요 파일 목록 출력
    # =========================================================
    print(f"\n{'='*60}")
    print(f"📦 Test 추론 시 필요한 파일들:")
    print(f"{'='*60}")
    print(f"1. {preprocessor_path} (전처리 객체 - 모든 PCA 포함)")
    print(f"2. {dict_A_path} (A 과거 이력)")
    print(f"3. {out_path} (학습된 CatBoost 모델)")
    print(f"4. optimal_preprocess_full.py (전처리 코드)")
    print(f"5. models/gap_label_same_rate_stats.json (LabelSameRate 통계)")
    print(f"6. models/gap_label_same_rate_month.csv (월별 통계)")
    print(f"7. models/gap_label_same_rate_bin.csv (Bin별 통계)")
    print(f"8. models/consecutive_00_retention_stats.json (Consecutive00 통계)")
    print(f"9. models/consecutive_00_retention_month.csv (월별 00 통계)")
    print(f"10. models/consecutive_00_retention_bin.csv (Bin별 00 통계)")
    print(f"11. {b_with_label_same_rate_path} (B 데이터 with 유지율 변수)")
    print(f"{'='*60}")
