import os
import json
import time
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.calibration import calibration_curve

from pre_a import (
    preprocess_a_features,
    build_reaction_time_stats_fast,
    build_primarykey_testdate_dict as build_pk_dict_a,
    add_primarykey_month_diff_features as add_month_diff_a,
    build_primary_label_history_with_date,
    add_primary_history_features_with_date,
    build_b_previous_label_dict_for_a,
    add_b_previous_label_to_a,
    add_month_prev_label_adjustment_feature,
    build_label_pattern_history as build_label_pattern_history_a,
    add_label_pattern_features as add_label_pattern_features_a,
    add_is_first_test_feature,
    add_holiday_season_feature,
    add_a1_condition_reaction_features,
    add_a2_condition_reaction_features,
    add_a3_condition_reaction_features,
    add_a4_condition_reaction_features,
)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    bin_totals = np.histogram(y_prob, bins=np.linspace(0, 1, n_bins + 1), density=False)[0]
    non_empty_bins = bin_totals > 0
    bin_weights = bin_totals / len(y_prob)
    bin_weights = bin_weights[non_empty_bins]
    prob_true = prob_true[:len(bin_weights)]
    prob_pred = prob_pred[:len(bin_weights)]
    return float(np.sum(bin_weights * np.abs(prob_true - prob_pred)))


def competition_metric(y_true, y_prob):
    auc = roc_auc_score(y_true, y_prob)
    brier = mean_squared_error(y_true, y_prob)
    ece = expected_calibration_error(y_true, y_prob)
    score = 0.5 * (1 - auc) + 0.25 * brier + 0.25 * ece
    return score, auc, brier, ece


def train_xgb(X_tr, y_tr, X_val, y_val, out_path, params):
    model = XGBClassifier(**params)
    model.fit(
        X_tr,
        y_tr,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    evals_result = model.evals_result()
    auc_history = []
    if (
        isinstance(evals_result, dict)
        and "validation_0" in evals_result
        and "auc" in evals_result["validation_0"]
    ):
        auc_history = [float(v) for v in evals_result["validation_0"]["auc"]]

    if auc_history:
        best_iter = int(np.argmax(auc_history))
        best_auc = float(auc_history[best_iter])
    else:
        best_iter = params.get("n_estimators", 0) - 1
        best_auc = float("nan")

    best_ntree_limit = max(best_iter + 1, 1)

    prob_val = model.predict_proba(X_val, iteration_range=(0, best_ntree_limit))[:, 1]
    score, auc, brier, ece = competition_metric(y_val, prob_val)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save_model(out_path)

    meta_path = out_path.replace(".json", "_meta.json")
    meta_payload = {
        "best_iteration": int(best_iter),
        "best_ntree_limit": int(best_ntree_limit),
        "best_auc": float(best_auc),
        "metric_history_length": len(auc_history),
    }
    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[A] Warning: failed to save XGB meta info → {exc}")

    print(
        f"  Best iteration={best_iter} (ntree_limit={best_ntree_limit}, val_auc={best_auc:.6f})"
    )

    return model, score, auc, brier, ece, model.feature_importances_




if __name__ == "__main__":
    t_total = time.time()
    RANDOM_SEED = 42
    N_FOLDS = 5
    HOLDOUT_RATIO = 0.2

    # =============================================================
    # 🔸 캐시 설정
    # =============================================================
    CACHE_DIR = "./cache"
    os.makedirs(CACHE_DIR, exist_ok=True)
    CACHE_FILE_A = os.path.join(CACHE_DIR, "preprocessed_A_5fold.csv")
    USE_CACHE = True  # False로 설정하면 강제로 재생성

    print("="*80)
    print(f"🚀 A 모델 학습 시작 ({N_FOLDS}-Fold Cross Validation)")
    print(f"  - Holdout ratio: {HOLDOUT_RATIO}")
    print("="*80)

    # =============================================================
    # 🔸 캐시된 전처리 데이터 확인
    # =============================================================
    if USE_CACHE and os.path.exists(CACHE_FILE_A):
        print(f"\n✅ 캐시된 전처리 데이터 발견: {CACHE_FILE_A}")
        print("📂 기존 전처리 데이터를 로드합니다...")
        dfA = pd.read_csv(CACHE_FILE_A)
        print(f"✅ 로드 완료! Shape: {dfA.shape}")
        print("⚡ 전처리 시간 절약!\n")
        
        # 캐시 로드 시 필요한 리소스 로드
        print("📂 저장된 리소스 로드 중...")
        hist_all_A_path = "./model_fold/primary_label_history.json"
        if os.path.exists(hist_all_A_path):
            with open(hist_all_A_path, "r", encoding="utf-8") as f:
                hist_all_A = json.load(f)
            print(f"✅ primary_label_history 로드: {len(hist_all_A)} keys")
        else:
            print("⚠️ primary_label_history.json not found, building from cache...")
            hist_all_A = build_primary_label_history_with_date(dfA, save_path=hist_all_A_path)
        
    else:
        if USE_CACHE:
            print(f"\n⚠️ 캐시 파일 없음. 새로 전처리를 시작합니다...")
        else:
            print(f"\n🔄 강제 재생성 모드. 새로 전처리를 시작합니다...")
        
        # -------------------------------------------------
        # Load data
        # -------------------------------------------------
        print("\n📥 원본 데이터 로드 중...")
        train_label = pd.read_csv("./data/train.csv")

        # B 데이터를 먼저 로드하여 A에 활용할 딕셔너리 생성
        # (전처리 없이 원본에서 PrimaryKey, Label만 사용)
        B_raw = pd.read_csv("./data/train/B.csv")
        dfB_raw = train_label[train_label["Test"] == "B"].merge(B_raw, on=["Test_id", "Test"], how="left")
        
        # B 데이터로부터 A에 사용할 라벨 딕셔너리 생성 및 저장 (any 방식)
        b_prev_label_dict = build_b_previous_label_dict_for_a(
            dfB_raw[["PrimaryKey", "Label"]], 
            save_path="./model_fold/b_previous_label_for_a.json"
        )

        # ===================== A =====================
        print("\n🔧 A 데이터 전처리 중...")
        A_feat = pd.read_csv("./data/train/A.csv")
        
        # A1 조건별 반응속도 파생변수 추가 (전처리 전에 원본 데이터에서 계산)
        print("[A] Adding A1 condition reaction features...")
        A_feat = add_a1_condition_reaction_features(A_feat)
        
        # A2 조건별 반응속도 파생변수 추가 (전처리 전에 원본 데이터에서 계산)
        print("[A] Adding A2 condition reaction features...")
        A_feat = add_a2_condition_reaction_features(A_feat)
        
        # A3 조건별·방향별·위치별 반응속도 파생변수 추가 (전처리 전에 원본 데이터에서 계산)
        print("[A] Adding A3 condition reaction features...")
        A_feat = add_a3_condition_reaction_features(A_feat)
        
        # A4 조건별 반응속도 파생변수 추가 (전처리 전에 원본 데이터에서 계산)
        print("[A] Adding A4 condition reaction features...")
        A_feat = add_a4_condition_reaction_features(A_feat)
        
        # A4 패턴 파생변수 추가 (일단 드랍)
        # print("[A] Adding A4 pattern features...")
        # A_feat = add_a4_pattern_features(A_feat)
        
        # A5 조건별 반응속도 파생변수 추가 (일단 드랍 - 함수가 정의되어 있지 않음)
        # print("[A] Adding A5 condition reaction features...")
        # A_feat = add_a5_condition_reaction_features(A_feat)
        
        # A5 패턴 파생변수 추가 (일단 드랍)
        # print("[A] Adding A5 pattern features...")
        # A_feat = add_a5_pattern_features(A_feat)
        
        rt_stats = build_reaction_time_stats_fast(A_feat, save_path="./model_fold/reaction_time_stats.json")
        A_feat = preprocess_a_features(A_feat, rt_stats=rt_stats, pk_date_dict=None)
        dfA = train_label[train_label["Test"] == "A"].merge(A_feat, on=["Test_id", "Test"], how="left")
        
        # B의 라벨 정보를 A에 추가 (any 방식: B에서 라벨 1이 있으면 1, 없으면 0)
        dfA = add_b_previous_label_to_a(dfA, b_prev_label_dict, out_col='b_previous_label')
        
        # Month diff features for A
        pk_dict_a = build_pk_dict_a(dfA[["PrimaryKey", "TestDate"]], save_path="./model_fold/primarykey_testdate_dict.json")
        dfA = add_month_diff_a(dfA, pk_dict_a)

        # pk_count 저장
        pk_count_a = dfA.groupby('PrimaryKey').size().to_dict()
        with open('./model_fold/pk_count_A.json', 'w', encoding='utf-8') as f:
            json.dump({str(k): int(v) for k,v in pk_count_a.items()}, f, ensure_ascii=False, indent=2)

        # dict_A 저장 (추론용)
        dict_A = dfA.groupby('PrimaryKey')['Label'].max().astype(int).to_dict()
        with open("./model_fold/dict_A.json", "w", encoding="utf-8") as f:
            json.dump({str(k): int(v) for k, v in dict_A.items()}, f, ensure_ascii=False, indent=2)

        # Primary label history (전체 데이터 기준으로 저장)
        hist_all_A = build_primary_label_history_with_date(dfA, save_path="./model_fold/primary_label_history.json")
        
        # 첫 시험 여부 추가 (전처리에 포함)
        dfA = add_is_first_test_feature(dfA, hist_all_A, out_col='is_first_test')
        dfA = add_holiday_season_feature(dfA, out_col='has_holiday_season', months_ahead=6)
        
        # =============================================================
        # 🔸 전처리 완료 후 캐시 저장
        # =============================================================
        print(f"\n💾 전처리 완료! 캐시에 저장 중: {CACHE_FILE_A}")
        dfA.to_csv(CACHE_FILE_A, index=False)
        print(f"✅ 캐시 저장 완료! Shape: {dfA.shape}")
        print(f"⚡ 다음 실행 시 전처리 시간을 절약할 수 있습니다.\n")
    
    # =============================================================
    # 🔸 사용하지 않을 파생변수 드랍 (캐시에 포함되어 있을 수 있음)
    # =============================================================
    # A4 패턴 변수 (feature_names_A.json 78-81번째) + 추가 변수들
    drop_cols_list = [
        'A4_color_repeat_rt_diff_mean',
        'A4_con_incon_switch_rate',
        'A4_con_incon_switch_rt_diff_mean',
        'A4_con_incon_repeat_rt_diff_mean',
        # 추가 드랍 변수들
        'A4_color_switch_rt_diff_mean',
        'A5_nonchange_incorrect_mean',
        'A5_poschange_incorrect_mean',
        'A5_colorchange_incorrect_mean',
        'A5_shapechange_incorrect_mean',
    ]
    
    # A4/A5 패턴 변수 전체 (혹시 다른 것들도 있을 수 있음)
    drop_pattern_cols = [col for col in dfA.columns if col in drop_cols_list or
                         'A4_con_rt_std' in col or 'A4_incon_rt_std' in col or 
                         'A4_con_incon_rt_var_ratio' in col or 'A4_rt_cv' in col or 'A4_color_switch_rate' in col or
                         'A4_switch_cost' in col or 'A4_correct_streak_max' in col or 'A4_incorrect_streak_max' in col or
                         'A4_correct_incorrect_switches' in col or 'A4_response_alternation_rate' in col or
                         'A5_change_pattern_entropy' in col or 'A5_change_switch_rate' in col or 
                         'A5_most_frequent_change' in col or 'A5_nonchange_rt_std' in col or 
                         'A5_poschange_rt_std' in col or 'A5_colorchange_rt_std' in col or 
                         'A5_shapechange_rt_std' in col or 'A5_rt_cv' in col or
                         'A5_correct_streak_max' in col or 'A5_incorrect_streak_max' in col or
                         'A5_response_alternation_rate' in col or 'A5_transition_time_diff_mean' in col or
                         'A5_condition_repetition_effect' in col]
    
    if drop_pattern_cols:
        print(f"\n🗑️ 불필요한 파생변수 드랍: {len(drop_pattern_cols)}개")
        dfA = dfA.drop(columns=drop_pattern_cols, errors='ignore')
        print(f"   드랍된 컬럼: {drop_pattern_cols}")

    # =========================================================
    # 🔸 5-Fold Cross Validation
    # =========================================================
    print("\n")
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    fold_scores = []
    fold_aucs = []
    
    os.makedirs("./model_fold", exist_ok=True)
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(dfA)):
        print(f"\n{'='*80}")
        print(f"📂 Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'='*80}")
        
        dfA_train_full = dfA.iloc[train_idx].copy()
        dfA_val_full = dfA.iloc[val_idx].copy()
        
        # 각 폴드 내에서 추가 홀드아웃 (0.2)
        dfA_tr, dfA_holdout = train_test_split(
            dfA_train_full, test_size=HOLDOUT_RATIO, stratify=dfA_train_full["Label"], random_state=RANDOM_SEED
        )
        
        print(f"  Train: {len(dfA_tr)} samples")
        print(f"  Holdout: {len(dfA_holdout)} samples")
        print(f"  Validation: {len(dfA_val_full)} samples")
        
        # 교차 방식으로 과거 라벨 파생(primary_past_label)
        half1, half2 = train_test_split(
            dfA_tr, test_size=0.5, stratify=dfA_tr['Label'], random_state=RANDOM_SEED
        )
        hist_1 = build_primary_label_history_with_date(half1, save_path=None)
        hist_2 = build_primary_label_history_with_date(half2, save_path=None)
        half1 = add_primary_history_features_with_date(half1, hist_2, out_col='primary_past_label')
        half2 = add_primary_history_features_with_date(half2, hist_1, out_col='primary_past_label')
        dfA_tr_cross = pd.concat([half1, half2], ignore_index=True)
        
        # 검증셋 매핑은 train 기준
        hist_train_A = build_primary_label_history_with_date(dfA_tr, save_path=None)
        dfA_val_full = add_primary_history_features_with_date(dfA_val_full, hist_train_A, out_col='primary_past_label')
        
        # 개월 수와 이전 라벨 기반 확률 값 변수 추가
        dfA_tr_cross = add_month_prev_label_adjustment_feature(dfA_tr_cross, hist_train_A, out_col='month_prev_label_adj')
        dfA_val_full = add_month_prev_label_adjustment_feature(dfA_val_full, hist_train_A, out_col='month_prev_label_adj')
        
        # 라벨 패턴 변수 추가
        dfA_tr_cross = add_label_pattern_features_a(dfA_tr_cross, hist_all_A, out_col_prefix='pattern')
        dfA_val_full = add_label_pattern_features_a(dfA_val_full, hist_all_A, out_col_prefix='pattern')

        yA_tr = dfA_tr_cross["Label"].values
        yA_val = dfA_val_full["Label"].values
        XA_tr_df = dfA_tr_cross.drop(columns=["Label", "Test_id", "Test"], errors="ignore").select_dtypes(include=[np.number]).fillna(-1)
        feature_names_A = XA_tr_df.columns.tolist()
        XA_tr = XA_tr_df.values
        XA_val_df = dfA_val_full.drop(columns=["Label", "Test_id", "Test"], errors="ignore").select_dtypes(include=[np.number]).fillna(-1)
        XA_val = XA_val_df[feature_names_A].values

        xgb_params = {
            "n_estimators": 2000,
            "learning_rate": 0.01,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "random_state": RANDOM_SEED,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
        }

        modelA, scoreA, aucA, brierA, eceA, fiA = train_xgb(
            XA_tr, yA_tr, XA_val, yA_val, out_path=f"./model_fold/xgb_A_fold{fold_idx}.json", params=xgb_params
        )
        
        print(f"  Score={scoreA:.6f} | AUC={aucA:.4f} | Brier={brierA:.4f} | ECE={eceA:.4f}")
        
        fold_scores.append(scoreA)
        fold_aucs.append(aucA)

    # Feature names 저장 (마지막 폴드 기준)
    with open("./model_fold/feature_names_A.json", "w", encoding="utf-8") as f:
        json.dump(feature_names_A, f, ensure_ascii=False, indent=2)

    # =========================================================
    # 🔸 최종 통계
    # =========================================================
    print(f"\n{'='*80}")
    print(f"📊 5-Fold Cross Validation 결과")
    print(f"{'='*80}")
    print(f"Score: {np.mean(fold_scores):.6f} ± {np.std(fold_scores):.6f}")
    print(f"AUC:   {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
    print(f"\nFold별 상세:")
    for i, (score, auc) in enumerate(zip(fold_scores, fold_aucs)):
        print(f"  Fold {i+1}: Score={score:.6f}, AUC={auc:.4f}")
    print(f"{'='*80}")

    print(f"\n✅ A 모델 5-Fold 학습 완료! Total time: {time.time() - t_total:.1f}s")
    print(f"\n📁 저장된 모델:")
    for i in range(N_FOLDS):
        print(f"  - model_fold/xgb_A_fold{i}.json")
        print(f"  - model_fold/xgb_A_fold{i}_meta.json")

