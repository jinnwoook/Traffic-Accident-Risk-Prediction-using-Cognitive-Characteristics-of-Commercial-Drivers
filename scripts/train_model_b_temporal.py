"""
이전 시험 피처를 활용한 학습
- PrimaryKey가 중복인 B 데이터만 사용
- 시간 순서상 가장 가까운 이전 시험의 피처를 현재 시험 피처에 추가
- [이전 피처 + 현재 피처] -> 현재 라벨 예측

시간 복잡도 최적화:
- 정렬 후 groupby로 이전 행 찾기 (O(n log n))
- 벡터화 연산 사용
"""

import os
import numpy as np
import pandas as pd
import json
import pickle
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings
warnings.filterwarnings('ignore')

from optimal_preprocess_full import BTestPreprocessor

def create_previous_feature_dataset(df_processed, df_original):
    """
    이전 시험 피처를 추가한 데이터셋 생성 (시간 복잡도 최적화)
    
    Parameters:
    -----------
    df_processed : DataFrame
        전처리된 B 데이터 (PCA features 포함)
    df_original : DataFrame
        원본 B 데이터 (TestDate, Label 포함)
    
    Returns:
    --------
    df_with_prev : DataFrame
        이전 피처가 추가된 데이터 (중복 PrimaryKey만)
    """
    print("\n" + "="*60)
    print("[이전 시험 피처 데이터셋 생성]")
    print("="*60)
    
    # TestDate와 Label 추가
    df_work = df_processed.copy()
    df_work['TestDate_original'] = df_original['TestDate'].values
    df_work['Label'] = df_original['Label'].values
    
    print(f"\n전체 데이터: {len(df_work)} rows")
    
    # PrimaryKey별 개수 계산 (벡터화)
    pk_counts = df_work.groupby('PrimaryKey').size()
    duplicate_pks = pk_counts[pk_counts > 1].index
    
    print(f"중복 PrimaryKey: {len(duplicate_pks)} keys")
    
    # 중복 PrimaryKey만 필터링
    df_dup = df_work[df_work['PrimaryKey'].isin(duplicate_pks)].copy()
    print(f"중복 데이터: {len(df_dup)} rows")
    
    if len(df_dup) == 0:
        print("⚠ 중복 데이터가 없습니다!")
        return pd.DataFrame()
    
    # TestDate로 정렬 (O(n log n))
    df_dup = df_dup.sort_values(['PrimaryKey', 'TestDate_original']).reset_index(drop=True)
    
    # 피처 컬럼 추출 (TestDate, Label, PrimaryKey 제외)
    feature_cols = [col for col in df_dup.columns 
                    if col not in ['TestDate', 'TestDate_original', 'Label', 'PrimaryKey', 'past_A_history']]
    
    print(f"\n피처 컬럼 수: {len(feature_cols)}")
    
    # 이전 행 찾기 (벡터화 - shift 사용)
    print("\n이전 시험 피처 추가 중...")
    
    # PrimaryKey가 변경되는 지점 찾기
    pk_changed = df_dup['PrimaryKey'] != df_dup['PrimaryKey'].shift(1)
    
    # 각 피처에 대해 이전 값 가져오기
    prev_features = {}
    for col in feature_cols:
        prev_col = f'prev_{col}'
        # shift(1)로 이전 행 가져오기
        prev_features[prev_col] = df_dup[col].shift(1)
        # PrimaryKey가 변경된 경우 NaN으로 설정
        prev_features[prev_col] = prev_features[prev_col].where(~pk_changed, np.nan)
    
    # DataFrame으로 변환
    df_prev = pd.DataFrame(prev_features, index=df_dup.index)
    
    # 현재 피처와 이전 피처 결합
    df_combined = pd.concat([
        df_dup[feature_cols + ['Label', 'PrimaryKey', 'TestDate_original', 'past_A_history']],
        df_prev
    ], axis=1)
    
    # 이전 피처가 없는 행 제거 (각 PrimaryKey의 첫 번째 행)
    df_combined = df_combined.dropna(subset=[f'prev_{feature_cols[0]}'])
    
    print(f"✓ 최종 데이터: {len(df_combined)} rows (이전 피처 있는 행만)")
    print(f"✓ 전체 피처 수: {len(feature_cols) * 2} (현재 {len(feature_cols)} + 이전 {len(feature_cols)})")
    
    # 통계
    label_dist = df_combined['Label'].value_counts()
    print(f"\nLabel 분포:")
    print(f"  Label 0: {label_dist.get(0, 0)} ({label_dist.get(0, 0)/len(df_combined)*100:.1f}%)")
    print(f"  Label 1: {label_dist.get(1, 0)} ({label_dist.get(1, 0)/len(df_combined)*100:.1f}%)")
    
    return df_combined


def prepare_features(df, feature_cols):
    """
    모델 학습용 피처 준비
    """
    X = df[feature_cols].copy()
    
    # Categorical 변환
    if 'past_A_history' in X.columns:
        X['past_A_history'] = X['past_A_history'].astype(str)
    
    # 결측치 처리
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    X[numeric_cols] = X[numeric_cols].fillna(-999)
    
    return X


def main():
    print("="*60)
    print("[이전 시험 피처 활용 학습]")
    print("="*60)
    
    # =========================================================
    # 데이터 로드
    # =========================================================
    print("\n[데이터 로드]")
    data_dir = r"C:\Users\82102\Downloads\5fold_with_preprocessing(0.18458)\data"
    
    train_label = pd.read_csv(os.path.join(data_dir, "train.csv"))
    
    # A 데이터
    A_feat = pd.read_csv(os.path.join(data_dir, "train/A.csv"))
    dfA = train_label[train_label["Test"] == "A"].merge(A_feat, on=["Test_id", "Test"], how="left")
    dict_A = dfA.groupby('PrimaryKey')['Label'].max().to_dict()
    print(f"✓ dict_A: {len(dict_A)} keys")
    
    # B 데이터
    B_feat = pd.read_csv(os.path.join(data_dir, "train/B.csv"))
    dfB = train_label[train_label["Test"] == "B"].merge(B_feat, on=["Test_id", "Test"], how="left")
    print(f"✓ B data: {dfB.shape}")
    
    # =========================================================
    # Preprocessor 로드 (이미 fit된 것 사용)
    # =========================================================
    print("\n[Preprocessor 로드]")
    
    preprocessor_path = "models/preprocessor.pkl"
    if os.path.exists(preprocessor_path):
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        print(f"✓ 기존 preprocessor 로드: {preprocessor_path}")
    else:
        print("⚠ Preprocessor 파일이 없습니다. 새로 fit합니다...")
        preprocessor = BTestPreprocessor(dict_A=dict_A)
        preprocessor.fit(dfB)
    
    # dict_A 업데이트
    preprocessor.dict_A = dict_A
    
    # =========================================================
    # 전처리
    # =========================================================
    print("\n[B 데이터 전처리]")
    dfB_processed = preprocessor.transform(dfB)
    print(f"✓ 전처리 완료: {dfB_processed.shape}")
    
    # =========================================================
    # 이전 피처 데이터셋 생성
    # =========================================================
    df_with_prev = create_previous_feature_dataset(dfB_processed, dfB)
    
    if len(df_with_prev) == 0:
        print("❌ 데이터가 없어 종료합니다.")
        return
    
    # =========================================================
    # Train/Val 분리 (PrimaryKey 기준 - 데이터 누수 방지)
    # =========================================================
    print("\n[Train/Val 분리 (PrimaryKey 기준)]")
    
    # 고유 PrimaryKey 추출
    unique_pks = df_with_prev['PrimaryKey'].unique()
    print(f"고유 PrimaryKey 수: {len(unique_pks)}")
    
    # PrimaryKey를 80:20으로 분리 (랜덤)
    np.random.seed(42)
    val_size = int(len(unique_pks) * 0.2)
    val_pks = np.random.choice(unique_pks, size=val_size, replace=False)
    train_pks = np.setdiff1d(unique_pks, val_pks)
    
    print(f"Train PrimaryKeys: {len(train_pks)}")
    print(f"Val PrimaryKeys: {len(val_pks)}")
    
    # PrimaryKey 기준으로 분리
    train_data = df_with_prev[df_with_prev['PrimaryKey'].isin(train_pks)].copy()
    val_data = df_with_prev[df_with_prev['PrimaryKey'].isin(val_pks)].copy()
    
    print(f"✓ Train: {len(train_data)} rows")
    print(f"✓ Val: {len(val_data)} rows")
    
    # 중복 확인
    overlap = set(train_pks) & set(val_pks)
    print(f"✓ Train/Val PrimaryKey 중복: {len(overlap)} (should be 0)")
    
    # 피처 컬럼 (현재 + 이전)
    feature_cols = [col for col in df_with_prev.columns 
                    if col not in ['Label', 'PrimaryKey', 'TestDate_original', 'TestDate']]
    
    print(f"✓ 사용 피처 수: {len(feature_cols)}")
    
    # X, y 준비
    X_train = prepare_features(train_data, feature_cols)
    y_train = train_data['Label'].values
    
    X_val = prepare_features(val_data, feature_cols)
    y_val = val_data['Label'].values
    
    print(f"\nX_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    
    # =========================================================
    # CatBoost 학습
    # =========================================================
    print("\n[CatBoost 학습]")
    
    # Categorical feature 인덱스
    cat_features = []
    if 'past_A_history' in X_train.columns:
        cat_features.append(X_train.columns.get_loc('past_A_history'))
    
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3,
        loss_function='Logloss',
        eval_metric='AUC',
        random_seed=42,
        task_type='GPU',
        devices='0',
        verbose=100,
        early_stopping_rounds=200
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        cat_features=cat_features,
        use_best_model=True,
        plot=False
    )
    
    # =========================================================
    # 평가
    # =========================================================
    print("\n[평가]")
    
    # Validation 예측
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_pred_proba)
    val_brier = brier_score_loss(y_val, val_pred_proba)
    
    print(f"\n✓ Validation AUC: {val_auc:.6f}")
    print(f"✓ Validation Brier: {val_brier:.6f}")
    
    # =========================================================
    # 모델 저장
    # =========================================================
    print("\n[모델 저장]")
    
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    model_filename = f"cat_prev_feat_{val_auc:.6f}.cbm"
    model_path = os.path.join(model_dir, model_filename)
    model.save_model(model_path)
    print(f"✓ 모델 저장: {model_path}")
    
    # Feature importance 저장
    feature_importance = model.get_feature_importance()
    feat_imp_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    feat_imp_path = os.path.join(model_dir, "feature_importance_prev_feat.csv")
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"✓ Feature importance 저장: {feat_imp_path}")
    
    print("\n" + "="*60)
    print("[완료!]")
    print("="*60)
    print(f"최종 Validation AUC: {val_auc:.6f}")
    print(f"최종 Validation Brier: {val_brier:.6f}")
    print("="*60)


if __name__ == "__main__":
    main()
