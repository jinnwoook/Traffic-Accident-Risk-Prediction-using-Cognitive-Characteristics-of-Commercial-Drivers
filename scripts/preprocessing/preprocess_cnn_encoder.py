import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from functools import lru_cache


# =========================================================
# 🔸 CNN Feature Extractor Model
# =========================================================

class CNNFeatureExtractor(nn.Module):
    """1D CNN AutoEncoder for sequence feature extraction"""
    def __init__(self, latent_dim=2):
        super(CNNFeatureExtractor, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder (1D CNN)
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc_encode = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, latent_dim),
        )
    
    def encode(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.squeeze(2)
        features = self.fc_encode(x)
        return features
    
    def forward(self, x):
        return self.encode(x)


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
    
    df[f"{name_prefix}_speed1_ones"] = speed1_ones
    df[f"{name_prefix}_speed2_ones"] = speed2_ones
    df[f"{name_prefix}_speed3_ones"] = speed3_ones


# =========================================================
# 🔸 CNN 특징 추출 (기존과 동일)
# =========================================================

def preprocess_sequence_to_list(seq_str):
    if pd.isna(seq_str) or seq_str == "":
        return []
    try:
        values = [float(x.strip()) for x in str(seq_str).split(",") if x.strip()]
        if len(values) == 0:
            return []
        values = np.array(values, dtype=np.float32)
        if values.max() != values.min():
            values = (values - values.min()) / (values.max() - values.min())
        return values.tolist()
    except:
        return []


def combine_a_sequences_for_cnn(row, seq_columns):
    combined = []
    for col in seq_columns:
        if col in row.index:
            seq = preprocess_sequence_to_list(row[col])
            combined.extend(seq)
    if len(combined) == 0:
        combined = [0.0]
    return np.array(combined, dtype=np.float32)


def extract_cnn_features(df, model_path="./model/cnn_feature_extractor.pth"):
    if not os.path.exists(model_path):
        print(f"⚠️ CNN model not found: {model_path}")
        return df
    
    a_seq_columns = [
        "A1-1", "A1-2", "A1-3", "A1-4",
        "A2-1", "A2-2", "A2-3", "A2-4",
        "A3-1", "A3-2", "A3-3", "A3-4", "A3-5", "A3-6", "A3-7",
        "A4-1", "A4-2", "A4-3", "A4-4", "A4-5",
        "A5-1", "A5-2", "A5-3"
    ]
    
    existing_cols = [col for col in a_seq_columns if col in df.columns]
    if len(existing_cols) == 0:
        return df
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        latent_dim = checkpoint.get('latent_dim', 2)
        
        model = CNNFeatureExtractor(latent_dim=latent_dim)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(device)
        model.eval()
        
        sequences = []
        for idx, row in df.iterrows():
            seq = combine_a_sequences_for_cnn(row, existing_cols)
            sequences.append(seq)
        
        batch_size = 128
        all_features = []
        
        with torch.no_grad():
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                max_len = max(len(s) for s in batch_seqs)
                padded = []
                for seq in batch_seqs:
                    if len(seq) < max_len:
                        pad_seq = np.zeros(max_len, dtype=np.float32)
                        pad_seq[:len(seq)] = seq
                        padded.append(pad_seq)
                    else:
                        padded.append(seq)
                
                batch_tensor = torch.FloatTensor(np.array(padded)).to(device)
                features = model.encode(batch_tensor)
                all_features.append(features.cpu().numpy())
        
        all_features = np.vstack(all_features)
        
        for i in range(latent_dim):
            df[f'a_cnn_feature_{i+1}'] = all_features[:, i]
        
        print(f"✅ CNN features extracted: {latent_dim} features added")
        
    except Exception as e:
        print(f"⚠️ Error extracting CNN features: {e}")
    
    return df


# =========================================================
# 🔸 최적화된 메인 전처리 함수
# =========================================================

def pp_fast(df, use_cnn=True, rt_stats=None):
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
            df[f"{col}_minus_num"] = vectorized_count_values(df[col], lambda v: v < 0)
    
    # 응답 1 카운트 (벡터화)
    ones_columns = ["A1-3", "A2-3", "A3-6", "A3-5", "A4-3", "A4-4", "A5-2", "A5-3"]
    for col in ones_columns:
        if col in df.columns:
            df[f"{col}_ONE_num"] = vectorized_count_values(df[col], lambda v: v == 1)
    
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
                    df[f"{col}_ratio_to_train_mean"] = row_abs_means / train_abs_mean
                    df[f"{col}_ratio_to_train_mean"] = df[f"{col}_ratio_to_train_mean"].fillna(1.0)
                    
                    # 반응 방향 (최적화)
                    df[f"{col}_delayed_reaction"] = fast_reaction_direction(df[col])
                    
                    created_count += 1
        
        print(f"✅ Reaction time features: {created_count} features")
    
    # A9 조합 파생변수
    a9_columns = ['A9-1', 'A9-2', 'A9-3', 'A9-5']
    if all(col in df.columns for col in a9_columns):
        df['A9_emotional_behavioral'] = df['A9-1'] + df['A9-2']
        df['A9_behavioral_judgment'] = df['A9-2'] + df['A9-3']
        df['A9_emotional_stress'] = df['A9-1'] + df['A9-5']
        df['A9_comprehensive_stability'] = df['A9-1'] + df['A9-2'] + df['A9-3']
        print(f"✅ A9 combination features: 4 features")
    
    # CNN 특징 추출
    if use_cnn:
        t_cnn = time.time()
        df = extract_cnn_features(df)
        print(f"⏱️ CNN time: {time.time()-t_cnn:.2f}s")
    else:
        print("ℹ️ CNN features disabled")
    
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


# =========================================================
# 🔸 기존 함수들 (변경 없음)
# =========================================================

def load_reaction_time_stats(load_path="./model/reaction_time_stats.json"):
    if not os.path.exists(load_path):
        print(f"⚠️ Reaction time stats not found")
        return {"A1-4": 0.0, "A2-4": 0.0, "A3-7": 0.0, "A4-5": 0.0}
    
    with open(load_path, 'r', encoding='utf-8') as f:
        rt_stats = json.load(f)
    
    print(f"✅ Reaction time stats loaded")
    return rt_stats


def build_primary_label_history(df, save_path="./model/primary_label_history.json"):
    if 'PrimaryKey' not in df.columns or 'Label' not in df.columns:
        print("⚠️ PrimaryKey or Label not found")
        return {}
    
    df_sorted = df.copy()
    if 'TestDate' in df_sorted.columns:
        df_sorted = df_sorted.sort_values('TestDate')
    
    primary_label_dict = df_sorted.groupby('PrimaryKey')['Label'].last().to_dict()
    
    total_keys = len(primary_label_dict)
    label_1_count = sum(1 for label in primary_label_dict.values() if label == 1)
    label_0_count = sum(1 for label in primary_label_dict.values() if label == 0)
    
    print(f"\n📊 Primary Label History:")
    print(f"   Total: {total_keys:,}, Label=1: {label_1_count:,}, Label=0: {label_0_count:,}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        primary_label_dict_str = {str(k): int(v) for k, v in primary_label_dict.items()}
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(primary_label_dict_str, f, ensure_ascii=False, indent=2)
        print(f"✅ Primary label history saved")
    
    return primary_label_dict


def load_primary_label_history(load_path="./model/primary_label_history.json"):
    if not os.path.exists(load_path):
        print(f"⚠️ Primary label history not found")
        return {}
    
    with open(load_path, 'r', encoding='utf-8') as f:
        primary_label_dict_str = json.load(f)
    
    primary_label_dict = {}
    for k, v in primary_label_dict_str.items():
        try:
            key = int(k) if k.isdigit() else k
        except:
            key = k
        primary_label_dict[key] = int(v)
    
    print(f"✅ Primary label history loaded ({len(primary_label_dict):,} keys)")
    return primary_label_dict


def add_primary_history_features(df, primary_label_history):
    df = df.copy()
    
    if 'PrimaryKey' not in df.columns:
        df['primary_past_label'] = 2
        return df
    
    df['primary_past_label'] = df['PrimaryKey'].map(
        lambda pk: primary_label_history.get(pk, 2)
    )
    
    return df


def build_a8_cluster_dict(df, save_path="./model/a8_cluster_dict.json"):
    if 'A8-1' not in df.columns or 'A8-2' not in df.columns or 'Label' not in df.columns:
        print("⚠️ A8-1, A8-2, or Label not found")
        return {}
    
    df_temp = df.copy()
    df_temp['A8_combination'] = df_temp['A8-1'].astype(str) + '_' + df_temp['A8-2'].astype(str)
    
    combination_stats = df_temp.groupby('A8_combination').agg({
        'Label': ['count', 'sum', 'mean']
    }).reset_index()
    combination_stats.columns = ['A8_combination', 'count', 'positive_count', 'positive_rate']
    combination_stats = combination_stats.sort_values('positive_rate', ascending=True).reset_index(drop=True)
    
    q1 = combination_stats['positive_rate'].quantile(0.25)
    q2 = combination_stats['positive_rate'].quantile(0.5)
    q3 = combination_stats['positive_rate'].quantile(0.75)
    
    combination_stats['cluster'] = pd.cut(
        combination_stats['positive_rate'], 
        bins=[-np.inf, q1, q2, q3, np.inf],
        labels=[1, 2, 3, 4]
    )
    combination_stats['cluster'] = combination_stats['cluster'].astype(int)
    
    a8_cluster_dict = {}
    for _, row in combination_stats.iterrows():
        a8_cluster_dict[row['A8_combination']] = int(row['cluster'])
    
    cluster_summary = combination_stats.groupby('cluster').agg({
        'positive_rate': ['count', 'min', 'max', 'mean']
    }).reset_index()
    cluster_summary.columns = ['cluster', 'count', 'min_rate', 'max_rate', 'mean_rate']
    
    print(f"\n📊 A8 Cluster: {len(a8_cluster_dict):,} combinations")
    for _, row in cluster_summary.iterrows():
        print(f"   Cluster {int(row['cluster'])}: {int(row['count'])}개 (사고율 {row['min_rate']:.4f}~{row['max_rate']:.4f})")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(a8_cluster_dict, f, ensure_ascii=False, indent=2)
        print(f"✅ A8 cluster dict saved")
    
    return a8_cluster_dict


def load_a8_cluster_dict(load_path="./model/a8_cluster_dict.json"):
    if not os.path.exists(load_path):
        print(f"⚠️ A8 cluster dict not found")
        return {}
    
    with open(load_path, 'r', encoding='utf-8') as f:
        a8_cluster_dict = json.load(f)
    
    print(f"✅ A8 cluster dict loaded ({len(a8_cluster_dict):,} combinations)")
    return a8_cluster_dict


def add_a8_cluster_feature(df, a8_cluster_dict):
    df = df.copy()
    
    if 'A8-1' not in df.columns or 'A8-2' not in df.columns:
        df['A8_cluster'] = 0
        return df
    
    df['A8_combination_temp'] = df['A8-1'].astype(str) + '_' + df['A8-2'].astype(str)
    df['A8_cluster'] = df['A8_combination_temp'].map(a8_cluster_dict).fillna(0).astype(int)
    df = df.drop('A8_combination_temp', axis=1)
    
    return df

