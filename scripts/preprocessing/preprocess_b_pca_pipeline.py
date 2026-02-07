"""
Optimal B Test Preprocessing Pipeline - FULL VERSION (B1-B10)
Combines B1-B10 test features with PCA dimensionality reduction

Feature Engineering:
- B1/B2: 23 features → 3 PCA components (62.77% variance)
- B3: 18 features → 4 PCA components (83.23% variance)
- B4: 20 features → 5 PCA components
- B5: 43 features → 7 PCA components (90.93% variance)
- B6/B7: 6 features → 1 PCA component (64.19% variance)
- B8: 37 features → 8 PCA components (99.31% variance)
- B9/B10: 75 features → 23 PCA components (category-wise)
  * Original: 57 → 13 PCs (6 categories)
  * Interaction: 18 → 10 PCs (6 categories)

Total: ~222 features → 51 PCA components (77.0% compression)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import norm
import ast
import warnings
warnings.filterwarnings('ignore')

def parse_list(list_str):
    """Parse comma-separated string or Python list string to integer list"""
    if pd.isna(list_str):
        return []
    if isinstance(list_str, str):
        try:
            return ast.literal_eval(list_str)
        except:
            return [int(x.strip()) for x in str(list_str).split(',') if x.strip()]
    return []


def convert_age_to_numeric(age_val):
    """
    Convert Age feature to numeric
    - '60a' -> 60
    - '60b' -> 65
    - '60' -> 60
    """
    if pd.isna(age_val):
        return np.nan
    
    s = str(age_val).strip()
    if not s:
        return np.nan
    
    try:
        # If ends with 'a', return base number
        if s[-1] == 'a':
            return int(s[:-1])
        # If ends with 'b', return base number + 5
        elif s[-1] == 'b':
            return int(s[:-1]) + 5
        # Otherwise, just convert to int
        else:
            return int(s)
    except:
        return np.nan


def add_past_history_features(df, dict_A=None, dict_B_past_label=None):
    """
    Add past history features (A and B) - OPTIMIZED VERSION
    벡터화 연산으로 빠르게 처리합니다 (4.py 방식)
    
    Parameters:
    -----------
    df : DataFrame
        B test data with PrimaryKey and Label
    dict_A : dict, optional
        Dictionary mapping PrimaryKey to A test Label (max)
    dict_B_past_label : dict, optional
        Dictionary mapping PrimaryKey to B past accident label
        (from training data for test inference)
        
    Returns:
    --------
    df : DataFrame with added columns:
        - past_A: A test Label for this PrimaryKey (if available)
        - past_A_history: Categorical ['A 기록 없음', 'A Label 0', 'A Label 1']
        - Past_Label: B test Label from other rows with same PrimaryKey
    """
    df = df.copy()
    
    # A 과거 이력 추가 (벡터화됨 - 이미 최적화됨)
    if dict_A is not None:
        df['past_A'] = df['PrimaryKey'].map(dict_A)
        
        # A 이력을 3개 그룹으로 분류
        choices_A = ['A 기록 없음', 'A Label 0', 'A Label 1']
        conditions_A = [
            df['past_A'].isna(),
            df['past_A'] == 0,
            df['past_A'] == 1
        ]
        df['past_A_history'] = np.select(conditions_A, choices_A, default='Error')
    else:
        df['past_A'] = np.nan
        df['past_A_history'] = 'A 기록 없음'
    
    # B 과거 이력 추가 (벡터화 연산으로 최적화)
    if 'Label' in df.columns:
        # Training mode: Calculate from same dataframe (본인 제외)
        # PrimaryKey별 Label 통계 계산 (한 번에 처리)
        pk_stats = df.groupby('PrimaryKey', sort=False)['Label'].agg(['count', 'sum'])
        pk_stats.columns = ['pk_count', 'pk_sum_label']
        
        # Index를 기준으로 join (merge보다 빠름)
        df = df.join(pk_stats, on='PrimaryKey')
        
        # 본인 제외 계산 (벡터 연산)
        other_count = df['pk_count'] - 1
        other_sum = df['pk_sum_label'] - df['Label']
        
        # Past_Label 계산 (벡터 연산)
        df['Past_Label'] = np.where(
            other_count == 0,  # 본인만 있는 경우
            np.nan,
            np.where(other_sum > 0, 1.0, 0.0)
        )
        
        # 임시 컬럼 제거
        df.drop(columns=['pk_count', 'pk_sum_label'], inplace=True)
    else:
        # Test mode: Use pre-calculated dictionary from training data
        if dict_B_past_label is not None:
            df['Past_Label'] = df['PrimaryKey'].map(dict_B_past_label)
        else:
            df['Past_Label'] = np.nan
    
    return df


# ============================================================================
# B1 Feature Extraction Functions
# ============================================================================

def calculate_b1_1_features(b1_1_list):
    """Extract features from B1-1 (central vision test responses)"""
    features = {}
    if not b1_1_list or len(b1_1_list) == 0:
        return {
            'b1_1_correct_count': 0,
            'b1_1_early_incorrect': 0,
            'b1_1_late_incorrect': 0,
            'b1_1_concentration_decline': 0
        }
    
    b1_1_arr = np.array(b1_1_list)
    features['b1_1_correct_count'] = np.sum(b1_1_arr == 1)
    
    # Early vs late errors
    half = len(b1_1_arr) // 2
    features['b1_1_early_incorrect'] = np.sum(b1_1_arr[:half] == 2)
    features['b1_1_late_incorrect'] = np.sum(b1_1_arr[half:] == 2)
    features['b1_1_concentration_decline'] = features['b1_1_late_incorrect'] - features['b1_1_early_incorrect']
    
    return features


def calculate_b1_2_features(b1_2_list):
    """Extract features from B1-2 (reaction times)"""
    features = {}
    if not b1_2_list or len(b1_2_list) == 0:
        return {
            'b1_2_avg_time': 0,
            'b1_2_std_time': 0,
            'b1_2_time_trend': 0,
            'b1_2_cv_time': 0
        }
    
    b1_2_arr = np.array(b1_2_list)
    features['b1_2_avg_time'] = np.mean(b1_2_arr)
    features['b1_2_std_time'] = np.std(b1_2_arr)
    
    # Time trend
    half = len(b1_2_arr) // 2
    first_half = np.mean(b1_2_arr[:half]) if half > 0 else 0
    second_half = np.mean(b1_2_arr[half:]) if len(b1_2_arr) > half else 0
    features['b1_2_time_trend'] = second_half - first_half
    
    # Coefficient of variation
    features['b1_2_cv_time'] = features['b1_2_std_time'] / features['b1_2_avg_time'] if features['b1_2_avg_time'] > 0 else 0
    
    return features


def calculate_b1_3_features(b1_3_list):
    """Extract features from B1-3 (change detection)"""
    features = {}
    if not b1_3_list or len(b1_3_list) == 0:
        return {
            'b1_3_correct_count': 0,
            'b1_3_max_streak': 0
        }
    
    b1_3_arr = np.array(b1_3_list)
    features['b1_3_correct_count'] = np.sum(b1_3_arr == 1)
    
    # Max correct streak
    streaks = []
    current = 0
    for val in b1_3_arr:
        if val == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    
    features['b1_3_max_streak'] = max(streaks) if streaks else 0
    
    return features


# ============================================================================
# B2 Feature Extraction Functions (mirror B1 structure)
# ============================================================================

def calculate_b2_1_features(b2_1_list):
    """Extract features from B2-1 (peripheral vision)"""
    features = {}
    if not b2_1_list or len(b2_1_list) == 0:
        return {
            'b2_1_correct_count': 0,
            'b2_1_early_incorrect': 0,
            'b2_1_late_incorrect': 0,
            'b2_1_concentration_decline': 0
        }
    
    b2_1_arr = np.array(b2_1_list)
    features['b2_1_correct_count'] = np.sum(b2_1_arr == 1)
    
    half = len(b2_1_arr) // 2
    features['b2_1_early_incorrect'] = np.sum(b2_1_arr[:half] == 2)
    features['b2_1_late_incorrect'] = np.sum(b2_1_arr[half:] == 2)
    features['b2_1_concentration_decline'] = features['b2_1_late_incorrect'] - features['b2_1_early_incorrect']
    
    return features


def calculate_b2_2_features(b2_2_list):
    """Extract features from B2-2 (reaction times)"""
    features = {}
    if not b2_2_list or len(b2_2_list) == 0:
        return {
            'b2_2_avg_time': 0,
            'b2_2_std_time': 0,
            'b2_2_time_trend': 0,
            'b2_2_cv_time': 0
        }
    
    b2_2_arr = np.array(b2_2_list)
    features['b2_2_avg_time'] = np.mean(b2_2_arr)
    features['b2_2_std_time'] = np.std(b2_2_arr)
    
    half = len(b2_2_arr) // 2
    first_half = np.mean(b2_2_arr[:half]) if half > 0 else 0
    second_half = np.mean(b2_2_arr[half:]) if len(b2_2_arr) > half else 0
    features['b2_2_time_trend'] = second_half - first_half
    
    features['b2_2_cv_time'] = features['b2_2_std_time'] / features['b2_2_avg_time'] if features['b2_2_avg_time'] > 0 else 0
    
    return features


def calculate_b2_3_features(b2_3_list):
    """Extract features from B2-3 (change detection)"""
    features = {}
    if not b2_3_list or len(b2_3_list) == 0:
        return {
            'b2_3_correct_count': 0,
            'b2_3_max_streak': 0
        }
    
    b2_3_arr = np.array(b2_3_list)
    features['b2_3_correct_count'] = np.sum(b2_3_arr == 1)
    
    streaks = []
    current = 0
    for val in b2_3_arr:
        if val == 1:
            current += 1
        else:
            if current > 0:
                streaks.append(current)
            current = 0
    if current > 0:
        streaks.append(current)
    
    features['b2_3_max_streak'] = max(streaks) if streaks else 0
    
    return features


def calculate_composite_features(b1_features, b2_features):
    """Calculate composite features comparing B1 and B2"""
    features = {}
    
    # Accuracy difference
    b1_accuracy = b1_features.get('b1_1_correct_count', 0) + b1_features.get('b1_3_correct_count', 0)
    b2_accuracy = b2_features.get('b2_1_correct_count', 0) + b2_features.get('b2_3_correct_count', 0)
    features['central_peripheral_diff'] = b1_accuracy - b2_accuracy
    
    # Reaction time difference
    b1_time = b1_features.get('b1_2_avg_time', 0)
    b2_time = b2_features.get('b2_2_avg_time', 0)
    features['reaction_time_diff'] = b1_time - b2_time
    
    # Concentration decline difference
    b1_decline = b1_features.get('b1_1_concentration_decline', 0)
    b2_decline = b2_features.get('b2_1_concentration_decline', 0)
    features['concentration_decline_diff'] = b1_decline - b2_decline
    
    return features


# ============================================================================
# B3 Feature Extraction Functions
# ============================================================================

def calculate_b3_features(b3_1_list, b3_2_list):
    """Extract features from B3 (traffic light reaction test)"""
    features = {}
    
    b3_1_arr = np.array(b3_1_list) if b3_1_list and len(b3_1_list) > 0 else np.array([])
    b3_2_arr = np.array(b3_2_list) if b3_2_list and len(b3_2_list) > 0 else np.array([])
    
    # Accuracy features
    if len(b3_1_arr) > 0:
        features['b3_correct_count'] = np.sum(b3_1_arr == 1)
        features['b3_incorrect_count'] = np.sum(b3_1_arr == 2)
    else:
        features['b3_correct_count'] = 0
        features['b3_incorrect_count'] = 0
    
    # Reaction time features
    if len(b3_2_arr) > 0:
        features['b3_avg_time'] = np.mean(b3_2_arr)
        features['b3_std_time'] = np.std(b3_2_arr)
        features['b3_min_time'] = np.min(b3_2_arr)
        features['b3_max_time'] = np.max(b3_2_arr)
        features['b3_median_time'] = np.median(b3_2_arr)
        features['b3_cv_time'] = features['b3_std_time'] / features['b3_avg_time'] if features['b3_avg_time'] > 0 else 0
    else:
        for key in ['b3_avg_time', 'b3_std_time', 'b3_min_time', 'b3_max_time', 'b3_median_time', 'b3_cv_time']:
            features[key] = 0
    
    # Combined accuracy-speed features
    if len(b3_1_arr) > 0 and len(b3_2_arr) > 0 and len(b3_1_arr) == len(b3_2_arr):
        correct_times = b3_2_arr[b3_1_arr == 1]
        incorrect_times = b3_2_arr[b3_1_arr == 2]
        
        features['b3_correct_avg_time'] = np.mean(correct_times) if len(correct_times) > 0 else 0
        features['b3_incorrect_avg_time'] = np.mean(incorrect_times) if len(incorrect_times) > 0 else 0
        
        features['b3_efficiency'] = features['b3_correct_count'] / features['b3_avg_time'] if features['b3_avg_time'] > 0 else 0
    else:
        features['b3_correct_avg_time'] = 0
        features['b3_incorrect_avg_time'] = 0
        features['b3_efficiency'] = 0
    
    # Streak features
    if len(b3_1_arr) > 0:
        correct_streaks = []
        current_correct = 0
        
        for val in b3_1_arr:
            if val == 1:
                current_correct += 1
            else:
                if current_correct > 0:
                    correct_streaks.append(current_correct)
                    current_correct = 0
        
        if current_correct > 0:
            correct_streaks.append(current_correct)
        
        features['b3_max_correct_streak'] = max(correct_streaks) if correct_streaks else 0
        features['b3_avg_correct_streak'] = np.mean(correct_streaks) if correct_streaks else 0
    else:
        features['b3_max_correct_streak'] = 0
        features['b3_avg_correct_streak'] = 0
    
    # Position features
    if len(b3_1_arr) >= 10:
        features['b3_first10_correct'] = np.sum(b3_1_arr[:10] == 1)
        features['b3_last10_correct'] = np.sum(b3_1_arr[-10:] == 1)
        features['b3_early_late_diff'] = features['b3_first10_correct'] - features['b3_last10_correct']
    else:
        features['b3_first10_correct'] = 0
        features['b3_last10_correct'] = 0
        features['b3_early_late_diff'] = 0
    
    # Time trend
    if len(b3_2_arr) >= 10:
        first_half = np.mean(b3_2_arr[:len(b3_2_arr)//2])
        second_half = np.mean(b3_2_arr[len(b3_2_arr)//2:])
        features['b3_time_trend'] = second_half - first_half
    else:
        features['b3_time_trend'] = 0
    
    return features


# ============================================================================
# B4 Feature Extraction Functions
# ============================================================================

def calculate_b4_features(b4_1_list, b4_2_list):
    """Extract features from B4 (Stroop test - cognitive interference)"""
    features = {}
    
    b4_1_arr = np.array(b4_1_list) if b4_1_list and len(b4_1_list) > 0 else np.array([])
    b4_2_arr = np.array(b4_2_list) if b4_2_list and len(b4_2_list) > 0 else np.array([])
    
    total_trials = len(b4_1_arr)
    
    # Basic accuracy
    if total_trials > 0:
        features['b4_correct_count'] = np.sum(b4_1_arr == 1)
        features['b4_incorrect_count'] = np.sum(b4_1_arr == 2)
    else:
        features['b4_correct_count'] = 0
        features['b4_incorrect_count'] = 0
    
    # Congruent vs Incongruent (first 30 congruent, next 30 incongruent)
    if total_trials >= 30:
        features['b4_congruent_correct'] = np.sum(b4_1_arr[:30] == 1)
        
        if total_trials >= 60:
            features['b4_incongruent_correct'] = np.sum(b4_1_arr[30:60] == 1)
            features['b4_stroop_effect'] = features['b4_congruent_correct'] - features['b4_incongruent_correct']
        else:
            features['b4_incongruent_correct'] = 0
            features['b4_stroop_effect'] = 0
    else:
        features['b4_congruent_correct'] = 0
        features['b4_incongruent_correct'] = 0
        features['b4_stroop_effect'] = 0
    
    # Position features
    if total_trials >= 10:
        features['b4_first10_correct'] = np.sum(b4_1_arr[:10] == 1)
        features['b4_last10_correct'] = np.sum(b4_1_arr[-10:] == 1)
    else:
        features['b4_first10_correct'] = 0
        features['b4_last10_correct'] = 0
    
    # Concentration decline
    if total_trials > 0:
        half = total_trials // 2
        first_half_correct = np.sum(b4_1_arr[:half] == 1)
        second_half_correct = np.sum(b4_1_arr[half:] == 1)
        features['b4_performance_change'] = second_half_correct - first_half_correct
    else:
        features['b4_performance_change'] = 0
    
    # Streak features
    if total_trials > 0:
        correct_streaks = []
        current_correct = 0
        
        for val in b4_1_arr:
            if val == 1:
                current_correct += 1
            else:
                if current_correct > 0:
                    correct_streaks.append(current_correct)
                    current_correct = 0
        
        if current_correct > 0:
            correct_streaks.append(current_correct)
        
        features['b4_max_correct_streak'] = max(correct_streaks) if correct_streaks else 0
        features['b4_avg_correct_streak'] = np.mean(correct_streaks) if correct_streaks else 0
    else:
        features['b4_max_correct_streak'] = 0
        features['b4_avg_correct_streak'] = 0
    
    # Reaction time features
    if len(b4_2_arr) > 0:
        features['b4_avg_reaction_time'] = np.mean(b4_2_arr)
        features['b4_std_reaction_time'] = np.std(b4_2_arr)
        features['b4_min_reaction_time'] = np.min(b4_2_arr)
        features['b4_median_reaction_time'] = np.median(b4_2_arr)
        features['b4_cv_reaction_time'] = features['b4_std_reaction_time'] / features['b4_avg_reaction_time'] if features['b4_avg_reaction_time'] > 0 else 0
        
        # Time trend
        if len(b4_2_arr) >= 10:
            first_half_time = np.mean(b4_2_arr[:len(b4_2_arr)//2])
            second_half_time = np.mean(b4_2_arr[len(b4_2_arr)//2:])
            features['b4_time_trend'] = second_half_time - first_half_time
        else:
            features['b4_time_trend'] = 0
        
        # Congruent vs Incongruent times
        if total_trials >= 30:
            congruent_time = np.mean(b4_2_arr[:30])
            features['b4_congruent_avg_time'] = congruent_time
            
            if total_trials >= 60:
                incongruent_time = np.mean(b4_2_arr[30:60])
                features['b4_incongruent_avg_time'] = incongruent_time
                features['b4_stroop_time_effect'] = incongruent_time - congruent_time
            else:
                features['b4_incongruent_avg_time'] = 0
                features['b4_stroop_time_effect'] = 0
        else:
            features['b4_congruent_avg_time'] = 0
            features['b4_incongruent_avg_time'] = 0
            features['b4_stroop_time_effect'] = 0
    else:
        for key in ['b4_avg_reaction_time', 'b4_std_reaction_time', 'b4_min_reaction_time',
                    'b4_median_reaction_time', 'b4_cv_reaction_time', 'b4_time_trend',
                    'b4_congruent_avg_time', 'b4_incongruent_avg_time', 'b4_stroop_time_effect']:
            features[key] = 0
    
    # Combined features
    if len(b4_1_arr) > 0 and len(b4_2_arr) > 0 and len(b4_1_arr) == len(b4_2_arr):
        correct_times = b4_2_arr[b4_1_arr == 1]
        
        if len(correct_times) > 0:
            features['b4_correct_avg_time'] = np.mean(correct_times)
        else:
            features['b4_correct_avg_time'] = 0
        
        features['b4_efficiency'] = features['b4_correct_count'] / features['b4_avg_reaction_time'] if features['b4_avg_reaction_time'] > 0 else 0
    else:
        features['b4_correct_avg_time'] = 0
        features['b4_efficiency'] = 0
    
    return features


# ============================================================================
# B5 Feature Extraction Functions
# ============================================================================

def calculate_b5_features(b5_1_list, b5_2_list):
    """Extract features from B5 (spatial judgment test)
    
    B5 test: Hexagonal road pathfinding, 20 trials
    B5-1: Response correctness (1=correct, 2=incorrect)
    B5-2: Reaction times
    
    Returns 43 significant features (based on correlation analysis)
    """
    features = {}
    
    b5_1_arr = np.array(b5_1_list) if b5_1_list and len(b5_1_list) > 0 else np.array([])
    b5_2_arr = np.array(b5_2_list) if b5_2_list and len(b5_2_list) > 0 else np.array([])
    
    total_trials = len(b5_1_arr)
    
    # Basic accuracy features
    if total_trials > 0:
        features['b5_correct_count'] = np.sum(b5_1_arr == 1)
        features['b5_incorrect_count'] = np.sum(b5_1_arr == 2)
    else:
        features['b5_correct_count'] = 0
        features['b5_incorrect_count'] = 0
    
    # Position-based features
    if total_trials >= 5:
        features['b5_first5_correct'] = np.sum(b5_1_arr[:5] == 1)
        features['b5_last5_correct'] = np.sum(b5_1_arr[-5:] == 1)
    else:
        features['b5_first5_correct'] = 0
        features['b5_last5_correct'] = 0
    
    if total_trials >= 10:
        features['b5_first10_correct'] = np.sum(b5_1_arr[:10] == 1)
        features['b5_last10_correct'] = np.sum(b5_1_arr[-10:] == 1)
        features['b5_early_late_diff'] = features['b5_first10_correct'] - features['b5_last10_correct']
    else:
        features['b5_first10_correct'] = 0
        features['b5_last10_correct'] = 0
        features['b5_early_late_diff'] = 0
    
    # Streak features (consistency patterns) - MOST SIGNIFICANT
    if total_trials > 0:
        correct_streaks = []
        incorrect_streaks = []
        current_correct = 0
        current_incorrect = 0
        
        for val in b5_1_arr:
            if val == 1:
                if current_incorrect > 0:
                    incorrect_streaks.append(current_incorrect)
                    current_incorrect = 0
                current_correct += 1
            else:
                if current_correct > 0:
                    correct_streaks.append(current_correct)
                    current_correct = 0
                current_incorrect += 1
        
        if current_correct > 0:
            correct_streaks.append(current_correct)
        if current_incorrect > 0:
            incorrect_streaks.append(current_incorrect)
        
        features['b5_max_correct_streak'] = max(correct_streaks) if correct_streaks else 0
        features['b5_avg_correct_streak'] = np.mean(correct_streaks) if correct_streaks else 0
        features['b5_num_correct_streaks'] = len(correct_streaks)
        
        features['b5_max_incorrect_streak'] = max(incorrect_streaks) if incorrect_streaks else 0
        features['b5_num_incorrect_streaks'] = len(incorrect_streaks)
        
        features['b5_total_streaks'] = len(correct_streaks) + len(incorrect_streaks)
    else:
        for key in ['b5_max_correct_streak', 'b5_avg_correct_streak', 'b5_num_correct_streaks',
                    'b5_max_incorrect_streak', 'b5_num_incorrect_streaks', 'b5_total_streaks']:
            features[key] = 0
    
    # Transition and consistency features - HIGHLY SIGNIFICANT
    if total_trials > 1:
        transitions = 0
        for i in range(len(b5_1_arr) - 1):
            if b5_1_arr[i] != b5_1_arr[i+1]:
                transitions += 1
        features['b5_transitions'] = transitions
        features['b5_consistency'] = 1 - (transitions / (total_trials - 1))
    else:
        features['b5_transitions'] = 0
        features['b5_consistency'] = 0
    
    # Reaction time features
    if len(b5_2_arr) > 0:
        features['b5_avg_time'] = np.mean(b5_2_arr)
        features['b5_median_time'] = np.median(b5_2_arr)
        features['b5_std_time'] = np.std(b5_2_arr)
        features['b5_min_time'] = np.min(b5_2_arr)
        features['b5_max_time'] = np.max(b5_2_arr)
        features['b5_cv_time'] = features['b5_std_time'] / features['b5_avg_time'] if features['b5_avg_time'] > 0 else 0
        
        # Percentiles
        features['b5_time_p25'] = np.percentile(b5_2_arr, 25)
        features['b5_time_p75'] = np.percentile(b5_2_arr, 75)
        features['b5_time_iqr'] = features['b5_time_p75'] - features['b5_time_p25']
        
        # Time trend
        if len(b5_2_arr) >= 10:
            first_half = np.mean(b5_2_arr[:len(b5_2_arr)//2])
            second_half = np.mean(b5_2_arr[len(b5_2_arr)//2:])
            features['b5_time_trend'] = second_half - first_half
        else:
            features['b5_time_trend'] = 0
    else:
        for key in ['b5_avg_time', 'b5_median_time', 'b5_std_time', 'b5_min_time', 'b5_max_time',
                    'b5_cv_time', 'b5_time_p25', 'b5_time_p75', 'b5_time_iqr', 'b5_time_trend']:
            features[key] = 0
    
    # Combined accuracy-speed features
    if len(b5_1_arr) > 0 and len(b5_2_arr) > 0 and len(b5_1_arr) == len(b5_2_arr):
        correct_times = b5_2_arr[b5_1_arr == 1]
        incorrect_times = b5_2_arr[b5_1_arr == 2]
        
        features['b5_correct_avg_time'] = np.mean(correct_times) if len(correct_times) > 0 else 0
        features['b5_incorrect_avg_time'] = np.mean(incorrect_times) if len(incorrect_times) > 0 else 0
        features['b5_time_accuracy_diff'] = features['b5_incorrect_avg_time'] - features['b5_correct_avg_time']
        
        features['b5_efficiency'] = features['b5_correct_count'] / features['b5_avg_time'] if features['b5_avg_time'] > 0 else 0
        features['b5_spatial_score'] = features['b5_correct_count'] * features['b5_consistency']
    else:
        features['b5_correct_avg_time'] = 0
        features['b5_incorrect_avg_time'] = 0
        features['b5_time_accuracy_diff'] = 0
        features['b5_efficiency'] = 0
        features['b5_spatial_score'] = 0
    
    # Error pattern features
    if total_trials >= 15:
        early_errors = np.sum(b5_1_arr[:5] == 2)
        middle_errors = np.sum(b5_1_arr[5:15] == 2)
        late_errors = np.sum(b5_1_arr[15:] == 2) if total_trials > 15 else 0
        
        features['b5_early_error_rate'] = early_errors / 5
        features['b5_middle_error_rate'] = middle_errors / 10
        features['b5_late_error_rate'] = late_errors / max(1, total_trials - 15)
    else:
        features['b5_early_error_rate'] = 0
        features['b5_middle_error_rate'] = 0
        features['b5_late_error_rate'] = 0
    
    return features


# ============================================================================
# B6/B7 Feature Extraction Functions
# ============================================================================

def calculate_b6_b7_features(b6_list, b7_list):
    """Extract 6 significant features from B6 and B7 (traffic sign recognition)
    
    B6: Basic traffic sign recognition (15 trials)
    B7: Advanced traffic sign recognition (15 trials)
    
    Returns only the 6 most significant features for PC1 generation
    """
    features = {}
    
    b6_arr = np.array(b6_list) if b6_list and len(b6_list) > 0 else np.array([])
    b7_arr = np.array(b7_list) if b7_list and len(b7_list) > 0 else np.array([])
    
    b6_total = len(b6_arr)
    b7_total = len(b7_arr)
    
    # B6 features
    if b6_total > 0:
        b6_correct_count = np.sum(b6_arr == 1)
        b6_correct_rate = b6_correct_count / b6_total
        
        # B6 consistency
        if b6_total > 1:
            transitions = 0
            for i in range(len(b6_arr) - 1):
                if b6_arr[i] != b6_arr[i+1]:
                    transitions += 1
            b6_consistency = 1 - (transitions / (b6_total - 1))
        else:
            b6_consistency = 0
    else:
        b6_correct_rate = 0
        b6_consistency = 0
    
    # B7 features
    if b7_total > 0:
        b7_correct_count = np.sum(b7_arr == 1)
        b7_correct_rate = b7_correct_count / b7_total
        
        # B7 consistency
        if b7_total > 1:
            transitions = 0
            for i in range(len(b7_arr) - 1):
                if b7_arr[i] != b7_arr[i+1]:
                    transitions += 1
            b7_consistency = 1 - (transitions / (b7_total - 1))
        else:
            b7_consistency = 0
        
        # B7 position features
        features['b7_correct_count'] = b7_correct_count
        features['b7_first5_correct'] = np.sum(b7_arr[:5] == 1) if b7_total >= 5 else 0
        features['b7_last5_correct'] = np.sum(b7_arr[-5:] == 1) if b7_total >= 5 else 0
    else:
        b7_correct_rate = 0
        b7_consistency = 0
        features['b7_correct_count'] = 0
        features['b7_first5_correct'] = 0
        features['b7_last5_correct'] = 0
    
    # Composite features (comparison between B6 and B7)
    features['b6_b7_accuracy_diff'] = b6_correct_rate - b7_correct_rate
    features['b6_b7_consistency_diff'] = b6_consistency - b7_consistency
    
    # Advanced visual ability (B7/B6 ratio)
    if b6_correct_rate > 0:
        features['advanced_visual_ability'] = b7_correct_rate / b6_correct_rate
    else:
        features['advanced_visual_ability'] = 0
    
    return features


def calculate_b8_features(b8_list):
    """
    Extract 37 significant features from B8 test (Lane Keeping Test)
    
    B8: 12 trials of lane position judgment (1=correct, 2=incorrect)
    Returns only the 37 statistically significant features (p<0.05)
    """
    features = {}
    
    # Convert to numpy array
    b8_arr = np.array(b8_list) if b8_list and len(b8_list) > 0 else np.array([])
    
    if len(b8_arr) == 0:
        # Return zeros for all 37 significant features
        feature_names = [
            # Basic accuracy
            'b8_correct_count', 'b8_correct_rate', 'b8_incorrect_count', 
            'b8_incorrect_rate', 'b8_accuracy_score',
            # Position-based (significant ones)
            'b8_first_third_correct', 'b8_middle_third_correct', 'b8_last_third_correct',
            'b8_first_half_correct', 'b8_last_half_correct',
            'b8_first_quarter_correct', 'b8_last_quarter_correct',
            'b8_early_accuracy', 'b8_late_accuracy', 'b8_middle_accuracy',
            'b8_learning_effect', 'b8_fatigue_effect',
            # Streaks
            'b8_max_correct_streak', 'b8_max_incorrect_streak',
            'b8_avg_correct_streak', 'b8_avg_incorrect_streak',
            'b8_correct_streak_ratio', 'b8_streak_consistency',
            'b8_alternation_rate', 'b8_pattern_stability',
            # Consistency metrics
            'b8_response_consistency', 'b8_performance_variance',
            'b8_consistency_score', 'b8_error_clustering',
            'b8_correct_clustering', 'b8_response_stability', 'b8_rolling_std',
            # Error analysis
            'b8_early_error_rate', 'b8_late_error_rate',
            'b8_error_recovery_rate', 'b8_consecutive_errors',
            # Advanced metrics
            'b8_lane_judgment_ability', 'b8_spatial_consistency'
        ]
        for name in feature_names:
            features[name] = 0.0
        return features
    
    # Correct answers are 1, incorrect are 2
    correct = (b8_arr == 1).astype(int)
    incorrect = (b8_arr == 2).astype(int)
    total = len(b8_arr)
    
    # === Basic Accuracy ===
    features['b8_correct_count'] = np.sum(correct)
    features['b8_correct_rate'] = np.mean(correct)
    features['b8_incorrect_count'] = np.sum(incorrect)
    features['b8_incorrect_rate'] = np.mean(incorrect)
    features['b8_accuracy_score'] = features['b8_correct_rate'] * 100
    
    # === Position-based Accuracy ===
    third = total // 3
    features['b8_first_third_correct'] = np.mean(correct[:third]) if third > 0 else 0
    features['b8_middle_third_correct'] = np.mean(correct[third:2*third]) if third > 0 else 0
    features['b8_last_third_correct'] = np.mean(correct[2*third:]) if total - 2*third > 0 else 0
    
    half = total // 2
    features['b8_first_half_correct'] = np.mean(correct[:half])
    features['b8_last_half_correct'] = np.mean(correct[half:])
    
    quarter = total // 4
    features['b8_first_quarter_correct'] = np.mean(correct[:quarter]) if quarter > 0 else 0
    features['b8_last_quarter_correct'] = np.mean(correct[-quarter:]) if quarter > 0 else 0
    
    features['b8_early_accuracy'] = np.mean(correct[:4]) if total >= 4 else features['b8_correct_rate']
    features['b8_late_accuracy'] = np.mean(correct[-4:]) if total >= 4 else features['b8_correct_rate']
    features['b8_middle_accuracy'] = np.mean(correct[4:8]) if total >= 8 else features['b8_correct_rate']
    
    features['b8_learning_effect'] = features['b8_last_half_correct'] - features['b8_first_half_correct']
    features['b8_fatigue_effect'] = features['b8_first_half_correct'] - features['b8_last_half_correct']
    
    # === Streaks ===
    def get_streaks(arr):
        if len(arr) == 0:
            return [0]
        streaks = []
        current_streak = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i-1] and arr[i] == 1:
                current_streak += 1
            else:
                if current_streak > 0 and arr[i-1] == 1:
                    streaks.append(current_streak)
                current_streak = 1 if arr[i] == 1 else 0
        if current_streak > 0 and arr[-1] == 1:
            streaks.append(current_streak)
        return streaks if streaks else [0]
    
    def get_incorrect_streaks(arr):
        if len(arr) == 0:
            return [0]
        streaks = []
        current_streak = 1
        for i in range(1, len(arr)):
            if arr[i] == arr[i-1] and arr[i] == 0:
                current_streak += 1
            else:
                if current_streak > 0 and arr[i-1] == 0:
                    streaks.append(current_streak)
                current_streak = 1 if arr[i] == 0 else 0
        if current_streak > 0 and arr[-1] == 0:
            streaks.append(current_streak)
        return streaks if streaks else [0]
    
    correct_streaks = get_streaks(correct)
    incorrect_streaks = get_incorrect_streaks(correct)
    
    features['b8_max_correct_streak'] = max(correct_streaks)
    features['b8_max_incorrect_streak'] = max(incorrect_streaks)
    features['b8_avg_correct_streak'] = np.mean(correct_streaks)
    features['b8_avg_incorrect_streak'] = np.mean(incorrect_streaks)
    features['b8_correct_streak_ratio'] = features['b8_max_correct_streak'] / total
    features['b8_streak_consistency'] = 1 - (np.std(correct_streaks) / (np.mean(correct_streaks) + 1e-6))
    
    alternations = np.sum(np.abs(np.diff(correct)))
    features['b8_alternation_rate'] = alternations / (total - 1) if total > 1 else 0
    features['b8_pattern_stability'] = 1 - features['b8_alternation_rate']
    
    # === Consistency Metrics ===
    if total >= 3:
        rolling_correct = [np.mean(correct[i:i+3]) for i in range(total-2)]
        features['b8_response_consistency'] = 1 - np.std(rolling_correct)
        features['b8_performance_variance'] = np.var(rolling_correct)
        features['b8_rolling_std'] = np.std(rolling_correct)
    else:
        features['b8_response_consistency'] = 1.0
        features['b8_performance_variance'] = 0.0
        features['b8_rolling_std'] = 0.0
    
    features['b8_consistency_score'] = features['b8_correct_rate'] * features['b8_response_consistency']
    
    # Error clustering
    if np.sum(incorrect) > 0:
        error_positions = np.where(incorrect == 1)[0]
        if len(error_positions) > 1:
            error_gaps = np.diff(error_positions)
            features['b8_error_clustering'] = 1 / (np.mean(error_gaps) + 1)
        else:
            features['b8_error_clustering'] = 0
    else:
        features['b8_error_clustering'] = 0
    
    if np.sum(correct) > 0:
        correct_positions = np.where(correct == 1)[0]
        if len(correct_positions) > 1:
            correct_gaps = np.diff(correct_positions)
            features['b8_correct_clustering'] = 1 / (np.mean(correct_gaps) + 1)
        else:
            features['b8_correct_clustering'] = 0
    else:
        features['b8_correct_clustering'] = 0
    
    features['b8_response_stability'] = 1 / (np.var(correct.astype(float)) + 0.01)
    
    # === Error Analysis ===
    features['b8_early_error_rate'] = 1 - features['b8_early_accuracy']
    features['b8_late_error_rate'] = 1 - features['b8_late_accuracy']
    
    recoveries = 0
    for i in range(1, len(correct)):
        if correct[i-1] == 0 and correct[i] == 1:
            recoveries += 1
    features['b8_error_recovery_rate'] = recoveries / max(np.sum(incorrect), 1)
    features['b8_consecutive_errors'] = features['b8_max_incorrect_streak']
    
    # === Advanced Metrics ===
    features['b8_lane_judgment_ability'] = (
        features['b8_correct_rate'] * 0.5 +
        features['b8_consistency_score'] * 0.3 +
        (1 - features['b8_error_clustering'] / (features['b8_error_clustering'] + 1)) * 0.2
    )
    
    features['b8_spatial_consistency'] = (
        features['b8_correct_rate'] * 0.6 +
        features['b8_pattern_stability'] * 0.4
    )
    
    return features


# ============================================================================
# B9/B10 Feature Extraction Functions
# ============================================================================

def calculate_b9_features(row):
    """Extract all B9 features (dual task assessment)
    
    B9 test: Dual task - Auditory (50 trials) + Visual (32 trials)
    Returns 28 features for modeling
    """
    features = {}
    
    # Raw values
    b9_aud_hit = row.get('B9-1', 0)
    b9_aud_miss = row.get('B9-2', 0)
    b9_aud_fa = row.get('B9-3', 0)
    b9_aud_cr = row.get('B9-4', 0)
    b9_vis_err = row.get('B9-5', 0)
    
    # Basic rates
    features['b9_aud_hit_rate'] = b9_aud_hit / 15
    features['b9_aud_miss_rate'] = b9_aud_miss / 15
    features['b9_aud_fa_rate'] = b9_aud_fa / 35
    features['b9_aud_cr_rate'] = b9_aud_cr / 35
    features['b9_aud_accuracy'] = (b9_aud_hit + b9_aud_cr) / 50
    features['b9_vis_error_rate'] = b9_vis_err / 32
    features['b9_vis_success_rate'] = 1 - features['b9_vis_error_rate']
    
    # Signal detection theory
    hit_rate = np.clip(features['b9_aud_hit_rate'], 0.01, 0.99)
    fa_rate = np.clip(features['b9_aud_fa_rate'], 0.01, 0.99)
    
    features['b9_aud_dprime'] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    features['b9_aud_sensitivity'] = features['b9_aud_dprime']
    features['b9_aud_beta'] = np.exp((norm.ppf(fa_rate)**2 - norm.ppf(hit_rate)**2) / 2)
    features['b9_aud_c'] = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    features['b9_aud_response_bias'] = features['b9_aud_c']
    
    # Precision, Recall, F1
    features['b9_aud_precision'] = b9_aud_hit / (b9_aud_hit + b9_aud_fa) if (b9_aud_hit + b9_aud_fa) > 0 else 0
    features['b9_aud_recall'] = features['b9_aud_hit_rate']
    features['b9_aud_f1_score'] = (2 * features['b9_aud_precision'] * features['b9_aud_recall'] / 
                                    (features['b9_aud_precision'] + features['b9_aud_recall'])) \
                                   if (features['b9_aud_precision'] + features['b9_aud_recall']) > 0 else 0
    
    features['b9_aud_hit_minus_fa'] = features['b9_aud_hit_rate'] - features['b9_aud_fa_rate']
    
    # Dual task performance
    features['b9_dual_task_performance'] = (features['b9_aud_accuracy'] + features['b9_vis_success_rate']) / 2
    features['b9_attention_distribution'] = 1 - np.abs(features['b9_aud_accuracy'] - features['b9_vis_success_rate'])
    
    # Cognitive load
    total_errors = b9_aud_miss + b9_aud_fa + b9_vis_err
    features['b9_cognitive_load_index'] = total_errors / 82
    
    # Multitasking efficiency
    features['b9_multitask_efficiency'] = (b9_aud_hit + (32 - b9_vis_err)) / 47
    
    # Advanced composite features
    features['b9_selective_attention'] = features['b9_aud_dprime']
    features['b9_selective_attention_quality'] = features['b9_aud_dprime']
    features['b9_divided_attention_capacity'] = features['b9_multitask_efficiency']
    features['b9_cognitive_flexibility'] = features['b9_attention_distribution']
    
    return features


def calculate_b10_features(row):
    """Extract all B10 features (triple task assessment)
    
    B10 test: Triple task - Auditory (80) + Visual1 (52) + Visual2 (20)
    Returns 41 features for modeling
    """
    features = {}
    
    # Raw values
    b10_aud_hit = row.get('B10-1', 0)
    b10_aud_miss = row.get('B10-2', 0)
    b10_aud_fa = row.get('B10-3', 0)
    b10_aud_cr = row.get('B10-4', 0)
    b10_vis1_err = row.get('B10-5', 0)
    b10_vis2_correct = row.get('B10-6', 0)
    
    # Basic rates
    features['b10_aud_hit_rate'] = b10_aud_hit / 20
    features['b10_aud_miss_rate'] = b10_aud_miss / 20
    features['b10_aud_fa_rate'] = b10_aud_fa / 60
    features['b10_aud_cr_rate'] = b10_aud_cr / 60
    features['b10_aud_accuracy'] = (b10_aud_hit + b10_aud_cr) / 80
    
    features['b10_vis1_error_rate'] = b10_vis1_err / 52
    features['b10_vis1_success_rate'] = 1 - features['b10_vis1_error_rate']
    features['b10_vis1_err'] = b10_vis1_err
    
    features['b10_vis2_correct_rate'] = b10_vis2_correct / 20
    features['b10_vis2_incorrect_rate'] = 1 - features['b10_vis2_correct_rate']
    
    # Signal detection theory
    hit_rate = np.clip(features['b10_aud_hit_rate'], 0.01, 0.99)
    fa_rate = np.clip(features['b10_aud_fa_rate'], 0.01, 0.99)
    
    features['b10_aud_dprime'] = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    features['b10_aud_sensitivity'] = features['b10_aud_dprime']
    features['b10_aud_beta'] = np.exp((norm.ppf(fa_rate)**2 - norm.ppf(hit_rate)**2) / 2)
    features['b10_aud_c'] = -0.5 * (norm.ppf(hit_rate) + norm.ppf(fa_rate))
    features['b10_aud_response_bias'] = features['b10_aud_c']
    features['b10_aud_roc_auc'] = features['b10_aud_dprime']
    
    # Precision, Recall, F1
    features['b10_aud_precision'] = b10_aud_hit / (b10_aud_hit + b10_aud_fa) if (b10_aud_hit + b10_aud_fa) > 0 else 0
    features['b10_aud_recall'] = features['b10_aud_hit_rate']
    features['b10_aud_f1_score'] = (2 * features['b10_aud_precision'] * features['b10_aud_recall'] / 
                                     (features['b10_aud_precision'] + features['b10_aud_recall'])) \
                                    if (features['b10_aud_precision'] + features['b10_aud_recall']) > 0 else 0
    
    features['b10_aud_hit_minus_fa'] = features['b10_aud_hit_rate'] - features['b10_aud_fa_rate']
    
    # Triple task performance
    avg_perf = (features['b10_aud_accuracy'] + 
                features['b10_vis1_success_rate'] + 
                features['b10_vis2_correct_rate']) / 3
    features['b10_triple_task_performance'] = avg_perf
    
    # Task balance
    perfs = [features['b10_aud_accuracy'], features['b10_vis1_success_rate'], features['b10_vis2_correct_rate']]
    features['b10_task_balance'] = 1 - (np.std(perfs) / np.mean(perfs)) if np.mean(perfs) > 0 else 0
    
    # Attention allocation
    attention_weights = np.array([features['b10_aud_accuracy'], 
                                  features['b10_vis1_success_rate'], 
                                  features['b10_vis2_correct_rate']])
    features['b10_attention_allocation_variance'] = np.var(attention_weights)
    features['b10_attention_bottleneck'] = np.min(attention_weights)
    
    # Cognitive overload
    total_errors = b10_aud_miss + b10_aud_fa + b10_vis1_err + (20 - b10_vis2_correct)
    features['b10_cognitive_overload_index'] = total_errors / 152
    
    # Task completion efficiency
    features['b10_task_completion_efficiency'] = (b10_aud_hit + (52 - b10_vis1_err) + b10_vis2_correct) / 92
    
    # Executive function
    features['b10_executive_function_score'] = features['b10_triple_task_performance'] * features['b10_task_balance']
    
    # Attentional control
    features['b10_attentional_control'] = (features['b10_aud_dprime'] + features['b10_task_balance']) / 2
    
    # Modality variance
    vis_avg = (features['b10_vis1_success_rate'] + features['b10_vis2_correct_rate']) / 2
    features['b10_modality_variance'] = np.abs(features['b10_aud_accuracy'] - vis_avg)
    
    # Fluid intelligence proxy
    features['b10_fluid_intelligence_proxy'] = features['b10_executive_function_score'] * (1 - features['b10_cognitive_overload_index'])
    
    return features


def calculate_b9_b10_interaction_features(b9_features, b10_features):
    """Calculate interaction features between B9 and B10
    
    Captures adaptation, resilience, and performance changes under increased load
    Returns 25 features
    """
    features = {}
    
    # Task load comparison
    features['b9b10_cognitive_load_increase'] = b10_features['b10_cognitive_overload_index'] - b9_features['b9_cognitive_load_index']
    features['b9b10_performance_degradation'] = b9_features['b9_dual_task_performance'] - b10_features['b10_triple_task_performance']
    features['b9b10_efficiency_ratio'] = b10_features['b10_task_completion_efficiency'] / (b9_features['b9_multitask_efficiency'] + 1e-6)
    features['b9b10_task_resilience'] = 1 - np.abs(features['b9b10_performance_degradation'])
    features['b9b10_task_addition_penalty'] = features['b9b10_performance_degradation']
    
    # Auditory task comparison
    features['b9b10_dprime_change'] = b10_features['b10_aud_dprime'] - b9_features['b9_aud_dprime']
    features['b9b10_aud_accuracy_drop'] = b9_features['b9_aud_accuracy'] - b10_features['b10_aud_accuracy']
    features['b9b10_hit_rate_consistency'] = 1 - np.abs(b9_features['b9_aud_hit_rate'] - b10_features['b10_aud_hit_rate'])
    features['b9b10_fa_rate_increase'] = b10_features['b10_aud_fa_rate'] - b9_features['b9_aud_fa_rate']
    features['b9b10_precision_maintenance'] = 1 - np.abs(b9_features['b9_aud_precision'] - b10_features['b10_aud_precision'])
    
    # Visual task impact
    features['b9b10_vis_error_increase'] = b10_features['b10_vis1_error_rate'] - b9_features['b9_vis_error_rate']
    features['b9b10_visual_interference'] = features['b9b10_vis_error_increase']
    
    # Attention distribution changes
    features['b9b10_attention_flexibility'] = 1 - np.abs(b9_features['b9_attention_distribution'] - b10_features.get('b10_task_balance', 0.5))
    features['b9b10_balance_degradation'] = b9_features['b9_attention_distribution'] - b10_features.get('b10_task_balance', 0.5)
    
    # Executive function capacity
    features['b9b10_executive_capacity'] = b10_features.get('b10_executive_function_score', 0) / (b9_features['b9_cognitive_load_index'] + 0.5)
    features['b9b10_working_memory_saturation'] = features['b9b10_cognitive_load_increase']
    
    # Multitasking scaling
    features['b9b10_multitask_scaling'] = b10_features['b10_task_completion_efficiency'] / (b9_features['b9_multitask_efficiency'] + 1e-6)
    
    # Composite resilience indicators
    features['b9b10_load_tolerance'] = 1 - features['b9b10_cognitive_load_increase']
    features['b9b10_cognitive_reserve'] = (b9_features['b9_multitask_efficiency'] + b10_features['b10_task_completion_efficiency']) / 2 - features['b9b10_cognitive_load_increase']
    features['b9b10_adaptive_performance'] = (features['b9b10_task_resilience'] + features['b9b10_hit_rate_consistency']) / 2
    
    # Signal detection robustness
    features['b9b10_signal_detection_robustness'] = np.minimum(b9_features['b9_aud_dprime'], b10_features['b10_aud_dprime'])
    
    # Task switching proficiency
    features['b9b10_task_switching_proficiency'] = 1 - np.abs(b9_features['b9_dual_task_performance'] - b10_features['b10_triple_task_performance'])
    
    # Risk indicators
    features['b9b10_overload_susceptibility'] = features['b9b10_cognitive_load_increase'] * (1 - features['b9b10_task_resilience'])
    features['b9b10_performance_collapse'] = np.maximum(0, features['b9b10_performance_degradation'])
    features['b9b10_bottleneck_severity'] = b10_features.get('b10_attention_bottleneck', 0.5) - b9_features['b9_dual_task_performance']
    
    return features


# ============================================================================
# BTestPreprocessor Class
# ============================================================================

class BTestPreprocessor:
    """
    Preprocessing pipeline for B test features with PCA dimensionality reduction
    
    Combines B1, B2, B3, B4, B5, B6/B7, and B8 cognitive test features into optimized PCA components.
    
    Architecture:
    - B1/B2 (vision): 23 features → 3 PCA (62.77% variance)
    - B3 (traffic light): 18 features → 4 PCA (83.23% variance)
    - B4 (Stroop): 20 features → 5 PCA 
    - B5 (spatial): 43 features → 7 PCA (90.93% variance)
    - B6/B7 (sign recognition): 6 features → 1 PCA (64.19% variance)
    - B8 (lane keeping): 37 features → 8 PCA (99.31% variance)
    
    Total: 147 raw features → 28 PCA components
    """
    
    def __init__(self, dict_A=None, dict_B_past_label=None):
        """
        Initialize scalers and PCA transformers
        
        Parameters:
        -----------
        dict_A : dict, optional
            Dictionary mapping PrimaryKey to A test Label (max)
            Used for adding A past history features
        dict_B_past_label : dict, optional
            Dictionary mapping PrimaryKey to B past accident label
            Used for test inference (from training data)
        """
        self.dict_A = dict_A
        self.dict_B_past_label = dict_B_past_label
        
        # B1/B2 (combined central and peripheral vision)
        self.b1b2_scaler = StandardScaler()
        self.b1b2_pca = PCA(n_components=3)
        self.b1b2_feature_names = [
            'b1_1_correct_count', 'b1_1_early_incorrect', 'b1_1_late_incorrect', 'b1_1_concentration_decline',
            'b1_2_avg_time', 'b1_2_std_time', 'b1_2_time_trend', 'b1_2_cv_time',
            'b1_3_correct_count', 'b1_3_max_streak',
            'b2_1_correct_count', 'b2_1_early_incorrect', 'b2_1_late_incorrect', 'b2_1_concentration_decline',
            'b2_2_avg_time', 'b2_2_std_time', 'b2_2_time_trend', 'b2_2_cv_time',
            'b2_3_correct_count', 'b2_3_max_streak',
            'central_peripheral_diff', 'reaction_time_diff', 'concentration_decline_diff'
        ]
        
        # B3 (traffic light reaction)
        self.b3_scaler = StandardScaler()
        self.b3_pca = PCA(n_components=4)
        self.b3_feature_names = [
            'b3_correct_count', 'b3_incorrect_count',
            'b3_avg_time', 'b3_std_time', 'b3_min_time', 'b3_max_time', 'b3_median_time', 'b3_cv_time',
            'b3_correct_avg_time', 'b3_incorrect_avg_time', 'b3_efficiency',
            'b3_max_correct_streak', 'b3_avg_correct_streak',
            'b3_first10_correct', 'b3_last10_correct', 'b3_early_late_diff',
            'b3_time_trend'
        ]
        
        # B4 (Stroop test)
        self.b4_scaler = StandardScaler()
        self.b4_pca = PCA(n_components=5)
        self.b4_feature_names = [
            'b4_correct_count', 'b4_incorrect_count',
            'b4_congruent_correct', 'b4_incongruent_correct', 'b4_stroop_effect',
            'b4_first10_correct', 'b4_last10_correct', 'b4_performance_change',
            'b4_max_correct_streak', 'b4_avg_correct_streak',
            'b4_avg_reaction_time', 'b4_std_reaction_time', 'b4_min_reaction_time',
            'b4_median_reaction_time', 'b4_cv_reaction_time', 'b4_time_trend',
            'b4_congruent_avg_time', 'b4_incongruent_avg_time', 'b4_stroop_time_effect',
            'b4_correct_avg_time', 'b4_efficiency'
        ]
        
        # B5 (spatial judgment) - NEW!
        self.b5_scaler = StandardScaler()
        self.b5_pca = PCA(n_components=7)
        self.b5_feature_names = [
            'b5_correct_count', 'b5_incorrect_count',
            'b5_first5_correct', 'b5_last5_correct',
            'b5_first10_correct', 'b5_last10_correct', 'b5_early_late_diff',
            'b5_max_correct_streak', 'b5_avg_correct_streak', 'b5_num_correct_streaks',
            'b5_max_incorrect_streak', 'b5_num_incorrect_streaks', 'b5_total_streaks',
            'b5_transitions', 'b5_consistency',
            'b5_avg_time', 'b5_median_time', 'b5_std_time', 'b5_min_time', 'b5_max_time',
            'b5_cv_time', 'b5_time_p25', 'b5_time_p75', 'b5_time_iqr', 'b5_time_trend',
            'b5_correct_avg_time', 'b5_incorrect_avg_time', 'b5_time_accuracy_diff',
            'b5_efficiency', 'b5_spatial_score',
            'b5_early_error_rate', 'b5_middle_error_rate', 'b5_late_error_rate'
        ]
        
        # B6/B7 (traffic sign recognition) - NEW!
        self.b6b7_scaler = StandardScaler()
        self.b6b7_pca = PCA(n_components=1)
        self.b6b7_feature_names = [
            'b6_b7_accuracy_diff',
            'b6_b7_consistency_diff',
            'b7_correct_count',
            'b7_first5_correct',
            'advanced_visual_ability',
            'b7_last5_correct'
        ]
        
        # B8 (lane keeping) - NEW!
        self.b8_scaler = StandardScaler()
        self.b8_pca = PCA(n_components=8)
        self.b8_feature_names = [
            'b8_correct_count', 'b8_correct_rate', 'b8_incorrect_count', 
            'b8_incorrect_rate', 'b8_accuracy_score',
            'b8_first_third_correct', 'b8_middle_third_correct', 'b8_last_third_correct',
            'b8_first_half_correct', 'b8_last_half_correct',
            'b8_first_quarter_correct', 'b8_last_quarter_correct',
            'b8_early_accuracy', 'b8_late_accuracy', 'b8_middle_accuracy',
            'b8_learning_effect', 'b8_fatigue_effect',
            'b8_max_correct_streak', 'b8_max_incorrect_streak',
            'b8_avg_correct_streak', 'b8_avg_incorrect_streak',
            'b8_correct_streak_ratio', 'b8_streak_consistency',
            'b8_alternation_rate', 'b8_pattern_stability',
            'b8_response_consistency', 'b8_performance_variance',
            'b8_consistency_score', 'b8_error_clustering',
            'b8_correct_clustering', 'b8_response_stability', 'b8_rolling_std',
            'b8_early_error_rate', 'b8_late_error_rate',
            'b8_error_recovery_rate', 'b8_consecutive_errors',
            'b8_lane_judgment_ability', 'b8_spatial_consistency'
        ]
        
        # B9/B10 (dual/triple task) - Category-wise PCA - NEW!
        self.b9b10_orig_scalers = {}
        self.b9b10_orig_pcas = {}
        self.b9b10_inter_scalers = {}
        self.b9b10_inter_pcas = {}
        
        # Define feature categories for B9/B10
        self.b9b10_original_categories = {
            'Auditory_Signal_Detection': {
                'n_components': 2,
                'features': ['b10_aud_sensitivity', 'b10_aud_dprime', 'b10_aud_precision', 
                            'b9_aud_precision', 'b9_aud_dprime', 'b9_selective_attention',
                            'b9_aud_sensitivity', 'b9_selective_attention_quality', 'b10_aud_roc_auc',
                            'b9_aud_hit_minus_fa', 'b10_aud_hit_minus_fa', 'b9_aud_f1_score']
            },
            'Auditory_Basic': {
                'n_components': 3,
                'features': ['b10_aud_accuracy', 'b9_aud_accuracy', 'b10_aud_miss_rate', 
                            'b9_aud_miss_rate', 'b10_aud_hit_rate', 'b9_aud_hit_rate',
                            'b10_aud_recall', 'b9_aud_recall', 'b10_aud_cr_rate',
                            'b9_aud_cr_rate', 'b10_aud_fa_rate', 'b9_aud_fa_rate',
                            'b10_aud_f1_score', 'b10_aud_beta', 'b9_aud_beta',
                            'b10_aud_response_bias', 'b9_aud_response_bias', 'b10_aud_c']
            },
            'Visual_Performance': {
                'n_components': 2,
                'features': ['b10_vis1_err', 'b10_vis1_success_rate', 'b10_vis1_error_rate',
                            'b9_vis_error_rate', 'b9_vis_success_rate', 'b10_vis2_correct_rate']
            },
            'Dual_Triple_Task': {
                'n_components': 3,
                'features': ['b10_task_completion_efficiency', 'b9_multitask_efficiency', 
                            'b9_dual_task_performance', 'b10_triple_task_performance',
                            'b10_task_balance', 'b9_attention_distribution',
                            'b10_attention_allocation_variance', 'b10_attentional_control',
                            'b9_divided_attention_capacity']
            },
            'Cognitive_Load': {
                'n_components': 2,
                'features': ['b9_cognitive_load_index', 'b10_cognitive_overload_index',
                            'b10_executive_function_score', 'b10_fluid_intelligence_proxy',
                            'b9_cognitive_flexibility', 'b10_attention_bottleneck',
                            'b10_modality_variance']
            },
            'Advanced_Composite': {
                'n_components': 1,
                'features': ['b9_selective_attention_quality', 'b9_divided_attention_capacity',
                            'b9_cognitive_flexibility', 'b10_executive_function_score',
                            'b10_fluid_intelligence_proxy']
            }
        }
        
        self.b9b10_interaction_categories = {
            'Task_Load_Comparison': {
                'n_components': 2,
                'features': ['b9b10_balance_degradation', 'b9b10_task_addition_penalty',
                            'b9b10_performance_degradation', 'b9b10_efficiency_ratio',
                            'b9b10_task_resilience']
            },
            'Performance_Adaptation': {
                'n_components': 1,
                'features': ['b9b10_adaptive_performance', 'b9b10_hit_rate_consistency',
                            'b9b10_precision_maintenance']
            },
            'Cognitive_Capacity': {
                'n_components': 2,
                'features': ['b9b10_executive_capacity', 'b9b10_cognitive_reserve',
                            'b9b10_working_memory_saturation']
            },
            'Signal_Detection_Robustness': {
                'n_components': 2,
                'features': ['b9b10_signal_detection_robustness', 'b9b10_dprime_change']
            },
            'Attention_Management': {
                'n_components': 2,
                'features': ['b9b10_task_switching_proficiency', 'b9b10_attention_flexibility',
                            'b9b10_bottleneck_severity']
            },
            'Performance_Change': {
                'n_components': 1,
                'features': ['b9b10_visual_interference', 'b9b10_vis_error_increase']
            }
        }
        
        # Initialize scalers and PCAs for each category
        for cat_name, cat_info in self.b9b10_original_categories.items():
            self.b9b10_orig_scalers[cat_name] = StandardScaler()
            self.b9b10_orig_pcas[cat_name] = PCA(n_components=cat_info['n_components'])
        
        for cat_name, cat_info in self.b9b10_interaction_categories.items():
            self.b9b10_inter_scalers[cat_name] = StandardScaler()
            self.b9b10_inter_pcas[cat_name] = PCA(n_components=cat_info['n_components'])
    
    def extract_b1b2_features(self, df):
        """Extract B1 and B2 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b1_1 = parse_list(row.get('B1-1', []))
            b1_2 = parse_list(row.get('B1-2', []))
            b1_3 = parse_list(row.get('B1-3', []))
            b2_1 = parse_list(row.get('B2-1', []))
            b2_2 = parse_list(row.get('B2-2', []))
            b2_3 = parse_list(row.get('B2-3', []))
            
            b1_features = {}
            b1_features.update(calculate_b1_1_features(b1_1))
            b1_features.update(calculate_b1_2_features(b1_2))
            b1_features.update(calculate_b1_3_features(b1_3))
            
            b2_features = {}
            b2_features.update(calculate_b2_1_features(b2_1))
            b2_features.update(calculate_b2_2_features(b2_2))
            b2_features.update(calculate_b2_3_features(b2_3))
            
            composite_features = calculate_composite_features(b1_features, b2_features)
            
            all_features = {**b1_features, **b2_features, **composite_features}
            features_list.append(all_features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b1b2_feature_names]
    
    def extract_b3_features(self, df):
        """Extract B3 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b3_1 = parse_list(row.get('B3-1', []))
            b3_2 = parse_list(row.get('B3-2', []))
            
            features = calculate_b3_features(b3_1, b3_2)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b3_feature_names]
    
    def extract_b4_features(self, df):
        """Extract B4 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b4_1 = parse_list(row.get('B4-1', []))
            b4_2 = parse_list(row.get('B4-2', []))
            
            features = calculate_b4_features(b4_1, b4_2)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b4_feature_names]
    
    def extract_b5_features(self, df):
        """Extract B5 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b5_1 = parse_list(row.get('B5-1', []))
            b5_2 = parse_list(row.get('B5-2', []))
            
            features = calculate_b5_features(b5_1, b5_2)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b5_feature_names]
    
    def extract_b6b7_features(self, df):
        """Extract B6/B7 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b6 = parse_list(row.get('B6', []))
            b7 = parse_list(row.get('B7', []))
            
            features = calculate_b6_b7_features(b6, b7)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b6b7_feature_names]
    
    def extract_b8_features(self, df):
        """Extract B8 features from dataframe"""
        features_list = []
        
        for idx, row in df.iterrows():
            b8 = parse_list(row.get('B8', []))
            
            features = calculate_b8_features(b8)
            features_list.append(features)
        
        df_features = pd.DataFrame(features_list)
        return df_features[self.b8_feature_names]
    
    def extract_b9b10_features(self, df):
        """Extract B9 and B10 features including interactions"""
        
        # Extract B9 features
        b9_features_list = []
        for idx, row in df.iterrows():
            b9_feat = calculate_b9_features(row)
            b9_features_list.append(b9_feat)
        b9_df = pd.DataFrame(b9_features_list)
        
        # Extract B10 features
        b10_features_list = []
        for idx, row in df.iterrows():
            b10_feat = calculate_b10_features(row)
            b10_features_list.append(b10_feat)
        b10_df = pd.DataFrame(b10_features_list)
        
        # Combine B9 and B10 features
        combined_df = pd.concat([b9_df, b10_df], axis=1)
        
        # Calculate interaction features
        interaction_features_list = []
        for i in range(len(df)):
            inter_feat = calculate_b9_b10_interaction_features(b9_df.iloc[i], b10_df.iloc[i])
            interaction_features_list.append(inter_feat)
        interaction_df = pd.DataFrame(interaction_features_list)
        
        # Combine all features
        all_features_df = pd.concat([combined_df, interaction_df], axis=1)
        
        return all_features_df
    
    def fit(self, df):
        """Fit the preprocessor on training data"""
        print("Extracting B1/B2 features...")
        b1b2_features = self.extract_b1b2_features(df)
        
        print("Extracting B3 features...")
        b3_features = self.extract_b3_features(df)
        
        print("Extracting B4 features...")
        b4_features = self.extract_b4_features(df)
        
        print("Extracting B5 features...")
        b5_features = self.extract_b5_features(df)
        
        print("Extracting B6/B7 features...")
        b6b7_features = self.extract_b6b7_features(df)
        
        print("Extracting B8 features...")
        b8_features = self.extract_b8_features(df)
        
        print("Fitting B1/B2 scaler and PCA...")
        b1b2_scaled = self.b1b2_scaler.fit_transform(b1b2_features)
        self.b1b2_pca.fit(b1b2_scaled)
        
        print("Fitting B3 scaler and PCA...")
        b3_scaled = self.b3_scaler.fit_transform(b3_features)
        self.b3_pca.fit(b3_scaled)
        
        print("Fitting B4 scaler and PCA...")
        b4_scaled = self.b4_scaler.fit_transform(b4_features)
        self.b4_pca.fit(b4_scaled)
        
        print("Fitting B5 scaler and PCA...")
        b5_scaled = self.b5_scaler.fit_transform(b5_features)
        self.b5_pca.fit(b5_scaled)
        
        print("Fitting B6/B7 scaler and PCA...")
        b6b7_scaled = self.b6b7_scaler.fit_transform(b6b7_features)
        self.b6b7_pca.fit(b6b7_scaled)
        
        print("Fitting B8 scaler and PCA...")
        b8_scaled = self.b8_scaler.fit_transform(b8_features)
        self.b8_pca.fit(b8_scaled)
        
        # B9/B10 - Category-wise PCA
        print("Extracting B9/B10 features...")
        b9b10_all_features = self.extract_b9b10_features(df)
        
        print("Fitting B9/B10 category-wise PCA...")
        
        # Fit original feature categories
        total_orig_pcs = 0
        for cat_name, cat_info in self.b9b10_original_categories.items():
            cat_features = []
            for feat in cat_info['features']:
                if feat in b9b10_all_features.columns:
                    cat_features.append(feat)
            
            if len(cat_features) > 0:
                X = b9b10_all_features[cat_features].fillna(0)
                X_scaled = self.b9b10_orig_scalers[cat_name].fit_transform(X)
                self.b9b10_orig_pcas[cat_name].fit(X_scaled)
                variance = np.sum(self.b9b10_orig_pcas[cat_name].explained_variance_ratio_) * 100
                total_orig_pcs += cat_info['n_components']
                print(f"  [{cat_name}] {len(cat_features)} features → {cat_info['n_components']} PCs ({variance:.1f}%)")
        
        # Fit interaction feature categories
        total_inter_pcs = 0
        for cat_name, cat_info in self.b9b10_interaction_categories.items():
            cat_features = []
            for feat in cat_info['features']:
                if feat in b9b10_all_features.columns:
                    cat_features.append(feat)
            
            if len(cat_features) > 0:
                X = b9b10_all_features[cat_features].fillna(0)
                X_scaled = self.b9b10_inter_scalers[cat_name].fit_transform(X)
                self.b9b10_inter_pcas[cat_name].fit(X_scaled)
                variance = np.sum(self.b9b10_inter_pcas[cat_name].explained_variance_ratio_) * 100
                total_inter_pcs += cat_info['n_components']
                print(f"  [{cat_name}] {len(cat_features)} features → {cat_info['n_components']} PCs ({variance:.1f}%)")
        
        print(f"✓ B1/B2 PCA: {len(self.b1b2_feature_names)} features → 3 components ({np.sum(self.b1b2_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B3 PCA: {len(self.b3_feature_names)} features → 4 components ({np.sum(self.b3_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B4 PCA: {len(self.b4_feature_names)} features → 5 components ({np.sum(self.b4_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B5 PCA: {len(self.b5_feature_names)} features → 7 components ({np.sum(self.b5_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B6/B7 PCA: {len(self.b6b7_feature_names)} features → 1 component ({np.sum(self.b6b7_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B8 PCA: {len(self.b8_feature_names)} features → 8 components ({np.sum(self.b8_pca.explained_variance_ratio_)*100:.2f}% variance)")
        print(f"✓ B9/B10 PCA: {total_orig_pcs + total_inter_pcs} components total (Original: {total_orig_pcs}, Interaction: {total_inter_pcs})")
        
        # Create dict_B_past_label from training data
        print("\nCreating B past accident dictionary...")
        if 'Label' in df.columns and 'PrimaryKey' in df.columns:
            # PrimaryKey별 과거 사고 여부 계산 (any accident = 1)
            pk_b_label = df.groupby('PrimaryKey')['Label'].max().to_dict()
            self.dict_B_past_label = pk_b_label
            print(f"✓ dict_B_past_label created: {len(self.dict_B_past_label)} unique PrimaryKeys")
        else:
            self.dict_B_past_label = {}
            print("⚠ Warning: No Label column found, dict_B_past_label is empty")
        
        return self
    
    def transform(self, df):
        """
        Transform data using fitted preprocessor
        
        Returns DataFrame with:
        - 51 PCA components (B1-B10)
        - Age_num (numeric age)
        - TestDate (datetime)
        - past_A_history (A test history category)
        - Past_Label (B test history - numeric)
        - PrimaryKey (for grouping)
        """
        df_work = df.copy()
        
        # ===== Add Age_num =====
        if 'Age' in df_work.columns:
            df_work['Age_num'] = df_work['Age'].apply(convert_age_to_numeric)
        else:
            df_work['Age_num'] = np.nan
        
        # ===== Add TestDate (only TestDate, no Year/Month) =====
        if 'TestDate' in df_work.columns:
            df_work['TestDate'] = pd.to_datetime(df_work['TestDate'].astype(str), format='%Y%m', errors='coerce')
        else:
            df_work['TestDate'] = pd.NaT
        
        # ===== Add past history features =====
        df_work = add_past_history_features(
            df_work, 
            dict_A=self.dict_A,
            dict_B_past_label=self.dict_B_past_label
        )
        
        # ===== Extract B test PCA features =====
        # Extract features
        b1b2_features = self.extract_b1b2_features(df_work)
        b3_features = self.extract_b3_features(df_work)
        b4_features = self.extract_b4_features(df_work)
        b5_features = self.extract_b5_features(df_work)
        b6b7_features = self.extract_b6b7_features(df_work)
        b8_features = self.extract_b8_features(df_work)
        
        # Scale and apply PCA
        b1b2_scaled = self.b1b2_scaler.transform(b1b2_features)
        b1b2_pca = self.b1b2_pca.transform(b1b2_scaled)
        
        b3_scaled = self.b3_scaler.transform(b3_features)
        b3_pca = self.b3_pca.transform(b3_scaled)
        
        b4_scaled = self.b4_scaler.transform(b4_features)
        b4_pca = self.b4_pca.transform(b4_scaled)
        
        b5_scaled = self.b5_scaler.transform(b5_features)
        b5_pca = self.b5_pca.transform(b5_scaled)
        
        b6b7_scaled = self.b6b7_scaler.transform(b6b7_features)
        b6b7_pca = self.b6b7_pca.transform(b6b7_scaled)
        
        b8_scaled = self.b8_scaler.transform(b8_features)
        b8_pca = self.b8_pca.transform(b8_scaled)
        
        # B9/B10 - Category-wise PCA transform
        b9b10_all_features = self.extract_b9b10_features(df_work)
        
        b9b10_pca_results = {}
        
        # Transform original features
        for cat_name, cat_info in self.b9b10_original_categories.items():
            cat_features = []
            for feat in cat_info['features']:
                if feat in b9b10_all_features.columns:
                    cat_features.append(feat)
            
            if len(cat_features) > 0:
                X = b9b10_all_features[cat_features].fillna(0)
                X_scaled = self.b9b10_orig_scalers[cat_name].transform(X)
                X_pca = self.b9b10_orig_pcas[cat_name].transform(X_scaled)
                
                for i in range(cat_info['n_components']):
                    b9b10_pca_results[f'{cat_name}_PC{i+1}'] = X_pca[:, i]
        
        # Transform interaction features
        for cat_name, cat_info in self.b9b10_interaction_categories.items():
            cat_features = []
            for feat in cat_info['features']:
                if feat in b9b10_all_features.columns:
                    cat_features.append(feat)
            
            if len(cat_features) > 0:
                X = b9b10_all_features[cat_features].fillna(0)
                X_scaled = self.b9b10_inter_scalers[cat_name].transform(X)
                X_pca = self.b9b10_inter_pcas[cat_name].transform(X_scaled)
                
                for i in range(cat_info['n_components']):
                    b9b10_pca_results[f'{cat_name}_PC{i+1}'] = X_pca[:, i]
        
        # Combine all PCA components
        result = pd.DataFrame({
            'PC1': b1b2_pca[:, 0],
            'PC2': b1b2_pca[:, 1],
            'PC3': b1b2_pca[:, 2],
            'B3_PC1': b3_pca[:, 0],
            'B3_PC2': b3_pca[:, 1],
            'B3_PC3': b3_pca[:, 2],
            'B3_PC4': b3_pca[:, 3],
            'B4_PC1': b4_pca[:, 0],
            'B4_PC2': b4_pca[:, 1],
            'B4_PC3': b4_pca[:, 2],
            'B4_PC4': b4_pca[:, 3],
            'B4_PC5': b4_pca[:, 4],
            'B5_PC1': b5_pca[:, 0],
            'B5_PC2': b5_pca[:, 1],
            'B5_PC3': b5_pca[:, 2],
            'B5_PC4': b5_pca[:, 3],
            'B5_PC5': b5_pca[:, 4],
            'B5_PC6': b5_pca[:, 5],
            'B5_PC7': b5_pca[:, 6],
            'B6B7_PC1': b6b7_pca[:, 0],
            'B8_PC1': b8_pca[:, 0],
            'B8_PC2': b8_pca[:, 1],
            'B8_PC3': b8_pca[:, 2],
            'B8_PC4': b8_pca[:, 3],
            'B8_PC5': b8_pca[:, 4],
            'B8_PC6': b8_pca[:, 5],
            'B8_PC7': b8_pca[:, 6],
            'B8_PC8': b8_pca[:, 7],
            **b9b10_pca_results  # Add all B9/B10 PCA components
        })
        
        # Add metadata columns (only keep necessary ones)
        result['Age_num'] = df_work['Age_num'].values
        result['TestDate'] = df_work['TestDate'].values
        result['PrimaryKey'] = df_work['PrimaryKey'].values if 'PrimaryKey' in df_work.columns else np.nan
        
        # Add past history features (only keep non-redundant ones)
        result['past_A_history'] = df_work['past_A_history'].values
        result['Past_Label'] = df_work['Past_Label'].values
        
        return result
    
    def fit_transform(self, df):
        """Fit and transform in one step"""
        self.fit(df)
        return self.transform(df)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Optimal B Test Preprocessing Pipeline (Complete)")
    print("="*80)
    
    # Load data
    print("\nLoading training data...")
    b_df = pd.read_csv('train/B.csv')
    
    # Filter to 2020+
    print("Filtering to 2020+ data...")
    b_df['TestDate'] = pd.to_datetime(b_df['TestDate'].astype(str), format='%Y%m')
    b_df = b_df[b_df['TestDate'] >= '2020-01-01'].copy()
    print(f"Samples: {len(b_df)}")
    
    # Initialize and fit preprocessor
    print("\nInitializing preprocessor...")
    preprocessor = BTestPreprocessor()
    
    print("\nFitting preprocessor...")
    preprocessor.fit(b_df)
    
    print("\nTransforming data...")
    pca_features = preprocessor.transform(b_df)
    
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(f"Output shape: {pca_features.shape}")
    print(f"Columns: {list(pca_features.columns)}")
    print("\nFirst 5 rows:")
    print(pca_features.head())
    
    print("\n✓ Preprocessing pipeline ready!")
    print(f"  Total PCA components: 28")
    print(f"  - B1/B2: 3 components (vision)")
    print(f"  - B3: 4 components (traffic light)")
    print(f"  - B4: 5 components (Stroop)")
    print(f"  - B5: 7 components (spatial judgment)")
    print(f"  - B6/B7: 1 component (sign recognition)")
    print(f"  - B8: 8 components (lane keeping)")
    print(f"  - B9/B10: 23 components (dual/triple task, category-wise)")
    print(f"\n  Total PCA components: {pca_features.shape[1]}")
    print(f"\n  Compression: ~222 features → {pca_features.shape[1]} components (77% reduction)")
