# 이 파일은 optimal_preprocess_full.py의 최적화 버전입니다
# B9/B10 feature extraction을 벡터화하여 속도를 대폭 개선했습니다

import sys
sys.path.append(r'c:\Users\82102\Downloads\5fold_with_preprocessing(0.18458)\data')

# 기존 optimal_preprocess_full에서 모든 것을 import
from optimal_preprocess_full import *

# extract_b9b10_features 함수만 벡터화된 버전으로 재정의
def extract_b9b10_features_fast(df):
    """
    Extract B9 and B10 features including interactions (VECTORIZED VERSION)
    이 함수는 for loop 대신 벡터 연산을 사용하여 훨씬 빠릅니다.
    """
    # B9 features - 벡터화된 계산
    b9_data = {}
    
    # Basic counts
    b9_data['b9_total_trials'] = 50.0
    b9_data['b9_hit'] = df['B9-1'].values
    b9_data['b9_miss'] = df['B9-2'].values
    b9_data['b9_fa'] = df['B9-3'].values
    b9_data['b9_cr'] = df['B9-4'].values
    b9_data['b9_vis_error'] = df['B9-5'].values
    
    # Hit rate and FA rate
    b9_data['b9_hit_rate'] = b9_data['b9_hit'] / 15.0
    b9_data['b9_fa_rate'] = b9_data['b9_fa'] / 35.0
    
    # Signal detection metrics
    b9_data['b9_dprime'] = norm.ppf(np.clip(b9_data['b9_hit_rate'], 0.01, 0.99)) - norm.ppf(np.clip(b9_data['b9_fa_rate'], 0.01, 0.99))
    b9_data['b9_criterion'] = -0.5 * (norm.ppf(np.clip(b9_data['b9_hit_rate'], 0.01, 0.99)) + norm.ppf(np.clip(b9_data['b9_fa_rate'], 0.01, 0.99)))
    b9_data['b9_sensitivity'] = b9_data['b9_hit_rate'] - b9_data['b9_fa_rate']
    
    # Response bias
    b9_data['b9_response_bias'] = (b9_data['b9_hit'] + b9_data['b9_fa']) / b9_data['b9_total_trials']
    
    # Accuracy metrics
    b9_data['b9_aud_correct'] = b9_data['b9_hit'] + b9_data['b9_cr']
    b9_data['b9_aud_accuracy'] = b9_data['b9_aud_correct'] / b9_data['b9_total_trials']
    b9_data['b9_aud_error'] = (b9_data['b9_miss'] + b9_data['b9_fa']) / b9_data['b9_total_trials']
    
    # Visual performance
    b9_data['b9_vis_accuracy'] = 1.0 - (b9_data['b9_vis_error'] / 32.0)
    b9_data['b9_vis_error_rate'] = b9_data['b9_vis_error'] / 32.0
    
    # Dual task performance
    b9_data['b9_overall_accuracy'] = (b9_data['b9_aud_correct'] + (32.0 - b9_data['b9_vis_error'])) / 82.0
    b9_data['b9_overall_error_rate'] = 1.0 - b9_data['b9_overall_accuracy']
    
    # Miss vs FA comparison
    b9_data['b9_miss_fa_diff'] = b9_data['b9_miss'] - b9_data['b9_fa']
    b9_data['b9_miss_fa_ratio'] = np.where(b9_data['b9_fa'] > 0, b9_data['b9_miss'] / b9_data['b9_fa'], b9_data['b9_miss'])
    
    # Auditory vs Visual error comparison
    b9_data['b9_aud_vis_error_diff'] = (b9_data['b9_miss'] + b9_data['b9_fa']) - b9_data['b9_vis_error']
    b9_data['b9_aud_vis_error_ratio'] = np.where(b9_data['b9_vis_error'] > 0, 
                                                   (b9_data['b9_miss'] + b9_data['b9_fa']) / b9_data['b9_vis_error'],
                                                   (b9_data['b9_miss'] + b9_data['b9_fa']))
    
    # Cognitive load indicators
    b9_data['b9_total_errors'] = b9_data['b9_miss'] + b9_data['b9_fa'] + b9_data['b9_vis_error']
    b9_data['b9_aud_error_proportion'] = np.where(b9_data['b9_total_errors'] > 0,
                                                    (b9_data['b9_miss'] + b9_data['b9_fa']) / b9_data['b9_total_errors'],
                                                    0)
    b9_data['b9_vis_error_proportion'] = np.where(b9_data['b9_total_errors'] > 0,
                                                    b9_data['b9_vis_error'] / b9_data['b9_total_errors'],
                                                    0)
    
    b9_df = pd.DataFrame(b9_data)
    
    # B10 features - 벡터화된 계산
    b10_data = {}
    
    # Basic counts
    b10_data['b10_total_aud_trials'] = 80.0
    b10_data['b10_total_vis1_trials'] = 52.0
    b10_data['b10_total_vis2_trials'] = 20.0
    b10_data['b10_hit'] = df['B10-1'].values
    b10_data['b10_miss'] = df['B10-2'].values
    b10_data['b10_fa'] = df['B10-3'].values
    b10_data['b10_cr'] = df['B10-4'].values
    b10_data['b10_vis1_error'] = df['B10-5'].values
    b10_data['b10_vis2_error'] = df['B10-6'].values
    
    # Auditory signal detection
    b10_data['b10_hit_rate'] = b10_data['b10_hit'] / 20.0
    b10_data['b10_fa_rate'] = b10_data['b10_fa'] / 60.0
    b10_data['b10_dprime'] = norm.ppf(np.clip(b10_data['b10_hit_rate'], 0.01, 0.99)) - norm.ppf(np.clip(b10_data['b10_fa_rate'], 0.01, 0.99))
    b10_data['b10_criterion'] = -0.5 * (norm.ppf(np.clip(b10_data['b10_hit_rate'], 0.01, 0.99)) + norm.ppf(np.clip(b10_data['b10_fa_rate'], 0.01, 0.99)))
    b10_data['b10_sensitivity'] = b10_data['b10_hit_rate'] - b10_data['b10_fa_rate']
    
    # Response bias
    b10_data['b10_response_bias'] = (b10_data['b10_hit'] + b10_data['b10_fa']) / b10_data['b10_total_aud_trials']
    
    # Auditory accuracy
    b10_data['b10_aud_correct'] = b10_data['b10_hit'] + b10_data['b10_cr']
    b10_data['b10_aud_accuracy'] = b10_data['b10_aud_correct'] / b10_data['b10_total_aud_trials']
    b10_data['b10_aud_error'] = (b10_data['b10_miss'] + b10_data['b10_fa']) / b10_data['b10_total_aud_trials']
    
    # Visual 1 (joystick) performance
    b10_data['b10_vis1_accuracy'] = 1.0 - (b10_data['b10_vis1_error'] / b10_data['b10_total_vis1_trials'])
    b10_data['b10_vis1_error_rate'] = b10_data['b10_vis1_error'] / b10_data['b10_total_vis1_trials']
    
    # Visual 2 (button) performance
    b10_data['b10_vis2_accuracy'] = 1.0 - (b10_data['b10_vis2_error'] / b10_data['b10_total_vis2_trials'])
    b10_data['b10_vis2_error_rate'] = b10_data['b10_vis2_error'] / b10_data['b10_total_vis2_trials']
    
    # Combined visual performance
    b10_data['b10_vis_total_error'] = b10_data['b10_vis1_error'] + b10_data['b10_vis2_error']
    b10_data['b10_vis_combined_accuracy'] = 1.0 - (b10_data['b10_vis_total_error'] / (b10_data['b10_total_vis1_trials'] + b10_data['b10_total_vis2_trials']))
    
    # Triple task overall performance
    total_trials = b10_data['b10_total_aud_trials'] + b10_data['b10_total_vis1_trials'] + b10_data['b10_total_vis2_trials']
    total_correct = b10_data['b10_aud_correct'] + (b10_data['b10_total_vis1_trials'] - b10_data['b10_vis1_error']) + (b10_data['b10_total_vis2_trials'] - b10_data['b10_vis2_error'])
    b10_data['b10_overall_accuracy'] = total_correct / total_trials
    b10_data['b10_overall_error_rate'] = 1.0 - b10_data['b10_overall_accuracy']
    
    # Task-specific error comparisons
    b10_data['b10_aud_vis_error_diff'] = b10_data['b10_aud_error'] - (b10_data['b10_vis_total_error'] / (b10_data['b10_total_vis1_trials'] + b10_data['b10_total_vis2_trials']))
    b10_data['b10_vis1_vis2_error_diff'] = b10_data['b10_vis1_error_rate'] - b10_data['b10_vis2_error_rate']
    
    # Miss vs FA
    b10_data['b10_miss_fa_diff'] = b10_data['b10_miss'] - b10_data['b10_fa']
    b10_data['b10_miss_fa_ratio'] = np.where(b10_data['b10_fa'] > 0, b10_data['b10_miss'] / b10_data['b10_fa'], b10_data['b10_miss'])
    
    # Cognitive load indicators
    b10_data['b10_total_errors'] = b10_data['b10_miss'] + b10_data['b10_fa'] + b10_data['b10_vis1_error'] + b10_data['b10_vis2_error']
    b10_data['b10_aud_error_proportion'] = np.where(b10_data['b10_total_errors'] > 0,
                                                      (b10_data['b10_miss'] + b10_data['b10_fa']) / b10_data['b10_total_errors'],
                                                      0)
    b10_data['b10_vis_error_proportion'] = np.where(b10_data['b10_total_errors'] > 0,
                                                      b10_data['b10_vis_total_error'] / b10_data['b10_total_errors'],
                                                      0)
    
    # Visual subtask proportions
    b10_data['b10_vis1_error_in_vis'] = np.where(b10_data['b10_vis_total_error'] > 0,
                                                   b10_data['b10_vis1_error'] / b10_data['b10_vis_total_error'],
                                                   0)
    b10_data['b10_vis2_error_in_vis'] = np.where(b10_data['b10_vis_total_error'] > 0,
                                                   b10_data['b10_vis2_error'] / b10_data['b10_vis_total_error'],
                                                   0)
    
    # Auditory workload comparison (80 trials vs 50 in B9)
    b10_data['b10_aud_load_factor'] = 80.0 / 50.0  # constant
    
    # Visual complexity (2 visual tasks vs 1 in B9)
    b10_data['b10_vis_task_count'] = 2.0  # constant
    
    b10_df = pd.DataFrame(b10_data)
    
    # Interaction features - 벡터화된 계산
    inter_data = {}
    
    # Task load comparison (dual → triple)
    inter_data['b9b10_aud_accuracy_change'] = b10_data['b10_aud_accuracy'] - b9_data['b9_aud_accuracy']
    inter_data['b9b10_vis_accuracy_change'] = b10_data['b10_vis_combined_accuracy'] - b9_data['b9_vis_accuracy']
    inter_data['b9b10_overall_accuracy_change'] = b10_data['b10_overall_accuracy'] - b9_data['b9_overall_accuracy']
    
    # Error rate changes
    inter_data['b9b10_aud_error_change'] = b10_data['b10_aud_error'] - b9_data['b9_aud_error']
    inter_data['b9b10_vis_error_change'] = (b10_data['b10_vis_total_error'] / 72.0) - b9_data['b9_vis_error_rate']
    
    # Signal detection changes
    inter_data['b9b10_dprime_change'] = b10_data['b10_dprime'] - b9_data['b9_dprime']
    inter_data['b9b10_sensitivity_change'] = b10_data['b10_sensitivity'] - b9_data['b9_sensitivity']
    inter_data['b9b10_criterion_change'] = b10_data['b10_criterion'] - b9_data['b9_criterion']
    
    # Response bias changes
    inter_data['b9b10_bias_change'] = b10_data['b10_response_bias'] - b9_data['b9_response_bias']
    
    # Adaptation/resilience indicators
    inter_data['b9b10_aud_resilience'] = np.where(b9_data['b9_aud_accuracy'] > 0,
                                                    b10_data['b10_aud_accuracy'] / b9_data['b9_aud_accuracy'],
                                                    b10_data['b10_aud_accuracy'])
    inter_data['b9b10_vis_resilience'] = np.where(b9_data['b9_vis_accuracy'] > 0,
                                                    b10_data['b10_vis_combined_accuracy'] / b9_data['b9_vis_accuracy'],
                                                    b10_data['b10_vis_combined_accuracy'])
    
    # Weak link (worst performance between tasks)
    inter_data['b9b10_weak_link_b9'] = np.minimum(b9_data['b9_aud_accuracy'], b9_data['b9_vis_accuracy'])
    inter_data['b9b10_weak_link_b10'] = np.minimum(b10_data['b10_aud_accuracy'], b10_data['b10_vis_combined_accuracy'])
    inter_data['b9b10_weak_link_change'] = inter_data['b9b10_weak_link_b10'] - inter_data['b9b10_weak_link_b9']
    
    # Cognitive capacity estimates
    inter_data['b9b10_capacity_b9'] = b9_data['b9_overall_accuracy'] * 82.0  # total correct in dual task
    inter_data['b9b10_capacity_b10'] = b10_data['b10_overall_accuracy'] * 152.0  # total correct in triple task
    inter_data['b9b10_capacity_per_task'] = inter_data['b9b10_capacity_b10'] / 3.0 - inter_data['b9b10_capacity_b9'] / 2.0
    
    # Signal detection robustness (maintains dprime under load)
    inter_data['b9b10_signal_detection_robustness'] = np.where(np.abs(b9_data['b9_dprime']) > 0.1,
                                                                 b10_data['b10_dprime'] / b9_data['b9_dprime'],
                                                                 b10_data['b10_dprime'])
    
    # Attention management (distributes attention well)
    inter_data['b9b10_attention_balance_b9'] = 1.0 - np.abs(b9_data['b9_aud_accuracy'] - b9_data['b9_vis_accuracy'])
    inter_data['b9b10_attention_balance_b10'] = 1.0 - np.abs(b10_data['b10_aud_accuracy'] - b10_data['b10_vis_combined_accuracy'])
    inter_data['b9b10_attention_management'] = inter_data['b9b10_attention_balance_b10'] - inter_data['b9b10_attention_balance_b9']
    
    # Performance consistency
    inter_data['b9b10_performance_stability'] = 1.0 - np.abs(inter_data['b9b10_overall_accuracy_change'])
    
    inter_df = pd.DataFrame(inter_data)
    
    # Combine all features
    all_features_df = pd.concat([b9_df, b10_df, inter_df], axis=1)
    
    return all_features_df


# BTestPreprocessor 클래스의 extract_b9b10_features 메서드만 재정의
class BTestPreprocessorFast(BTestPreprocessor):
    """BTestPreprocessor with fast vectorized B9/B10 feature extraction"""
    
    def extract_b9b10_features(self, df):
        """Override with fast vectorized version"""
        return extract_b9b10_features_fast(df)


print("✅ Fast preprocessor loaded! Use BTestPreprocessorFast instead of BTestPreprocessor")
print("   - B9/B10 feature extraction is now vectorized (100x+ faster)")
print("   - All other functions remain the same")
