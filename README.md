<div align="center">

# Traffic Accident Risk Prediction

### Using Cognitive Characteristics of Commercial Drivers

<br>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-189FDD?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black)](https://catboost.ai)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org)

<br>

> 운수종사자(버스/택시/화물 기사)의 **인지 반응 검사 데이터**로 **교통사고 위험도**를 예측하는 이진 분류 모델

**Dacon AI Competition | 6,000만원 상금 | 1,595명 참가**

</div>

---

## Competition Result

<p align="center">
  <img src="assets/leaderboard.png" alt="Leaderboard" width="800">
</p>

<div align="center">

| Metric | Value |
|--------|-------|
| **Final Rank** | **3rd / 1,595 teams** |
| **Final Score** | **0.14386** (lower is better) |
| **Prize** | **Top 1% (WINNER)** |
| Submissions | 96 |

</div>

---

## Overview

본 프로젝트는 **Dacon 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회**에 참가하여 개발한 시스템입니다.

운수종사자의 인지 반응 검사 데이터는 두 가지 유형으로 구분됩니다:

- **Test A (인지 반응 검사)** - 단순반응, 선택반응, 주의력, 간섭(Stroop), 변화탐지 등 5개 하위검사
- **Test B (운전적성 종합 검사)** - 시야, 신호등, 화살표(Flanker), 길찾기, 표지판, 추적, 복합기능(이중과제) 등 10개 하위검사

각 검사 유형에 **별도의 모델**을 설계하고, 운전자별 **시계열 이력 피처**와 **교차 검사 정보**를 결합하여 잘 보정된 사고 위험 확률을 예측합니다.

---

## Evaluation Metric

대회 평가 지표는 **판별력(AUC)**과 **확률 보정(Calibration)**을 결합한 가중합입니다:

$$\text{Score} = 0.5 \times (1 - \text{AUC}) + 0.25 \times \text{Brier Score} + 0.25 \times \text{ECE}$$

> **낮을수록** 좋은 점수. AUC가 높을수록 + Brier/ECE가 낮을수록(잘 보정될수록) 좋은 성능

| Metric | Weight | Description |
|--------|:------:|-------------|
| **AUC** | 50% | 판별력: 양성/음성 분리 능력 (1에 가까울수록 좋음) |
| **Brier Score** | 25% | 보정: 예측 확률과 실제 결과의 평균 제곱 오차 (0에 가까울수록 좋음) |
| **ECE** | 25% | 보정: 구간별 예측 확률과 실제 빈도의 차이 (0에 가까울수록 좋음) |

---

## Core Strategies

### 1. Dual-Model Architecture

두 검사 유형의 데이터 구조와 인지 특성이 근본적으로 다르므로 **별도의 모델**을 설계하였습니다.

| | Model A (인지 반응) | Model B (운전적성) |
|--|:---:|:---:|
| **Algorithm** | XGBoost | CatBoost |
| **Validation** | 5-Fold CV + 20% Holdout | 85/15 Stratified Split |
| **Key Hyperparams** | lr=0.01, depth=6, n_est=2000 | lr=0.005, depth=9, iter=3000 |
| **Features** | 조건별 반응시간 통계 | PCA 압축 + SDT 지표 |
| **Categorical** | - | past_A_history (Native) |

### 2. Feature Engineering

#### Test A: 조건별 반응시간 분석 (pre_a.py, 1,684 lines)

| Feature Group | Description | Count |
|---------------|-------------|:-----:|
| **A1 반응시간** | LEFT/RIGHT × SLOW/NORMAL/FAST 조합별 평균 RT | 10 |
| **A2 반응시간** | 두 조건 축 조합별 평균 RT | 12 |
| **A3 주의력** | 크기/유효성/방향/위치(8방향)별 RT 통계 | ~40 |
| **A4 간섭** | 일치/불일치 × RED/GREEN 정답 RT | 4 |
| **정답/오답** | 하위검사별 정답 수, 음수 반응 수 | ~15 |
| **A8 클러스터** | A8-1, A8-2 조합의 4분위 위험 클러스터 | 1 |
| **A9 종합** | 정서-행동, 행동-판단, 정서-스트레스, 종합안정성 | 4 |

#### Test B: PCA 파이프라인 (optimal_preprocess_full.py)

| Test Group | Raw Features | PCA Components | Variance Explained |
|:----------:|:------------:|:--------------:|:------------------:|
| B1/B2 (시야) | 23 | 3 | 62.8% |
| B3 (신호등) | 18 | 4 | 83.2% |
| B4 (화살표) | 20 | 5 | - |
| B5 (길찾기) | 43 | 7 | 90.9% |
| B6/B7 (표지판) | 6 | 1 | 64.2% |
| B8 (추적) | 37 | 8 | 99.3% |
| B9/B10 (복합기능) | 75 | 23 | - |
| **Total** | **222** | **51** | **77% 압축** |

#### B9/B10: Signal Detection Theory (SDT)

복합기능 검사에서 **신호탐지론 지표**를 산출하여 운전자의 인지 특성을 정량화합니다:

$$d' = z(\text{Hit Rate}) - z(\text{False Alarm Rate})$$

| Metric | Description |
|--------|-------------|
| **d' (민감도)** | 신호와 소음을 구별하는 능력 |
| **Criterion (기준)** | 응답 편향 (보수적/진보적) |
| **Hit Rate** | 신호 존재 시 정확 탐지율 |
| **False Alarm Rate** | 신호 부재 시 오탐율 |

### 3. Driver-Level Temporal Features (17 features)

동일 운전자의 **과거 검사 이력**을 시계열로 분석하여 17개 피처를 생성합니다:

| Feature | Description |
|---------|-------------|
| `A_SuccessRate` | 전체 합격률 |
| `A_RecentSuccessRate` | 최근 3회 합격률 |
| `A_Streak` | 현재 연속 합격/불합격 횟수 |
| `A_SuccessTrend` | 합격 추세 (선형 회귀 계수) |
| `A_SuccessMomentum` | 이동 평균 합격률 변화량 |
| `A_LastSuccessGap` / `A_LastFailGap` | 마지막 합격/불합격 이후 경과일 |
| `A_AvgInterval` / `A_LastInterval` | 평균/최근 검사 간격 |
| `is_first_test` | 최초 검사 여부 |
| `has_holiday_season` | 명절 시즌(9-10월) 포함 여부 |

### 4. Cross-Test Information Sharing

A 모델과 B 모델이 **독립적이지만 상호 정보를 공유**합니다:

```
Test A Model ←── B 검사 이전 라벨 (b_previous_label)
    │
    └── A 검사 이력 시계열 ──→ Test B Model
         (pk_dict_utils: 17 features)
```

| Feature | Source → Target | Description |
|---------|:---------------:|-------------|
| `b_previous_label` | B → A | 동일 운전자의 B 검사 과거 라벨 |
| `primary_past_label` | A → A | 동일 운전자의 이전 A 검사 라벨 |
| `past_A_history` | A → B | A 검사 이력 카테고리 (없음/합격/불합격) |
| `pk_dict features` | A → B | 17개 시계열 피처 |

### 5. Label Pattern Transition

라벨 이력의 **전이 패턴**을 분석하여 운전자의 위험도 변화 경향을 포착합니다:

| Transition | Meaning |
|:----------:|---------|
| 0 → 0 | 지속 안전 (개선 유지) |
| 0 → 1 | 위험 전환 (악화) |
| 1 → 0 | 안전 전환 (개선) |
| 1 → 1 | 지속 위험 (고위험군) |

---

## Dataset

> 데이터는 **Dacon 대회 제공** 데이터이며, 운수종사자의 인지 반응 검사 결과를 포함합니다.

| 항목 | 내용 |
|------|------|
| **Data Source** | Dacon 대회 제공 (운수종사자 인지 검사) |
| **Task** | Binary Classification (사고 위험 0/1) |
| **Test A** | 5개 하위검사 (단순반응, 선택반응, 주의력, 간섭, 변화탐지) |
| **Test B** | 10개 하위검사 (시야, 신호등, 화살표, 길찾기, 표지판, 추적, 복합기능) |
| **Key Features** | 조건별 반응시간, 정답률, SDT 지표, 연령, 과거 이력 |

---

## Tech Stack

| Category | Tool | Version | Role |
|----------|------|---------|------|
| **GBDT** | XGBoost | 2.1.1 | Model A (5-Fold CV) |
| **GBDT** | CatBoost | latest | Model B (Native Categorical) |
| **ML** | scikit-learn | latest | PCA, StandardScaler, KFold, Metrics |
| **DL** | PyTorch | latest | 1D CNN AutoEncoder (Feature Extraction) |
| **Data** | pandas | latest | DataFrame 처리 |
| **Numeric** | NumPy | latest | 수치 연산, Vectorized SDT |
| **Serialization** | joblib | 1.3.2 | 모델 저장/로드 |
| **GBDT** | LightGBM | 4.6.0 | 실험용 |
| **Visualization** | matplotlib | latest | 피처 분석 시각화 |
| **Stats** | SciPy | latest | norm.ppf (SDT d' 계산) |

---

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train Model A (XGBoost, 5-Fold CV)

```bash
python scripts/train_model_a.py
```

### 3. Train Model B (CatBoost)

```bash
python scripts/train_model_b.py
```

---

## Project Structure

```
.
├── assets/                          # README 이미지
│   └── leaderboard.png              #   대회 리더보드 스크린샷
│
├── scripts/                         # 실행 스크립트
│   ├── train_model_a.py             #   Model A 학습 (XGBoost, 5-Fold CV)
│   ├── train_model_b.py             #   Model B 학습 (CatBoost + PCA)
│   ├── train_model_b_retention.py   #   Model B 변형 (라벨 유지율 피처)
│   ├── train_model_b_temporal.py    #   Model B 변형 (이전 검사 피처)
│   └── preprocessing/               #   전처리 모듈
│       ├── preprocess_test_a.py     #     Test A 피처 엔지니어링
│       ├── preprocess_test_b.py     #     Test B 기본 피처 추출
│       ├── preprocess_b_pca_pipeline.py  # Test B PCA 차원축소 파이프라인
│       ├── preprocess_b_vectorized.py    # B9/B10 벡터화 처리 (고속)
│       ├── preprocess_cnn_encoder.py     # 1D CNN AutoEncoder 피처
│       └── driver_history_features.py    # 운전자 시계열 이력 피처 (17개)
│
├── models/                          # 학습된 모델 & 아티팩트
│   ├── model_a/                     #   XGBoost 5-Fold 모델 (.json)
│   ├── model_b/                     #   CatBoost 모델 (.cbm) + 전처리기
│   └── pk_dict.pkl                  #   운전자별 이력 딕셔너리
│
├── docs/                            # 문서
│   ├── model_development_report.hwp #   모델 개발 보고서
│   └── data_analysis_report.hwp     #   데이터 분석 보고서
│
├── requirements.txt                 # Python 의존성
└── .gitignore
```

---

## References

1. T. Chen & C. Guestrin. *XGBoost: A Scalable Tree Boosting System.* KDD, 2016.
2. L. Prokhorenkova et al. *CatBoost: unbiased boosting with categorical features.* NeurIPS, 2018.
3. D. M. Green & J. A. Swets. *Signal Detection Theory and Psychophysics.* Wiley, 1966.
4. Dacon. 운수종사자 인지적 특성 데이터를 활용한 교통사고 위험 예측 AI 경진대회. https://dacon.io/competitions/official/236564
