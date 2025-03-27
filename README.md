# Finance-insights
Studying Financial Knowledge
# 📊 LSTM 기반 주가 예측: Sentiment 비교 (No Sentiment vs. VIX Sentiment)

## 🧠 개요 (Overview)
이 프로젝트는 LSTM(Long Short-Term Memory) 모델을 기반으로 주가(종가, Close)를 예측합니다.  
특히, **감성 정보(Sentiment)**를 입력에 포함했을 때와 포함하지 않았을 때의 예측 성능을 비교합니다.

- **No Sentiment**: 감성 정보 없이 기본 가격/거래량 데이터만 사용
- **VIX Sentiment**: 시장 변동성을 나타내는 **VIX 지수**를 감성 지표로 추가하여 입력

## ⚙️ 모델 설명
본 모델은 다음과 같은 딥러닝 구조로 구성됩니다:
- LSTM(64 units, return_sequences=True)
- Dropout(0.2)
- LSTM(32 units)
- Dropout(0.2)
- Dense(1): 종가 예측

이 구조는 시계열 데이터를 효과적으로 학습하며, 과적합을 방지하기 위해 Dropout과 L2 정규화를 사용합니다.

## 📐 입력 피처 (Features)
- 공통 입력: `Open`, `High`, `Low`, `Close`, `Volume`
- 감성 입력: `Sentiment` (없거나, 또는 VIX 지수)

## 🧪 평가 지표 설명 (Evaluation Metrics)

| 지표 | 설명 |
|------|------|
| **RMSE** (Root Mean Squared Error) | 예측 오차의 표준편차. 값이 작을수록 예측이 실제 값에 가까움 |
| **MAE** (Mean Absolute Error) | 예측 값과 실제 값 사이의 절대 오차 평균. 직관적이고 노이즈에 덜 민감 |
| **MAPE** (Mean Absolute Percentage Error) | 예측이 실제 값에서 평균적으로 몇 % 정도 벗어났는지를 백분율로 표현 |
| **Accuracy (±5%)** | 예측값이 실제값의 ±5% 이내에 들어올 확률 (상대 오차 기준 정확도) |

## 🔍 비교 목적
- 감성 정보(VIX)가 주가 예측에 도움이 되는지 확인
- 기본 LSTM 구조 기반 성능 비교
- 추후 **FinBERT 기반 감성 지수**와도 비교할 수 있는 베이스라인 마련

## 📊 시각화 제공
- 각 지표(RMSE, MAE, MAPE, Accuracy)에 대해 **막대그래프 시각화**
- 결과를 `results/LSTM_results.csv`로 저장

---


## 코드 설명
### ✅ 코드 1: 기본 LSTM 모델 + 성능 지표 시각화

#### 📌 개요
- 기본적인 LSTM 구조로 종가(Close)를 단변량으로 예측
- 감성 정보(Sentiment)는 0으로 고정
- Rolling Window 방식으로 학습 및 검증 수행
- 시각화는 성능 비교용 바 차트 위주로 구성

#### 🧱 모델 구조
- LSTM(64, return_sequences=True)
- Dropout(0.2)
- LSTM(32)
- Dropout(0.2)
- Dense(1)

#### 📈 평가 지표
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Accuracy (±5% 오차 허용)

#### 📊 시각화 기능
- `plot_metric()` 함수로 지표별 바 차트 출력
  - RMSE, MAE, MAPE, Accuracy 비교 가능
- 학습 곡선 시각화는 없음

#### 📌 특징 요약
- 가장 기본적인 LSTM 예측 구조
- 감성 정보는 아직 비활성화된 상태
- 모델 성능 지표는 정리되어 있으나 학습 곡선 분석은 불가
### ✅ 코드 2: 기본 LSTM + 학습 곡선 시각화

#### 📌 개요
- 코드 1의 구조를 기반으로 하되, 각 윈도우 학습 결과(`loss`, `val_loss`) 추적 및 시각화 가능
- 평균 학습 손실 시각화를 통해 **과적합 여부 분석**에 용이

#### 🧱 모델 구조
- LSTM(64, return_sequences=True)
- Dropout(0.2)
- LSTM(32)
- Dropout(0.2)
- Dense(1)

#### 📈 평가 지표
- RMSE
- MAE
- MAPE
- Accuracy (±5%)

#### 📊 시각화 기능
- 성능 지표 비교 (bar chart)
- **Rolling Window 학습 손실 곡선 시각화**
  - Epoch별 평균 Training/Validation Loss
- `history_list` 활용

#### 📌 특징 요약
- 모델 구조는 동일하지만, 학습 로그 추적 기능 강화
- 학습 안정성, 과적합 여부를 확인하고자 할 때 적합
### ✅ 코드 3: 개선된 양방향 LSTM + 성능 안정화 + 통합 시각화

#### 📌 개요
- Bidirectional LSTM 도입으로 예측 정확도 향상
- 학습 로그 수집 및 평균 loss 시각화 포함
- 데이터 부족 시 NaN 처리로 **오류 발생 최소화**
- 시각화 함수 구조 간소화 (`plot_avg_loss`, `plot_metric` 등)

#### 🧱 모델 구조
- Bidirectional(LSTM(64, return_sequences=True))
- Dropout(0.2)
- LSTM(32)
- Dropout(0.2)
- Dense(1)

#### 📈 평가 지표
- RMSE
- Accuracy (±5%)
- MAE, MAPE는 최근 버전에서 추가되어 **더욱 정밀한 평가 가능**

#### 📊 시각화 기능
- 성능 지표 바 차트
- 학습 손실 곡선 (각 윈도우 평균 loss, val_loss)
- 시각화 함수 통합 및 개선

#### 🛡️ 안정성 강화
- try-except 구문을 통해 데이터 부족 시 자동 NaN 처리
- 불완전한 시계열에도 견고한 실행 보장

#### 📌 특징 요약
- **예측 성능과 코드 안정성 모두 고려한 구조**
- 확장성과 유지보수 측면에서 가장 적합
- 추후 FinBERT 감성지표 추가 시도에 적합한 베이스라인



#### 
일간 뉴스 5가지 각각 감정분석한 값을 평균내어 진행
감정분석의 경우 단순 모델과 앙상블 모델을 비교하여 진행
좋아! 아래는 두 코드의 차이를 README.md에 정리하기 좋은 형태로 정리한 내용이야.
목표는 뉴스 기반 감성 분석에 사용한 두 방법(1. Ensemble 방식, 2. 단일 모델 방식)을 비교해서 설명하는 것이고, 표 형식도 함께 제공할게.

⸻

🧠 뉴스 감성 분석 방식 비교 (Ensemble vs Single Model)

본 프로젝트에서는 뉴스 타이틀을 기반으로 주가 예측용 감성 점수를 생성하기 위해 두 가지 접근을 사용했습니다:

항목	앙상블 방식 (dual-model ensemble)	단일 모델 방식 (single-model)
🔍 사용 모델	ProsusAI/finbert + yiyanghkust/finbert-tone	ProsusAI/finbert
🎯 감정 점수 방식	두 모델의 점수 평균	감정 레이블을 수치로 변환 (90, 50, 10)
🧮 점수 계산 방식	score = (model1_score + model2_score) / 2	positive: 90, neutral: 50, negative: 10
📊 결과 값	Ensemble_Sentiment_Score (0–100)	sentiment_mean, sentiment_mean_scaled
📁 결과 파일	- meta_news_with_ensemble_sentiment.csv- meta_daily_ensemble_sentiment_avg.csv	- meta_news_with_sentiment.csv- meta_daily_sentiment_avg.csv
📦 장점	다양한 모델의 판단 반영으로 노이즈 보완 가능	빠르고 간단한 처리, 재현성 높음
⚠️ 단점	속도 느림, 리소스 많이 소모	단일 모델 한계 존재


⸻
