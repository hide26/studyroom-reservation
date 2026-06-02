# HAWK — Hongik Anomaly WatchKeeper

다변량 시계열 이상탐지 웹앱. 6개 탐지기의 합의 투표로 이상을 판정하고, 합성 이상 주입으로 탐지 품질을 자동 평가한다.

> 시계열분석 프로젝트 2 · 임의의 CSV를 업로드하면 **FastAPI 백엔드**가 스스로 학습해 이상을 탐지하고, 그 탐지가 적절한지 다중 평가지표로 검증하는 풀스택 웹앱.

배포 링크: `(여기에 배포 URL)`
데모 영상(3분): `(여기에 영상 링크)`

---

## 한눈에 보는 특징

- **FastAPI + pandas + scikit-learn 백엔드.** 6개 탐지 알고리즘, 합성 이상 주입, 합의 투표, 3개 평가지표를 서버에서 계산한다. scikit-learn `IsolationForest`는 정식 사용.
- **파일이 바뀌면 스스로 재수립.** 백엔드가 업로드된 데이터의 분포·계절성을 매번 다시 측정하고, 6개 탐지기의 임계값을 그 데이터에 맞춰 F1 최대 지점으로 자동 재추정한다. 직전 파일 대비 데이터 드리프트도 감지.
- **라벨 없는 데이터를 정직하게 평가.** 임의 CSV에는 정답 라벨이 없다. 그래서 깨끗한 구간에 **알려진 이상을 합성 주입**(스파이크·레벨시프트·분산폭발·드리프트)하고, 탐지기가 그걸 잡는지로 Precision/Recall/F1과 Confusion Matrix를 계산한다.
- **합의 기반 탐지.** 6개 탐지기의 투표를 집계해 "N개 이상이 동의한 지점"만 신뢰한다. 단일 알고리즘보다 견고하다.
- **평가지표를 비판적으로.** point-wise · point-adjusted · affiliation 세 지표를 나란히 보여주고, point-adjusted가 점수를 부풀린다는 학계 비판(Kim et al. 2022)을 대시보드에서 직접 수치로 드러낸다.

---

## 실행 방법

### 로컬 실행 (권장)

```bash
# 1. 백엔드 의존성 설치
cd backend
pip install -r requirements.txt

# 2. 서버 실행 (프론트엔드까지 같은 서버가 서빙)
uvicorn app.main:app --reload --port 8000

# 3. 브라우저에서 접속
#    http://localhost:8000
```

백엔드가 `frontend/` 정적 파일을 함께 서빙하므로 **하나의 서버**로 끝난다. 별도 프론트엔드 서버가 필요 없다.

### Docker

```bash
docker compose up --build
# http://localhost:8000
```

### 프론트엔드를 따로 띄우고 싶을 때
프론트(`frontend/index.html`)를 다른 정적 호스팅에 올렸다면, 백엔드 주소를 알려준다:
```
http://your-frontend/?api=https://your-backend.com
```
또는 `index.html`에 `<script>window.HAWK_API='https://your-backend.com'</script>` 추가.

### 사용 흐름
1. **데모 데이터 분석** 버튼으로 즉시 체험하거나, CSV를 드래그&드롭.
2. 백엔드가 시간축·수치 변수 자동 감지 → 6개 탐지기 학습 → 임계값 자동 수립.
3. 대시보드에서 탐지 결과·평가지표·알고리즘 비교·판단 근거 확인.
4. 합의 임계(1~4표)·활성 탐지기를 바꾸면 즉시 재계산.

### AI 인사이트 (선택)
"AI 인사이트" 탭에서 GPT가 **탐지 결과를 해석**해 준다. 중요한 설계 원칙:
- **AI는 해석만 한다.** 이상 탐지·점수·F1 계산은 전부 HAWK가 수행하고, AI는 그 수치를 읽어 설명·우선순위·다음 단계만 제시한다.
- **환각 차단**: 시스템 프롬프트가 "제공된 수치 외의 값을 만들지 마라"를 강제. AI는 주어진 숫자만 인용한다.
- **중급자 대상**: 기본 통계 용어는 그대로 쓰되 생소한 개념(Matrix Profile, point-adjusted F1 등)은 한 줄로 풀어준다.
- **비판적 시각**: point-wise F1과 point-adjusted F1 격차가 0.15 이상이면 Kim et al.(2022) 경고를 자동 삽입.

**키 설정 — 방법 A: `.env` 파일 (권장)**
```bash
cd backend
cp .env.example .env      # .env 파일 생성
# .env 를 열어 OPENAI_API_KEY=sk-... 에 실제 키를 붙여넣고 저장
uvicorn app.main:app --port 8000
```
`.env` 의 `OPENAI_API_KEY` 줄을 실제 키(`sk-`로 시작)로 바꾸기만 하면 된다. 서버가 시작될 때 자동으로 읽는다. `.env` 는 `.gitignore` 에 등록돼 GitHub에 올라가지 않는다.

**키 설정 — 방법 B: 환경변수 직접 지정**
```bash
export OPENAI_API_KEY="sk-..."
uvicorn app.main:app --port 8000
```

키가 제대로 인식됐는지는 헤더의 `API` 표시 또는 `/api/health` 의 `openai_key_set` 값으로 확인할 수 있다.

**키가 없어도 작동한다.** 키가 없거나 placeholder 그대로면 완성된 프롬프트를 화면에 띄워주므로, "프롬프트 복사" 버튼으로 복사해 ChatGPT(GPT-4o)에 붙여넣으면 동일한 인사이트를 받을 수 있다. 호출 중 오류(크레딧 부족·키 오류 등)가 나도 사람이 읽을 수 있는 진단 메시지와 함께 프롬프트 복사 폴백을 제공한다.

---

## API 명세

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/api/health` | 상태·탐지기 목록·Darts 가용 여부 |
| POST | `/api/analyze` | CSV 업로드(multipart) → 전체 탐지·평가 결과 JSON |
| GET | `/api/demo?eval_mode=auto` | 내장 데모 데이터로 분석 |
| GET | `/api/benchmark` | 실제 라벨 벤치마크(NAB 스타일) — 합성 vs 실제 갭 측정 |
| POST | `/api/explain` | 합의 N 변경 시 근거 텍스트만 재생성(경량) |
| POST | `/api/insight` | AI 인사이트 생성 (OpenAI 해석) |

`/api/analyze` 응답은 변수별 6개 탐지기 점수·임계값·ROC, 합의 투표, 합성주입 라벨, 메타데이터를 포함한다.

---

## CSV 입력 형식

- 첫 행은 헤더. 구분자(`,` `\t` `;` `|`)는 자동 감지.
- 시간축 컬럼은 자동 감지(날짜 파싱 성공률 + 컬럼명 힌트). 없으면 행 인덱스 사용.
- 나머지 수치형 컬럼은 모두 분석 대상 변수로 인식. 결측치는 선형보간.
- 권장 길이: 50행 이상.

```csv
timestamp,temp_sensor,pressure,vibration,flow_rate
2026-01-01 00:00,22.13,101.2,0.51,80.3
2026-01-01 01:00,22.88,101.5,0.49,79.1
...
```

---

## 아키텍처

```
backend/  (FastAPI)
├── app/
│   ├── main.py              엔드포인트 + CORS + 정적 프론트 서빙
│   ├── detectors/
│   │   ├── base.py          BaseDetector — Strategy 패턴 인터페이스
│   │   └── algorithms.py    6개 탐지기 (NumPy + scikit-learn)
│   ├── core/
│   │   ├── parser.py        CSV 파싱·시간축 자동감지·데모 생성
│   │   ├── pipeline.py      합성주입·합의·3개 평가지표·임계값 스윕
│   │   └── analysis.py      전체 오케스트레이터 + 근거 텍스트 생성
│   └── schemas/models.py    Pydantic 스키마
└── requirements.txt

frontend/  (정적, 빌드 불필요)
├── index.html               대시보드 + 방법론 탭
└── app.js                   API 호출 + SVG 차트 렌더링
```

**Strategy 패턴**: 모든 탐지기는 `BaseDetector.score(series)→[0,1]` 인터페이스를 따른다. 그래서 알고리즘 추가·비교·임계값 조정이 일관되게 처리된다.

---

## 탐지 알고리즘 (References)

여섯 탐지기는 서로 다른 이상 유형(점 이상 · 맥락 이상 · 패턴 이상 · 변화점)을 포착하도록 **의도적으로 다양하게** 골랐다. 다양성이 합의 투표의 신뢰도를 높인다.

| 탐지기 | 계열 | 근거 문헌 | 왜 골랐나 |
|---|---|---|---|
| **Robust Z-Score** | 통계 | Iglewicz & Hoaglin (1993), *How to Detect and Handle Outliers* | 중앙값·MAD 기반이라 이상치가 임계값을 오염시키지 않는다. 점 이상의 가장 단순·강건한 1차 방어선. |
| **IQR Fence** | 통계 | Tukey (1977), *Exploratory Data Analysis* | 분포 가정이 없고 비대칭 분포에 강해, 임의 CSV에 안전한 기본기. |
| **Isolation Forest** | ML | Liu, Ting & Zhou (2008), *Isolation Forest*, ICDM | 무작위 분할로 고립되는 점을 찾는다. scikit-learn 정식 구현 + 슬라이딩 윈도우 특징으로 다변량·맥락 이상에 강하다. |
| **Forecast Residual (AR)** | 예측 | Box & Jenkins (1970); **수업 회귀모형 자료** | 과거값으로 다음 값을 자기회귀 예측하고 잔차가 크면 이상. 수업에서 배운 예측을 이상탐지로 직접 연결한 탐지기. 시간 구조를 명시적으로 사용. |
| **Level Shift / CUSUM** | 변화점 | Page (1954), *Continuous Inspection Schemes*, Biometrika | 좌우 윈도우 평균 차로 체제 전환(센서 고장 등)을 잡는다. 다른 탐지기가 놓치는 구조적 변화점 담당. |
| **Matrix Profile** | SOTA | Yeh et al. (2016), *Matrix Profile I*, ICDM | 시계열 이상탐지의 표준 baseline. 서브시퀀스 최근접 거리로 "닮은꼴 없는 패턴(discord)"을 탐지. 파라미터가 길이 m 하나뿐이고 도메인 가정이 없어 임의 CSV에 가장 적합. |

### 수업 내용과의 연결
업로드된 수업 자료(RNN/LSTM/GRU, Transformer, N-BEATS/N-HiTS, 회귀·트리계열 예측, Darts, 백테스팅, 공변량/chunk/lag)는 모두 **예측(Forecasting)** 이다. 본 프로젝트는 그 예측 관점을 이상탐지로 가져온다 — *예측값과 실제값의 잔차가 크면 이상*이라는 Forecast Residual 탐지기가 그 다리다. 수업의 잔차분석(Residual/ACF/Distribution)과 `check_seasonality`(자기상관 주기 추정)가 각각 잔차 기반 탐지와 Matrix Profile의 서브시퀀스 길이 추정에 그대로 쓰였다. `requirements.txt`의 Darts는 예측기반 탐지를 RNN/N-BEATS로 확장할 수 있도록 옵션으로 남겨두었다.

---

## 평가 방법론 (References & 비판)

이상탐지 평가는 라벨 희소성·구간성 때문에 단일 지표로 충분하지 않다. 세 지표를 **함께** 제시해 서로의 맹점을 드러낸다.

### Point-wise F1
각 시점을 독립적으로 정상/이상 분류로 보고 TP·FP·FN을 센다. 가장 보수적이고 해석이 명확하지만, 구간 이상에서 약간의 위치 오차에도 가혹하다.

### Point-adjusted F1 — ⚠️ 비판과 함께 제시
> Xu et al. (2018), *Unsupervised Anomaly Detection via VAE for Seasonal KPIs*, WWW — 정답 이상 구간 안에서 하나라도 탐지하면 구간 전체를 정탐으로 인정.

이 보정은 시계열 이상탐지에서 널리 쓰이지만 심각한 함정이 있다.

> **Kim et al. (2022), *Towards a Rigorous Evaluation of Time-series Anomaly Detection*, AAAI** — point-adjustment가 **무작위·자명한 예측조차** 높은 F1을 만든다는 것을 이론·실험으로 증명했다.

본 대시보드는 이 지표를 **의도적으로 표시하되 그 한계를 함께 경고**한다. 실제로 데모 데이터에서 무작위 점수(15% 무작위 발화)를 넣으면 point-wise F1은 약 0.1~0.2에 불과하지만 point-adjusted F1은 약 0.5~0.75로 **3~4배 부풀려진다**(검증 로그에서 재현됨). 이 격차를 평가 패널에서 실시간으로 보여준다 — "지표를 맹신하지 말라"는 살아있는 증거다.

### Affiliation-based F1
> Huet, Navarro & Rossi (2022), *Local Evaluation of Time Series Anomaly Detection Algorithms*, KDD

예측과 정답 이벤트 사이의 **시간적 거리**로 정밀도·재현율을 정의한다. 파라미터가 없고, "거의 맞춤"을 부분 점수로 인정해 point-wise의 가혹함과 point-adjusted의 관대함 사이를 메운다.

### Range-based 지표 (미채택, 참고)
> Tatbul et al. (2018), *Precision and Recall for Time Series*, NeurIPS — 구간의 겹침·위치·기수를 고려한 정밀도/재현율.

표현력은 크지만 파라미터(α, bias 함수 등)가 많아 affiliation을 메인 보완 지표로 채택했다.

---

## 합성 이상 주입 (평가의 토대)

라벨 없는 CSV를 정량 평가하기 위해, 데이터 표준편차에 비례한 4가지 이상을 재현 가능한 시드로 주입한다.

| 유형 | 설명 | 어떤 탐지기를 겨냥하나 |
|---|---|---|
| Spike | 단일 점 급등/급락 | Robust Z, IQR |
| Level Shift | 구간 평균이 통째로 이동 | Level Shift, Forecast |
| Variance Burst | 구간 변동성 폭발 | Isolation Forest |
| Drift | 점진적 추세 이탈 | Forecast, Matrix Profile |

이 주입 라벨이 임계값 자동 수립(F1 최대 스윕)과 모든 평가지표의 기준이 된다.

---

## 배포

백엔드가 프론트까지 서빙하므로 **컨테이너 하나**만 올리면 된다.

```bash
# Railway / Render / Fly.io 등에 Docker로 배포
docker compose up --build      # 로컬 확인
# 플랫폼에 backend Dockerfile 지정, 포트 8000 노출
```

기본 의존성(FastAPI·pandas·scikit-learn·NumPy)은 가벼워 무료 티어에서도 동작한다. (Darts/PyTorch는 `requirements.txt`에서 주석 처리 — 필요 시 활성화하되 메모리가 큰 플랜이 필요하다.)

**OpenAI 키는 배포 환경에서 따로 넣어야 한다.** `.env` 파일은 `.gitignore`에 있어 GitHub·배포 서버에 올라가지 않는다. 그래서 배포 플랫폼의 대시보드에서 환경변수로 직접 등록한다.
- Render: 서비스 → Environment → `OPENAI_API_KEY` 추가
- Railway: 프로젝트 → Variables → `OPENAI_API_KEY` 추가
- Fly.io: `fly secrets set OPENAI_API_KEY=sk-...`

키를 넣지 않아도 앱은 정상 동작하며, AI 인사이트 탭만 "프롬프트 복사" 모드로 바뀐다.

### 배포 전 체크리스트 / 흔한 오류
- **AI 인사이트가 "키 없음"으로 뜸** → 배포 플랫폼 환경변수에 `OPENAI_API_KEY`를 등록했는지 확인. 등록 후 재배포 필요. `/api/health`의 `openai_key_set`으로 확인 가능.
- **빌드 중 메모리 초과로 죽음** → 무료 티어(512MB)에서 가끔 발생. Dockerfile 빌드는 `--no-cache-dir`로 이미 최소화돼 있으나, 그래도 실패하면 한 단계 위 플랜을 쓰거나 numpy/pandas 버전을 고정한다.
- **첫 요청이 느림(콜드 스타트)** → 무료 티어는 유휴 시 슬립된다. 첫 분석이 10~30초 걸릴 수 있으나 정상이다.
- **`darts_available: false`** → 정상이다. Darts는 선택 의존성이며, 없어도 6개 탐지기는 전부 동작한다.

배포가 실패할 경우를 대비해 전체 소스를 zip으로도 제출한다. `pip install -r requirements.txt && uvicorn app.main:app` 두 줄로 어디서든 실행된다.

---

## 알려진 한계

- 합성 주입 평가는 *주입한 유형*에 대한 탐지력만 측정한다. 실제 도메인 이상이 이와 다르면 지표가 그대로 옮겨지지 않을 수 있다(affiliation·point-wise 병기로 과신 방지).
- 다변량 상관 이상(여러 변수가 함께 비정상)은 변수별 독립 탐지 후 합산하는 방식이라 부분적으로만 포착된다.
- Matrix Profile은 O(n²)이라 수만 점 이상에서는 느려질 수 있다(수천 점까지 쾌적).

---

## 참고문헌

- Iglewicz, B., & Hoaglin, D. (1993). *How to Detect and Handle Outliers*. ASQC Quality Press.
- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM*.
- Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*.
- Page, E. S. (1954). Continuous Inspection Schemes. *Biometrika*.
- Yeh, C.-C. M., et al. (2016). Matrix Profile I: All Pairs Similarity Joins for Time Series. *ICDM*.
- Xu, H., et al. (2018). Unsupervised Anomaly Detection via VAE for Seasonal KPIs in Web Applications. *WWW*.
- Kim, S., et al. (2022). Towards a Rigorous Evaluation of Time-series Anomaly Detection. *AAAI*.
- Huet, A., Navarro, J. M., & Rossi, D. (2022). Local Evaluation of Time Series Anomaly Detection Algorithms. *KDD*.
- Tatbul, N., et al. (2018). Precision and Recall for Time Series. *NeurIPS*.
