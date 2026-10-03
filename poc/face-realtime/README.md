# 얼굴 움직임 실시간 탐지 PoC

AI 면접/발표 실시간 코칭 서비스를 위한 PoC입니다. 웹캠 영상에서 고개 방향, 시선, 표정을 실시간으로 탐지하는 것이 목표입니다.

> **결과 요약은 [SUMMARY.md](SUMMARY.md)에 있습니다.** 처음 보는 분은 여기부터 읽어 주세요.

| 단계 | 내용 | 상태 |
|---|---|---|
| 1 | AI(비전 모델)로 프레임을 해석할 때의 지연 측정 | ✅ `step1_ai_latency.py` · [사용법](#1단계-ai-분석-지연-측정) · [결과](RESULTS.md#1단계-ai-분석-지연) |
| 2 | MediaPipe Face Landmarker로 랜드마크·고개 회전 각도 추출 | ✅ `step2_face_landmarks.py` · [사용법](#2단계-얼굴-랜드마크고개-각도) · [결과](RESULTS.md#2단계-mediapipe-face-landmarker) |
| 3 | 규칙 기반 탐지 (고개 돌림·숙임, 깜빡임 과다, 얼굴 이탈) | ✅ `step3_coaching.py` · [사용법](#3단계-규칙-기반-실시간-코칭) · [결과](RESULTS.md#3단계-규칙-기반-탐지) |
| 4 | FPS·지연(ms)을 화면에 표시하고 1단계와 비교 | ✅ `step4_benchmark.py` · [사용법](#4단계-실시간-처리-속도-측정과-1단계-비교) · [결과](RESULTS.md#4단계-처리-속도-비교) |
| 5 | 결과 캡처와 사용 기술 정리 | ✅ [요약 문서](SUMMARY.md) (결론 · 단계별 정리 · 사용 기술 · 한계 · 다음 할 일) |

---

## 1단계: AI 분석 지연 측정

웹캠 프레임 1장을 JPEG로 인코딩해 **Gemini (`gemini-3.5-flash-lite`)** 에 보냅니다. 고개 방향·시선·표정을 JSON으로 받는 과정을 **10회** 반복하며 다음 세 가지 시간을 잽니다.

- (a) 캡처 + JPEG 인코딩 시간
- (b) API 요청부터 응답까지 걸린 시간
- (c) 전체 시간 (a + b)

### 결과물 (`results/`)
- 터미널: 회차별 표와 평균/최소/최대
- `ai_latency.csv`: 회차별 측정값과 AI 응답 원문
- `ai_latency_capture.png`: 마지막 프레임 위에 "AI 분석 평균 지연: X.XX초"와 AI 응답을 표시한 이미지

---

## ⚠️ WSL과 웹캠

WSL2(Ubuntu)에서는 기본적으로 웹캠(`/dev/video*`)이 보이지 않습니다. 실행 방법은 아래 중 하나를 고르세요.

| 방법 | 웹캠 | 설명 |
|---|---|---|
| **A. Windows Python으로 실행 (권장)** | ✅ | 웹캠이 바로 잡힙니다. 2~4단계의 실시간 화면도 같은 방법으로 실행합니다. |
| B. WSL에서 `--image` 옵션으로 실행 | ❌ | 사진 파일로 대신합니다. 1단계는 API 지연 (b)만 의미가 있고, 2단계는 사진 1장의 각도만 확인할 수 있습니다. |
| C. usbipd-win으로 웹캠을 WSL에 연결 | △ | 기본 WSL 커널에는 웹캠 드라이버(uvcvideo)가 없어 커널을 다시 빌드해야 할 수 있습니다. 권장하지 않습니다. |

---

## 준비: API 키

1. [Google AI Studio](https://aistudio.google.com/apikey)에서 API 키를 발급받습니다.
2. `.env.example`을 `.env`로 복사하고 키를 넣습니다. `.env`는 `.gitignore`에 들어 있어 커밋되지 않습니다.

```
GEMINI_API_KEY=발급받은_키
```

---

## A. Windows에서 실행 (권장)

Windows에 Python 3.10 이상이 설치되어 있어야 합니다(`py --version`으로 확인).

이 폴더는 WSL 안에 있으므로 Windows 쪽으로 복사해서 쓰는 것을 권장합니다. WSL 경로에서 바로 가상환경을 만들면 느립니다.

```bat
:: 명령 프롬프트(cmd). WSL의 venv는 Linux용이라 복사에서 제외(/XD)
robocopy "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime" C:\dev\face-realtime /E /XD venv __pycache__ results
cd /d C:\dev\face-realtime

py -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

python step1_ai_latency.py
```

> - OneDrive 폴더(바탕 화면 등)는 피하세요. venv 파일 수천 개가 동기화됩니다.
> - PowerShell이라면 활성화만 `.\venv\Scripts\Activate.ps1`로 바꾸면 됩니다. 실행 정책 오류가 나면 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`를 실행하세요.
> - `.env`도 함께 복사되었는지 확인하세요. 측정이 끝나면 `results/` 폴더를 WSL 쪽으로 다시 복사해 두면 됩니다.

## B. WSL(Ubuntu)에서 실행 (`--image`)

```bash
cd ~/sibop/poc/face-realtime
sudo apt install -y python3-venv       # 처음 한 번만 (venv 생성 오류가 날 때)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

python step1_ai_latency.py --image 내얼굴.jpg
```

> `import cv2`에서 `libGL.so.1` 오류가 나면 `sudo apt install -y libgl1`을 실행하세요.

---

## 1단계 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | gemini-3.5-flash-lite | 사용할 Gemini 모델 |
| `--runs` | 10 | 반복 횟수 |
| `--camera` | 0 | 웹캠 번호. 외장 캠이면 1, 2 … |
| `--image` | 없음 | 웹캠 대신 쓸 이미지 파일 |
| `--interval` | 4.0 | 회차 사이 대기(초). 측정값에는 포함되지 않습니다. |

## 1단계 참고

- **요청 한도**: 무료 등급은 분당 요청 수가 제한되어 있어 회차 사이에 4초씩 쉽니다. 그래도 429(요청 한도 초과)나 503(서버 혼잡)이 나면 30초 기다린 뒤 해당 회차를 처음부터 다시 측정합니다(최대 5번). 쉬는 시간은 측정값에 들어가지 않으며, 5번을 넘기면 그때까지 측정한 회차만 저장합니다.
- **1회차 지연**: 1회차는 연결을 맺는 시간이 더해져 더 느릴 수 있습니다. 실제 사용 환경과 같다고 보고 측정값에서 빼지 않았습니다.
- **모델**: 원래 `gemini-2.5-flash`로 계획했지만, 신규 사용자는 404(사용 불가)가 납니다. API가 안내한 `gemini-3.8-flash`는 503(서버 혼잡)이 잦고 응답이 20~40초 걸려서, `gemini-3.5-flash-lite`를 기본값으로 씁니다. `--model`로 바꿀 수 있습니다.
- **thinking**: 모델에 따라 thinking(추론)이 기본으로 켜져 있으면 그 시간도 응답 시간에 포함됩니다.
- **한글 표시**: OpenCV `putText`는 한글을 그리지 못하므로 Pillow로 그립니다. 폰트는 맑은 고딕을 먼저 찾고, 없으면 나눔고딕(`sudo apt install fonts-nanum`)을 씁니다.

---

## 2단계: 얼굴 랜드마크·고개 각도

**MediaPipe Face Landmarker**로 웹캠 영상에서 얼굴 점 478개를 찾습니다. 화면에는 다음 값을 실시간으로 표시합니다.
- 고개 각도: **Yaw(좌우)·Pitch(위아래)·Roll(기울기)**
- 표정 점수: **눈 깜빡임(L/R)·미소** (0~1)

API 키는 필요 없고, 모든 처리가 내 PC 안에서 돌아갑니다.

### Windows에 반영하기 (1단계 환경이 이미 있을 때)
OpenCV가 `opencv-contrib-python`으로 바뀌었습니다. 기존 `opencv-python`을 먼저 지워야 충돌하지 않습니다.

```bat
cd /d C:\dev\face-realtime
venv\Scripts\activate
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\*.py" .
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\requirements.txt" .
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\README.md" .
pip uninstall -y opencv-python
pip install -r requirements.txt

python step2_face_landmarks.py --log
```

처음 실행하면 얼굴 모델(`models/face_landmarker.task`, 약 4MB)을 자동으로 내려받습니다.

### 키 조작
| 키 | 동작 |
|---|---|
| `c` | 지금 자세를 0°로 보정합니다. 카메라를 정면으로 본 상태에서 누르세요. |
| `s` | 지금 화면을 `results/step2_landmarks.png`로 저장합니다. |
| `q` | 종료합니다. |

### 옵션
| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--camera` | 0 | 웹캠 번호 |
| `--log` | 꺼짐 | 프레임마다 시간·각도(보정 전/후)·표정 점수를 `results/step2_angles.csv`에 기록합니다. |
| `--image` | 없음 | 사진 1장을 분석해 값을 출력하고 `results/step2_image.png`를 저장합니다. 웹캠이 없는 WSL에서 씁니다. |

### 값 읽는 법
- 화면은 **거울 모드**(좌우 반전)입니다. 내가 움직이는 방향과 화면 속 방향이 같습니다.
- **Roll**: 화면에서 얼굴이 반시계 방향으로 기울면 +입니다.
- **Yaw**: 왼쪽으로 돌리면 −, 오른쪽으로 돌리면 +입니다. **Pitch**: 고개를 숙이면 +입니다. 실측값은 [RESULTS.md](RESULTS.md)에 있습니다.
- 노트북 카메라는 보통 아래에서 올려다보기 때문에, 정면을 봐도 Pitch가 0°가 아닐 수 있습니다. 그래서 시작할 때 `c`로 보정하는 것을 권장합니다.
- 화면 글자는 영어로 씁니다. 프레임마다 Pillow로 한글을 그리면 속도가 떨어지기 때문입니다.

---

## 3단계: 규칙 기반 실시간 코칭

2단계의 각도와 깜빡임 점수를 아래 규칙으로 판정하고, 걸리면 화면에 **한글 경고**를 띄웁니다.

| 경고 | 조건 |
|---|---|
| 정면을 보세요 | 좌우로 **20° 이상** 돌린 상태가 **2초** 이어짐 |
| 고개를 드세요 | **15° 이상** 숙인 상태가 **2초** 이어짐 |
| 긴장 상태입니다 (깜빡임 과다) | 최근 **1분** 동안 깜빡임이 **25회 초과** |
| 화면 안으로 들어와 주세요 | 얼굴이 **2초** 이상 보이지 않음 |

- **깜빡임 1회 세는 법**: 두 눈 점수의 평균이 0.5 이상(감음)이 됐다가 0.3 미만(뜸)으로 돌아오면 1회입니다. 감은 시간이 0.5초를 넘으면 깜빡임이 아니라 일부러 감은 것으로 보고 세지 않습니다.
- **경고가 사라지는 시점**: 자세가 돌아온 뒤 1초가 지나면 사라집니다.
- **기준값을 바꾸려면**: [step3_coaching.py](step3_coaching.py) 맨 위의 상수를 고치면 됩니다. 기준값을 정한 근거는 [RESULTS.md](RESULTS.md)에 있습니다.

### 자동 보정
시작하면 "정면을 보고 잠시 기다려 주세요" 문구가 뜹니다. **정면을 보고 2초 동안 가만히 있으면** 그 자세가 0°로 잡힙니다. 다만 아래 경우에는 보정을 확정하지 않고 계속 기다립니다.
- 고개를 돌리고 있을 때 (카메라 기준 좌우 15° 초과)
- 자세가 흔들릴 때 (2초 동안 3° 넘게 움직임)

이렇게 해서, 딴 데를 보는 자세가 "정면"으로 잡히는 일을 막습니다. 실행 중에 `c`를 누르면 다시 보정합니다.

### Windows에서 실행
```bat
cd /d C:\dev\face-realtime
venv\Scripts\activate
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\*.py" .
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\requirements.txt" .
pip install -r requirements.txt

python step3_coaching.py --log
```

| 키 / 옵션 | 동작 |
|---|---|
| `c` | 다시 보정합니다. |
| `s` | 지금 화면을 `results/step3_coaching.png`로 저장합니다. |
| `q` | 종료합니다. |
| `--log` | 종료할 때 경고 기록(경고 종류, 시작·끝 시각, 지속 시간)과 총 깜빡임 횟수를 `results/step3_events.csv`에 저장합니다. |

### 테스트
규칙과 보정은 웹캠 없이 테스트할 수 있습니다. 2단계 실측 기록(`results/step2_angles.csv`)을 재생하는 테스트도 들어 있습니다.

```bash
pytest -q
```

---

## 4단계: 실시간 처리 속도 측정과 1단계 비교

3단계 코칭 화면을 그대로 띄우고, 오른쪽 위에 **FPS와 처리 시간(ms)** 을 표시합니다. 보정이 끝나면 **30초** 동안 측정하고 자동으로 종료합니다. 그다음 1단계 Gemini 결과(`results/ai_latency.csv`)와 비교합니다.

### 무엇을 재나
| 구간 | 내용 | 1단계에서 대응하는 값 |
|---|---|---|
| ① 캡처 | `cap.read()`: 웹캠이 다음 프레임을 줄 때까지 기다리는 시간 | (a) 캡처+인코딩 |
| ② 얼굴 분석 | 좌우 반전, 변환, MediaPipe 분석 | (b) API 응답 |
| ③ 규칙+그리기 | 3단계 규칙 판정과 경고 그리기 | — |
| ④ 처리 합계 | ②+③ | (c) 전체 |

- **① 캡처는 처리 합계에 넣지 않습니다.** 웹캠이 30 FPS면 우리 코드가 아무리 빨라도 약 33ms를 기다리기 때문입니다.
- **화면 FPS**는 웹캠 한계(보통 30)에서 멈춥니다. **처리 가능 FPS**(= 1000 ÷ 처리 합계 평균)는 웹캠이 더 빠르다면 얼마까지 처리할 수 있는지를 보여 줍니다.
- 각 구간은 평균, 중앙값, **p95**(95번째 백분위), 최대를 냅니다. p95는 "프레임 20개 중 19개는 이 시간 안에 처리됐다"는 뜻이라, 실시간 서비스에서는 평균보다 중요합니다.
- 맨 처음 프레임은 모델을 처음 불러오느라 느려서, 측정 구간 밖에서 따로 출력합니다.

### Windows에서 실행
```bat
cd /d C:\dev\face-realtime
venv\Scripts\activate
copy /Y "\\wsl$\Ubuntu\home\yeojin\sibop\poc\face-realtime\*.py" .
python step4_benchmark.py
```

정면을 보고 2초 동안 가만히 있으면 보정이 끝나고, 화면에 `REC 0.0/30s`가 표시되며 측정이 시작됩니다. 측정 중에는 평소처럼 계시면 됩니다. 경고가 떠도 괜찮습니다.

| 키 / 옵션 | 동작 |
|---|---|
| `c` | 다시 보정합니다. 측정도 처음부터 다시 합니다. |
| `q` | 측정을 중단합니다. 그때까지 잰 결과는 저장합니다. |
| `--seconds` | 측정 시간(초). 기본값은 30입니다. |

### 결과물 (`results/`)
- `step4_frames.csv`: 프레임별 구간 시간(ms)과 FPS
- `step4_comparison.csv`: 1단계 Gemini와의 비교표
- `step4_comparison.png`: 비교 막대그래프 (처리 시간은 로그 축, FPS는 일반 축)
- `step4_capture.png`: 측정 중간(15초)에 자동으로 저장한 화면
