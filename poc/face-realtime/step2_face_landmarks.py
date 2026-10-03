"""2단계: MediaPipe Face Landmarker로 얼굴 랜드마크와 고개 회전 각도(Yaw/Pitch/Roll)를 추출한다.

키: c = 지금 자세를 0°로 보정 / s = 화면 캡처 저장 / q = 종료

실행 예)
  python step2_face_landmarks.py                 # 웹캠 (Windows 권장)
  python step2_face_landmarks.py --log           # 프레임별 각도·표정 점수를 CSV로 기록
  python step2_face_landmarks.py --image me.jpg  # 웹캠이 없을 때 사진 1장 분석 (WSL 등)
"""
import argparse
import csv
import math
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
             "face_landmarker/face_landmarker/float16/latest/face_landmarker.task")
ANGLES = ("yaw", "pitch", "roll")
WINDOW = "Step2 - Face Landmarker (c: calibrate, s: save, q: quit)"


def ensure_model():
    """얼굴 모델 파일이 없으면 Google 공식 저장소에서 내려받는다."""
    if not MODEL_PATH.exists():
        print("얼굴 모델(face_landmarker.task) 다운로드 중...")
        MODEL_PATH.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_landmarker(video):
    """FaceLandmarker 생성. 웹캠은 VIDEO 모드(프레임 간 추적), 사진은 IMAGE 모드."""
    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_faces=1,
        output_face_blendshapes=True,               # 표정 점수 (깜빡임, 미소 등)
        output_facial_transformation_matrixes=True,  # 고개 회전 계산용 4x4 행렬
    )
    return vision.FaceLandmarker.create_from_options(options)


def to_mp_image(frame):
    """OpenCV(BGR) 프레임 → MediaPipe 이미지(RGB)"""
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


def head_angles(matrix):
    """4x4 변환 행렬의 회전 부분 → (yaw, pitch, roll) 각도(°)"""
    r = np.array(matrix)[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def analyze(result):
    """탐지 결과 → 각도·표정 점수 dict. 얼굴이 없으면 None."""
    if not result.face_landmarks:
        return None
    yaw, pitch, roll = head_angles(result.facial_transformation_matrixes[0])
    s = {c.category_name: c.score for c in result.face_blendshapes[0]}
    return {
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "blink_l": s["eyeBlinkLeft"], "blink_r": s["eyeBlinkRight"],
        "smile": (s["mouthSmileLeft"] + s["mouthSmileRight"]) / 2,
    }


def put(frame, text, y, color=(255, 255, 255)):
    """검은 테두리가 있는 글자 (밝은 배경에서도 잘 보이게)"""
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw(frame, landmarks, info, angles, calibrated):
    """랜드마크 점과 각도·표정 점수를 프레임에 그린다. (FPS·지연 표시는 4단계에서)"""
    h, w = frame.shape[:2]
    for p in landmarks:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
    put(frame, f"Yaw   {angles['yaw']:+6.1f}", 30, (0, 255, 255))
    put(frame, f"Pitch {angles['pitch']:+6.1f}", 58, (0, 255, 255))
    put(frame, f"Roll  {angles['roll']:+6.1f}", 86, (0, 255, 255))
    put(frame, f"Blink L {info['blink_l']:.2f}  R {info['blink_r']:.2f}", 120)
    put(frame, f"Smile {info['smile']:.2f}", 148)
    put(frame, "calibrated" if calibrated else "press c to calibrate", h - 15,
        (0, 255, 0) if calibrated else (200, 200, 200))


def run_image(path):
    """사진 1장을 분석해 각도·표정 점수를 출력하고 결과 이미지를 저장한다."""
    frame = cv2.imread(path)
    if frame is None:
        sys.exit(f"이미지를 읽을 수 없습니다: {path}")
    with create_landmarker(video=False) as landmarker:
        result = landmarker.detect(to_mp_image(frame))
    info = analyze(result)
    if info is None:
        sys.exit("얼굴을 찾지 못했습니다.")
    print("  ".join(f"{k}={v:+.1f}°" for k, v in info.items() if k in ANGLES))
    print(f"blink_l={info['blink_l']:.2f}  blink_r={info['blink_r']:.2f}  smile={info['smile']:.2f}")
    draw(frame, result.face_landmarks[0], info, info, calibrated=False)
    out = RESULTS_DIR / "step2_image.png"
    cv2.imwrite(str(out), frame)
    print(f"저장 완료: {out}")


def run_webcam(camera, log):
    """웹캠 영상에서 실시간으로 랜드마크·각도를 표시한다."""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit("웹캠을 열 수 없습니다. WSL이라면 Windows에서 실행하거나 --image 옵션을 사용하세요.")

    # --log: 프레임별 각도(보정 전/후)·표정 점수 기록
    writer, log_file = None, None
    if log:
        log_file = open(RESULTS_DIR / "step2_angles.csv", "w", newline="", encoding="utf-8")
        writer = csv.writer(log_file)
        writer.writerow(["time_s", "yaw_raw", "pitch_raw", "roll_raw", "yaw", "pitch", "roll",
                         "blink_l", "blink_r", "smile"])

    baseline = dict.fromkeys(ANGLES, 0.0)  # c 키로 잡은 기준 자세
    calibrated, start, last_ts = False, time.perf_counter(), -1
    with create_landmarker(video=True) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드: 내가 움직이는 방향 = 화면 방향

            # VIDEO 모드는 타임스탬프(ms)가 계속 증가해야 함
            ts = max(int((time.perf_counter() - start) * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(to_mp_image(frame), ts)
            info = analyze(result)

            if info:
                angles = {k: info[k] - baseline[k] for k in ANGLES}
                draw(frame, result.face_landmarks[0], info, angles, calibrated)
                if writer:
                    writer.writerow([f"{ts / 1000:.3f}"]
                                    + [f"{info[k]:.2f}" for k in ANGLES]
                                    + [f"{angles[k]:.2f}" for k in ANGLES]
                                    + [f"{info[k]:.3f}" for k in ("blink_l", "blink_r", "smile")])
            else:
                put(frame, "No face", 30, (0, 0, 255))

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c") and info:
                baseline = {k: info[k] for k in ANGLES}
                calibrated = True
                print("기준 자세 보정: " + "  ".join(f"{k}={v:+.1f}°" for k, v in baseline.items()))
            if key == ord("s"):
                out = RESULTS_DIR / "step2_landmarks.png"
                cv2.imwrite(str(out), frame)
                print(f"캡처 저장: {out}")

    cap.release()
    cv2.destroyAllWindows()
    if log_file:
        log_file.close()
        print(f"각도 기록 저장: {RESULTS_DIR / 'step2_angles.csv'}")


def main():
    parser = argparse.ArgumentParser(description="MediaPipe Face Landmarker 랜드마크·고개 각도 추출")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    parser.add_argument("--image", help="웹캠 대신 분석할 이미지 파일 경로")
    parser.add_argument("--log", action="store_true", help="프레임별 각도·표정 점수를 results/step2_angles.csv에 기록")
    args = parser.parse_args()

    ensure_model()
    RESULTS_DIR.mkdir(exist_ok=True)
    if args.image:
        run_image(args.image)
    else:
        run_webcam(args.camera, args.log)


if __name__ == "__main__":
    main()
