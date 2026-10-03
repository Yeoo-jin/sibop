"""3단계: 규칙 기반 실시간 코칭 (고개 돌림·숙임, 깜빡임 과다, 얼굴 이탈)

시작 후 2초 동안 정면 자세를 자동 보정한 뒤, 규칙에 걸리면 한글 경고를 띄운다.
키: c = 다시 보정 / s = 화면 캡처 저장 / q = 종료

실행 예)
  python step3_coaching.py         # 웹캠 (Windows 권장)
  python step3_coaching.py --log   # 경고 기록을 results/step3_events.csv에 저장
"""
import argparse
import csv
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from step2_face_landmarks import (ANGLES, RESULTS_DIR, analyze, create_landmarker,
                                  ensure_model, put, to_mp_image)

# --- 규칙 기준값 (2단계 실측으로 정함, RESULTS.md 참고) ---
YAW_LIMIT = 20.0        # 좌우로 이 각도(°) 이상 돌리면
PITCH_LIMIT = 15.0      # 이 각도(°) 이상 숙이면
HOLD_SEC = 2.0          # 위 상태(또는 얼굴 이탈)가 이 시간 이상 이어지면 경고
BLINK_CLOSE = 0.5       # 두 눈 평균 점수가 이 값 이상이면 '감음'
BLINK_OPEN = 0.3        # 이 값 미만으로 내려가면 '뜸' (고개를 돌리면 뜬 눈도 0.2~0.28까지 올라감)
BLINK_MAX_SEC = 0.5     # 이보다 오래 감으면 깜빡임이 아니라 '눈 감음' → 세지 않음
BLINK_WINDOW_SEC = 60.0
BLINK_LIMIT = 25        # 최근 1분 깜빡임이 이 횟수를 넘으면 경고
MIN_SHOW_SEC = 1.0      # 상태가 풀려도 경고를 이 시간만큼 더 보여줌 (깜빡거림 방지)
CALIB_SEC = 2.0         # 시작 후 이 시간 동안의 평균 자세를 정면(0°)으로 잡음
CALIB_MAX_YAW = 15.0    # 보정 중 카메라 기준 좌우 각도가 이보다 크면 정면이 아니라고 보고 다시 보정
CALIB_MAX_PITCH = 25.0  # Pitch는 카메라 높이에 따라 원래 클 수 있어 넉넉하게
CALIB_MAX_STD = 3.0     # 보정 중 각도가 이보다 흔들리면(표준편차 °) 다시 보정

ALERTS = {
    "away": "화면 안으로 들어와 주세요",
    "turn": "정면을 보세요",
    "down": "고개를 드세요",
    "blink": "긴장 상태입니다 (깜빡임 과다)",
}
CALIB_MSG = "정면을 보고 잠시 기다려 주세요 (보정 중)"
FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]
WINDOW = "Step3 - Coaching (c: recalibrate, s: save, q: quit)"


class Calibrator:
    """CALIB_SEC 동안 모은 각도의 평균을 정면(0°) 기준으로 잡는다.
    고개를 돌리고 있거나 움직이면 확정하지 않고, 최근 CALIB_SEC 동안 정면으로 가만히 있을 때까지 기다린다."""

    def __init__(self):
        self.baseline = dict.fromkeys(ANGLES, 0.0)
        self.restart()

    def restart(self):
        self.samples, self.t0, self.active = deque(), None, True

    def feed(self, t, info):
        if self.t0 is None:
            self.t0 = t
        self.samples.append((t, [info[k] for k in ANGLES]))
        while t - self.samples[0][0] > CALIB_SEC:  # 최근 CALIB_SEC만 남김
            self.samples.popleft()
        if t - self.t0 < CALIB_SEC:
            return
        values = np.array([v for _, v in self.samples])
        mean, std = values.mean(axis=0), values.std(axis=0)
        yaw, pitch = mean[ANGLES.index("yaw")], mean[ANGLES.index("pitch")]
        if abs(yaw) > CALIB_MAX_YAW or abs(pitch) > CALIB_MAX_PITCH or std.max() > CALIB_MAX_STD:
            return  # 아직 정면이 아니거나 흔들림 → 다음 프레임에 최근 2초로 다시 확인
        self.baseline = dict(zip(ANGLES, mean))
        self.active = False

    def apply(self, info):
        """보정된 각도 = 현재 각도 − 기준 각도"""
        return {k: info[k] - self.baseline[k] for k in ANGLES}


class CoachRules:
    """프레임마다 각도·깜빡임을 받아 켜진 경고 목록을 돌려준다."""

    def __init__(self):
        self.since = {}           # 조건이 연속으로 참이 되기 시작한 시각
        self.held = {}            # 조건이 이어진 시간(초) — 화면 표시용
        self.last_on = {}         # 경고 조건이 마지막으로 참이던 시각
        self.blinks = deque()     # 최근 1분 깜빡임 시각
        self.total_blinks = 0
        self.closed_at = None     # 눈을 감기 시작한 시각
        self.open_events = {}     # 진행 중인 경고: 이름 → 시작 시각
        self.events = []          # 끝난 경고: (이름, 시작, 끝)

    @property
    def blink_count(self):
        return len(self.blinks)

    def reset_timers(self):
        """다시 보정할 때 지속 시간 타이머를 초기화"""
        self.since.clear()

    def _hold(self, name, cond, t):
        """cond가 연속으로 참인 시간(초). 거짓이 되면 0부터 다시 잰다."""
        if not cond:
            self.since.pop(name, None)
            return 0.0
        self.since.setdefault(name, t)
        return t - self.since[name]

    def _count_blink(self, score, t):
        """감음(≥0.5) → 뜸(<0.3) 한 번을 깜빡임 1회로 센다. 두 기준을 달리 둬서 흔들림에 중복으로 세지 않음."""
        if self.closed_at is None and score >= BLINK_CLOSE:
            self.closed_at = t
        elif self.closed_at is not None and score < BLINK_OPEN:
            if t - self.closed_at <= BLINK_MAX_SEC:
                self.blinks.append(t)
                self.total_blinks += 1
            self.closed_at = None
        while self.blinks and t - self.blinks[0] > BLINK_WINDOW_SEC:
            self.blinks.popleft()

    def update(self, t, angles, info):
        """한 프레임 판정. angles(보정된 각도)·info가 None이면 얼굴 없음."""
        face = angles is not None
        self.held = {
            "away": self._hold("away", not face, t),
            "turn": self._hold("turn", face and abs(angles["yaw"]) >= YAW_LIMIT, t),
            "down": self._hold("down", face and angles["pitch"] >= PITCH_LIMIT, t),
        }
        if face:
            self._count_blink((info["blink_l"] + info["blink_r"]) / 2, t)
        else:
            self._count_blink(0.0, t)  # 얼굴이 없어도 1분 지난 깜빡임은 정리

        on = {k for k, sec in self.held.items() if sec >= HOLD_SEC}
        if self.blink_count > BLINK_LIMIT:
            on.add("blink")
        for k in on:
            self.last_on[k] = t

        active = [k for k in ALERTS if k in self.last_on and t - self.last_on[k] < MIN_SHOW_SEC]
        self._track_events(active, t)
        return active

    def _track_events(self, active, t):
        for k in active:
            self.open_events.setdefault(k, t)
        for k in [k for k in self.open_events if k not in active]:
            self.events.append((k, self.open_events.pop(k), t))

    def finish(self, t):
        """종료 시 진행 중인 경고를 마무리해 기록"""
        self._track_events([], t)


def render_label(text, color):
    """한글 문구를 반투명 박스가 있는 이미지로 미리 만들어 둔다. (프레임마다 Pillow를 쓰면 느림)"""
    font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)
    font = ImageFont.truetype(font_path, 30) if font_path else ImageFont.load_default(30)
    x0, y0, x1, y1 = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=font)
    pad = 14
    img = Image.new("RGBA", (x1 - x0 + pad * 2, y1 - y0 + pad * 2), color + (190,))
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=(255, 255, 255, 255))
    rgba = np.array(img)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (rgba[:, :, 3:] / 255.0).astype(np.float32)
    return bgr, alpha


def paste(frame, label, y):
    """미리 만든 문구 이미지를 프레임 가운데(세로 위치 y)에 합성"""
    bgr, alpha = label
    h, w = bgr.shape[:2]
    fh, fw = frame.shape[:2]
    x = max((fw - w) // 2, 0)
    w, h = min(w, fw - x), min(h, fh - y)
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    frame[y:y + h, x:x + w] = (bgr[:h, :w] * alpha[:h, :w] + roi * (1 - alpha[:h, :w])).astype(np.uint8)


def draw_status(frame, angles, rules):
    """좌측 상단에 각도·지속 시간·깜빡임 횟수를 작게 표시 (영어: 속도 유지)"""
    held = rules.held
    put(frame, f"Yaw {angles['yaw']:+5.1f}  Turn {held.get('turn', 0):.1f}/{HOLD_SEC:.0f}s", 28, (0, 255, 255))
    put(frame, f"Pitch {angles['pitch']:+5.1f}  Down {held.get('down', 0):.1f}/{HOLD_SEC:.0f}s", 56, (0, 255, 255))
    put(frame, f"Blinks/min {rules.blink_count}/{BLINK_LIMIT}", 84)


def save_events(rules):
    path = RESULTS_DIR / "step3_events.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀에서 한글 안 깨지게
        writer = csv.writer(f)
        writer.writerow(["alert", "message", "start_s", "end_s", "duration_s"])
        for name, start, end in rules.events:
            writer.writerow([name, ALERTS[name], f"{start:.2f}", f"{end:.2f}", f"{end - start:.2f}"])
    print(f"경고 기록 저장: {path} ({len(rules.events)}건, 총 깜빡임 {rules.total_blinks}회)")


def main():
    parser = argparse.ArgumentParser(description="규칙 기반 실시간 면접 코칭")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    parser.add_argument("--log", action="store_true", help="경고 기록을 results/step3_events.csv에 저장")
    args = parser.parse_args()

    ensure_model()
    RESULTS_DIR.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("웹캠을 열 수 없습니다. WSL이라면 Windows에서 실행하세요.")

    labels = {k: render_label(msg, (200, 30, 30)) for k, msg in ALERTS.items()}
    calib_label = render_label(CALIB_MSG, (30, 90, 200))
    cal, rules = Calibrator(), CoachRules()
    start, last_ts, t = time.perf_counter(), -1, 0.0

    with create_landmarker(video=True) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)  # 거울 모드
            t = time.perf_counter() - start
            ts = max(int(t * 1000), last_ts + 1)  # VIDEO 모드는 타임스탬프가 계속 증가해야 함
            last_ts = ts
            result = landmarker.detect_for_video(to_mp_image(frame), ts)
            info = analyze(result)

            h = frame.shape[0]
            if cal.active:
                # 보정 중: 얼굴이 보일 때만 샘플을 모으고 규칙 판정은 쉼
                if info:
                    cal.feed(t, info)
                paste(frame, calib_label, h - 70)
            else:
                angles = cal.apply(info) if info else None
                active = rules.update(t, angles, info)
                if angles:
                    draw_status(frame, angles, rules)
                for i, k in enumerate(active):  # 경고는 아래쪽부터 위로 쌓음
                    paste(frame, labels[k], h - 70 - i * 62)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                cal.restart()
                rules.reset_timers()
                print("다시 보정합니다. 정면을 봐 주세요.")
            if key == ord("s"):
                out = RESULTS_DIR / "step3_coaching.png"
                cv2.imwrite(str(out), frame)
                print(f"캡처 저장: {out}")

    cap.release()
    cv2.destroyAllWindows()
    rules.finish(t)
    if args.log:
        save_events(rules)


if __name__ == "__main__":
    main()
