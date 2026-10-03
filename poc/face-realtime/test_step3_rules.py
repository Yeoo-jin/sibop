"""3단계 규칙(CoachRules)·보정(Calibrator) 테스트. 실행: pytest -q"""
import csv
from pathlib import Path

import pytest

from step3_coaching import Calibrator, CoachRules

FPS = 30


class Feeder:
    """가짜 프레임을 30fps로 규칙에 넣어 주는 도우미"""

    def __init__(self):
        self.rules, self.t, self.active = CoachRules(), 0.0, []

    def run(self, sec, yaw=0.0, pitch=0.0, blink=0.0, face=True):
        for _ in range(round(sec * FPS)):
            self.frame(yaw, pitch, blink, face)
        return self.active

    def frame(self, yaw=0.0, pitch=0.0, blink=0.0, face=True):
        self.t += 1 / FPS
        angles = {"yaw": yaw, "pitch": pitch} if face else None
        info = {"blink_l": blink, "blink_r": blink} if face else None
        self.active = self.rules.update(self.t, angles, info)
        return self.active

    def blink(self, closed_sec=0.15):
        """눈 감음(0.8) → 뜸(0.05) 한 번"""
        self.run(closed_sec, blink=0.8)
        self.run(0.2, blink=0.05)


# --- 고개 돌림 ---
def test_turn_needs_two_seconds():
    f = Feeder()
    assert "turn" not in f.run(1.9, yaw=25)
    assert "turn" in f.run(0.2, yaw=25)


def test_turn_either_direction():
    assert "turn" in Feeder().run(2.1, yaw=-25)


def test_small_turn_ignored():
    assert "turn" not in Feeder().run(5, yaw=15)


def test_turn_timer_resets_when_back_to_front():
    f = Feeder()
    f.run(1.5, yaw=25)
    f.frame(yaw=0)
    assert "turn" not in f.run(1.5, yaw=25)


# --- 고개 숙임 ---
def test_head_down():
    f = Feeder()
    assert "down" not in f.run(1.9, pitch=16)
    assert "down" in f.run(0.2, pitch=16)


def test_head_up_is_not_down():
    # 고개를 돌릴 때 Pitch가 −로 움직여도 숙임이 아님 (2단계 실측)
    assert "down" not in Feeder().run(3, pitch=-16)


# --- 최소 표시 시간 ---
def test_alert_stays_one_second_after_release():
    f = Feeder()
    f.run(2.1, yaw=25)
    assert "turn" in f.run(0.9, yaw=0)
    assert "turn" not in f.run(0.2, yaw=0)


# --- 깜빡임 ---
def test_jittery_blink_counts_once():
    f = Feeder()
    for score in [0.1, 0.55, 0.45, 0.52, 0.35, 0.51, 0.25, 0.1]:
        f.frame(blink=score)
    assert f.rules.blink_count == 1


def test_long_eye_close_is_not_blink():
    f = Feeder()
    f.blink(closed_sec=1.0)
    assert f.rules.blink_count == 0


def test_too_many_blinks():
    f = Feeder()
    for _ in range(25):
        f.blink()
    assert f.rules.blink_count == 25
    assert "blink" not in f.active
    f.blink()
    assert "blink" in f.active


def test_old_blinks_expire_after_a_minute():
    f = Feeder()
    for _ in range(26):
        f.blink()
    f.run(61)
    assert f.rules.blink_count == 0
    assert "blink" not in f.active


# --- 얼굴 이탈 ---
def test_face_away():
    f = Feeder()
    assert "away" not in f.run(1.9, face=False)
    assert "away" in f.run(0.2, face=False)


# --- 이벤트 기록 ---
def test_events_recorded_with_duration():
    f = Feeder()
    f.run(1, yaw=0)
    f.run(5, yaw=25)            # 1초 시작 → 3초에 경고 시작
    f.run(2, yaw=0)             # 6초에 자세 복귀 → 7초에 경고 종료
    f.rules.finish(f.t)
    (name, start, end), = f.rules.events
    assert name == "turn"
    assert start == pytest.approx(3.0, abs=0.1)
    assert end == pytest.approx(7.0, abs=0.1)


def test_finish_closes_open_events():
    f = Feeder()
    f.run(3, face=False)
    f.rules.finish(f.t)
    assert [e[0] for e in f.rules.events] == ["away"]


# --- 자동 보정 ---
def test_calibrator_uses_average_of_first_two_seconds():
    cal = Calibrator()
    t = 0.0
    while cal.active:
        cal.feed(t, {"yaw": 10.0, "pitch": 4.0, "roll": 0.0})
        t += 1 / FPS
    assert t == pytest.approx(2.0, abs=0.1)
    angles = cal.apply({"yaw": 30.0, "pitch": 4.0, "roll": 0.0})
    assert angles["yaw"] == pytest.approx(20.0)
    assert angles["pitch"] == pytest.approx(0.0)


def feed_calibrator(cal, sec, yaw=0.0, pitch=0.0, t=0.0):
    for _ in range(round(sec * FPS)):
        cal.feed(t, {"yaw": yaw, "pitch": pitch, "roll": 0.0})
        t += 1 / FPS
    return t


def test_calibrator_rejects_turned_head():
    # 보정 중 고개를 돌리고 있으면 그 자세를 정면으로 잡지 않고 다시 기다림
    cal = Calibrator()
    t = feed_calibrator(cal, 3, yaw=-25)
    assert cal.active
    feed_calibrator(cal, 2.2, yaw=-2, t=t)
    assert not cal.active
    assert cal.baseline["yaw"] == pytest.approx(-2.0)


def test_calibrator_rejects_moving_head():
    cal = Calibrator()
    t = 0.0
    for i in range(70):  # 2초 넘게 좌우로 ±10° 흔들기
        cal.feed(t, {"yaw": 10.0 if i % 2 else -10.0, "pitch": 0.0, "roll": 0.0})
        t += 1 / FPS
    assert cal.active


def test_calibrator_allows_camera_below_face():
    # 노트북 카메라가 아래에 있어 정면이어도 Pitch가 큰 경우는 허용
    cal = Calibrator()
    feed_calibrator(cal, 2.2, pitch=18)
    assert not cal.active


# --- 2단계 실측 기록으로 재생 ---
LOG = Path(__file__).parent / "results" / "step2_angles.csv"


def replay(calibrate):
    """2단계 실측 기록을 규칙에 재생. calibrate=False면 원래 각도(카메라 기준) 그대로 판정."""
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    cal, rules = Calibrator(), CoachRules()
    for r in rows:
        t = float(r["time_s"])
        info = {k: float(r[f"{k}_raw"]) for k in ("yaw", "pitch", "roll")}
        info.update(blink_l=float(r["blink_l"]), blink_r=float(r["blink_r"]))
        if calibrate and cal.active:
            cal.feed(t, info)
            continue
        rules.update(t, cal.apply(info), info)
    rules.finish(t)
    return cal, rules


needs_log = pytest.mark.skipif(not LOG.exists(), reason="2단계 기록 없음")


@needs_log
def test_replay_step2_log_alerts():
    """2단계 실측(왼쪽 8~13초, 오른쪽 15~17초, 숙임 23~25초)에서 돌림·숙임 경고가 떠야 한다."""
    _, rules = replay(calibrate=False)
    turns = [(s, e) for n, s, e in rules.events if n == "turn"]
    downs = [(s, e) for n, s, e in rules.events if n == "down"]
    assert any(8 < s < 12 for s, _ in turns)
    assert any(15 < s < 19 for s, _ in turns)
    assert any(23 < s < 27 for s, _ in downs)
    assert rules.total_blinks == 13  # 일부러 오래 감은 2번(1.4초, 2.0초)은 제외


@needs_log
def test_replay_step2_log_calibration_skips_turned_start():
    """기록 초반 2~19초는 좌우로 돌린 구간 → 보정은 그 뒤 정면 구간에서 확정되어야 한다."""
    cal, rules = replay(calibrate=True)
    assert abs(cal.baseline["yaw"]) < 5
    assert "turn" not in {n for n, _, _ in rules.events}  # 보정 후에는 정면만 봤음
