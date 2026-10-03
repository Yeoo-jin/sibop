"""4단계: 실시간 처리 속도(FPS·ms)를 화면에 표시하고 1단계(Gemini)와 비교한다.

3단계 코칭 화면을 그대로 띄우고, 보정이 끝나면 정해진 시간(기본 30초) 동안 측정한 뒤 자동 종료한다.
키: c = 다시 보정 / q = 측정 중단

실행 예)
  python step4_benchmark.py              # 30초 측정
  python step4_benchmark.py --seconds 60
"""
import argparse
import csv
import os
import sys
import time
import unicodedata
from collections import deque

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")  # 창 없이 이미지 파일로만 그림
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

from step2_face_landmarks import (RESULTS_DIR, analyze, create_landmarker, ensure_model,
                                  put, to_mp_image)
from step3_coaching import (ALERTS, CALIB_MSG, FONT_PATHS, Calibrator, CoachRules,
                            draw_status, paste, render_label)

AI_CSV = RESULTS_DIR / "ai_latency.csv"
PARTS = ("capture_ms", "face_ms", "draw_ms", "total_ms")  # ① 캡처 ② 얼굴 분석 ③ 규칙+그리기 ④ 처리 합계(②+③)
PART_NAMES = {"capture_ms": "① 캡처(웹캠 대기)", "face_ms": "② 얼굴 분석",
              "draw_ms": "③ 규칙+그리기", "total_ms": "④ 처리 합계(②+③)"}
WINDOW = "Step4 - Benchmark (c: recalibrate, q: stop)"


def pad(text, width):
    """한글(2칸 폭)을 고려해 왼쪽 정렬 (터미널 표 정렬용)"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return text + " " * max(width - w, 0)


def summarize(values):
    """ms 목록 → 평균·중앙값·95번째 백분위·최대"""
    v = np.array(values)
    return {"avg": v.mean(), "median": np.median(v), "p95": np.percentile(v, 95), "max": v.max()}


def put_right(frame, text, y, color=(255, 255, 255)):
    """오른쪽 정렬 글자"""
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = frame.shape[1] - w - 10
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def run(camera, seconds):
    """웹캠 루프를 돌며 프레임별 구간 시간을 잰다. (측정 행 목록, 첫 프레임 얼굴 분석 ms) 반환"""
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        sys.exit("웹캠을 열 수 없습니다. WSL이라면 Windows에서 실행하세요.")

    labels = {k: render_label(msg, (200, 30, 30)) for k, msg in ALERTS.items()}
    calib_label = render_label(CALIB_MSG, (30, 90, 200))
    cal, rules = Calibrator(), CoachRules()
    recent = deque()           # 최근 1초 프레임: (시각, 얼굴 분석 ms, 처리 합계 ms) — 화면 표시용
    rows, first_face_ms = [], None
    start, last_ts, measure_start, captured = time.perf_counter(), -1, None, False

    with create_landmarker(video=True) as landmarker:
        while True:
            t0 = time.perf_counter()
            ok, frame = cap.read()                                  # ① 캡처
            t1 = time.perf_counter()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break

            frame = cv2.flip(frame, 1)                              # ② 얼굴 분석 (반전·변환 포함)
            t = t1 - start
            ts = max(int(t * 1000), last_ts + 1)
            last_ts = ts
            result = landmarker.detect_for_video(to_mp_image(frame), ts)
            info = analyze(result)
            t2 = time.perf_counter()

            h = frame.shape[0]                                      # ③ 규칙 + 그리기 (3단계와 동일)
            if cal.active:
                if info:
                    cal.feed(t, info)
                paste(frame, calib_label, h - 70)
                if not cal.active:
                    measure_start = t                               # 보정이 끝나면 측정 시작
            else:
                angles = cal.apply(info) if info else None
                active = rules.update(t, angles, info)
                if angles:
                    draw_status(frame, angles, rules)
                for i, k in enumerate(active):
                    paste(frame, labels[k], h - 70 - i * 62)
            t3 = time.perf_counter()

            ms = {"capture_ms": (t1 - t0) * 1000, "face_ms": (t2 - t1) * 1000, "draw_ms": (t3 - t2) * 1000}
            ms["total_ms"] = ms["face_ms"] + ms["draw_ms"]
            if first_face_ms is None:
                first_face_ms = ms["face_ms"]  # 첫 프레임: 모델 준비 시간 포함 (1단계 1회차의 연결 시간과 같은 성격)

            # 최근 1초 평균으로 FPS·ms 표시 (숫자가 너무 흔들리지 않게)
            recent.append((t, ms["face_ms"], ms["total_ms"]))
            while t - recent[0][0] > 1.0:
                recent.popleft()
            fps = (len(recent) - 1) / (t - recent[0][0]) if t > recent[0][0] else 0.0
            put_right(frame, f"FPS {fps:4.1f}", 28, (0, 255, 0))
            put_right(frame, f"Face {np.mean([r[1] for r in recent]):5.1f}ms", 56, (0, 255, 0))
            put_right(frame, f"Total {np.mean([r[2] for r in recent]):5.1f}ms", 84, (0, 255, 0))

            if measure_start is not None:
                elapsed = t - measure_start
                rows.append({"frame": len(rows) + 1, "time_s": round(elapsed, 3),
                             **{k: round(v, 2) for k, v in ms.items()}, "fps": round(fps, 1)})
                put_right(frame, f"REC {elapsed:4.1f}/{seconds:.0f}s", 112, (0, 0, 255))
                if not captured and elapsed >= seconds / 2:  # 측정 중간에 화면 자동 캡처
                    cv2.imwrite(str(RESULTS_DIR / "step4_capture.png"), frame)
                    captured = True
                if elapsed >= seconds:
                    break

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("측정을 중단했습니다.")
                break
            if key == ord("c"):  # 다시 보정하면 측정도 처음부터
                cal.restart()
                rules.reset_timers()
                rows, measure_start, captured = [], None, False
                print("다시 보정합니다. 정면을 봐 주세요.")

    cap.release()
    cv2.destroyAllWindows()
    return rows, first_face_ms


def print_summary(rows, first_face_ms):
    """구간별 평균·중앙값·p95·최대 표 출력. 구간별 요약 dict 반환"""
    stats = {k: summarize([r[k] for r in rows]) for k in PARTS}
    duration = rows[-1]["time_s"]
    print(f"\n[MediaPipe 실시간 처리] {len(rows)}프레임 / {duration:.1f}초")
    print(f"{pad('구간', 18)} |    평균 |  중앙값 |     p95 |    최대  (ms)")
    print("-" * 66)
    for k in PARTS:
        s = stats[k]
        print(f"{pad(PART_NAMES[k], 18)} | {s['avg']:7.1f} | {s['median']:7.1f} | {s['p95']:7.1f} | {s['max']:7.1f}")
    print("-" * 66)
    print(f"화면 FPS: {len(rows) / duration:.1f}  (웹캠 한계 근처에서 멈춤)")
    print(f"처리 가능 FPS: {1000 / stats['total_ms']['avg']:.0f}  (= 1000 / 처리 합계 평균)")
    print(f"첫 프레임 얼굴 분석: {first_face_ms:.1f}ms  (모델 준비 포함, 측정 구간 밖)")
    return stats


def compare(stats, rows):
    """1단계 Gemini 결과와 비교표를 출력·저장. 비교 dict 반환 (1단계 결과가 없으면 None)"""
    if not AI_CSV.exists():
        print(f"\n1단계 결과({AI_CSV.name})가 없어 비교를 건너뜁니다.")
        return None
    ai = list(csv.DictReader(open(AI_CSV, encoding="utf-8-sig")))
    g_face = np.mean([float(r["api_s"]) for r in ai]) * 1000
    g_total = np.mean([float(r["total_s"]) for r in ai]) * 1000
    g_p95 = np.percentile([float(r["total_s"]) for r in ai], 95) * 1000
    m_face, m_total, m_p95 = stats["face_ms"]["avg"], stats["total_ms"]["avg"], stats["total_ms"]["p95"]
    table = [
        ("프레임당 분석 시간 평균(ms)", g_face, m_face),
        ("프레임당 처리 합계 평균(ms)", g_total, m_total),
        ("프레임당 처리 합계 p95(ms)", g_p95, m_p95),
        ("처리 가능 FPS", 1000 / g_total, 1000 / m_total),
        ("측정 횟수(프레임)", len(ai), len(rows)),
    ]
    speedup = g_total / m_total

    print(f"\n[1단계 Gemini vs 4단계 MediaPipe]")
    print(f"{pad('항목', 28)} |    Gemini | MediaPipe")
    print("-" * 50)
    for name, g, m in table:
        print(f"{pad(name, 28)} | {g:9,.1f} | {m:9,.1f}")
    print("-" * 50)
    print(f"→ MediaPipe가 프레임당 처리 기준 약 {speedup:,.0f}배 빠릅니다.")

    path = RESULTS_DIR / "step4_comparison.csv"
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "gemini", "mediapipe"])
        for name, g, m in table:
            writer.writerow([name, round(g, 2), round(m, 2)])
        writer.writerow(["배수(Gemini 처리 합계 / MediaPipe 처리 합계)", round(speedup, 1), ""])
    print(f"비교표 저장: {path}")
    return {"g_total": g_total, "m_total": m_total, "speedup": speedup}


def plot(cmp):
    """비교 막대그래프: 왼쪽 처리 시간(로그 축), 오른쪽 처리 가능 FPS"""
    font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)
    if font_path:  # 한글 폰트 (없으면 한글이 네모로 보임)
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()

    names = ["Gemini\n(1단계)", "MediaPipe\n(4단계)"]
    colors = ["#b5b3ad", "#2a78d6"]  # 비교 대상은 회색, 강조 대상은 파랑
    ink, muted, grid = "#0b0b0b", "#898781", "#e1e0d9"
    ms = [cmp["g_total"], cmp["m_total"]]
    fps = [1000 / v for v in ms]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="#fcfcfb")
    panels = [(axes[0], ms, "프레임당 처리 시간 (ms, 로그 축)", lambda v: f"{v:,.1f}ms"),
              (axes[1], fps, "처리 가능 FPS (초당 프레임)", lambda v: f"{v:,.1f}")]
    for ax, values, title, fmt in panels:
        ax.set_facecolor("#fcfcfb")
        bars = ax.barh(names, values, color=colors, height=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {fmt(v)}",
                    va="center", color=ink, fontsize=11)
        ax.set_title(title, color=ink, fontsize=12, loc="left")
        ax.invert_yaxis()
        ax.tick_params(colors=muted)
        ax.grid(axis="x", color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(grid)
    axes[0].set_xscale("log")
    axes[0].set_xlim(1, max(ms) * 20)
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axes[1].axvline(30, color=muted, linestyle="--", linewidth=1)  # 일반 웹캠 최대 FPS
    axes[1].text(30, 0.5, " 웹캠 한계 30 FPS", color=muted, fontsize=9, va="center",
                 transform=axes[1].get_xaxis_transform())  # x는 데이터, y는 축 비율(두 막대 사이)
    axes[1].set_xlim(0, max(fps) * 1.35)
    fig.suptitle(f"MediaPipe가 프레임당 약 {cmp['speedup']:,.0f}배 빠름", x=0.02, ha="left",
                 fontsize=15, color=ink)
    fig.tight_layout()
    path = RESULTS_DIR / "step4_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"그래프 저장: {path}")


def main():
    parser = argparse.ArgumentParser(description="실시간 처리 속도 측정과 1단계(Gemini) 비교")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    parser.add_argument("--seconds", type=float, default=30.0, help="보정 후 측정 시간(초, 기본 30)")
    args = parser.parse_args()

    ensure_model()
    RESULTS_DIR.mkdir(exist_ok=True)
    rows, first_face_ms = run(args.camera, args.seconds)
    if not rows:
        sys.exit("측정된 프레임이 없습니다. 정면을 보고 보정이 끝날 때까지 기다려 주세요.")

    path = RESULTS_DIR / "step4_frames.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"프레임별 기록 저장: {path}")

    stats = print_summary(rows, first_face_ms)
    cmp = compare(stats, rows)
    if cmp:
        plot(cmp)


if __name__ == "__main__":
    main()
