"""1단계: AI(비전 모델)로 웹캠 프레임을 해석할 때 얼마나 느린지 측정한다.

실행 예)
  python step1_ai_latency.py                 # 웹캠 사용 (Windows 권장)
  python step1_ai_latency.py --image me.jpg  # 웹캠이 없을 때 (WSL 등)
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
MODEL = "gemini-3.5-flash-lite"  # 2.5-flash는 신규 사용자 404, 3.8-flash는 503 잦음 (--model로 변경 가능)
PROMPT = (
    "이 사진 속 사람의 얼굴을 분석해서 아래 JSON 형식으로만 답해줘. 좌/우는 사진 속 인물 기준이야.\n"
    '{"head_direction": "정면|좌|우|아래 중 하나", "gaze": "시선 방향(짧게)", "expression": "표정(짧게)"}'
)
# 한글 폰트 후보 (Windows → WSL에서 보이는 Windows 폰트 → Ubuntu 나눔폰트)
FONT_PATHS = [
    "C:/Windows/Fonts/malgun.ttf",
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
]


def capture_jpeg(cap, image_path):
    """프레임 1장을 얻어 JPEG로 인코딩한다. (원본 프레임, JPEG 바이트) 반환"""
    if cap is not None:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("웹캠 프레임을 읽지 못했습니다.")
    else:
        frame = cv2.imread(str(image_path))
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return frame, buf.tobytes()


def ask_gemini(client, model, jpeg):
    """Gemini에 이미지를 보내고 응답 텍스트(JSON 문자열)를 받는다."""
    res = client.models.generate_content(
        model=model,
        contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),  # 도구 미사용 → AFC 경고 끄기
        ),
    )
    return res.text


def summarize(text):
    """응답 JSON을 '고개/시선/표정' 한 줄 요약으로 바꾼다. 파싱 실패 시 원문 그대로."""
    try:
        d = json.loads(text)
        return f"{d.get('head_direction')} / {d.get('gaze')} / {d.get('expression')}"
    except (json.JSONDecodeError, TypeError, AttributeError):
        return (text or "").replace("\n", " ")


def draw_result(frame, model, avg_sec, runs, answer):
    """프레임 위에 평균 지연과 AI 응답을 한글로 그린다. (OpenCV putText는 한글 불가 → Pillow 사용)"""
    font_path = next((p for p in FONT_PATHS if os.path.exists(p)), None)
    font = ImageFont.truetype(font_path, 24) if font_path else ImageFont.load_default(24)

    lines = [f"AI 분석 평균 지연: {avg_sec:.2f}초  ({model}, {runs}회)"]
    try:
        d = json.loads(answer)
        lines += [f"고개: {d.get('head_direction')}", f"시선: {d.get('gaze')}", f"표정: {d.get('expression')}"]
    except (json.JSONDecodeError, TypeError, AttributeError):
        lines.append(f"응답: {answer}")

    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([0, 0, img.width, 20 + 34 * len(lines)], fill=(0, 0, 0, 160))  # 반투명 배경
    for i, line in enumerate(lines):
        color = (255, 220, 0) if i == 0 else (255, 255, 255)
        draw.text((12, 12 + 34 * i), line, font=font, fill=color)
    img = Image.alpha_composite(img, overlay).convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def print_table(rows):
    """회차별 측정 결과와 평균/최소/최대를 표로 출력한다."""
    print("\n  회차 | 캡처+인코딩(s) | API 응답(s) |   전체(s) | AI 응답")  # 한글은 2칸 폭이라 직접 정렬
    print("-" * 90)
    for r in rows:
        print(f"{r['run']:>6} | {r['capture_s']:>14.3f} | {r['api_s']:>11.3f} | {r['total_s']:>9.3f} | {summarize(r['response'])}")
    print("-" * 90)
    for name, fn in [("평균", lambda v: sum(v) / len(v)), ("최소", min), ("최대", max)]:
        vals = [fn([r[k] for r in rows]) for k in ("capture_s", "api_s", "total_s")]
        print(f"  {name} | {vals[0]:>14.3f} | {vals[1]:>11.3f} | {vals[2]:>9.3f} |")


def main():
    parser = argparse.ArgumentParser(description="AI 비전 모델 프레임 해석 지연 측정")
    parser.add_argument("--model", default=MODEL, help=f"Gemini 모델 (기본 {MODEL})")
    parser.add_argument("--runs", type=int, default=10, help="반복 횟수 (기본 10)")
    parser.add_argument("--camera", type=int, default=0, help="웹캠 번호 (기본 0)")
    parser.add_argument("--image", help="웹캠 대신 사용할 이미지 파일 경로")
    parser.add_argument("--interval", type=float, default=4.0, help="회차 사이 대기(초). 무료 등급 요청 한도 대비, 측정에서 제외")
    args = parser.parse_args()

    # API 키 읽기 (.env → GEMINI_API_KEY)
    load_dotenv(BASE_DIR / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        sys.exit("GEMINI_API_KEY가 없습니다. .env.example을 .env로 복사한 뒤 키를 넣어주세요.")
    client = genai.Client(api_key=api_key)

    # 입력 준비: 이미지 파일 또는 웹캠
    cap = None
    if args.image:
        if cv2.imread(args.image) is None:
            sys.exit(f"이미지를 읽을 수 없습니다: {args.image}")
    else:
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            sys.exit("웹캠을 열 수 없습니다. WSL이라면 Windows에서 실행하거나 --image 옵션을 사용하세요.")
        for _ in range(10):  # 노출/초점이 안정될 때까지 초기 프레임 버리기
            cap.read()

    rows, frame, retries = [], None, 0
    print(f"{args.model}로 {args.runs}회 측정 시작...")
    while len(rows) < args.runs:
        t0 = time.perf_counter()
        frame, jpeg = capture_jpeg(cap, args.image)   # (a) 캡처 + 인코딩
        t1 = time.perf_counter()
        try:
            answer = ask_gemini(client, args.model, jpeg)         # (b) API 요청 ~ 응답
        except errors.APIError as e:
            # 요청 한도 초과(429)·서버 혼잡(503)이면 잠시 쉬고 이번 회차를 처음부터 다시 측정
            if e.code not in (429, 503):
                raise
            if retries >= 5:
                print(f"  재시도 5회 초과 → {len(rows)}회까지의 결과만 저장합니다.")
                break
            retries += 1
            print(f"  일시적 오류({e.code}) → 30초 후 이번 회차 재측정 ({retries}/5)")
            time.sleep(30)
            continue
        t2 = time.perf_counter()

        rows.append({"run": len(rows) + 1, "capture_s": t1 - t0, "api_s": t2 - t1,
                     "total_s": t2 - t0, "response": answer})   # (c) 전체
        print(f"  {len(rows)}/{args.runs}회 완료: {t2 - t0:.2f}초")
        if len(rows) < args.runs:
            time.sleep(args.interval)

    if cap is not None:
        cap.release()

    if not rows:
        sys.exit("측정된 회차가 없습니다. 잠시 후 다시 하거나 --model로 다른 모델을 지정하세요.")
    print_table(rows)

    # 결과 저장: CSV + 마지막 프레임 캡처
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "ai_latency.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig: 엑셀에서 한글 안 깨지게
        writer = csv.DictWriter(f, fieldnames=["run", "capture_s", "api_s", "total_s", "response"])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, **{k: round(r[k], 4) for k in ("capture_s", "api_s", "total_s")}})

    avg_total = sum(r["total_s"] for r in rows) / len(rows)
    png_path = RESULTS_DIR / "ai_latency_capture.png"
    cv2.imwrite(str(png_path), draw_result(frame, args.model, avg_total, len(rows), rows[-1]["response"]))
    print(f"\n저장 완료: {csv_path}\n          {png_path}")


if __name__ == "__main__":
    main()
