"""Gemini 2 RGB, MediaPipe and aligned-depth diagnostic.

Run from the repository root on Raspberry Pi:

    python -m src.tests.orbbec_rgbd_test

Controls: Q/Esc quits, M toggles mirroring, D toggles the depth panel.
This test does not initialize LEDs, servos, MiDaS, YOLO, or the main runtime.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VISION_DIR = PROJECT_ROOT / "src" / "vision"
for path in (PROJECT_ROOT, VISION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gestures import GestureRecognizer  # noqa: E402
from orbbec_camera import OrbbecCamera  # noqa: E402


WINDOW_NAME = "Gemini 2 RGB-D diagnostic"
HAND_SAMPLE_RADIUS = 4


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--color-width", type=int, default=1280)
    parser.add_argument("--color-height", type=int, default=720)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=400)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--software-align",
        action="store_true",
        help="Skip hardware D2C and force software depth-to-color alignment.",
    )
    return parser.parse_args()


def depth_color_map(depth_mm, min_mm=150.0, max_mm=5000.0):
    valid = depth_mm > 0
    clipped = np.clip(depth_mm, min_mm, max_mm)
    normalized = ((clipped - min_mm) * 255.0 / (max_mm - min_mm)).astype(np.uint8)
    normalized[~valid] = 0
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def landmark_depth_stats(depth_mm, hand_landmarks):
    height, width = depth_mm.shape
    rows = []
    for name, index in (("MCP", 5), ("PIP", 6), ("DIP", 7), ("TIP", 8)):
        landmark = hand_landmarks.landmark[index]
        x = int(np.clip(landmark.x, 0.0, 1.0) * (width - 1))
        y = int(np.clip(landmark.y, 0.0, 1.0) * (height - 1))
        radius = HAND_SAMPLE_RADIUS
        patch = depth_mm[
            max(0, y - radius):min(height, y + radius + 1),
            max(0, x - radius):min(width, x + radius + 1),
        ]
        valid = patch[patch > 0]
        ratio = len(valid) / max(1, patch.size)
        median = float(np.median(valid)) if len(valid) else None
        rows.append((name, x, y, ratio, median))
    return rows


def draw_text(image, text, row, color=(255, 255, 255)):
    y = 28 + row * 25
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4)
    cv2.putText(image, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1)


def main():
    args = parse_args()
    camera = OrbbecCamera(
        color_width=args.color_width,
        color_height=args.color_height,
        depth_width=args.depth_width,
        depth_height=args.depth_height,
        fps=args.fps,
        prefer_hardware_align=not args.software_align,
    )
    recognizer = GestureRecognizer()
    hands_api = mp.solutions.hands
    drawing = mp.solutions.drawing_utils
    hands = hands_api.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    mirror = args.mirror
    show_depth = True
    fps_ema = 0.0
    previous_time = time.monotonic()
    print(f"[Orbbec] alignment={camera.alignment_mode}", flush=True)
    print(f"[Orbbec] intrinsics={camera.intrinsics}", flush=True)

    try:
        while True:
            ok, packet = camera.read()
            if not ok:
                continue

            color = packet.color_bgr
            depth = packet.depth_mm
            if mirror:
                color = cv2.flip(color, 1)
                depth = cv2.flip(depth, 1)

            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True

            overlay = color.copy()
            state = recognizer.extract_full_state(results)
            depth_rows = []
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                drawing.draw_landmarks(overlay, hand, hands_api.HAND_CONNECTIONS)
                depth_rows = landmark_depth_stats(depth, hand)
                for name, x, y, ratio, median in depth_rows:
                    status_color = (50, 220, 50) if ratio >= 0.6 else (0, 80, 255)
                    cv2.circle(overlay, (x, y), HAND_SAMPLE_RADIUS + 2, status_color, 2)
                    label = f"{name}:{median:.0f}mm" if median is not None else f"{name}:invalid"
                    cv2.putText(
                        overlay, label, (x + 7, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, status_color, 1,
                    )

            now = time.monotonic()
            instant_fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            fps_ema = instant_fps if fps_ema == 0 else fps_ema * 0.9 + instant_fps * 0.1

            valid_ratio = float(np.count_nonzero(depth)) / max(1, depth.size)
            draw_text(overlay, f"RGB {color.shape[1]}x{color.shape[0]}  FPS {fps_ema:.1f}", 0)
            draw_text(
                overlay,
                f"D2C {camera.alignment_mode}  valid depth {valid_ratio * 100:.1f}%",
                1,
            )
            draw_text(overlay, f"gesture: {state.get('gesture') or '-'}", 2, (0, 255, 255))
            if depth_rows:
                qualities = "  ".join(f"{name}:{ratio * 100:.0f}%" for name, _, _, ratio, _ in depth_rows)
                draw_text(overlay, f"depth quality {qualities}", 3)

            if show_depth:
                depth_view = depth_color_map(depth)
                # The panels use the same dimensions because depth is D2C-aligned.
                display = np.hstack((overlay, depth_view))
            else:
                display = overlay

            max_display_width = 1800
            if display.shape[1] > max_display_width:
                scale = max_display_width / display.shape[1]
                display = cv2.resize(display, None, fx=scale, fy=scale)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("d"):
                show_depth = not show_depth
    finally:
        hands.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
