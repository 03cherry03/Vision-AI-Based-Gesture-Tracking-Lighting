import math
import os
import threading
import time
from collections import deque

import cv2
import numpy as np


HAND_WRIST_TO_MIDDLE_MCP_M = 0.10
RAY_STEP_PX = 4
RAY_MAX_STEPS = 300
DEPTH_HIT_THRESHOLD = min(
    0.5,
    max(0.0, float(os.environ.get("PI_DEPTH_HIT_THRESHOLD", "0.12"))),
)
RAY_START_MARGIN_PX = max(
    0.0,
    float(os.environ.get("PI_POINT_RAY_START_MARGIN_PX", "35")),
)
DEPTH_UPDATE_INTERVAL = max(
    1,
    int(os.environ.get("PI_DEPTH_UPDATE_INTERVAL", "4")),
)
STABLE_SECONDS = 3.0
STABLE_STD_PX = 55.0
JITTER_RESET_PX = 140.0
BUFFER_MAXLEN = 50
EMA_ALPHA = 0.25
DEPTH_ASYNC = os.environ.get("PI_DEPTH_ASYNC", "0") != "0"
DEPTH_ASYNC_FPS = max(0.5, float(os.environ.get("PI_DEPTH_ASYNC_FPS", "4.0")))
DEPTH_RESULT_MAX_AGE_SECONDS = max(
    0.05,
    float(os.environ.get("PI_DEPTH_RESULT_MAX_AGE", "0.75")),
)
TORCH_NUM_THREADS = max(1, int(os.environ.get("PI_TORCH_NUM_THREADS", "2")))
HAND_DEPTH_SAMPLE_RADIUS = max(
    1, int(os.environ.get("PI_HAND_DEPTH_SAMPLE_RADIUS", "5"))
)
HAND_DEPTH_MIN_VALID_RATIO = min(
    1.0, max(0.0, float(os.environ.get("PI_HAND_DEPTH_MIN_VALID_RATIO", "0.35")))
)
HAND_MASK_DILATE_PX = max(
    0, int(os.environ.get("PI_HAND_DEPTH_MASK_DILATE_PX", "6"))
)
HAND_MASK_DEPTH_TOLERANCE_MM = max(
    20.0, float(os.environ.get("PI_HAND_MASK_DEPTH_TOLERANCE_MM", "140"))
)
BACKGROUND_DEPTH_PATCH_RADIUS = max(
    1, int(os.environ.get("PI_BACKGROUND_DEPTH_PATCH_RADIUS", "4"))
)
BACKGROUND_DEPTH_MIN_VALID_RATIO = min(
    1.0,
    max(0.0, float(os.environ.get("PI_BACKGROUND_DEPTH_MIN_VALID_RATIO", "0.55"))),
)
BACKGROUND_DEPTH_MAX_MAD_MM = max(
    1.0, float(os.environ.get("PI_BACKGROUND_DEPTH_MAX_MAD_MM", "80"))
)
BACKGROUND_DEPTH_MAX_STEP_JUMP_MM = max(
    1.0, float(os.environ.get("PI_BACKGROUND_DEPTH_MAX_STEP_JUMP_MM", "160"))
)
BACKGROUND_DEPTH_HIT_TOLERANCE = min(
    0.5,
    max(0.0, float(os.environ.get("PI_BACKGROUND_DEPTH_HIT_TOLERANCE", "0.08"))),
)
BACKGROUND_DEPTH_CONFIRM_STEPS = max(
    1, int(os.environ.get("PI_BACKGROUND_DEPTH_CONFIRM_STEPS", "3"))
)

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
)


class DepthEstimator:
    def __init__(self):
        import torch

        self.torch = torch
        try:
            torch.set_num_threads(TORCH_NUM_THREADS)
        except RuntimeError as exc:
            print(f"[Pointing] could not set torch threads: {exc}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[Pointing] MiDaS loading... first run can take a while")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        self.depth_scale = 1.0
        self.is_calibrated = False
        print(f"[Pointing] MiDaS loaded ({self.device})")

    def estimate(self, frame_bgr):
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb).to(self.device)

        with self.torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = self.torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_map = prediction.cpu().numpy()
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 1e-6:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        return depth_map.astype(np.float32)

    def calibrate(self, depth_map, wrist_px, middle_mcp_px, cam_fx):
        wx, wy = wrist_px
        mx, my = middle_mcp_px
        px_dist = math.hypot(mx - wx, my - wy)
        if px_dist < 20:
            return False

        estimated_depth_m = HAND_WRIST_TO_MIDDLE_MCP_M * cam_fx / px_dist
        h, w = depth_map.shape
        wx_c = int(np.clip(wx, 5, w - 5))
        wy_c = int(np.clip(wy, 5, h - 5))
        midas_val = float(np.mean(depth_map[wy_c - 5:wy_c + 5, wx_c - 5:wx_c + 5]))
        if midas_val < 0.01:
            return False

        self.depth_scale = estimated_depth_m * midas_val
        self.is_calibrated = True
        print(
            f"[Pointing] calibrated scale={self.depth_scale:.4f} "
            f"hand_depth={estimated_depth_m:.2f}m midas={midas_val:.3f}"
        )
        return True

    def midas_to_meters(self, midas_val):
        if midas_val < 0.001:
            return 99.0
        return self.depth_scale / midas_val


class MotorAngleTracker: #0520_v2m
    def __init__(self, cam_fx, cam_fy, cx, cy):
        self.fx = cam_fx
        self.fy = cam_fy
        self.cx = cx
        self.cy = cy
        self.buf_x = deque(maxlen=BUFFER_MAXLEN)
        self.buf_y = deque(maxlen=BUFFER_MAXLEN)
        self.stable_start = None
        self.confirmed_target = None
        self.confirmed_angles = None
        self.ema_x = None
        self.ema_y = None
        self.grace_until = 0.0
        self.last_pointing_time = 0.0
        self.pointing_grace_seconds = float(os.environ.get("PI_POINTING_GRACE_SECONDS", "0.35"))
        self.entry_grace_seconds = float(
            os.environ.get("PI_POINT_ENTRY_GRACE_SECONDS", "1.0")
        )

    def update(self, raw, is_pointing):
        result = {
            "display_target": None,
            "confirmed": self.confirmed_target,
            "pan_deg": self.confirmed_angles[0] if self.confirmed_angles else None,
            "tilt_deg": self.confirmed_angles[1] if self.confirmed_angles else None,
            "stable_ratio": 0.0,
            "std_px": 0.0,
            "stable_rejected": False,
        }

        if self.confirmed_target is not None:
            result["display_target"] = self.confirmed_target
            result["stable_ratio"] = 1.0
            return result

        # 포인트모드 진입 직후 유예시간 — 피스→포인트 전환 중 잘못된 데이터 방지
        if time.time() < self.grace_until: #0520_v2m
            return result

        now = time.time()

        if is_pointing and raw is not None:
            self.last_pointing_time = now
        else:
            if now - self.last_pointing_time <= self.pointing_grace_seconds:
                result["display_target"] = (
                    (int(self.ema_x), int(self.ema_y))
                    if self.ema_x is not None and self.ema_y is not None
                    else None
                )
                if self.stable_start is not None:
                    elapsed = now - self.stable_start
                    result["stable_ratio"] = min(elapsed / STABLE_SECONDS, 1.0)
                return result

            self._reset_buffer()
            return result

        rx, ry = raw
        self.buf_x.append(rx)
        self.buf_y.append(ry)

        if self.ema_x is None:
            self.ema_x, self.ema_y = float(rx), float(ry)
        else:
            self.ema_x = EMA_ALPHA * rx + (1 - EMA_ALPHA) * self.ema_x
            self.ema_y = EMA_ALPHA * ry + (1 - EMA_ALPHA) * self.ema_y
        result["display_target"] = (int(self.ema_x), int(self.ema_y))

        if self.stable_start is None:
            self.stable_start = time.time()

        elapsed = time.time() - self.stable_start
        result["stable_ratio"] = min(elapsed / STABLE_SECONDS, 1.0)

        if len(self.buf_x) >= 5:
            result["std_px"] = float(np.hypot(np.std(self.buf_x), np.std(self.buf_y)))
            if result["std_px"] > JITTER_RESET_PX:
                print(f"[Pointing] too much jitter, reset buffer std={result['std_px']:.1f}px")
                self._reset_buffer()
                result["stable_ratio"] = 0.0
                result["stable_rejected"] = True
                return result

        if elapsed >= STABLE_SECONDS:
            if result["std_px"] > STABLE_STD_PX: #0520_v2m
                print(f"[Pointing] not stable enough, retry std={result['std_px']:.1f}px")
                self._reset_buffer()          # ← 버퍼도 비워서 과거 흔들림 데이터 제거
                result["stable_ratio"] = 0.0
                result["stable_rejected"] = True
                return result

            tx = int(np.mean(self.buf_x))
            ty = int(np.mean(self.buf_y))
            pan, tilt = self._to_angles(tx, ty)
            self.confirmed_target = (tx, ty)
            self.confirmed_angles = (pan, tilt)
            result.update({
                "confirmed": self.confirmed_target,
                "pan_deg": pan,
                "tilt_deg": tilt,
                "stable_ratio": 1.0,
            })
            print(
                f"[Pointing] target confirmed pixel=({tx},{ty}) "
                f"pan={pan:+.2f} tilt={tilt:+.2f} std={result['std_px']:.1f}px"
            )
            self._reset_buffer()

        return result

    def _to_angles(self, tx, ty):
        pan_deg = math.degrees(math.atan((tx - self.cx) / self.fx))
        tilt_deg = math.degrees(math.atan((ty - self.cy) / self.fy))
        return pan_deg, tilt_deg

    def _reset_buffer(self):
        self.buf_x.clear()
        self.buf_y.clear()
        self.stable_start = None
        self.ema_x = None
        self.ema_y = None

    def reset(self, clear_confirmed=False): #0520_v2m
        self._reset_buffer()
        if clear_confirmed:
            self.confirmed_target = None
            self.confirmed_angles = None
            self.grace_until = time.time() + self.entry_grace_seconds


class PointingTargetEstimator:
    WRIST = 0
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9

    def __init__(
        self,
        frame_w,
        frame_h,
        cam_fx=None,
        cam_fy=None,
        cx=None,
        cy=None,
        ray_mode=None,
        enable_midas=True,
    ):
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.cam_fx = float(cam_fx) if cam_fx is not None else frame_w * 0.7
        self.cam_fy = float(cam_fy) if cam_fy is not None else self.cam_fx
        self.cx = float(cx) if cx is not None else frame_w / 2.0
        self.cy = float(cy) if cy is not None else frame_h / 2.0
        self.ray_mode = (
            ray_mode or os.environ.get("PI_POINT_RAY_MODE", "mcp_tip")
        ).strip().lower()
        if self.ray_mode not in {"mcp_tip", "finger_axis"}:
            print(f"[Pointing] unknown ray mode {self.ray_mode!r}; using mcp_tip")
            self.ray_mode = "mcp_tip"

        self.depth_est = None
        self.depth_error = None
        if enable_midas:
            try:
                self.depth_est = DepthEstimator()
            except ModuleNotFoundError as e:
                self.depth_error = str(e)
                print(f"[Pointing] MiDaS disabled, using 2D ray fallback: {e}")
            except Exception as e:
                self.depth_error = str(e)
                print(f"[Pointing] MiDaS disabled, using 2D ray fallback: {e}")
        else:
            print("[Pointing] Gemini metric depth selected; MiDaS disabled")

        self.tracker = MotorAngleTracker(self.cam_fx, self.cam_fy, self.cx, self.cy)
        self.depth_map = None
        self.frame_count = 0
        self.async_depth = bool(self.depth_est is not None and DEPTH_ASYNC)
        self._last_result = self._empty_result()
        self._generation = 0
        self._job_sequence = 0
        self._consumed_sequence = 0
        self._pending_job = None
        self._async_result = None
        self._latest_depth_captured_at = None
        self._latest_depth_error = None
        self._async_stop = False
        self._async_condition = threading.Condition()
        self._async_thread = None
        print(
            f"[Pointing] ray config hit_threshold={DEPTH_HIT_THRESHOLD:g} "
            f"start_margin={RAY_START_MARGIN_PX:g}px "
            f"depth_update_interval={DEPTH_UPDATE_INTERVAL}"
        )
        if self.async_depth:
            self._async_thread = threading.Thread(
                target=self._depth_worker,
                daemon=True,
            )
            self._async_thread.start()
            print(
                f"[Pointing] async MiDaS enabled "
                f"fps={DEPTH_ASYNC_FPS:g} max_age={DEPTH_RESULT_MAX_AGE_SECONDS:g}s"
            )

    def _empty_result(self):
        return {
            "display_target": None,
            "confirmed": None,
            "pan_deg": None,
            "tilt_deg": None,
            "stable_ratio": 0.0,
            "std_px": 0.0,
            "stable_rejected": False,
            "raw_target": None,
            "ray_start_px": None,
            "ray_tip_px": None,
            "calibrated": False,
            "depth_available": self.depth_est is not None,
            "used_depth_hit": False,
            "hit_method": "waiting_for_depth" if self.depth_est is not None else "2d_fallback_no_depth",
            "depth_error": self.depth_error,
            "depth_map": None,
            "async_pending": self.async_depth,
            "ray_mode": self.ray_mode,
            "depth_source": "midas" if self.depth_est is not None else "none",
            "hand_depth_mm": None,
            "hand_depth_samples_mm": {},
            "hand_depth_quality": {},
            "hand_depth_points_px": {},
            "hand_depth_valid_ratio": 0.0,
            "hand_mask": None,
            "surface_depth_mm": None,
            "ray_depth_mm": None,
            "surface_valid_ratio": 0.0,
        }

    def update(
        self,
        frame_bgr,
        hand_landmarks,
        depth_mm=None,
        camera_intrinsics=None,
    ):
        self.frame_count += 1
        if camera_intrinsics is not None:
            self._set_camera_intrinsics(camera_intrinsics)
        points = self._key_points(hand_landmarks.landmark)
        ray_start, ray_tip = self._ray_segment(points)

        if depth_mm is not None:
            return self._update_from_metric_depth(depth_mm, hand_landmarks, points, ray_start, ray_tip)

        if self.depth_est is None:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)
            result = self.tracker.update(raw_target, True)
            result.update(
                {
                    "raw_target": raw_target,
                    "ray_start_px": ray_start,
                    "ray_tip_px": ray_tip,
                    "calibrated": False,
                    "depth_available": False,
                    "used_depth_hit": False,
                    "hit_method": "2d_fallback_no_depth",
                    "depth_error": self.depth_error,
                    "depth_map": None,
                    "async_pending": False,
                    "ray_mode": self.ray_mode,
                }
            )
            self._last_result = result
            return result

        if self.async_depth:
            self._submit_depth_job(frame_bgr)
            return self._consume_async_result(points, ray_start, ray_tip)

        if self.depth_map is None or self.frame_count % DEPTH_UPDATE_INTERVAL == 0:
            self.depth_map = self.depth_est.estimate(frame_bgr)
        payload = self._target_from_depth(
            self.depth_map,
            points,
            ray_start,
            ray_tip,
        )
        result = self.tracker.update(payload["raw_target"], True)
        result.update(payload)
        result["async_pending"] = False
        self._last_result = result
        return result

    def _set_camera_intrinsics(self, intrinsics):
        """Apply intrinsics already scaled to the runtime RGB frame."""
        try:
            fx = float(intrinsics["fx"])
            fy = float(intrinsics["fy"])
            cx = float(intrinsics["cx"])
            cy = float(intrinsics["cy"])
        except (KeyError, TypeError, ValueError):
            return
        if fx <= 0 or fy <= 0:
            return
        self.cam_fx = fx
        self.cam_fy = fy
        self.cx = cx
        self.cy = cy
        self.tracker.fx = fx
        self.tracker.fy = fy
        self.tracker.cx = cx
        self.tracker.cy = cy

    def _update_from_metric_depth(
        self, depth_mm, hand_landmarks, points, ray_start, ray_tip
    ):
        """Keep hand depth, but remove the hand from the background-search map.

        Stage 3 intentionally does not perform background ray intersection yet.
        The masked metric map produced here is the input for that next stage.
        """
        depth_mm = np.asarray(depth_mm, dtype=np.float32)
        if depth_mm.shape != (self.frame_h, self.frame_w):
            depth_mm = cv2.resize(
                depth_mm,
                (self.frame_w, self.frame_h),
                interpolation=cv2.INTER_NEAREST,
            )

        samples, quality, hand_depth_mm, valid_ratio = self._sample_hand_depth(
            depth_mm, points
        )
        hand_mask = self._hand_mask(
            hand_landmarks.landmark,
            depth_mm,
            hand_depth_mm,
        )
        background_depth_mm = depth_mm.copy()
        background_depth_mm[hand_mask] = 0.0

        hit = self._find_metric_background_hit(
            depth_mm,
            hand_mask,
            hand_depth_mm,
            ray_start,
            ray_tip,
        )
        raw_target = hit["target"] if hit is not None else None
        result = self.tracker.update(raw_target, raw_target is not None)
        result.update(
            {
                "raw_target": raw_target,
                "ray_start_px": ray_start,
                "ray_tip_px": ray_tip,
                "calibrated": True,
                "depth_available": True,
                "used_depth_hit": raw_target is not None,
                "hit_method": (
                    "gemini_metric_background_hit"
                    if raw_target is not None
                    else "gemini_metric_no_surface"
                ),
                "depth_error": None,
                "depth_map": background_depth_mm,
                "async_pending": False,
                "ray_mode": self.ray_mode,
                "depth_source": "gemini_metric",
                "hand_depth_mm": hand_depth_mm,
                "hand_depth_samples_mm": samples,
                "hand_depth_quality": quality,
                "hand_depth_points_px": points,
                "hand_depth_valid_ratio": valid_ratio,
                "hand_mask": hand_mask,
                "surface_depth_mm": (
                    hit["surface_depth_mm"] if hit is not None else None
                ),
                "ray_depth_mm": hit["ray_depth_mm"] if hit is not None else None,
                "surface_valid_ratio": (
                    hit["valid_ratio"] if hit is not None else 0.0
                ),
                "camera_intrinsics": {
                    "fx": self.cam_fx,
                    "fy": self.cam_fy,
                    "cx": self.cx,
                    "cy": self.cy,
                },
            }
        )
        self.depth_map = background_depth_mm
        self._last_result = result
        return result

    def _hand_mask(self, landmarks, depth_mm=None, hand_depth_mm=None):
        pixels = np.asarray(
            [
                (
                    int(np.clip(lm.x, 0.0, 1.0) * (self.frame_w - 1)),
                    int(np.clip(lm.y, 0.0, 1.0) * (self.frame_h - 1)),
                )
                for lm in landmarks
            ],
            dtype=np.int32,
        )
        core = np.zeros((self.frame_h, self.frame_w), dtype=np.uint8)
        if len(pixels) < 21:
            return core.astype(bool)

        palm_width = float(np.linalg.norm(pixels[5] - pixels[17]))
        line_width = max(3, int(round(palm_width * 0.18)))
        joint_radius = max(2, int(round(line_width * 0.65)))

        # Fill only the palm; model each finger as connected capsules instead
        # of filling one convex hull around the entire hand.
        palm = pixels[[0, 5, 9, 13, 17]]
        cv2.fillConvexPoly(core, cv2.convexHull(palm), 255)
        for start_index, end_index in HAND_CONNECTIONS:
            cv2.line(
                core,
                tuple(pixels[start_index]),
                tuple(pixels[end_index]),
                255,
                line_width,
            )
        for pixel in pixels:
            cv2.circle(core, tuple(pixel), joint_radius, 255, -1)

        support = core
        if HAND_MASK_DILATE_PX > 0:
            kernel_size = HAND_MASK_DILATE_PX * 2 + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            support = cv2.dilate(core, kernel)

        if depth_mm is None or hand_depth_mm is None:
            return support.astype(bool)

        tolerance = max(
            HAND_MASK_DEPTH_TOLERANCE_MM,
            float(hand_depth_mm) * 0.22,
        )
        depth_consistent = (
            np.isfinite(depth_mm)
            & (depth_mm > 0)
            & (np.abs(depth_mm - float(hand_depth_mm)) <= tolerance)
        )
        return ((core > 0) | ((support > 0) & depth_consistent))

    def _sample_hand_depth(self, depth_mm, points):
        samples = {}
        quality = {}
        valid_count = 0
        total_count = 0
        previous_depth = None
        radii = {
            "wrist": HAND_DEPTH_SAMPLE_RADIUS,
            "index_mcp": HAND_DEPTH_SAMPLE_RADIUS,
            "index_pip": max(2, HAND_DEPTH_SAMPLE_RADIUS - 1),
            "index_dip": max(2, HAND_DEPTH_SAMPLE_RADIUS - 2),
            "index_tip": max(2, HAND_DEPTH_SAMPLE_RADIUS - 2),
        }
        for name in ("wrist", "index_mcp", "index_pip", "index_dip", "index_tip"):
            x, y = points[name]
            radius = radii[name]
            patch = depth_mm[
                max(0, y - radius):min(self.frame_h, y + radius + 1),
                max(0, x - radius):min(self.frame_w, x + radius + 1),
            ]
            valid = patch[np.isfinite(patch) & (patch > 0)]
            valid_count += int(valid.size)
            total_count += int(patch.size)
            if not valid.size:
                samples[name] = None
                quality[name] = {
                    "valid_ratio": 0.0,
                    "inlier_ratio": 0.0,
                    "mad_mm": None,
                    "reliable": False,
                }
                continue

            center = previous_depth if previous_depth is not None else float(np.median(valid))
            tolerance = max(90.0, center * 0.18)
            inliers = valid[np.abs(valid - center) <= tolerance]
            min_inliers = max(3, int(math.ceil(valid.size * 0.25)))
            reliable = inliers.size >= min_inliers
            value = float(np.median(inliers)) if reliable else None
            mad_mm = (
                float(np.median(np.abs(inliers - value)))
                if reliable
                else None
            )
            samples[name] = value
            quality[name] = {
                "valid_ratio": float(valid.size / max(1, patch.size)),
                "inlier_ratio": float(inliers.size / max(1, valid.size)),
                "mad_mm": mad_mm,
                "reliable": bool(reliable and mad_mm <= 60.0),
            }
            if quality[name]["reliable"]:
                previous_depth = value

        raw_valid_ratio = valid_count / max(1, total_count)
        quality_ratio = sum(
            item["valid_ratio"] * item["inlier_ratio"]
            if item["reliable"]
            else 0.0
            for item in quality.values()
        ) / max(1, len(quality))
        # Use proximal landmarks for the stage-4 ray origin. DIP/TIP are still
        # measured and reported, but their depth often contains background
        # bleed and is reserved for the quality-gated 3D-ray stage.
        proximal = [
            samples[name]
            for name in ("wrist", "index_mcp", "index_pip")
            if samples[name] is not None and quality[name]["reliable"]
        ]
        proximal_center = float(np.median(proximal)) if proximal else None
        proximal_inliers = (
            [
                value
                for value in proximal
                if abs(value - proximal_center)
                <= max(120.0, proximal_center * 0.25)
            ]
            if proximal_center is not None
            else []
        )
        hand_depth = None
        if (
            len(proximal_inliers) >= 2
            and raw_valid_ratio >= HAND_DEPTH_MIN_VALID_RATIO
        ):
            # The RGB ray starts at index MCP, so use its aligned metric depth
            # when it agrees with the robust proximal-hand estimate.
            mcp_depth = samples["index_mcp"]
            if mcp_depth in proximal_inliers:
                hand_depth = float(mcp_depth)
            else:
                hand_depth = float(np.median(proximal_inliers))
        return samples, quality, hand_depth, quality_ratio

    def _find_metric_background_hit(
        self, depth_mm, hand_mask, hand_depth_mm, start_px, tip_px
    ):
        """Find the first stable metric-depth surface along the RGB finger ray.

        This is a 2.5D stage: the image direction comes from RGB landmarks and
        the ray distance grows from the measured hand depth. Joint-to-joint 3D
        direction is deliberately deferred until landmark depth is reliable.
        """
        if hand_depth_mm is None or not np.isfinite(hand_depth_mm):
            return None

        mx, my = start_px
        tx, ty = tip_px
        dx = tx - mx
        dy = ty - my
        finger_length_px = math.hypot(dx, dy)
        if finger_length_px < 1.0:
            return None

        ux = dx / finger_length_px
        uy = dy / finger_length_px
        start_offset = finger_length_px + RAY_START_MARGIN_PX
        radius = BACKGROUND_DEPTH_PATCH_RADIUS
        candidates = []

        for step in range(RAY_MAX_STEPS):
            distance_px = start_offset + step * RAY_STEP_PX
            cx = int(round(mx + ux * distance_px))
            cy = int(round(my + uy * distance_px))
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                break
            if hand_mask[cy, cx]:
                candidates.clear()
                continue

            x1 = max(0, cx - radius)
            x2 = min(self.frame_w, cx + radius + 1)
            y1 = max(0, cy - radius)
            y2 = min(self.frame_h, cy + radius + 1)
            patch = depth_mm[y1:y2, x1:x2]
            allowed = ~hand_mask[y1:y2, x1:x2]
            valid = patch[allowed & np.isfinite(patch) & (patch > 0)]
            allowed_count = int(np.count_nonzero(allowed))
            valid_ratio = valid.size / max(1, allowed_count)
            if valid_ratio < BACKGROUND_DEPTH_MIN_VALID_RATIO:
                candidates.clear()
                continue

            surface_depth_mm = float(np.median(valid))
            mad_mm = float(np.median(np.abs(valid - surface_depth_mm)))
            if mad_mm > BACKGROUND_DEPTH_MAX_MAD_MM:
                candidates.clear()
                continue

            # At the fingertip (distance/finger_length == 1), the ray depth
            # equals the measured hand depth. It grows along the RGB direction
            # until it reaches a measured background surface.
            ray_depth_mm = float(hand_depth_mm) * (
                distance_px / finger_length_px
            )
            hit = ray_depth_mm >= surface_depth_mm * (
                1.0 - BACKGROUND_DEPTH_HIT_TOLERANCE
            )
            if not hit:
                candidates.clear()
                continue

            if (
                candidates
                and abs(surface_depth_mm - candidates[-1]["surface_depth_mm"])
                > BACKGROUND_DEPTH_MAX_STEP_JUMP_MM
            ):
                candidates.clear()

            candidates.append(
                {
                    "target": (cx, cy),
                    "surface_depth_mm": surface_depth_mm,
                    "ray_depth_mm": ray_depth_mm,
                    "valid_ratio": valid_ratio,
                }
            )
            if len(candidates) >= BACKGROUND_DEPTH_CONFIRM_STEPS:
                return candidates[0]

        return None

    def _submit_depth_job(self, frame_bgr):
        with self._async_condition:
            self._job_sequence += 1
            self._pending_job = {
                "sequence": self._job_sequence,
                "generation": self._generation,
                "captured_at": time.monotonic(),
                "frame": frame_bgr.copy(),
            }
            self._async_condition.notify()

    def _depth_worker(self):
        last_start = 0.0
        interval = 1.0 / DEPTH_ASYNC_FPS
        while True:
            with self._async_condition:
                while self._pending_job is None and not self._async_stop:
                    self._async_condition.wait()
                if self._async_stop:
                    return
                job = self._pending_job
                self._pending_job = None

                while True:
                    remaining = interval - (time.monotonic() - last_start)
                    if remaining <= 0 or self._async_stop:
                        break
                    self._async_condition.wait(timeout=remaining)
                    if self._pending_job is not None:
                        job = self._pending_job
                        self._pending_job = None
                if self._async_stop:
                    return

            last_start = time.monotonic()
            try:
                depth_map = self.depth_est.estimate(job["frame"])
                payload = {
                    "depth_map": depth_map,
                    "depth_error": None,
                    "completed_at": time.monotonic(),
                }
            except Exception as exc:
                payload = {
                    "depth_map": None,
                    "depth_error": str(exc),
                    "completed_at": time.monotonic(),
                }

            payload.update(
                {
                    "sequence": job["sequence"],
                    "generation": job["generation"],
                    "captured_at": job["captured_at"],
                }
            )
            with self._async_condition:
                self._async_result = payload

    def _consume_async_result(self, points, ray_start, ray_tip):
        with self._async_condition:
            payload = dict(self._async_result) if self._async_result else None

        if (
            payload is not None
            and payload["generation"] == self._generation
            and payload["sequence"] > self._consumed_sequence
        ):
            self._consumed_sequence = payload["sequence"]
            if payload.get("depth_map") is not None:
                self.depth_map = payload["depth_map"]
                self._latest_depth_captured_at = payload["captured_at"]
                self._latest_depth_error = None
            else:
                self._latest_depth_error = payload.get("depth_error")

        now = time.monotonic()
        depth_age = (
            max(0.0, now - self._latest_depth_captured_at)
            if self._latest_depth_captured_at is not None
            else None
        )
        depth_is_fresh = (
            self.depth_map is not None
            and depth_age is not None
            and depth_age <= DEPTH_RESULT_MAX_AGE_SECONDS
        )

        # MiDaS remains rate-limited in the worker, but the lightweight ray
        # intersection and EMA use the current hand landmarks every frame.
        if depth_is_fresh:
            target_payload = self._target_from_depth(
                self.depth_map,
                points,
                ray_start,
                ray_tip,
            )
            result = self.tracker.update(target_payload["raw_target"], True)
            result.update(target_payload)
            result["async_pending"] = False
            result["depth_age_seconds"] = depth_age
            self._last_result = result
            return result

        # If depth inference failed before producing any usable map, retain the
        # existing 2D fallback while continuing to request fresh depth frames.
        if self._latest_depth_error:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)
            result = self.tracker.update(raw_target, True)
            result.update(
                {
                    "raw_target": raw_target,
                    "ray_start_px": ray_start,
                    "ray_tip_px": ray_tip,
                    "calibrated": False,
                    "depth_available": False,
                    "used_depth_hit": False,
                    "hit_method": "2d_fallback_depth_error",
                    "depth_error": self._latest_depth_error,
                    "depth_map": None,
                    "async_pending": False,
                    "ray_mode": self.ray_mode,
                }
            )
            self._last_result = result
            return result

        # Before the first fresh depth map (or while a previous map is stale),
        # keep the last target but draw the ray from the current hand pose.
        result = dict(self._last_result)
        result["ray_start_px"] = ray_start
        result["ray_tip_px"] = ray_tip
        result["async_pending"] = True
        result["ray_mode"] = self.ray_mode
        if depth_age is not None:
            result["depth_age_seconds"] = depth_age
        return result

    def _target_from_depth(self, depth_map, points, ray_start, ray_tip):
        if not self.depth_est.is_calibrated:
            self.depth_est.calibrate(
                depth_map,
                points["wrist"],
                points["middle_mcp"],
                self.cam_fx,
            )

        raw_target = self._march(depth_map, ray_start, ray_tip)
        used_depth_hit = raw_target is not None
        if raw_target is None:
            raw_target = self._project_to_screen_edge(ray_start, ray_tip)

        return {
            "raw_target": raw_target,
            "ray_start_px": ray_start,
            "ray_tip_px": ray_tip,
            "calibrated": self.depth_est.is_calibrated,
            "depth_available": True,
            "used_depth_hit": used_depth_hit,
            "hit_method": "depth_march" if used_depth_hit else "2d_fallback_after_depth",
            "depth_error": None,
            "depth_map": depth_map,
            "ray_mode": self.ray_mode,
        }

    def reset(self, clear_confirmed=False):
        self.tracker.reset(clear_confirmed=clear_confirmed)
        self._last_result = self._empty_result()
        self.depth_map = None
        self._latest_depth_captured_at = None
        self._latest_depth_error = None
        with self._async_condition:
            self._generation += 1
            self._pending_job = None
            self._async_result = None
            self._consumed_sequence = self._job_sequence

    def close(self):
        if not self.async_depth:
            return
        with self._async_condition:
            self._async_stop = True
            self._async_condition.notify_all()
        if self._async_thread is not None:
            self._async_thread.join(timeout=2.0)

    def _key_points(self, landmarks):
        def to_px(lm):
            return (
                int(np.clip(lm.x, 0.0, 1.0) * (self.frame_w - 1)),
                int(np.clip(lm.y, 0.0, 1.0) * (self.frame_h - 1)),
            )

        return {
            "wrist": to_px(landmarks[self.WRIST]),
            "index_mcp": to_px(landmarks[self.INDEX_MCP]),
            "index_pip": to_px(landmarks[self.INDEX_PIP]),
            "index_dip": to_px(landmarks[self.INDEX_DIP]),
            "index_tip": to_px(landmarks[self.INDEX_TIP]),
            "middle_mcp": to_px(landmarks[self.MIDDLE_MCP]),
        }

    def _ray_segment(self, points):
        if self.ray_mode == "mcp_tip":
            return points["index_mcp"], points["index_tip"]

        joints = np.asarray(
            [
                points["index_pip"],
                points["index_dip"],
                points["index_tip"],
            ],
            dtype=np.float32,
        )
        centered = joints - joints.mean(axis=0)
        _, _, axes = np.linalg.svd(centered, full_matrices=False)
        direction = axes[0]
        forward = joints[-1] - joints[0]
        if float(np.dot(direction, forward)) < 0:
            direction = -direction
        segment_length = max(float(np.linalg.norm(forward)), 1.0)
        start = joints[0]
        tip = start + direction * segment_length
        return tuple(start.tolist()), tuple(tip.tolist())

    def _project_to_screen_edge(self, start_px, tip_px):
        mx, my = start_px
        tx, ty = tip_px
        dx = tx - mx
        dy = ty - my
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None

        ux = dx / dist
        uy = dy / dist
        start = dist + 10
        last = None
        max_steps = int(max(self.frame_w, self.frame_h) / RAY_STEP_PX) + 50
        for step in range(max_steps):
            t = start + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                return last
            last = (cx, cy)
        return last

    def _march(self, depth_map, start_px, tip_px):
        mx, my = start_px
        tx, ty = tip_px
        dx = tx - mx
        dy = ty - my
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            return None

        ux = dx / dist
        uy = dy / dist
        mx_c = int(np.clip(mx, 0, self.frame_w - 1))
        my_c = int(np.clip(my, 0, self.frame_h - 1))
        start_midas = float(depth_map[my_c, mx_c])
        start_depth_m = self.depth_est.midas_to_meters(start_midas)
        start_offset = dist + RAY_START_MARGIN_PX

        for step in range(RAY_MAX_STEPS):
            t = start_offset + step * RAY_STEP_PX
            cx = int(mx + ux * t)
            cy = int(my + uy * t)
            if cx < 0 or cx >= self.frame_w or cy < 0 or cy >= self.frame_h:
                return None

            surface_midas = float(depth_map[cy, cx])
            surface_depth_m = self.depth_est.midas_to_meters(surface_midas)
            ray_depth_m = start_depth_m * (t / dist)
            if ray_depth_m >= surface_depth_m * (1.0 - DEPTH_HIT_THRESHOLD):
                return (cx, cy)
        return None
