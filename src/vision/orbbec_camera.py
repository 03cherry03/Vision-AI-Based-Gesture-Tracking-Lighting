"""Orbbec Gemini 2 synchronized RGB-D camera adapter.

This module intentionally has no hard import-time dependency on pyorbbecsdk.
The main runtime can keep using the Raspberry Pi camera until the Orbbec
backend is explicitly selected and tested.

Depth returned by :meth:`OrbbecCamera.read` is aligned to the color image and
expressed in millimetres. A value of 0 means that the camera did not provide a
valid measurement for that color pixel.
"""

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass
class RGBDFrame:
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    depth_valid_mask: np.ndarray
    timestamp_ms: float
    intrinsics: Optional[CameraIntrinsics] = None


def _first_profile(profile_list: Any) -> Any:
    """Return the first SDK profile across wrapper API variants."""
    try:
        return profile_list[0]
    except (TypeError, AttributeError):
        pass
    for count_name in ("get_count", "count"):
        count = getattr(profile_list, count_name, None)
        if callable(count) and count() > 0:
            return profile_list.get_profile(0)
    raise RuntimeError("The Orbbec SDK returned an empty stream profile list")


def _video_profile(profile_list: Any, width: int, height: int, fmt: Any, fps: int) -> Any:
    try:
        return profile_list.get_video_stream_profile(width, height, fmt, fps)
    except Exception:
        # A profile combination differs between firmware revisions. Let the
        # SDK choose its default rather than failing before the diagnostic can
        # display the device capabilities.
        return _first_profile(profile_list)


class OrbbecCamera:
    """Capture synchronized color and depth frames from an Orbbec camera.

    Hardware depth-to-color alignment is preferred. If the selected device or
    profile cannot provide it, the adapter falls back to software D2C.
    """

    name = "orbbec_rgbd"

    def __init__(
        self,
        color_width: int = 1280,
        color_height: int = 720,
        depth_width: int = 640,
        depth_height: int = 400,
        fps: int = 30,
        timeout_ms: int = 1000,
        prefer_hardware_align: bool = True,
    ):
        try:
            import pyorbbecsdk as ob
        except ImportError as exc:
            raise RuntimeError(
                "pyorbbecsdk is not installed. Install the Orbbec SDK Python "
                "wrapper supplied for the Raspberry Pi architecture first."
            ) from exc

        self.ob = ob
        self.timeout_ms = int(timeout_ms)
        self.pipeline = ob.Pipeline()
        self.config = ob.Config()
        self.align_filter = None
        self.alignment_mode = "none"
        self._released = False

        color_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
        color_profile = _video_profile(
            color_profiles, color_width, color_height, ob.OBFormat.RGB, fps
        )

        depth_profile = None
        if prefer_hardware_align:
            try:
                compatible = self.pipeline.get_d2c_depth_profile_list(
                    color_profile, ob.OBAlignMode.HW_MODE
                )
                depth_profile = _first_profile(compatible)
                self.config.set_align_mode(ob.OBAlignMode.HW_MODE)
                self.alignment_mode = "hardware_d2c"
            except Exception:
                depth_profile = None

        if depth_profile is None:
            depth_profiles = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
            depth_profile = _video_profile(
                depth_profiles, depth_width, depth_height, ob.OBFormat.Y16, fps
            )
            self.align_filter = ob.AlignFilter(
                align_to_stream=ob.OBStreamType.COLOR_STREAM
            )
            self.alignment_mode = "software_d2c"

        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)
        aggregate_mode = getattr(
            ob.OBFrameAggregateOutputMode, "FULL_FRAME_REQUIRE", None
        )
        if aggregate_mode is not None:
            self.config.set_frame_aggregate_output_mode(aggregate_mode)

        try:
            self.pipeline.enable_frame_sync()
        except Exception:
            # Some SDK/firmware combinations synchronize the configured
            # FrameSet automatically and do not expose this call.
            pass

        try:
            self.pipeline.start(self.config)
            self.intrinsics = self._read_color_intrinsics()
        except Exception:
            self.release()
            raise

    def _read_color_intrinsics(self) -> Optional[CameraIntrinsics]:
        try:
            params = self.pipeline.get_camera_param()
            item = params.rgb_intrinsic
            return CameraIntrinsics(
                fx=float(item.fx),
                fy=float(item.fy),
                cx=float(item.cx),
                cy=float(item.cy),
                width=int(item.width),
                height=int(item.height),
            )
        except Exception:
            return None

    def read(self):
        """Return ``(True, RGBDFrame)`` or ``(False, None)`` on timeout."""
        frames = self.pipeline.wait_for_frames(self.timeout_ms)
        if frames is None:
            return False, None

        if self.align_filter is not None:
            aligned = self.align_filter.process(frames)
            if aligned is None:
                return False, None
            frames = aligned.as_frame_set()
            if frames is None:
                return False, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return False, None

        color_bgr = self._color_to_bgr(color_frame)
        depth_mm = self._depth_to_mm(depth_frame)
        if color_bgr is None or depth_mm is None:
            return False, None

        # D2C should already produce matching dimensions. Resize only as a
        # compatibility fallback and preserve invalid zeros with nearest-neighbour.
        color_h, color_w = color_bgr.shape[:2]
        if depth_mm.shape != (color_h, color_w):
            depth_mm = cv2.resize(
                depth_mm, (color_w, color_h), interpolation=cv2.INTER_NEAREST
            )

        timestamp = self._timestamp_ms(color_frame)
        packet = RGBDFrame(
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            depth_valid_mask=depth_mm > 0,
            timestamp_ms=timestamp,
            intrinsics=self.intrinsics,
        )
        return True, packet

    def _color_to_bgr(self, frame: Any) -> Optional[np.ndarray]:
        width = int(frame.get_width())
        height = int(frame.get_height())
        fmt = frame.get_format()
        data = np.frombuffer(frame.get_data(), dtype=np.uint8)
        ob_format = self.ob.OBFormat

        if fmt == ob_format.RGB:
            rgb = data.reshape((height, width, 3))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if hasattr(ob_format, "BGR") and fmt == ob_format.BGR:
            return data.reshape((height, width, 3)).copy()
        if fmt == ob_format.MJPG:
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        if fmt == ob_format.YUYV:
            yuyv = data.reshape((height, width, 2))
            return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUY2)
        if hasattr(ob_format, "UYVY") and fmt == ob_format.UYVY:
            uyvy = data.reshape((height, width, 2))
            return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
        raise RuntimeError(f"Unsupported Orbbec color format: {fmt}")

    @staticmethod
    def _depth_to_mm(frame: Any) -> np.ndarray:
        width = int(frame.get_width())
        height = int(frame.get_height())
        raw = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape((height, width))
        scale = float(frame.get_depth_scale())
        return raw.astype(np.float32) * scale

    @staticmethod
    def _timestamp_ms(frame: Any) -> float:
        for name in ("get_timestamp_us", "get_timestamp"):
            getter = getattr(frame, name, None)
            if callable(getter):
                value = float(getter())
                return value / 1000.0 if name.endswith("_us") else value
        return 0.0

    def release(self):
        if self._released:
            return
        self._released = True
        try:
            self.pipeline.stop()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

