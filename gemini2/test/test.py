import cv2
import numpy as np

from pyorbbecsdk import (
    Pipeline,
    Config,
    OBSensorType,
    OBFormat,
    OBFrameAggregateOutputMode
)

pipeline = Pipeline()
config = Config()

# --------------------------------------------------
# RGB
# Gemini 2 최대 RGB: 1920x1080 @ 30 FPS
# --------------------------------------------------
color_profiles = pipeline.get_stream_profile_list(
    OBSensorType.COLOR_SENSOR
)

color_profile = color_profiles.get_video_stream_profile(
    1920,
    1080,
    OBFormat.MJPG,
    30
)

# --------------------------------------------------
# DEPTH
# Gemini 2 최대 Depth: 1280x800 @ 30 FPS
# --------------------------------------------------
depth_profiles = pipeline.get_stream_profile_list(
    OBSensorType.DEPTH_SENSOR
)

depth_profile = depth_profiles.get_video_stream_profile(
    1280,
    800,
    OBFormat.Y16,
    30
)

config.enable_stream(color_profile)
config.enable_stream(depth_profile)

# RGB + Depth가 같은 Frameset으로 오도록 설정
config.set_frame_aggregate_output_mode(
    OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE
)

# --------------------------------------------------
# Gemini 2 권장 Depth 필터 가져오기
# Spatial / Temporal 등이 장치에 따라 포함될 수 있음
# --------------------------------------------------
device = pipeline.get_device()
depth_sensor = device.get_sensor(OBSensorType.DEPTH_SENSOR)

filters = depth_sensor.get_recommended_filters()

print("=== Recommended Depth Filters ===")
for i, f in enumerate(filters):
    try:
        print(i, type(f).__name__, "enabled =", f.is_enabled())
    except Exception:
        print(i, type(f).__name__)

pipeline.start(config)

print()
print("RGB : 1920x1080 @ 30 FPS")
print("Depth: 1280x800 @ 30 FPS")
print("Press q to quit.")

MIN_DEPTH_MM = 150
MAX_DEPTH_MM = 5000

try:
    while True:

        frames = pipeline.wait_for_frames(1000)

        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if color_frame is None or depth_frame is None:
            continue

        # ==================================================
        # 1. RGB 변환
        # ==================================================
        color_data = np.frombuffer(
            color_frame.get_data(),
            dtype=np.uint8
        )

        # MJPEG decode
        color_img = cv2.imdecode(
            color_data,
            cv2.IMREAD_COLOR
        )

        if color_img is None:
            continue

        # ==================================================
        # 2. DEPTH FILTER
        # ==================================================
        filtered_frame = depth_frame

        for f in filters:
            try:
                if f.is_enabled():
                    result = f.process(filtered_frame)

                    if result is not None:
                        filtered_frame = result
            except Exception:
                pass

        # 필터 결과를 DepthFrame으로 변환
        try:
            filtered_depth = filtered_frame.as_depth_frame()
        except Exception:
            filtered_depth = depth_frame

        # ==================================================
        # 3. Depth numpy 변환
        # ==================================================
        dw = filtered_depth.get_width()
        dh = filtered_depth.get_height()

        depth_data = np.frombuffer(
            filtered_depth.get_data(),
            dtype=np.uint16
        ).reshape((dh, dw))

        # ==================================================
        # 4. 고정 거리 범위로 색상 표시
        # frame마다 normalize하지 않음
        # ==================================================
        clipped = np.clip(
            depth_data,
            MIN_DEPTH_MM,
            MAX_DEPTH_MM
        )

        depth_vis = (
            (clipped - MIN_DEPTH_MM)
            / (MAX_DEPTH_MM - MIN_DEPTH_MM)
            * 255
        ).astype(np.uint8)

        depth_vis[depth_data == 0] = 0

        depth_color = cv2.applyColorMap(
            depth_vis,
            cv2.COLORMAP_JET
        )

        # ==================================================
        # 화면 표시용으로 RGB 크기 축소
        # 실제 RGB 데이터 자체는 1920x1080 유지
        # ==================================================
        display_rgb = cv2.resize(
            color_img,
            (960, 540)
        )

        display_depth = cv2.resize(
            depth_color,
            (864, 540)
        )

        cv2.putText(
            display_rgb,
            "RGB 1920x1080 @ 30",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            display_depth,
            "Filtered Depth 1280x800 @ 30",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Gemini 2 RGB",
            display_rgb
        )

        cv2.imshow(
            "Gemini 2 Filtered Depth",
            display_depth
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
