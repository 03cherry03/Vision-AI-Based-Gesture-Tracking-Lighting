"""Unit tests for Gemini metric-depth background surface search."""

import unittest

import numpy as np

from src.vision.pointing_target import PointingTargetEstimator


class GeminiDepthSurfaceTest(unittest.TestCase):
    def setUp(self):
        self.estimator = PointingTargetEstimator.__new__(PointingTargetEstimator)
        self.estimator.frame_w = 640
        self.estimator.frame_h = 360

    def test_uniform_background_hit_is_not_forced_to_screen_edge(self):
        depth_mm = np.full((360, 640), 2000.0, dtype=np.float32)
        hand_mask = np.zeros((360, 640), dtype=bool)
        hand_mask[150:211, 270:371] = True

        hit = self.estimator._find_metric_background_hit(
            depth_mm,
            hand_mask,
            hand_depth_mm=500.0,
            start_px=(300, 180),
            tip_px=(350, 180),
        )

        self.assertIsNotNone(hit)
        self.assertGreater(hit["target"][0], 450)
        self.assertLess(hit["target"][0], 550)
        self.assertEqual(hit["target"][1], 180)
        self.assertAlmostEqual(hit["surface_depth_mm"], 2000.0)

    def test_missing_hand_depth_does_not_create_fallback_target(self):
        depth_mm = np.full((360, 640), 2000.0, dtype=np.float32)
        hand_mask = np.zeros((360, 640), dtype=bool)

        hit = self.estimator._find_metric_background_hit(
            depth_mm,
            hand_mask,
            hand_depth_mm=None,
            start_px=(300, 180),
            tip_px=(350, 180),
        )

        self.assertIsNone(hit)

    def test_distal_background_outlier_is_rejected(self):
        depth_mm = np.full((360, 640), 2000.0, dtype=np.float32)
        points = {
            "wrist": (100, 180),
            "index_mcp": (130, 180),
            "index_pip": (160, 180),
            "index_dip": (190, 180),
            "index_tip": (220, 180),
        }
        for name, value in (
            ("wrist", 500.0),
            ("index_mcp", 510.0),
            ("index_pip", 520.0),
            ("index_dip", 530.0),
        ):
            x, y = points[name]
            depth_mm[y - 6:y + 7, x - 6:x + 7] = value

        samples, quality, hand_depth, valid_ratio = (
            self.estimator._sample_hand_depth(depth_mm, points)
        )

        self.assertEqual(hand_depth, 510.0)
        self.assertIsNone(samples["index_tip"])
        self.assertFalse(quality["index_tip"]["reliable"])
        self.assertGreater(valid_ratio, 0.7)
        self.assertLess(valid_ratio, 0.9)


if __name__ == "__main__":
    unittest.main()
