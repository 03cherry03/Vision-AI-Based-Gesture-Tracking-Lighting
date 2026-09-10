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


if __name__ == "__main__":
    unittest.main()
