"""Test CanonicalObs schema and standard action validation."""

import numpy as np
from benchlink.schema import CanonicalObs, STANDARD_ACTION_DIM, is_valid_standard_action


class TestCanonicalObs:
    """CanonicalObs construction and field behavior."""

    def test_construct_with_all_fields(self):
        obs = CanonicalObs(
            rgb_static=np.zeros((256, 256, 3), dtype=np.uint8),
            rgb_gripper=np.zeros((128, 128, 3), dtype=np.uint8),
            depth=np.zeros((256, 256), dtype=np.float32),
            proprio=np.zeros(7, dtype=np.float32),
            tactile_img=np.zeros((224, 224, 3), dtype=np.uint8),
            tactile_feat=np.zeros(64, dtype=np.float32),
            language="test task",
        )
        assert obs.rgb_static.shape == (256, 256, 3)
        assert obs.proprio.shape == (7,)
        assert obs.language == "test task"

    def test_default_none(self):
        obs = CanonicalObs()
        assert obs.rgb_static is None
        assert obs.rgb_gripper is None
        assert obs.language is None

    def test_extra_dict(self):
        obs = CanonicalObs(extra={"custom": "value"})
        assert obs.extra["custom"] == "value"


class TestStandardAction:
    """Standard action format and validation."""

    def test_dimension(self):
        assert STANDARD_ACTION_DIM == 7

    def test_valid_action(self):
        action = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
        assert is_valid_standard_action(action) is True

    def test_invalid_shape(self):
        action = np.array([0.1, 0.0], dtype=np.float64)
        assert is_valid_standard_action(action) is False

    def test_invalid_dtype(self):
        action = np.zeros(7, dtype=np.int32)
        assert is_valid_standard_action(action) is False

    def test_nan_rejected(self):
        action = np.array([0.1, np.nan, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
        assert is_valid_standard_action(action) is False

    def test_inf_rejected(self):
        action = np.array([np.inf, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], dtype=np.float64)
        assert is_valid_standard_action(action) is False
