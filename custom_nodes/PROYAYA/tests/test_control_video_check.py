import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from control_video_check import PROYAYA_ControlVideoCheck  # noqa: E402


class ControlVideoCheckTests(unittest.TestCase):
    def setUp(self):
        self.node = PROYAYA_ControlVideoCheck()
        self.image = torch.zeros((330, 4, 4, 3), dtype=torch.float32)

    def test_accepts_24fps_and_crops_extra_tail(self):
        (result,) = self.node.validate(self.image, 24.0, 328)
        self.assertEqual(tuple(result.shape), (328, 4, 4, 3))

    def test_rejects_non_24fps(self):
        with self.assertRaisesRegex(ValueError, "24 fps"):
            self.node.validate(self.image, 30.0, 328)

    def test_rejects_short_control_video(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            self.node.validate(self.image[:100], 24.0, 328)


if __name__ == "__main__":
    unittest.main()
