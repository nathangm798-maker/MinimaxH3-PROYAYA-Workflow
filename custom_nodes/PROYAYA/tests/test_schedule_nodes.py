"""Check editable schedules against the installed native H3 schedulers."""

import importlib
import unittest

import test_runtime as base

torch = base.torch
schedule_nodes = importlib.import_module(base.package_name + ".schedule_nodes")


class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.sampling = base.AVSampling()
        self.sampling.set_parameters(shift=12.0, audio_shift=3.0)

    def test_all_native_schedulers_have_requested_length_and_exact_endpoints(self):
        before = self.sampling.sigmas.clone()
        for name in base.comfy.samplers.SCHEDULER_NAMES:
            for steps in (1, 5, 8, 17, 40):
                for start in (0.2, 0.9, 1.0):
                    with self.subTest(name=name, steps=steps, start=start):
                        result = schedule_nodes.schedule_from_start(self.sampling, name, steps, start)
                        self.assertEqual(len(result), steps + 1)
                        self.assertEqual(result[0], torch.tensor(start, dtype=torch.float32))
                        self.assertEqual(result[-1], 0)
                        self.assertTrue(bool((result[:-1] > result[1:]).all()))
                        self.assertEqual(result.device.type, "cpu")
        torch.testing.assert_close(before, self.sampling.sigmas, rtol=0, atol=0)

    def test_full_simple_schedule_is_native_bit_exact(self):
        for steps in (8, 20, 64):
            expected = base.comfy.samplers.calculate_sigmas(self.sampling, "simple", steps)
            actual = schedule_nodes.schedule_from_start(self.sampling, "simple", steps, 1.0)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_more_steps_than_native_grid_can_still_progress(self):
        for start in (0.9, 1.0):
            result = schedule_nodes.schedule_from_start(self.sampling, "simple", 1500, start)
            self.assertEqual(len(result), 1501)
            self.assertTrue(bool((result[:-1] > result[1:]).all()))

    def test_invalid_counts_and_nonphysical_start_rejected(self):
        for steps, start in ((0, .9), (1.5, .9), (True, .9), (8, 0), (8, 1.01), (8, float("nan"))):
            with self.subTest(steps=steps, start=start), self.assertRaises(ValueError):
                schedule_nodes.schedule_from_start(self.sampling, "simple", steps, start)


if __name__ == "__main__":
    unittest.main()
