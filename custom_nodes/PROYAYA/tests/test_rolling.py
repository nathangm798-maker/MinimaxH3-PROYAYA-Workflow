"""Numerical contracts for forward, same-sigma H3 AV rolling (no model)."""

import importlib
from pathlib import Path
import sys
import types
import unittest

import torch


PACKAGE = "_h3_rolling_contract_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[PACKAGE] = package
windows = importlib.import_module(PACKAGE + ".windows")
rolling = importlib.import_module(PACKAGE + ".rolling")
Window = windows.Window
RollingConfig = rolling.RollingConfig
CPURollingFusion = rolling.CPURollingFusion


def three_windows():
    return [Window(start, 124, 1, 1) for start in (0, 102, 204)]


def random_state(fusion, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return windows.pack([torch.randn(shape, generator=generator) for shape in fusion.shapes])


class RollingContracts(unittest.TestCase):
    def test_inactive_single_window_and_zero_strength_are_bit_exact_v1(self):
        cases = [
            (three_windows(), RollingConfig(False, 0.45), 0.25),
            (three_windows(), RollingConfig(True, 0.0), 0.25),
            (three_windows()[:1], RollingConfig(True, 0.45), 0.25),
            (three_windows()[:1], RollingConfig(True, 0.45), 1.0),
        ]
        for chosen, config, sigma in cases:
            with self.subTest(windows=len(chosen), config=config, sigma=sigma):
                baseline = windows.CPUFusion(chosen)
                candidate = CPURollingFusion(chosen, config)
                state = random_state(candidate, 11)
                noise = random_state(candidate, 22)
                candidate.start_run(noise)
                candidate.sigma = sigma
                before_rng = torch.random.get_rng_state().clone()
                predictor = lambda index, local: local.square() * 0.125 + index
                expected = baseline.predict(state, predictor)
                actual = candidate.predict(state, predictor)
                self.assertTrue(torch.equal(actual, expected))
                self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))
                self.assertEqual(candidate.report()["video_latent_positions"], 0)
                self.assertEqual(candidate.report()["audio_latent_positions"], 0)

    def test_sigma_one_carries_no_donor_x0_information(self):
        """At sigma=1 the resample is pure noise, so the donor x0 cannot transfer.

        The flat curve still runs the mechanism at sigma=1 (effective_strength is
        the full strength), but donor_share is exactly zero, so two completely
        different donor predictions must leave window 2's input identical.
        """
        seen = {}
        for donor_value in (4.0, 9.0):
            fusion = CPURollingFusion(three_windows(), RollingConfig(True, 0.45, "resample"))
            noise = random_state(fusion, 22)
            fusion.start_run(noise, noise_scale=1.0)
            fusion.sigma = 1.0
            received = []

            def predict(index, local, donor=donor_value):
                received.append(windows.unpack(local.clone(), fusion.windows[index].shapes))
                return torch.full_like(local, donor)

            fusion.predict(noise.clone(), predict)  # sampler holds exactly noise at step 1
            seen[donor_value] = received[1]
            self.assertAlmostEqual(fusion.report()["steps"][0]["effective_strength"], 0.45)
            self.assertAlmostEqual(fusion.report()["steps"][0]["donor_share"], 0.0)

        torch.testing.assert_close(seen[4.0][0], seen[9.0][0])
        torch.testing.assert_close(seen[4.0][1], seen[9.0][1])

    def test_hard_mode_injects_donor_x0_even_at_sigma_one(self):
        """Hard mode skips the (1-sigma) rescale, so sigma=1 does transfer."""
        fusion = CPURollingFusion(three_windows(), RollingConfig(True, 1.0, "hard"))
        state = torch.zeros_like(random_state(fusion, 11))
        noise = random_state(fusion, 22)  # nonzero: must be ignored in hard mode
        fusion.start_run(noise, noise_scale=3.0)
        fusion.sigma = 1.0
        seen = []

        def predict(index, local):
            seen.append([part.clone() for part in windows.unpack(local, fusion.windows[index].shapes)])
            return torch.full_like(local, 4.0 * (index + 1))

        fusion.predict(state, predict)
        # beta=1 in hard mode: window 2's overlap is exactly window 1's x0 (4.0),
        # regardless of sigma=1, noise_scale=3.0 and the nonzero global noise.
        for index, expected in ((1, 4.0), (2, 8.0)):
            self.assertTrue(torch.equal(seen[index][0][:, :, :7],
                                        torch.full_like(seen[index][0][:, :, :7], expected)))
            self.assertTrue(torch.equal(seen[index][1][..., :37],
                                        torch.full_like(seen[index][1][..., :37], expected)))
        steps = fusion.report()["steps"]
        self.assertEqual(steps[0]["donor_share"], 1.0)
        self.assertEqual(fusion.report()["mode"], "hard")
        self.assertEqual(fusion.report()["transfer_bound"], "beta")
        self.assertEqual(fusion.report()["video_latent_positions"], 14)
        # Half strength must land exactly between the donor x0 and the local value.
        half = CPURollingFusion(three_windows(), RollingConfig(True, 0.5, "hard"))
        half_state = torch.full_like(state, 2.0)
        half.start_run(torch.zeros_like(noise), noise_scale=1.0)
        half.sigma = 1.0
        got = []

        def predict_half(index, local):
            got.append([p.clone() for p in windows.unpack(local, half.windows[index].shapes)])
            return torch.full_like(local, 4.0 * (index + 1))

        half.predict(half_state, predict_half)
        torch.testing.assert_close(got[1][0][:, :, :7], torch.full_like(got[1][0][:, :, :7], 3.0))
        torch.testing.assert_close(got[1][1][..., :37], torch.full_like(got[1][1][..., :37], 3.0))

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            RollingConfig(True, 0.5, "blend")

    def test_same_sigma_previous_prediction_reaches_next_window_on_both_grids(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig(True, 1.0))
        state = torch.zeros_like(random_state(fusion, 11))
        noise = torch.zeros_like(state)
        fusion.start_run(noise)
        fusion.sigma = 0.25
        seen = []

        def predict(index, local):
            seen.append([part.clone() for part in windows.unpack(local, fusion.windows[index].shapes)])
            return torch.full_like(local, float(index + 1))

        result = fusion.predict(state, predict)
        # At sigma=.25 and beta=1, donor x0 contributes .75. The third
        # window must get window 2's prediction (2), not window 1's (1).
        for index, expected in ((1, 0.75), (2, 1.5)):
            self.assertTrue(torch.equal(seen[index][0][:, :, :7], torch.full_like(seen[index][0][:, :, :7], expected)))
            self.assertTrue(torch.equal(seen[index][1][..., :37], torch.full_like(seen[index][1][..., :37], expected)))
            self.assertEqual(torch.count_nonzero(seen[index][0][:, :, 7:]).item(), 0)
            self.assertEqual(torch.count_nonzero(seen[index][1][..., 37:]).item(), 0)
        self.assertEqual(fusion.report()["video_latent_positions"], 14)
        self.assertEqual(fusion.report()["audio_latent_positions"], 74)
        # Final assembly keeps the V1 overlap weighting, rather than replacing
        # shared output with the latest writer or concatenating duplicate frames.
        expected = windows.CPUFusion(fusion.windows).predict(state, lambda i, x: torch.full_like(x, i + 1))
        self.assertTrue(torch.equal(result, expected))
        self.assertEqual([tuple(s.shape) for s in windows.unpack(result, fusion.shapes)],
                         [(1, 24, 97, 1, 1), (1, 32, 2, 547)])

    def test_global_state_noise_and_rng_are_not_mutated_by_destructive_predictor(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig())
        state, noise = random_state(fusion, 11), random_state(fusion, 22)
        before_state, before_noise = state.clone(), noise.clone()
        before_rng = torch.random.get_rng_state().clone()
        fusion.start_run(noise)
        self.assertIs(fusion.noise, noise)
        fusion.sigma = 0.4
        fusion.predict(state, lambda index, local: local.fill_(float(index + 2)))
        self.assertTrue(torch.equal(state, before_state))
        self.assertTrue(torch.equal(noise, before_noise))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))

    def test_original_coordinate_noise_and_model_noise_scale_are_used(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig(True, 0.4))
        state = torch.full_like(random_state(fusion, 11), 8.0)
        noise_streams = []
        for kind, shape in enumerate(fusion.shapes):
            dim = 2 if kind == 0 else 3
            values = torch.arange(shape[dim], dtype=torch.float32)
            view = [1] * len(shape)
            view[dim] = shape[dim]
            noise_streams.append(values.reshape(view).expand(shape).clone())
        fusion.start_run(windows.pack(noise_streams), noise_scale=2.5)
        fusion.sigma = 0.75  # flat curve gives beta = .4 at every sigma
        seen = []

        def predict(index, local):
            seen.append(windows.unpack(local.clone(), fusion.windows[index].shapes))
            return torch.full_like(local, 4.0)

        fusion.predict(state, predict)
        expected_video = 0.6 * 8.0 + 0.4 * (0.25 * 4.0 + 0.75 * 2.5 * torch.arange(30, 37))
        expected_audio = 0.6 * 8.0 + 0.4 * (0.25 * 4.0 + 0.75 * 2.5 * torch.arange(170, 207))
        torch.testing.assert_close(seen[1][0][0, 0, :7, 0, 0], expected_video)
        torch.testing.assert_close(seen[1][1][0, 0, 0, :37], expected_audio)
        self.assertAlmostEqual(fusion.report()["steps"][0]["effective_strength"], 0.4)
        self.assertAlmostEqual(fusion.report()["steps"][0]["donor_share"], 0.25)
        self.assertEqual(fusion.report()["noise_scale"], 2.5)

    def test_variable_windows_and_three_way_overlap_use_latest_writer(self):
        chosen = [Window(0, 158, 1, 1), Window(51, 124, 1, 1), Window(102, 141, 1, 1)]
        fusion = CPURollingFusion(chosen, RollingConfig(True, 1.0))
        state = torch.zeros_like(random_state(fusion, 11))
        fusion.start_run(torch.zeros_like(state))
        fusion.sigma = 0.5
        seen = []

        def predict(index, local):
            seen.append(windows.unpack(local.clone(), chosen[index].shapes))
            return torch.full_like(local, 10.0 * (index + 1))

        fusion.predict(state, predict)
        # Window 3 overlaps both earlier windows. Every shared position belongs
        # to the most recent prediction (window 2: x0=20), including triple areas.
        self.assertTrue(bool((seen[2][0][:, :, :22] == 10.0).all()))
        self.assertTrue(bool((seen[2][1][..., :122] == 10.0).all()))
        self.assertEqual(torch.count_nonzero(seen[2][0][:, :, 22:]).item(), 0)
        self.assertEqual(torch.count_nonzero(seen[2][1][..., 122:]).item(), 0)
        self.assertEqual(fusion.shapes, [(1, 24, 72, 1, 1), (1, 32, 2, 405)])

    def test_zero_noise_scale_uses_clean_donor_and_disabled_is_bit_exact_v1(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig(True, 1.0))
        state, noise = random_state(fusion, 11), random_state(fusion, 22)
        fusion.start_run(noise, noise_scale=0.0)
        fusion.sigma = 0.5
        received = []

        def predict(index, local):
            received.append(windows.unpack(local.clone(), fusion.windows[index].shapes))
            return torch.full_like(local, 2.0 * (index + 1))

        fusion.predict(state, predict)
        # beta=1: sigma=.5 and zero model noise scale leaves .5*x0 exactly,
        # regardless of the original global state and nonzero random noise.
        for index, expected in ((1, 1.0), (2, 2.0)):
            self.assertTrue(bool((received[index][0][:, :, :7] == expected).all()))
            self.assertTrue(bool((received[index][1][..., :37] == expected).all()))
        self.assertEqual(fusion.report()["noise_scale"], 0.0)
        for config in (RollingConfig(False, 0.45), RollingConfig(True, 0.0)):
            with self.subTest(config=config):
                disabled = CPURollingFusion(three_windows(), config)
                disabled.start_run(noise, noise_scale=0.0)
                disabled.sigma = 0.5
                predictor = lambda index, local: local.square() * 0.125 + index
                actual = disabled.predict(state, predictor)
                expected = windows.CPUFusion(three_windows()).predict(state, predictor)
                self.assertTrue(torch.equal(actual, expected))

    def test_prediction_cache_resets_each_sigma_and_run(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig(True, 1.0))
        state = random_state(fusion, 11)
        noise = random_state(fusion, 22)
        first_inputs = []

        def predict(index, local):
            if index == 0:
                first_inputs.append(local.clone())
            return torch.full_like(local, 99.0)

        fusion.start_run(noise)
        for sigma in (0.8, 0.4):
            fusion.sigma = sigma
            fusion.predict(state, predict)
        self.assertEqual(len(fusion.steps), 2)
        fusion.finish_run()
        self.assertIsNone(fusion.noise)
        self.assertIsNone(fusion.sigma)
        fusion.start_run(noise * 2)
        self.assertEqual(fusion.steps, [])
        fusion.sigma = 0.2
        fusion.predict(state, predict)
        original_head = windows.pack(windows.crop(windows.unpack(state, fusion.shapes), fusion.windows[0]))
        self.assertTrue(all(torch.equal(value, original_head) for value in first_inputs))
        self.assertEqual(len(fusion.steps), 1)

    def test_failure_does_not_leak_partial_predictions_into_retry(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig())
        state, noise = random_state(fusion, 11), random_state(fusion, 22)
        fusion.start_run(noise)
        fusion.sigma = 0.4

        def fail(index, local):
            if index == 1:
                raise RuntimeError("intentional predictor failure")
            return local.fill_(999.0)

        with self.assertRaisesRegex(RuntimeError, "intentional"):
            fusion.predict(state, fail)
        self.assertEqual(fusion.steps, [])
        fresh = CPURollingFusion(three_windows(), RollingConfig())
        fresh.start_run(noise)
        fresh.sigma = 0.4
        predictor = lambda index, local: local * 0.25 + index
        self.assertTrue(torch.equal(fusion.predict(state, predictor), fresh.predict(state, predictor)))

    def test_active_rolling_rejects_out_of_order_execution(self):
        fusion = CPURollingFusion(three_windows(), RollingConfig())
        state = random_state(fusion, 11)
        fusion.start_run(state.clone())
        fusion.sigma = 0.5
        with self.assertRaisesRegex(ValueError, "chronological"):
            fusion.predict(state, lambda index, local: local, order=[2, 1, 0])

    def test_invalid_config_sigma_and_noise_scale_fail_before_prediction(self):
        for strength in (-0.1, 1.01, float("inf"), float("nan"), True, "0.5"):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                RollingConfig(True, strength)
        with self.assertRaises(ValueError):
            RollingConfig("true", 0.45)
        fusion = CPURollingFusion(three_windows(), RollingConfig())
        state = random_state(fusion, 11)
        for scale in (-1, float("inf"), float("nan")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                fusion.start_run(state, noise_scale=scale)
        with self.assertRaises(ValueError):
            fusion.start_run(state.double())
        fusion.start_run(state)
        for sigma in (None, -0.1, 1.01, float("inf"), float("nan")):
            fusion.sigma = sigma
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                fusion.predict(state, lambda index, local: self.fail("invalid sigma reached predictor"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
