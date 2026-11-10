"""CPU integration checks against the installed ComfyUI sampler implementations.

No checkpoint, video, CUDA allocation, or generated media is used. Run with the
portable ComfyUI Python, or set COMFYUI_ROOT for another compatible checkout.
"""

import contextlib
import importlib
import math
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


sys.dont_write_bytecode = True
COMFYUI_ROOT = Path(os.environ.get("COMFYUI_ROOT", r"D:\ComfyUI_windows_portable\ComfyUI"))
sys.path.insert(0, str(COMFYUI_ROOT))
saved_argv = sys.argv[:]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options

comfy.options.enable_args_parsing()
import torch
import comfy.conds
import comfy.hooks
import comfy.model_sampling
import comfy.samplers
from comfy.nested_tensor import NestedTensor

package_name = "_h3_window_runtime_test"
package = types.ModuleType(package_name)
package.__path__ = [str(Path(__file__).resolve().parents[1])]
sys.modules[package_name] = package
runtime = importlib.import_module(package_name + ".runtime")
windows = importlib.import_module(package_name + ".windows")
sys.argv = [arg for arg in saved_argv if arg != "--cpu"]


class AVSampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


class FakeBase:
    def __init__(self):
        self.model_sampling = AVSampling()
        self.model_sampling.set_parameters(shift=12.0, audio_shift=3.0)
        self.latent_shapes = [("original",)]
        self.output_shapes = []
        self.input_calls = 0

    def process_latent_in(self, latent):
        self.input_calls += 1
        return latent

    def process_latent_out(self, latent):
        self.output_shapes.append(tuple(tuple(s) for s in self.latent_shapes))
        result = latent.clone()
        video_size = math.prod(self.latent_shapes[0])
        result[..., video_size:] /= self.model_sampling.audio_scale
        return result


class FakePatcher:
    def __init__(self):
        self.model = FakeBase()
        self.model_options = {"transformer_options": {}}
        self.load_device = torch.device("cpu")
        self.offload_device = torch.device("cpu")
        self.hook_mode = comfy.hooks.EnumHookMode.MinVram
        self.pre_runs = 0
        self.cleanups = 0
        self.restores = 0

    def is_dynamic(self):
        return False

    def model_dtype(self):
        return torch.float32

    def get_model_object(self, name):
        return getattr(self.model, name)

    def pre_run(self):
        self.pre_runs += 1

    def cleanup(self):
        self.cleanups += 1

    def restore_hook_patches(self):
        self.restores += 1


def conditioning(marker):
    return [[torch.zeros(1, 1, 1), {"marker": marker, "model_conds": {}}]]


def entries(count):
    return [{"window": windows.Window(i * 102, 124, 2, 2),
             "positive": conditioning(i + 1), "negative": conditioning(-i - 1)}
            for i in range(count)]


class NativeHarness:
    """Replace model work and loading, keeping CFGGuider and KSAMPLER real."""

    def __init__(self, testcase, fail_marker=None):
        self.testcase = testcase
        self.fail_marker = fail_marker
        self.predictions = []
        self.prepared = []
        self.load_shapes = []
        self.cleanup_calls = []
        self.solver_values = []
        self.stack = contextlib.ExitStack()

    def __enter__(self):
        patches = [
            mock.patch("comfy.sampler_helpers.prepare_sampling", side_effect=self.prepare_sampling),
            mock.patch("comfy.sampler_helpers.cleanup_models", side_effect=self.cleanup),
            mock.patch("comfy.sampler_helpers.prepare_model_patcher"),
            mock.patch("comfy.sampler_helpers.prepare_model_patcher_multigpu_clones", return_value=[]),
            mock.patch("comfy.model_management.cuda_device_context", side_effect=lambda _: contextlib.nullcontext()),
            mock.patch("comfy.samplers.process_conds", side_effect=self.process_conds),
            mock.patch("comfy.samplers.sampling_function", side_effect=self.predict),
        ]
        for patch in patches:
            self.stack.enter_context(patch)
        self.previous_trace = sys.gettrace()
        sys.settrace(self.trace_solver)
        return self

    def __exit__(self, *args):
        sys.settrace(self.previous_trace)
        return self.stack.__exit__(*args)

    def trace_solver(self, frame, event, arg):
        if frame.f_code.co_name in ("sample_euler", "res_multistep"):
            if event == "line":
                for name in ("x", "old_denoised", "denoised", "sigmas"):
                    value = frame.f_locals.get(name)
                    if isinstance(value, torch.Tensor):
                        self.testcase.assertEqual(value.device.type, "cpu", name)
                        self.solver_values.append((frame.f_code.co_name, name, tuple(value.shape)))
            return self.trace_solver
        return None

    def prepare_sampling(self, patcher, shape, conds, model_options):
        self.load_shapes.append(tuple(shape))
        return patcher.model, conds, ["fake-loaded-model"]

    def cleanup(self, conds, loaded):
        self.cleanup_calls.append(list(loaded))

    def process_conds(self, base, noise, conds, device, latent_image=None,
                      denoise_mask=None, seed=None, latent_shapes=None):
        self.prepared.append((tuple(noise.shape), tuple(tuple(s) for s in latent_shapes)))
        result = {}
        for name, group in conds.items():
            result[name] = []
            for cond in group:
                value = cond.copy()
                value["model_conds"] = {"c_crossattn": comfy.conds.CONDRegular(
                    torch.full((1, 2, 3), float(value["marker"])))}
                result[name].append(value)
        return result

    def predict(self, base, x, timestep, uncond, cond, cond_scale, model_options=None, seed=None):
        marker = cond[0]["marker"]
        shapes = tuple(tuple(s) for s in base.latent_shapes)
        self.testcase.assertEqual(tuple(x.shape), (1, 1, sum(math.prod(s) for s in shapes)))
        self.testcase.assertEqual(x.device.type, "cpu")
        self.testcase.assertEqual(timestep.device.type, "cpu")
        self.testcase.assertEqual(uncond[0]["marker"], -marker)
        self.testcase.assertEqual(model_options["transformer_options"]["sample_sigmas"].device.type, "cpu")
        self.predictions.append((marker, shapes, tuple(x.shape), float(timestep[0])))
        if marker == self.fail_marker:
            raise RuntimeError("injected window prediction failure")
        return 0.125 * x + marker * 0.01 + timestep.reshape(-1, 1, 1) * 0.02


class RuntimeIntegrationTests(unittest.TestCase):
    sigmas = torch.tensor([1.0, 0.7, 0.35, 0.1, 0.0])

    def inputs(self, shapes):
        generator = torch.Generator("cpu").manual_seed(123)
        noise = NestedTensor([torch.randn(s, generator=generator) for s in shapes])
        latent = NestedTensor([torch.zeros(s) for s in shapes])
        return noise, latent

    def assert_streams_close(self, actual, expected):
        for left, right in zip(actual.unbind(), expected.unbind()):
            torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7)

    def test_single_window_matches_stock_guider_for_both_native_samplers(self):
        for sampler_name in ("euler", "res_multistep"):
            with self.subTest(sampler=sampler_name):
                entry = entries(1)
                noise, latent = self.inputs(entry[0]["window"].shapes)
                stock_patcher, joint_patcher = FakePatcher(), FakePatcher()
                stock = comfy.samplers.CFGGuider(stock_patcher)
                stock.set_conds(entry[0]["positive"], entry[0]["negative"])
                stock.set_cfg(1.0)
                joint = runtime.CPUWindowGuider(joint_patcher, entry, 1.0)
                original_shapes = joint_patcher.model.latent_shapes
                original_options = joint.model_options
                with NativeHarness(self) as harness:
                    expected = stock.sample(noise, latent, comfy.samplers.sampler_object(sampler_name),
                                            self.sigmas, disable_pbar=True, seed=123)
                    actual = joint.sample(noise, latent, comfy.samplers.sampler_object(sampler_name),
                                          self.sigmas, disable_pbar=True, seed=123)
                self.assert_streams_close(actual, expected)
                self.assertIs(joint_patcher.model.latent_shapes, original_shapes)
                self.assertIs(joint.model_options, original_options)
                self.assertEqual(joint_patcher.cleanups, 1)
                self.assertEqual(joint_patcher.restores, 1)
                self.assertEqual(joint.window_calls, 4)
                self.assertEqual(len(joint_patcher.model.output_shapes), 1)
                self.assertEqual(joint_patcher.model.input_calls, 0)
                self.assertTrue(harness.solver_values)

    def test_three_windows_route_conditions_and_keep_solver_and_history_on_cpu(self):
        for sampler_name in ("euler", "res_multistep"):
            with self.subTest(sampler=sampler_name):
                entry = entries(3)
                patcher = FakePatcher()
                guider = runtime.CPUWindowGuider(patcher, entry, 1.0)
                noise, latent = self.inputs(guider.fusion.shapes)
                callbacks = []

                def callback(step, denoised, state, total):
                    callbacks.append((step, denoised, state, total))
                    for tensor in state.unbind():
                        self.assertEqual(tensor.device.type, "cpu")

                with NativeHarness(self) as harness:
                    output = guider.sample(noise, latent, comfy.samplers.sampler_object(sampler_name),
                                           self.sigmas, callback=callback, disable_pbar=True, seed=123)
                global_shapes = tuple(tuple(s) for s in guider.fusion.shapes)
                local_shapes = tuple(tuple(s) for s in entry[0]["window"].shapes)
                local_pack = (1, 1, sum(math.prod(s) for s in local_shapes))
                global_pack = (1, 1, sum(math.prod(s) for s in global_shapes))
                self.assertEqual([p[0] for p in harness.predictions], [1, 2, 3] * 4)
                self.assertTrue(all(p[1] == local_shapes for p in harness.predictions))
                self.assertTrue(all(p[2] == local_pack for p in harness.predictions))
                self.assertEqual(harness.load_shapes, [local_pack])
                self.assertEqual(len(harness.prepared), 3)
                self.assertEqual(patcher.model.output_shapes, [global_shapes])
                self.assertEqual(guider.window_calls, 12)
                self.assertEqual(len(callbacks), 4)
                self.assertTrue(all(shape == global_pack for _, name, shape in harness.solver_values
                                    if name in ("x", "old_denoised", "denoised")))
                if sampler_name == "res_multistep":
                    self.assertTrue(any(name == "old_denoised" for _, name, _ in harness.solver_values))
                last_denoised = callbacks[-1][1].unbind()
                torch.testing.assert_close(output.unbind()[0], last_denoised[0])
                torch.testing.assert_close(output.unbind()[1], last_denoised[1] / 4.0)
                self.assertEqual(harness.cleanup_calls, [["fake-loaded-model"]])
                self.assertEqual(guider.window_conds, [])
                self.assertFalse(hasattr(guider, "inner_model"))

    def test_prediction_failure_restores_shapes_options_and_cleans_up(self):
        patcher = FakePatcher()
        guider = runtime.CPUWindowGuider(patcher, entries(3), 1.0)
        original_shapes = patcher.model.latent_shapes
        original_options = guider.model_options
        original_mode = patcher.hook_mode
        noise, latent = self.inputs(guider.fusion.shapes)
        with NativeHarness(self, fail_marker=2) as harness:
            with self.assertRaisesRegex(RuntimeError, "injected window prediction failure"):
                guider.sample(noise, latent, comfy.samplers.sampler_object("res_multistep"),
                              self.sigmas, disable_pbar=True, seed=123)
        self.assertIs(patcher.model.latent_shapes, original_shapes)
        self.assertIs(guider.model_options, original_options)
        self.assertIs(patcher.hook_mode, original_mode)
        self.assertEqual(patcher.cleanups, 1)
        self.assertEqual(patcher.restores, 1)
        self.assertEqual(harness.cleanup_calls, [["fake-loaded-model"]])
        self.assertEqual(patcher.model.output_shapes, [])
        self.assertEqual(guider.window_conds, [])
        self.assertFalse(hasattr(guider, "inner_model"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
