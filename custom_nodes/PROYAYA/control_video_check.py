"""Validate the IMAGE control stream used by the MiniMax H3 ControlNet path."""

from __future__ import annotations

import math


class PROYAYA_ControlVideoCheck:
    """Reject ControlNet inputs whose timeline cannot match the H3 24 fps path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "ControlNet control-video IMAGE frames."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 0.001, "max": 1000.0, "step": 0.001, "tooltip": "Actual control-video frame rate; H3 ControlNet requires 24 fps."}),
                "expected_frames": ("INT", {"default": 328, "min": 1, "max": 1000000, "step": 1, "tooltip": "Target frame count; the control video must cover this many frames."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("control_video",)
    FUNCTION = "validate"
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = "Validate a ControlNet IMAGE stream at 24 fps and reject a shorter control video instead of silently repeating its last frame."

    def validate(self, image, fps, expected_frames):
        if getattr(image, "ndim", None) != 4:
            raise ValueError("ControlNet control video must be IMAGE shaped [frames, height, width, channels].")
        if not isinstance(expected_frames, int) or isinstance(expected_frames, bool):
            raise ValueError("ControlNet target frame count must be an integer.")
        if expected_frames < 1:
            raise ValueError("ControlNet target frame count must be greater than zero.")
        try:
            actual_fps = float(fps)
        except (TypeError, ValueError):
            raise ValueError("ControlNet control-video frame rate is invalid.") from None
        if not math.isfinite(actual_fps) or actual_fps <= 0:
            raise ValueError("ControlNet control-video frame rate must be positive.")
        if abs(actual_fps - 24.0) > 1e-3:
            raise ValueError(f"ControlNet control video must be 24 fps; received {actual_fps:g} fps. Convert the video before connecting it.")
        actual_frames = int(image.shape[0])
        if actual_frames < expected_frames:
            raise ValueError(f"ControlNet control video is too short: need at least {expected_frames} frames, but received {actual_frames}.")
        return (image[:expected_frames],)


NODE_CLASS_MAPPINGS = {"PROYAYA_ControlVideoCheck": PROYAYA_ControlVideoCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"PROYAYA_ControlVideoCheck": "۞▷Control Video Check◁۞"}
