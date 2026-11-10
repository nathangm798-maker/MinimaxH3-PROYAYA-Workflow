# Third-party attribution

The intra-step rolling design and renoise/blend formula are adapted from:

- Project: https://github.com/user2318/ComfyUI-CustomNodeKit
- Author: user2318
- File: `wan_scail2_rolling.py`
- Source revision: `5099660055cfe96a26366150f171129fd5578c39`
- Copyright (c) 2026 user2318
- License: MIT; the complete notice is included in `LICENSE.CustomNodeKit`.

The H3 port uses the existing V1 CPU AV fusion and native ComfyUI integration.
It changes noise sourcing to the original global H3 NOISE, commits post-CFG
predictions, operates on separate H3 video/audio coordinates, and retains only
the overlap needed by the next window. It does not include the donor's Wan
RoPE patches, mask streams, anchor prepend, or long-range memory system.
