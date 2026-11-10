"""Normalize the four checked-in PROYAYA workflows for release.

This keeps the existing graph and parameters, then applies the release rules
that must stay identical across workflow variants: widget metadata is rebuilt
from the executable values, stale VHS previews and local model hashes are
removed, workflow title overrides are cleared, and the ControlNet variant gets the
24fps/frame-coverage guard.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


WORKFLOW_NAMES = (
    "基础工作流_最终版.json",
    "YOLO遮罩版_最终版.json",
    "ControlNet版_最终版.json",
    "二采版_最终版.json",
)
CURRENT_COMFY = "0.39.0"

WIDGET_KEYS = {
    "PROYYImageScale": ("file", "width", "height", "upscale_method", "crop", "single_side", "target_mp", "read_frames", "skip_frames"),
    "PROYYMiniMaxH3ReferenceToVideo": ("auto_segment", "auto_prompt", "window_frames", "manual_lengths", "overlap", "keep_final", "width", "height", "length", "text_weight", "image_weight", "video_weight", "video_encode", "unload", "preview_only"),
    "PROYYMaskedSampler": ("high_cfg", "cfg_split", "low_cfg", "fusion_preserve", "rolling_enabled", "rolling_strength", "rolling_mode", "rolling_front_high", "rolling_front_low", "unload"),
    "LoraLoaderModelOnly": ("lora_name", "strength_model"),
    "UNETLoader": ("unet_name", "weight_dtype"),
    "RandomNoise": ("noise_seed", "control_after_generate"),
    "KSamplerSelect": ("sampler_name",),
    "BasicScheduler": ("scheduler", "steps", "denoise"),
    "ManualSigmas": ("sigmas",),
    "ImageFromBatch": ("batch_index", "length"),
    "CLIPLoader": ("clip_name", "type", "device"),
    "VAELoader": ("vae_name",),
    "LoadImage": ("image", "upload"),
    "ComfyMathExpression": ("expression",),
    "PrimitiveInt": ("value", "fixed"),
    "LoadVideo": ("file", "upload"),
    "ModelPatchLoader": ("name",),
    "MiniMaxH3FunControlNetApply": ("strength", "start_percent", "end_percent"),
    "PROYAYA_YoloPersonMask": ("model_name", "conf", "expand", "feather", "largest_only", "max_det", "fill_missing", "device", "emit_preview", "fill_holes"),
    "PROYYTiledLatentUpscale": ("model_name", "scale", "tile_size", "tile_overlap", "align", "enable_temporal_chunking", "precision", "device"),
    "PROYAYA_ControlVideoCheck": ("fps", "expected_frames"),
}

PREFIXES = {
    "基础工作流_最终版.json": "PROYAYA_Base",
    "YOLO遮罩版_最终版.json": "PROYAYA_YOLO",
    "ControlNet版_最终版.json": "PROYAYA_ControlNet",
    "二采版_最终版.json": "PROYAYA_SecondPass",
}


def node_map(data):
    return {int(node["id"]): node for node in data.get("nodes", [])}


def set_input_link(node, name, link):
    for item in node.get("inputs", []):
        if item.get("name") == name:
            item["link"] = link
            return
    raise KeyError(f"input {name!r} not found on node {node['id']}")


def set_output_links(node, output_index, links):
    outputs = node.get("outputs", [])
    while len(outputs) <= output_index:
        outputs.append({"name": "", "type": "", "links": []})
    outputs[output_index]["links"] = list(links) if links else None


def remove_link(data, link_id):
    data["links"] = [link for link in data.get("links", []) if int(link[0]) != int(link_id)]
    for node in data.get("nodes", []):
        for output in node.get("outputs", []):
            if output.get("links"):
                output["links"] = [value for value in output["links"] if int(value) != int(link_id)] or None
        for item in node.get("inputs", []):
            if item.get("link") == link_id:
                item["link"] = None


def link(data, link_id, source_id, source_slot, target_id, target_slot, type_name):
    data.setdefault("links", []).append([link_id, source_id, source_slot, target_id, target_slot, type_name])
    nodes = node_map(data)
    set_output_links(nodes[source_id], source_slot, (nodes[source_id].get("outputs", [])[source_slot].get("links") or []) + [link_id])
    nodes[target_id].get("inputs", [])[target_slot]["link"] = link_id


def sync_named_widgets(node):
    values = node.get("widgets_values")
    if values is None:
        node.pop("widgets_values_named", None)
        return
    if isinstance(values, dict):
        named = copy.deepcopy(values)
        named.pop("videopreview", None)
        node["widgets_values"] = copy.deepcopy(named)
        node["widgets_values_named"] = copy.deepcopy(named)
        return
    keys = WIDGET_KEYS.get(node.get("type"))
    if not keys:
        return
    node["widgets_values_named"] = {key: value for key, value in zip(keys, values)}


def clean_metadata(data, filename):
    nodes = node_map(data)
    for node in data.get("nodes", []):
        node_type = node.get("type")
        node.pop("title", None)
        props = node.get("properties")
        if isinstance(props, dict):
            props.pop("models", None)
            if props.get("cnr_id") == "comfy-core" and node_type.startswith("PROYAYA"):
                props.pop("cnr_id", None)
                props.pop("ver", None)
            elif props.get("cnr_id") == "comfy-core":
                props["ver"] = CURRENT_COMFY
        sync_named_widgets(node)

    # Linked length widgets should show the same release default as the live plan.
    if 30 in nodes and len(nodes[30].get("widgets_values", [])) >= 2:
        nodes[30]["widgets_values"][1] = 328
        sync_named_widgets(nodes[30])
    if 41 in nodes and len(nodes[41].get("widgets_values", [])) >= 2:
        nodes[41]["widgets_values"][1] = 328
        sync_named_widgets(nodes[41])
    if 18 in nodes and len(nodes[18].get("widgets_values", [])) >= 9:
        nodes[18]["widgets_values"][6:8] = [800, 1344]
        sync_named_widgets(nodes[18])

    output = nodes.get(39, {}).get("widgets_values")
    if isinstance(output, dict):
        output["filename_prefix"] = PREFIXES[filename]
        output.pop("videopreview", None)
        nodes[39]["widgets_values_named"] = copy.deepcopy(output)

    extra = data.get("extra")
    if isinstance(extra, dict):
        for key in ("anomalous_hashes", "VHS_latentpreview", "VHS_latentpreviewrate", "VHS_MetadataImage", "VHS_KeepIntermediate"):
            extra.pop(key, None)
        extra["frontendVersion"] = CURRENT_COMFY

    data["revision"] = 0


def add_control_guard(data):
    nodes = node_map(data)
    if 47 in nodes:
        return
    old_link = next(link for link in data["links"] if link[1:5] == [45, 0, 46, 3])
    remove_link(data, old_link[0])
    guard = {
        "id": 47,
        "type": "PROYAYA_ControlVideoCheck",
        "pos": [1370, 430],
        "size": [360, 170],
        "flags": {},
        "order": max(int(node.get("order", 0)) for node in data["nodes"]) + 1,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": 42},
            {"name": "fps", "type": "FLOAT", "link": 43},
            {"name": "expected_frames", "type": "INT", "link": 44},
        ],
        "outputs": [{"name": "control_video", "type": "IMAGE", "links": [45]}],
        "properties": {"Node name for S&R": "PROYAYA_ControlVideoCheck"},
        "widgets_values": [24.0, 328],
        "widgets_values_named": {"fps": 24.0, "expected_frames": 328},
    }
    data["nodes"].append(guard)
    data["links"].extend([
        [42, 45, 0, 47, 0, "IMAGE"],
        [43, 45, 2, 47, 1, "FLOAT"],
        [44, 37, 1, 47, 2, "INT"],
        [45, 47, 0, 46, 3, "IMAGE"],
    ])
    nodes = node_map(data)
    set_output_links(nodes[45], 0, [42])
    set_output_links(nodes[45], 2, [43])
    set_output_links(nodes[37], 1, (nodes[37]["outputs"][1].get("links") or []) + [44])
    set_output_links(nodes[47], 0, [45])
    set_input_link(nodes[47], "image", 42)
    set_input_link(nodes[47], "fps", 43)
    set_input_link(nodes[47], "expected_frames", 44)
    set_input_link(nodes[46], "control_video", 45)
    data["last_node_id"] = 47
    data["last_link_id"] = 45


def update_locales(package_dir):
    locale_dir = package_dir / "locales"
    en_path = locale_dir / "en" / "nodeDefs.json"
    zh_path = locale_dir / "zh" / "nodeDefs.json"
    en = json.loads(en_path.read_text(encoding="utf-8"))
    zh = json.loads(zh_path.read_text(encoding="utf-8"))
    en["PROYYImageScale"]["inputs"]["file"]["name"] = "Video file"
    zh["PROYYImageScale"]["inputs"]["file"]["name"] = "视频文件"
    en["PROYYMaskedSampler"]["outputs"]["0"]["name"] = "joint latent"
    zh["PROYYMaskedSampler"]["outputs"]["0"]["name"] = "联合 latent"
    guard_en = {
        "display_name": "۞▷Control Video Check◁۞",
        "inputs": {
            "image": {"name": "Control video frames"},
            "fps": {"name": "Frame rate"},
            "expected_frames": {"name": "Expected frames"},
        },
        "outputs": {"0": {"name": "control video"}},
    }
    guard_zh = {
        "display_name": "۞▷Control Video Check◁۞",
        "inputs": {
            "image": {"name": "控制视频帧"},
            "fps": {"name": "帧率"},
            "expected_frames": {"name": "目标帧数"},
        },
        "outputs": {"0": {"name": "控制视频"}},
    }
    en["PROYAYA_ControlVideoCheck"] = guard_en
    zh["PROYAYA_ControlVideoCheck"] = guard_zh
    en_path.write_text(json.dumps(en, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zh_path.write_text(json.dumps(zh, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    for filename in WORKFLOW_NAMES:
        path = args.input_dir / filename
        data = json.loads(path.read_text(encoding="utf-8"))
        clean_metadata(data, filename)
        if filename == "ControlNet版_最终版.json":
            add_control_guard(data)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    update_locales(args.package_dir)


if __name__ == "__main__":
    main()
