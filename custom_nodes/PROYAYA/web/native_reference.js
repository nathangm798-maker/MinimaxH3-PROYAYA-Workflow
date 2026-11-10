import { app } from "../../scripts/app.js";

const hookedAutoWidgets = new WeakSet();
const pendingRefreshes = new WeakSet();
const pairedSlotsByVideo = new WeakMap();
const synchronizingPairs = new WeakSet();
const pairedGroup = "main_ref_video_audios";
const encoderTypes = new Set(["PROYYMiniMaxH3ReferenceToVideo"]);
const referenceGroups = [
    ["ref_images", "ref_image_", "IMAGE"],
    ["main_ref_videos", "main_ref_video_", "VIDEO"],
    ["ref_videos", "ref_video_", "VIDEO"],
    ["ref_audios", "ref_audio_", "AUDIO"],
    ["prompts", "prompt_", "STRING"],
];

function updateManualLengths(node) {
    const auto = node.widgets?.find((widget) => widget.name === "auto_segment");
    const manual = node.widgets?.find((widget) => widget.name === "manual_lengths");
    if (!auto || !manual) return;
    manual.disabled = auto.value === true;
    node.updateComputedDisabled?.();
    node.setDirtyCanvas(true, true);
}

function hookAutoWidget(node) {
    const auto = node.widgets?.find((widget) => widget.name === "auto_segment");
    if (!auto || hookedAutoWidgets.has(auto)) return;
    hookedAutoWidgets.add(auto);
    const callback = auto.callback;
    auto.callback = function () {
        const result = callback?.apply(this, arguments);
        updateManualLengths(node);
        return result;
    };
}

function referenceInputs(node, group) {
    return (node.inputs || []).filter((input) => input.name.startsWith(`${group}.`));
}

function reorderInputs(node, inputs) {
    const moves = node.inputs.flatMap((input, slot) => {
        const link = node.getInputLink(slot);
        const target = inputs.indexOf(input);
        return link && target !== slot ? [{ link, target }] : [];
    });
    // Park endpoints outside the occupied range before applying a permutation.
    // LLink's setter updates the frontend's link store, including target indices.
    moves.forEach(({ link }, index) => { link.target_slot = inputs.length + index; });
    node.inputs.splice(0, node.inputs.length, ...inputs);
    moves.forEach(({ link, target }) => { link.target_slot = target; });
}

function normalizeReferenceInputs(node) {
    if (!node.graph || app.configuringGraph) return;
    for (const [group, prefix, type] of referenceGroups) {
        const inputs = referenceInputs(node, group);
        const connected = inputs.filter((input) => input.link != null);
        const nextOrdinal = connected.reduce((max, input) => Math.max(max, Number(input.name.match(/\d+$/)?.[0]) + 1), 0);
        const spareName = `${group}.${prefix}${nextOrdinal}`;
        let spare = inputs.find((input) => input.link == null && input.name === spareName);
        for (const input of inputs) {
            if (input.link == null && input !== spare) node.removeInput(node.inputs.indexOf(input));
        }
        if (!spare) spare = node.addInput(spareName, type);
    }
}

function orderReferenceInputs(node) {
    if (!node.graph || app.configuringGraph) return;
    const order = ["clip", "vae", "audio_vae", "ref_images", "main_ref_videos", "ref_videos", pairedGroup, "ref_audios", "prompts"];
    const rank = (input) => {
        const value = order.indexOf(input.name.split(".")[0]);
        return value < 0 ? order.length : value;
    };
    const inputs = [...node.inputs].sort((a, b) => rank(a) - rank(b)
        || Number(a.name.match(/\d+$/)?.[0] || 0) - Number(b.name.match(/\d+$/)?.[0] || 0));
    reorderInputs(node, inputs);
}

function synchronizePairedAudio(node) {
    if (!node.graph || app.configuringGraph) return;
    const videos = [
        ...referenceInputs(node, "main_ref_videos"),
        ...referenceInputs(node, "ref_videos"),
    ].filter((input) => input.link != null);
    const oldInputs = referenceInputs(node, pairedGroup);
    let previous = pairedSlotsByVideo.get(node);
    if (!previous) {
        previous = new Map();
        for (const input of oldInputs) {
            const ordinal = Number(input.name.match(/\d+$/)?.[0]);
            if (videos[ordinal]) previous.set(videos[ordinal].link, input);
        }
    }

    const next = new Map();
    const inputs = [];
    let suffix = oldInputs.reduce((max, input) => Math.max(max, Number(input.name.match(/\d+$/)?.[0]) + 1), 0);
    for (const video of videos) {
        let input = previous.get(video.link) || (inputs.length === 0 ? previous.get(null) : undefined);
        if (!input) {
            input = oldInputs.find((candidate) => candidate.link == null && !inputs.includes(candidate) && ![...previous.values()].includes(candidate));
        }
        if (!input) input = node.addInput(`${pairedGroup}.main_ref_video_audio_${suffix++}`, "AUDIO");
        inputs.push(input);
        next.set(video.link, input);
    }
    if (!inputs.length) {
        const input = videos.length === 0 && previous.size === 0 ? oldInputs[0] : oldInputs.find((slot) => slot.link == null);
        inputs.push(input || node.addInput(`${pairedGroup}.main_ref_video_audio_${suffix++}`, "AUDIO"));
        next.set(null, inputs[0]);
    }

    const autogrow = node.comfyDynamic?.autogrow?.[pairedGroup];
    if (autogrow) {
        // One stable optional slot per video permits holes without Autogrow
        // shifting a later soundtrack onto an earlier video on disconnect.
        autogrow.min = inputs.length;
        autogrow.max = inputs.length;
    }
    pairedSlotsByVideo.set(node, next);
    synchronizingPairs.add(node);
    try {
        for (const input of oldInputs) {
            if (!inputs.includes(input)) node.removeInput(node.inputs.indexOf(input));
        }
    } finally {
        synchronizingPairs.delete(node);
    }
    const otherInputs = node.inputs.filter((input) => !input.name.startsWith(`${pairedGroup}.`));
    let insertion = otherInputs.findIndex((input) => /^(ref_audios\.|prompts\.)/.test(input.name));
    if (insertion === -1) insertion = otherInputs.length;
    otherInputs.splice(insertion, 0, ...inputs);
    reorderInputs(node, otherInputs);
    inputs.forEach((input, index) => { input.name = `${pairedGroup}.main_ref_video_audio_${index}`; });
}

function labelReferences(inputs, kind, firstNumber = 1, suffix = "") {
    let number = firstNumber;
    let changed = false;
    for (const input of inputs) {
        const label = `<${kind} ${number}>${suffix}`;
        if (input.label !== label) {
            input.label = label;
            changed = true;
        }
        // An Autogrow spare displays the next number without reserving it.
        if (input.link != null) number++;
    }
    return { nextNumber: number, changed };
}

function updateReferenceLabels(node) {
    const pictures = labelReferences(referenceInputs(node, "ref_images"), "picture");
    const mainVideos = labelReferences(referenceInputs(node, "main_ref_videos"), "video", 1, " · main");
    // 辅助视频去掉后缀，编号接着主视频往下排（<video 2>、<video 3>…），
    // 与 H3 原生「主视频排在辅助视频前、共用同一套编号」一致。
    const videos = labelReferences(referenceInputs(node, "ref_videos"), "video", mainVideos.nextNumber);
    // 配对音轨统一成 <audio n> · main，未接入时也不再用「（未接入）」占位文案。
    // 只有已接入的槽才占用编号，与 labelReferences 对 Autogrow 空槽的处理一致。
    let audioNumber = 1;
    let pairedChanged = false;
    referenceInputs(node, pairedGroup).forEach((input) => {
        const label = `<audio ${audioNumber}> · main`;
        if (input.label !== label) {
            input.label = label;
            pairedChanged = true;
        }
        if (input.link != null) audioNumber++;
    });
    const mainAudio = { changed: pairedChanged };
    const audio = labelReferences(referenceInputs(node, "ref_audios"), "audio", audioNumber);
    referenceInputs(node, "prompts").forEach((input) => { input.label = "prompt"; });
    if ([pictures, mainVideos, videos, mainAudio, audio].some((group) => group.changed)) {
        node.graph?.trigger("node:slot-label:changed", { nodeId: node.id, slotType: 1 });
        node.setDirtyCanvas(true, true);
    }
}

function refreshNode(node) {
    hookAutoWidget(node);
    updateManualLengths(node);
    normalizeReferenceInputs(node);
    orderReferenceInputs(node);
    synchronizePairedAudio(node);
    updateReferenceLabels(node);
}

function scheduleRefresh(node) {
    if (pendingRefreshes.has(node)) return;
    pendingRefreshes.add(node);
    requestAnimationFrame(() => {
        pendingRefreshes.delete(node);
        refreshNode(node);
    });
}

app.registerExtension({
    name: "proyaya.reference",
    afterConfigureGraph() {
        const graphs = [app.rootGraph, ...app.rootGraph.subgraphs.values()];
        for (const graph of graphs) {
            for (const node of graph._nodes) {
                if (encoderTypes.has(node.type)) scheduleRefresh(node);
            }
        }
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!encoderTypes.has(nodeData.name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            // Keep native Autogrow in charge of socket growth, order and link slots.
            const onConnectionsChange = this.onConnectionsChange;
            this.onConnectionsChange = function () {
                if (synchronizingPairs.has(this) && arguments[4]?.name?.startsWith(`${pairedGroup}.`)) return;
                const connectionResult = onConnectionsChange?.apply(this, arguments);
                scheduleRefresh(this);
                return connectionResult;
            };
            refreshNode(this);
            return result;
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            pairedSlotsByVideo.delete(this);
            refreshNode(this);
            scheduleRefresh(this);
            return result;
        };
    },
});
