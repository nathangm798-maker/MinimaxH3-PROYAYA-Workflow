import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const TARGET_NODE = "PROYYPreviewAny";
const WIDGET_NAME = "分段信息";
const MONO = 'ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", "Courier New", monospace';

// Two palettes with the same keys. The preview resolves the active one from the
// page background so the text stays readable after a theme switch.
const PALETTES = {
    dark: {
        base: "#d6dbe2", dim: "#79828e", key: "#7fb2ff", str: "#8fd98a", num: "#f2a65a",
        bool: "#d08cf5", idx: "#58d1ff", head: "#ffd166", warn: "#ff6b61", sep: "#5c3a3a",
        ov73: "#f2a65a", ov22: "#49d3b4", ovother: "#9aa4b2",
    },
    light: {
        base: "#242a32", dim: "#77808c", key: "#1a4fd0", str: "#186b28", num: "#9a4b00",
        bool: "#7c26a8", idx: "#0a5f8a", head: "#7a4f00", warn: "#b81414", sep: "#c9c2c2",
        ov73: "#9a4b00", ov22: "#0b6c58", ovother: "#5c6672",
    },
};

function escapeHTML(text) {
    return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function span(color, text, extra = "") {
    return `<span style="color:${color}${extra}">${escapeHTML(text)}</span>`;
}

// Colour quoted strings, numbers and JSON literals inside a value fragment.
// Numbers glued to word characters (file names such as _s4_00001_) stay plain,
// otherwise a path would light up like a token stream.
function renderValue(text, P) {
    const re = /("(?:[^"\\]|\\.)*")|((?<![A-Za-z0-9_.])-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![A-Za-z0-9_.]))|(\btrue\b|\bfalse\b|\bnull\b)/g;
    let out = "";
    let last = 0;
    let match;
    while ((match = re.exec(text)) !== null) {
        out += escapeHTML(text.slice(last, match.index));
        if (match[1]) out += span(P.str, match[1]);
        else if (match[2]) out += span(P.num, match[2]);
        else out += span(P.bool, match[3]);
        last = match.index + match[0].length;
    }
    return out + escapeHTML(text.slice(last));
}

// One colour per distinct prompt key. Keys are picked from a curated ramp so two
// different prompts never end up with neighbouring hues.
const KEY_RAMP = {
    dark: ["#c792ea", "#7bd88f", "#ff9ecb", "#8aa4ff", "#a8d660", "#6fd0a8"],
    light: ["#6f2fa0", "#1b6b2f", "#a12b63", "#2b3fb0", "#4a6b00", "#0b6c58"],
};

function keyColorFor(key, dark, taken) {
    const ramp = dark ? KEY_RAMP.dark : KEY_RAMP.light;
    let hash = 0;
    for (let i = 0; i < key.length; i++) hash = (hash * 31 + key.charCodeAt(i)) % 9973;
    let index = hash % ramp.length;
    for (let probe = 0; probe < ramp.length && taken.has(ramp[index]); probe++) {
        index = (index + 1) % ramp.length;
    }
    taken.add(ramp[index]);
    return ramp[index];
}

function renderTable(line, P, state) {
    const cells = line.split(/\s*\|\s*/);
    const header = cells[0].trim() === "窗";
    const last = cells.length - 1;
    // Column layout of plan_report: 序号 | 帧区间 | 采样帧 | 实际重叠 | 有效帧 | 补帧 | 提示词
    const overlapColumn = cells.length === 7 ? 3 : -1;
    const painted = cells.map((cell, index) => {
        if (header) return span(P.head, cell, ";font-weight:700");
        if (index === 0) return span(P.idx, cell, ";font-weight:700");
        if (index === last) return span(state.keyColor(cell.trim()), cell, ";font-weight:700");
        if (index === 1) return span(P.base, cell);
        if (index === overlapColumn) {
            const value = cell.trim();
            const color = value === "73" ? P.ov73 : value === "22" ? P.ov22 : P.ovother;
            return span(color, cell, ";font-weight:700");
        }
        return span(P.num, cell);
    });
    return painted.join(span(P.dim, " | "));
}

function decorate(line, P, state) {
    const trimmed = line.trim();

    if (/^[!=\-─═]{3,}$/.test(trimmed)) return span(P.sep, line, ";opacity:.9");
    if (/⚠|警告|不匹配|失败|错误/.test(line)) return span(P.warn, line, ";font-weight:700");
    if (/^[{}\[\],]+\s*$/.test(trimmed)) return span(P.dim, line);

    let match = line.match(/^(\s*)"((?:[^"\\]|\\.)*)"(\s*:\s*)(.*)$/);
    if (match) {
        let value = match[4];
        let tail = "";
        if (value.endsWith(",")) {
            tail = ",";
            value = value.slice(0, -1);
        }
        return match[1]
            + span(P.key, `"${match[2]}"`)
            + span(P.dim, match[3])
            + renderValue(value, P)
            + (tail ? span(P.dim, tail) : "");
    }

    if ((line.match(/\s\|\s/g) || []).length >= 2) return renderTable(line, P, state);

    match = line.match(/^(\s*)([^：:|\s][^：:|]{0,15})：(.*)$/);
    if (match) return match[1] + span(P.key, match[2] + "：") + renderValue(match[3], P);

    return span(P.base, line);
}

function renderText(text, P) {
    const colors = new Map();
    const taken = new Set();
    const state = {
        keyColor(key) {
            if (!colors.has(key)) colors.set(key, keyColorFor(key, Boolean(P.dark), taken));
            return colors.get(key);
        },
    };
    return text.split("\n").map((line) => (line.trim() ? decorate(line, P, state) : "")).join("\n");
}

function resolvePalette() {
    let dark = document.documentElement.classList.contains("dark-theme")
        || document.body.classList.contains("dark-theme");
    const background = getComputedStyle(document.body).backgroundColor;
    const rgb = background.match(/rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)(?:[,\s/]+([\d.]+))?/);
    if (rgb) {
        const alpha = rgb[4] === undefined ? 1 : parseFloat(rgb[4]);
        if (alpha > 0.01) {
            const luma = 0.2126 * +rgb[1] + 0.7152 * +rgb[2] + 0.0722 * +rgb[3];
            dark = luma < 128;
        }
    }
    return { ...(dark ? PALETTES.dark : PALETTES.light), dark };
}

function stopWheelBubbling(host) {
    host.addEventListener("wheel", (event) => {
        const scrollableY = host.scrollHeight > host.clientHeight + 1;
        const scrollableX = host.scrollWidth > host.clientWidth + 1;
        if (scrollableY || (event.deltaX !== 0 && scrollableX)) event.stopPropagation();
    }, { passive: true });
}

// Rich preview: a real <pre> so individual spans can carry colour.
function buildRichPreview(node) {
    if (node._planPre) return true;
    if (typeof node.addDOMWidget !== "function") return false;

    const host = document.createElement("div");
    host.style.cssText = [
        "width:100%", "height:100%", "box-sizing:border-box",
        "overflow:auto", "overscroll-behavior:contain",
        "padding:6px 8px", "border-radius:6px",
        "background:rgba(127,127,127,0.08)",
        "user-select:text", "-webkit-user-select:text",
    ].join(";");

    const pre = document.createElement("pre");
    pre.style.cssText = [
        "margin:0", "padding:0", "white-space:pre", "tab-size:4",
        `font-family:${MONO}`, "font-size:12.5px", "line-height:1.5",
        "user-select:text", "-webkit-user-select:text",
    ].join(";");
    host.appendChild(pre);
    stopWheelBubbling(host);

    const widget = node.addDOMWidget(WIDGET_NAME, "preview", host);
    widget.serialize = false;
    widget.options = widget.options || {};
    widget.options.serialize = false;
    widget.options.getValue = () => "";
    widget.options.setValue = () => {};
    widget.options.getMinHeight = () => 260;
    widget.options.hideOnZoom = false;

    node._planPre = pre;
    node._planHost = host;
    return true;
}

// Fallback for frontends without DOM widgets: keep the old monochrome box.
function buildPlainPreview(node) {
    if (node.planWidget) return;
    node.planWidget = ComfyWidgets.STRING(node, WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
    node.planWidget.options.serialize = false;
    node.planWidget.inputEl.readOnly = true;
    node.planWidget.inputEl.style.fontFamily = "monospace";
}

app.registerExtension({
    name: "proyaya.planPreview",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== TARGET_NODE) return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            original?.apply(this, arguments);
            const text = ((message && message.text) || []).join("\n");

            if (buildRichPreview(this)) {
                const palette = resolvePalette();
                this._planHost.style.color = palette.base;
                this._planPre.innerHTML = renderText(text, palette);
            } else {
                buildPlainPreview(this);
                this.planWidget.value = text;
            }
            this.setSize([Math.max(this.size[0], 640), Math.max(this.size[1], 340)]);
        };
    },
});
