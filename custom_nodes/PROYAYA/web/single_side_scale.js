import { app } from "../../scripts/app.js";

const NODE_NAME = "PROYYImageScale";
const AUTO_SUFFIX = "（自动）";

function findWidget(node, name) {
    return (node.widgets || []).find((w) => w.name === name);
}

// 布尔控件在不同版本里可能是 true/false 或 "true"/"false"
function isLocked(toggle) {
    const v = toggle?.value;
    return v === true || v === "true" || v === 1;
}

/**
 * 只做一件事：把 height 控件的 disabled 置位。
 *
 * ComfyUI 前端（LGraphNode）本身已经支持这个标记：
 *   drawWidgets():  s.computedDisabled && (ctx.globalAlpha *= .5)          → 自动半透明变灰
 *   NumberWidget:   get _displayValue() { return computedDisabled ? "" }    → 不再显示数值
 *   getWidgetOnPos(): if (widget.computedDisabled) continue                → 鼠标完全点不到
 * 所以不要覆盖 widget.draw —— 前端用
 *   typeof s.draw === "function" ? s.draw(...) : drawWidget(...)
 * 一旦我们加了 draw，就会接管绘制而原生 drawWidget 不会被调用，结果是整行空白。
 */
function applyState(node) {
    const heightWidget = findWidget(node, "height");
    const toggleWidget = findWidget(node, "single_side");
    if (!heightWidget || !toggleWidget) return false;

    if (heightWidget.__baseLabel === undefined) {
        heightWidget.__baseLabel = heightWidget.label || "短边 / 高";
    }

    const locked = isLocked(toggleWidget);
    heightWidget.disabled = locked;
    heightWidget.computedDisabled = locked;
    heightWidget.label = locked
        ? heightWidget.__baseLabel + AUTO_SUFFIX
        : heightWidget.__baseLabel;

    node.setDirtyCanvas(true, true);
    return true;
}

function setup(node) {
    if (node.__singleSideReady) {
        applyState(node);
        return true;
    }

    const toggleWidget = findWidget(node, "single_side");
    if (!toggleWidget || !findWidget(node, "height")) return false;

    node.__singleSideReady = true;

    const originalSetValue = toggleWidget.setValue;
    toggleWidget.setValue = function () {
        const result = originalSetValue?.apply(this, arguments);
        requestAnimationFrame(() => applyState(node));
        return result;
    };

    // 有些版本走 callback 而不是 setValue
    const originalCallback = toggleWidget.callback;
    toggleWidget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        requestAnimationFrame(() => applyState(node));
        return result;
    };

    applyState(node);
    return true;
}

app.registerExtension({
    name: "proyaya.singleSideScale",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated
                ? originalOnNodeCreated.apply(this, arguments)
                : undefined;
            const node = this;
            if (!setup(node)) {
                let tries = 0;
                const tick = () => {
                    if (setup(node) || ++tries > 30) return;
                    requestAnimationFrame(tick);
                };
                requestAnimationFrame(tick);
            }
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure
                ? originalOnConfigure.apply(this, arguments)
                : undefined;
            const node = this;
            requestAnimationFrame(() => setup(node));
            return result;
        };
    },
});
