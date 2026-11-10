"""PROYYPreviewAny: text preview that is also shown after the run."""


class PROYYPreviewAny:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"source": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ()
    FUNCTION = "show"
    OUTPUT_NODE = True
    CATEGORY = "PROYAYA/H3"
    DESCRIPTION = "Show plan_report / run_report text in a read-only monospace box."

    def show(self, source):
        return {"ui": {"text": [source]}, "result": ()}