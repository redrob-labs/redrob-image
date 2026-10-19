"""OCR engines behind one interface: engine(img: PIL.Image) -> list[Line].

Line = dict(text=str, box=[x0, y0, x1, y1] | None, conf=float | None)
Boxes are in pixel coordinates of the input image. VLM engines return box=None and
lines in the order the model wrote them.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _bbox(pts) -> list[float]:
    a = np.asarray(pts, dtype=float).reshape(-1, 2)
    return [float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())]


class PaddleEngine:
    name = "paddleocr"

    def __init__(self, device: str = "gpu"):
        from paddleocr import PaddleOCR

        self.ocr = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device=device,
        )

    def __call__(self, img: Image.Image) -> list[dict]:
        arr = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()  # BGR like cv2
        out = []
        for res in self.ocr.predict(arr):
            r = res.json["res"] if hasattr(res, "json") else res
            for t, p, s in zip(r["rec_texts"], r["rec_polys"], r["rec_scores"]):
                if t.strip():
                    out.append(dict(text=t, box=_bbox(p), conf=float(s)))
        return out


class EasyEngine:
    name = "easyocr"

    def __init__(self, gpu: bool = True):
        import easyocr

        self.reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)

    def __call__(self, img: Image.Image) -> list[dict]:
        arr = np.asarray(img.convert("RGB"))
        return [dict(text=t, box=_bbox(b), conf=float(c)) for b, t, c in self.reader.readtext(arr) if t.strip()]


class QwenVLEngine:
    """Qwen3-VL Instruct (Apache-2.0) used as a transcriber."""

    PROMPT = (
        "Transcribe all text visible in this image exactly as written, character by character, "
        "one text line per output line. Do not correct spelling, do not translate, do not add anything else."
    )

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.name = "qwen3vl-" + model_id.split("-")[-2].lower()
        self.torch = torch
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="cuda"
        ).eval()
        self.proc = AutoProcessor.from_pretrained(model_id)

    def __call__(self, img: Image.Image) -> list[dict]:
        img = img.convert("RGB")
        if max(img.size) > 1024:
            s = 1024 / max(img.size)
            img = img.resize((round(img.width * s), round(img.height * s)), Image.BICUBIC)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": self.PROMPT}]}]
        inputs = self.proc.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
        ).to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=160, do_sample=False)
        text = self.proc.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return [dict(text=l.strip(), box=None, conf=None) for l in text.splitlines() if l.strip()]


def make(name: str):
    if name == "paddleocr":
        return PaddleEngine()
    if name == "easyocr":
        return EasyEngine()
    if name.startswith("qwen3vl"):
        size = name.split("-")[1].upper() if "-" in name else "8B"
        return QwenVLEngine(f"Qwen/Qwen3-VL-{size}-Instruct")
    raise ValueError(name)
