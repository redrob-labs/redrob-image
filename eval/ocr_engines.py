"""OCR engines behind one interface: engine(img: PIL.Image) -> list[Line].

Line = dict(text=str, box=[x0, y0, x1, y1] | None, conf=float | None)
Boxes are in pixel coordinates of the input image. VLM engines return box=None and
lines in the order the model wrote them.
"""
from __future__ import annotations

import json
import os

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


class SubprocessEngine:
    """Runs an engine inside another venv through eval/ocr_worker.py (images passed as PNG files)."""

    def __init__(self, name: str, python: str):
        import subprocess
        import tempfile

        self.name = name
        self.tmp = tempfile.mkdtemp(prefix="ocrw-")
        self.n = 0
        self.p = subprocess.Popen([python, os.path.join(os.path.dirname(__file__), "ocr_worker.py"), name],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        ready = self.p.stdout.readline().strip()
        if ready != "READY":
            raise RuntimeError(f"{name} worker failed to start: {ready!r}")

    def __call__(self, img: Image.Image) -> list[dict]:
        path = getattr(img, "filename", "") or ""
        if not path:
            path = os.path.join(self.tmp, f"{self.n % 4}.png")
            self.n += 1
            img.save(path)
        self.p.stdin.write(os.path.abspath(path) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def __del__(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=30)
        except Exception:
            pass


class NemotronEngine:
    """NVIDIA Nemotron OCR v2, multilingual checkpoint (NVIDIA Open Model License)."""

    name = "nemotron-ocr-v2"

    def __init__(self):
        from nemotron_ocr.inference.pipeline_v2 import NemotronOCRV2

        self.ocr = NemotronOCRV2(lang="multi")
        self.tmp = None

    def __call__(self, img: Image.Image) -> list[dict]:
        path = getattr(img, "filename", "") or ""
        if not path:
            import tempfile

            self.tmp = self.tmp or tempfile.mkdtemp(prefix="nemo-")
            path = os.path.join(self.tmp, "x.png")
            img.save(path)
        W, H = img.size
        out = []
        for p in self.ocr(path, merge_level="line"):
            t = p.get("text", "")
            if not t.strip():
                continue
            # Boxes come normalized to [0, 1]; scale to pixels.
            x0, y0, x1, y1 = (p.get(k) for k in ("left", "upper", "right", "lower"))
            box = None if x0 is None else [x0 * W, y0 * H, x1 * W, y1 * H]
            out.append(dict(text=t, box=box, conf=p.get("confidence")))
        return out


def make(name: str):
    py = os.environ.get("REDROB_PADDLE_PY")
    if name == "paddleocr" and py and not os.environ.get("REDROB_OCR_INPROC"):
        return SubprocessEngine(name, py)
    if name == "paddleocr":
        return PaddleEngine()
    if name == "easyocr":
        return EasyEngine()
    if name == "nemotron-ocr-v2":
        return NemotronEngine()
    if name.startswith("qwen3vl"):
        size = name.split("-")[1].upper() if "-" in name else "8B"
        return QwenVLEngine(f"Qwen/Qwen3-VL-{size}-Instruct")
    raise ValueError(name)
