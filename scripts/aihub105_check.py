"""Check every shard sample: image decodes at its stated size, every text box lies inside it,
and text height at 512 training resolution. Prints counts only."""
import io, json, sys, tarfile
from pathlib import Path
from PIL import Image

out = Path(sys.argv[1]).expanduser()
n = bad_size = bad_box = 0
small = 0
bad_examples = []
for tar in sorted(out.glob("*/*.tar")):
    with tarfile.open(tar) as tf:
        files = {m.name: m for m in tf.getmembers()}
        for name in files:
            if not name.endswith(".json"):
                continue
            meta = json.load(tf.extractfile(files[name]))
            im = Image.open(io.BytesIO(tf.extractfile(files[name[:-5] + ".jpg"]).read()))
            n += 1
            if im.size != (meta["width"], meta["height"]):
                bad_size += 1
            sc = 512 / (meta["width"] * meta["height"]) ** 0.5
            for b in meta["text_blocks"]:
                x0, y0, x1, y1 = b["bbox_xyxy"]
                if x0 < -1 or y0 < -1 or x1 > meta["width"] + 1 or y1 > meta["height"] + 1 or x1 <= x0 or y1 <= y0:
                    bad_box += 1
                    if len(bad_examples) < 3:
                        bad_examples.append({"key": name[:-5], "size": [meta["width"], meta["height"]],
                                             "bbox": [round(v) for v in b["bbox_xyxy"]]})
                if meta["kind"] == "crop" and min(x1 - x0, y1 - y0) * sc < 19.5:
                    small += 1
print(json.dumps({"samples": n, "bad_size": bad_size, "bad_box": bad_box, "crop_text_below_min_px": small,
                  "bad_box_examples": bad_examples}))
