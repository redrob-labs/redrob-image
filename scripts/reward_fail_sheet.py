"""Side-by-side sheet of reward-check pairs the reward got wrong (bad >= good)."""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

run = Path(sys.argv[1])
out = Path(sys.argv[2])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
font_path = sys.argv[4] if len(sys.argv) > 4 else "/usr/share/fonts/truetype/noto/NotoSansKR-Bold.ttf"

rewards = {}
for line in (run / "rewards.jsonl").read_text().splitlines():
    r = json.loads(line)
    rewards[r["key"]] = r
pairs = [json.loads(l) for l in (run / "pairs.jsonl").read_text().splitlines()]

fails = []
for p in pairs:
    g = rewards.get(f"good/{p['id']}")
    b = rewards.get(f"bad/{p['pair_id']}")
    if g and b and b["total"] >= g["total"]:
        fails.append((p, g, b))

# one per error kind first, then fill
picked, seen = [], set()
for f in fails:
    if f[0]["kind"] not in seen:
        picked.append(f)
        seen.add(f[0]["kind"])
for f in fails:
    if len(picked) >= n:
        break
    if f not in picked:
        picked.append(f)
picked = picked[:n]

W = 512
try:
    font = ImageFont.truetype(font_path, 22)
except OSError:
    font = ImageFont.load_default()
rows = []
for p, g, b in picked:
    gi = Image.open(run / "images/good" / f"{p['id']}.png").convert("RGB")
    bi = Image.open(run / "images/bad" / f"{p['pair_id']}.png").convert("RGB")
    h = int(gi.height * W / gi.width)
    gi, bi = gi.resize((W, h)), bi.resize((W, h))
    row = Image.new("RGB", (W * 2 + 8, h + 36), "white")
    row.paste(gi, (0, 36))
    row.paste(bi, (W + 8, 36))
    d = ImageDraw.Draw(row)
    d.text((6, 4), f"{p['target_text']}  {g['total']:.2f}", fill="black", font=font)
    d.text((W + 14, 4), f"{p['wrong']}  {b['total']:.2f}  ({p['kind']})", fill="black", font=font)
    rows.append(row)
sheet = Image.new("RGB", (W * 2 + 8, sum(r.height for r in rows) + 8 * len(rows)), "white")
y = 0
for r in rows:
    sheet.paste(r, (0, y))
    y += r.height + 8
out.parent.mkdir(parents=True, exist_ok=True)
sheet.save(out)
print(f"fails {len(fails)}/{len(pairs)}")
for p, g, b in picked:
    print(p["target_text"], p["wrong"], p["kind"], round(g["total"], 3), round(b["total"], 3), g["word_best"], "|", b["word_best"])
