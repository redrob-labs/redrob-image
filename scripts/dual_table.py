"""Print a qwen-only vs dual-OCR table for the pilot runs (run on the gpu-box)."""
import json
import os

R = os.path.expanduser("~/runs")
B = os.path.expanduser("~/redrob-image-eval/repo/outputs")
runs = [("Base", B + "/baseline-base"), ("Base+r32", R + "/pilot-r32/final-eval-Z-Image"),
        ("Base+r64v2", R + "/pilot-r64v2/final-eval-Z-Image"), ("Turbo", B + "/baseline-turbo"),
        ("Turbo+r32", R + "/pilot-r32/final-eval-Z-Image-Turbo"),
        ("Turbo+r64v2", R + "/pilot-r64v2/final-eval-Z-Image-Turbo")]
runs += [("r32 " + s, R + "/pilot-r32/eval/step-" + s) for s in ["2500", "5000", "7500", "10000", "11492"]]
runs += [("r64v2 " + s, R + "/pilot-r64v2/eval/step-" + s) for s in ["2500", "5000", "7500", "10000", "12216"]]
print("run|n|qwen CER|dual CER|dual seen|dual unseen|qwen EM|dual EM|dual EM seen|dual EM unseen")
for name, d in runs:
    q = json.load(open(d + "/report.json"))["metrics"]
    m = json.load(open(d + "/report_dual.json"))["metrics"]
    o, s, u = m["overall"], m["split"]["seen"], m["split"]["unseen"]
    print(f"{name}|{o['n']}|{q['overall']['cer']:.3f}|{o['cer']:.3f}|{s['cer']:.3f}|{u['cer']:.3f}|"
          f"{q['overall']['em']:.2f}|{o['em']:.2f}|{s['em']:.2f}|{u['em']:.2f}")
