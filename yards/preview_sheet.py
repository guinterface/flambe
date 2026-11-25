"""Contact sheet of a yard's 12 preview views. usage: preview_sheet.py <yard> [...]"""
import sys, os, glob
from PIL import Image
SY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for y in sys.argv[1:]:
    d = f"{SY}/yards/{y}/preview"; files = sorted(glob.glob(f"{d}/images/*.png"))[:12]
    if not files: print(y, "no preview images"); continue
    W, H = 400, 225; cols = 4; rows = (len(files) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * H), "white")
    for k, f in enumerate(files):
        im = Image.open(f).convert("RGB").resize((W, H)); sheet.paste(im, ((k % cols) * W, (k // cols) * H))
    out = f"{d}/sheet.png"; sheet.save(out); print(out)
