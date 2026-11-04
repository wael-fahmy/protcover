"""Measure a compiled PDF against RECOMB's format rules: font sizes per page and text outside the
1-inch margins. Usage: pdf_audit.py FILE [--show]  (--show prints sample text of each violation)."""
import collections
import sys

import fitz

path = sys.argv[1]
show = "--show" in sys.argv
doc = fitz.open(path)
M = 72.0
EPS = 0.05
print(path, "pages", len(doc), "size", doc[0].rect)

sizes_by_page = {}
below = collections.defaultdict(list)
outside = collections.defaultdict(list)
for pno, page in enumerate(doc, 1):
    W, H = page.rect.width, page.rect.height
    d = page.get_text("rawdict")
    hist = collections.Counter()
    for b in d["blocks"]:
        if b["type"] != 0:
            continue
        for line in b["lines"]:
            for sp in line["spans"]:
                chars = sp.get("chars", [])
                txt = "".join(c["c"] for c in chars)
                if not txt.strip():
                    continue
                size = round(sp["size"], 2)
                hist[size] += len(txt)
                # Sub- and superscripts are set small by design: the origin is shifted from the line.
                if size < 9.95:
                    below[pno].append((size, txt[:50], sp["font"]))
                for c in chars:
                    x0, y0, x1, y1 = c["bbox"]
                    if c["c"].strip() and (x0 < M - EPS or x1 > W - M + EPS or y0 < M - EPS or y1 > H - M + EPS):
                        outside[pno].append((round(x0, 2), round(x1, 2), round(y0, 2), round(y1, 2), txt[:50]))
                        break
    sizes_by_page[pno] = hist

print("\npage: characters per font size (sizes below 10 marked)")
for p, h in sizes_by_page.items():
    tot = sum(h.values())
    small = sum(v for k, v in h.items() if k < 9.95)
    top = sorted(h.items(), key=lambda kv: -kv[1])[:5]
    print(f"{p:3d} total {tot:5d} below10 {small:5d} ({100 * small / max(tot, 1):4.1f}%)  ", top)

print("\npage: spans with characters outside the 1-inch margins")
for p in sorted(outside):
    xs = [o for o in outside[p]]
    print(p, len(xs), "left", min(o[0] for o in xs), "right", max(o[1] for o in xs), "top", min(o[2] for o in xs), "bottom", max(o[3] for o in xs))
    if show:
        for o in xs[:6]:
            print("    ", o)

if show:
    print("\nbelow-10pt spans (page 1-13, excluding very short sub/superscripts)")
    for p in sorted(below):
        if p > 13:
            continue
        long_ = [b for b in below[p] if len(b[1].strip()) > 3]
        print(p, len(below[p]), "long:", long_[:6])
