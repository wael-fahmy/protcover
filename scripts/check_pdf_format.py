"""Measure a compiled PDF against RECOMB's format rules (US Letter, 1-inch margins, minimum 10pt).

Two checks, because LaTeX settings do not show what a document class or package does to the output:
- font sizes per page, read from the PDF (10 TeX points are 9.96 PDF points; 5 to 7pt text is
  math sub- and superscripts);
- the ink on each page, rasterized at 300 dpi: its bounding box must lie inside the margins.

Needs PyMuPDF, which the analysis does not use, so it is not in requirements.txt:
  uv venv pdfenv && uv pip install --python pdfenv pymupdf

Usage: python scripts/check_pdf_format.py paper/main.pdf [--last-page N] [--show]
"""
import argparse
import collections

import pymupdf

MARGIN, DPI = 72.0, 300
MIN_SIZE = 9.95  # PDF points; 10 TeX points are 9.963


def ink_box(page):
    """Bounding box of the non-white pixels, in PDF points, or None for a blank page."""
    pix = page.get_pixmap(dpi=DPI, colorspace=pymupdf.csGRAY, alpha=False)
    s, w, h = pix.samples, pix.width, pix.height
    rows = [y for y in range(h) if min(s[y * w:(y + 1) * w]) < 200]
    if not rows:
        return None
    x0, x1 = w, 0
    for y in rows:
        row = s[y * w:(y + 1) * w]
        for x in range(0, min(x0, w)):
            if row[x] < 200:
                x0 = x
                break
        for x in range(w - 1, max(x1, 0), -1):
            if row[x] < 200:
                x1 = x
                break
    k = 72.0 / DPI
    return x0 * k, rows[0] * k, (x1 + 1) * k, (rows[-1] + 1) * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--last-page", type=int, default=None, help="check only pages up to this one")
    ap.add_argument("--show", action="store_true", help="print the below-minimum text of each page")
    args = ap.parse_args()
    doc = pymupdf.open(args.pdf)
    print(args.pdf, "pages", len(doc), "size", doc[0].rect.width, "x", doc[0].rect.height)
    bad_ink = bad_font = 0
    for no, page in enumerate(doc, 1):
        if args.last_page and no > args.last_page:
            break
        W, H = page.rect.width, page.rect.height
        hist = collections.Counter()
        small = []
        for b in page.get_text("rawdict")["blocks"]:
            if b["type"] != 0:
                continue
            for line in b["lines"]:
                for sp in line["spans"]:
                    txt = "".join(c["c"] for c in sp.get("chars", []))
                    if txt.strip():
                        hist[round(sp["size"], 2)] += len(txt)
                        # Spans of up to three characters at 8pt or less are math sub- and superscripts.
                        if sp["size"] < MIN_SIZE and (len(txt.strip()) > 3 or sp["size"] > 8):
                            small.append((round(sp["size"], 2), txt.strip()[:40]))
        box = ink_box(page)
        out = box and (box[0] < MARGIN - 0.1 or box[1] < MARGIN - 0.1 or box[2] > W - MARGIN + 0.1 or box[3] > H - MARGIN + 0.1)
        bad_ink += bool(out)
        bad_font += bool(small)
        sizes = sorted(hist.items(), key=lambda kv: -kv[1])[:3]
        print(f"{no:3d} ink " + ("none" if not box else "x %.1f-%.1f y %.1f-%.1f" % (box[0], box[2], box[1], box[3]))
              + ("  OUTSIDE" if out else "") + f"  small spans {len(small):4d}  {sizes}")
        if args.show and small:
            print("     ", small[:8])
    print(f"pages with ink outside the margins: {bad_ink}; pages with text spans below 10pt: {bad_font}")


if __name__ == "__main__":
    main()
