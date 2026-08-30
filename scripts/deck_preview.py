"""Rasterise deck slides without PowerPoint.

PowerPoint is a sandboxed GUI application, not a renderer. Driving it over
AppleScript fails with -9074 on paths it has no grant for, and a failed open
leaves a modal that blocks every later command -- which is a person having to
walk over and click something before a layout check can run. LibreOffice
converts headless, from any path, with no dialog:

    uv run --with pymupdf python scripts/deck_preview.py 28
    uv run --with pymupdf python scripts/deck_preview.py            # every slide

Writes PNGs to docs/deck/preview/ (git-ignored scratch).

One machine setup step, and it is not optional for a Chinese deck: LibreOffice
on macOS enumerates only its *own* font directory, so every CJK glyph comes out
blank no matter what is installed in ~/Library/Fonts. Copy the faces in once:

    brew install --cask libreoffice font-noto-sans-cjk font-noto-sans-tc
    cp ~/Library/Fonts/NotoSansCJK.ttc ~/Library/Fonts/"NotoSansTC[wght].ttf" \
       /Applications/LibreOffice.app/Contents/Resources/fonts/truetype/

Without that the preview silently drops Chinese, which reads like a bug in the
deck and is not one.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "deck" / "aoi-agent-journey.zh-TW.pptx"
OUT = ROOT / "docs" / "deck" / "preview"
SOFFICE = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")


def soffice() -> Path | None:
    if SOFFICE.exists():
        return SOFFICE
    found = shutil.which("soffice") or shutil.which("libreoffice")
    return Path(found) if found else None


def to_pdf(deck: Path, binary: Path) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="deck-preview-"))
    subprocess.run([str(binary), "--headless", "--convert-to", "pdf",
                    "--outdir", str(tmp), str(deck)],
                   check=True, capture_output=True, timeout=600)
    pdf = next(tmp.glob("*.pdf"), None)
    if pdf is None:
        raise SystemExit("LibreOffice produced no PDF")
    return pdf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("slides", nargs="*", type=int, help="1-based slide numbers; default all")
    ap.add_argument("--deck", type=Path, default=DECK)
    ap.add_argument("--dpi", type=int, default=110)
    args = ap.parse_args()

    binary = soffice()
    if binary is None:
        print("LibreOffice not found. Install it once:\n"
              "  brew install --cask libreoffice\n"
              "It renders headless, so no dialog and no sandbox grant is needed.",
              file=sys.stderr)
        return 2
    if not args.deck.exists():
        print(f"{args.deck} not found; build the deck first", file=sys.stderr)
        return 2

    import pymupdf

    pdf = to_pdf(args.deck, binary)
    doc = pymupdf.open(pdf)
    wanted = args.slides or range(1, doc.page_count + 1)
    OUT.mkdir(parents=True, exist_ok=True)
    for n in wanted:
        if not 1 <= n <= doc.page_count:
            print(f"slide {n} is outside 1..{doc.page_count}", file=sys.stderr)
            continue
        target = OUT / f"slide{n:02d}.png"
        doc[n - 1].get_pixmap(dpi=args.dpi).save(target)
        print(f"{target.relative_to(ROOT)}  {target.stat().st_size // 1024} KB")
    doc.close()
    shutil.rmtree(pdf.parent, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
