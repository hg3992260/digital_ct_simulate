from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    src = repo_root / "resources" / "Atom.jpeg"
    dst = repo_root / "resources" / "Atom.ico"

    if not src.exists():
        raise FileNotFoundError(f"missing source image: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = im.convert("RGBA")
        im.save(
            dst,
            format="ICO",
            sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
        )

    print(f"wrote: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

