from __future__ import annotations

import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
RELEASE_DIR = ROOT / "releases"
SKIP_PARTS = {".git", ".venv", "__pycache__", ".pytest_cache", ".generated", "runs", "releases"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return not any(part in SKIP_PARTS for part in relative.parts) and path.suffix.lower() not in SKIP_SUFFIXES


def main() -> None:
    RELEASE_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d")
    archive = RELEASE_DIR / f"FLUID-Space-v{VERSION}-source-{stamp}.zip"
    prefix = Path(f"FLUID-Space-v{VERSION}")
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and should_include(path))
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in files:
            relative = path.relative_to(ROOT)
            info = zipfile.ZipInfo.from_file(path, str(prefix / relative))
            if path.suffix in {".sh", ".command"}:
                info.external_attr = (0o100755 & 0xFFFF) << 16
            with path.open("rb") as handle:
                bundle.writestr(info, handle.read(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    print(f"Created: {archive}")
    print(f"Files: {len(files)}")
    print(f"Size: {archive.stat().st_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
