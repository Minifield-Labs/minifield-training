"""Build a portable notebook containing an exact source snapshot."""

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import cast
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "examples/kaggle_magicbox_lfm350m_tpu_v5e_8.ipynb"


def payload() -> bytes:
    """Package the standalone source and lockfile."""
    paths = [
        ROOT / name
        for name in (
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "examples/__init__.py",
        )
    ]
    paths += sorted((ROOT / "src/minifield_training").rglob("*.py"))
    paths += [ROOT / "src/minifield_training/py.typed"]
    paths += sorted((ROOT / "examples/magicbox").glob("*.py"))
    buffer = io.BytesIO()
    with zipfile.ZipFile(
        buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as archive:
        for path in paths:
            info = zipfile.ZipInfo(
                path.relative_to(ROOT).as_posix(),
                date_time=(2026, 9, 27, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes())
    return buffer.getvalue()


def notebook() -> dict[str, object]:
    """Fill the reviewed notebook template with the exact source snapshot."""
    source_bytes = payload()
    template = (ROOT / "examples/magicbox/notebook_template.json").read_text()
    template = template.replace(
        "__SOURCE_SHA256__", hashlib.sha256(source_bytes).hexdigest()
    )
    template = template.replace(
        "__SOURCE_ARCHIVE__", base64.b64encode(source_bytes).decode()
    )
    return cast(dict[str, object], json.loads(template))


def main() -> None:
    """Write a stable source snapshot and executable notebook cells."""
    DESTINATION.write_text(
        json.dumps(notebook(), ensure_ascii=False, indent=1) + "\n"
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
