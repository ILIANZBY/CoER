"""Build an allowlisted source archive without Git or local runtime metadata.

This is a release safeguard, not a substitute for the authors' manual review.
No source file is modified and existing archives are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = (
    "training",
    "cotrain",
    "config",
    "model_configs",
    "verl",
    "AgentDyn/src",
    "corl-evaluation/evaluation",
    "corl-evaluation/scripts",
    "corl-evaluation/tests",
    "corl-evaluation/third_party/InjecAgent/src",
    "corl-evaluation/third_party/InjecAgent/data",
    "tests/release",
    "tests/cotrain",
    "docs/assets/paper",
)
SOURCE_FILES = (
    ".gitignore",
    "README.md",
    "LICENSE",
    "Notice.txt",
    "THIRD_PARTY.md",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-test.txt",
    "vllm_serve_patched.py",
    "AgentDyn/LICENSE",
    "AgentDyn/README.md",
    "AgentDyn/pyproject.toml",
    "corl-evaluation/README.md",
    "corl-evaluation/LICENSE",
    "corl-evaluation/pyproject.toml",
    "docs/anonymous_release.md",
    "docs/publishing.md",
    "scripts/export_anonymous.py",
    "scripts/__init__.py",
    "corl-evaluation/third_party/InjecAgent/LICENCE",
    "corl-evaluation/third_party/InjecAgent/requirements.txt",
)
SKIP_PARTS = {
    "__pycache__",
    "logs",
    "results",
    "runs",
    "wandb",
    "checkpoints",
    "outputs",
    "rollout_logs",
    "node_modules",
    "venv",
    "dist",
    "build",
}
SKIP_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".pdf",
    ".tex",
    ".ipynb",
    ".log",
    ".pt",
    ".bin",
    ".safetensors",
    ".parquet",
    ".zip",
    ".gz",
    ".mp4",
    ".pptx",
    ".docx",
}
IMAGE_PATHS = {
    f"docs/assets/paper/{name}.png"
    for name in (
        "figure1-adaptive-ipi",
        "figure2-corl-framework",
    )
}
PATTERNS = {
    "personal-home-path": re.compile(r"/(?:Users|home)/(?!(?:user|ray|ubuntu|runner)(?:/|\b))[^\s/\"']+"),
    "private-storage": re.compile(r"/(?:mnt/(?:bn|bd|hdfs)/|opt/tiger/|root/\.ssh/)"),
    "internal-service": re.compile(
        r"(?:https?://)?[^\s/\"']*\.(?:byteintl|bytedance|byted|larkoffice|feishu)\.(?:net|com|cn|org)", re.I
    ),
    "private-endpoint": re.compile(
        r"https?://(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)"
    ),
    "ipv6-endpoint": re.compile(r"https?://\[[0-9a-fA-F:]+\]"),
    "credential": re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}|hf_[A-Za-z0-9]{24,}"
        r"|gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{24,})\b"
        r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
}

# Narrow, content-pinned exceptions for reviewed executable benchmark fixtures.
# The key fixtures contain only a header string, never key material. InjecAgent
# paths are simulated tool responses from the public benchmark, not host paths.
# Changes to these files invalidate the exception and require another review.
REVIEWED_FIXTURES = {
    "AgentDyn/src/agentdojo/data/suites/dailylife/include/filesystem.yaml": (
        "fd49ec81e9efeaa2df88a27037fd51a850454fdfa74f8d363b6ab72d6fc1478c",
        {"credential"},
    ),
    "AgentDyn/src/agentdojo/default_suites/v1/dailylife/injection_tasks.py": (
        "592e4009f00d9ab029469584197bd059260ec041775c474258dbc5facb33efe0",
        {"credential"},
    ),
    "corl-evaluation/third_party/InjecAgent/data/attacker_simulated_responses.json": (
        "b1da2e1fb75f266c069b832fef6738c295f01d8962fae15bdac9ff6625e5594e",
        {"personal-home-path"},
    ),
}


def eligible(path: Path) -> bool:
    return not (
        any(part.startswith(".") or part in SKIP_PARTS for part in path.parts) or path.suffix.lower() in SKIP_SUFFIXES
    )


def collect_files(root: Path) -> list[Path]:
    files: set[Path] = set()
    for item in SOURCE_FILES:
        path = root / item
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Required release file missing or symlink: {item}")
        files.add(Path(item))
    for item in SOURCE_DIRS:
        directory = root / item
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"Required release directory missing or symlink: {item}")
        for path in directory.rglob("*"):
            relative = path.relative_to(root)
            if not eligible(relative):
                continue
            if path.is_symlink():
                raise ValueError(f"Symlink requires manual review: {relative}")
            if path.is_file():
                files.add(relative)
    return sorted(files)


def scan_text(relative: Path, content: bytes) -> list[str]:
    if relative.as_posix() in IMAGE_PATHS:
        # Exactly the manually inspected paper images; no arbitrary binary files.
        if not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return [f"{relative}: invalid PNG"]
        return []
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return [f"{relative}: unreviewed binary file"]
    findings = []
    digest, allowed_kinds = REVIEWED_FIXTURES.get(relative.as_posix(), (None, set()))
    if hashlib.sha256(content).hexdigest() != digest:
        allowed_kinds = set()
    for number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in PATTERNS.items():
            if kind not in allowed_kinds and pattern.search(line):
                # Never echo the matching secret/path itself.
                findings.append(f"{relative}:{number}: {kind}")
    return findings


def build_archive(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing release archive")
    payloads = {}
    findings = []
    for relative in collect_files(root):
        content = (root / relative).read_bytes()
        findings.extend(scan_text(relative, content))
        payloads[relative.as_posix()] = content
    if findings:
        raise ValueError("Release scan requires review:\n" + "\n".join(findings))
    manifest = {
        "schema_version": 1,
        "kind": "anonymous-source-only",
        "raw_results_included": False,
        "manual_review_required": True,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()},
    }
    payloads["SOURCE_MANIFEST.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(payloads.items()):
            info = zipfile.ZipInfo("corl/" + name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            mode = 0o755 if name.endswith(".sh") else 0o644
            info.external_attr = (0o100000 | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, content)
    return {"files": len(payloads), "bytes": output.stat().st_size, "manual_review_required": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_archive(ROOT, args.output), indent=2))


if __name__ == "__main__":
    main()
