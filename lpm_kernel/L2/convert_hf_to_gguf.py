"""Compatibility entry point for the pinned llama.cpp converter.

Do not import the repository's historical gguf-py: it predates Qwen3.
"""
from pathlib import Path
import runpy
import sys
import subprocess

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    llama_root = root / "llama.cpp"
    converter = llama_root / "convert_hf_to_gguf.py"
    if not converter.is_file():
        raise SystemExit("Pinned converter missing. Run bash scripts/build_llama.sh first.")
    expected = (root / "dependencies/llama.cpp.version").read_text().strip()
    actual = subprocess.check_output(["git", "-C", str(llama_root), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise SystemExit("Converter revision mismatch. Run bash scripts/build_llama.sh first.")
    sys.path.insert(0, str(llama_root / "gguf-py"))
    sys.path.insert(0, str(llama_root))
    runpy.run_path(str(converter), run_name="__main__")
