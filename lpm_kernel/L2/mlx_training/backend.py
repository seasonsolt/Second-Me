"""Choose training without allowing PyTorch to silently select MPS."""
import importlib.util
import platform


def resolve_training_backend(params):
    requested = params.get("training_backend", "auto")
    if requested not in ("auto", "mlx", "pytorch"):
        raise ValueError("training_backend must be auto, mlx, or pytorch")
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    if requested == "mlx" or (requested == "auto" and apple_silicon):
        if not apple_silicon:
            raise RuntimeError("MLX training requires native macOS on Apple Silicon")
        if any(importlib.util.find_spec(name) is None for name in ("mlx", "mlx_lm")):
            raise RuntimeError("Apple Silicon training requires mlx-lm; install the project training dependencies")
        return "mlx"
    return "pytorch"


def training_capabilities(params):
    """Read-only capabilities for UI; do not overwrite the user's auto preference."""
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    mlx_available = apple_silicon and all(
        importlib.util.find_spec(name) is not None for name in ("mlx", "mlx_lm")
    )
    error = None
    try:
        backend = resolve_training_backend(params)
    except (ValueError, RuntimeError) as exc:
        backend = None
        error = str(exc)
    return {
        "apple_silicon": apple_silicon, "mlx_available": mlx_available,
        "selected_training_backend": backend, "training_backend_error": error,
    }


def normalize_training_language(value):
    """Match existing L2 templates while honoring the configured generation language."""
    language = (value or "English").strip().split("/")[-1].strip()
    if language.lower() in {"chinese", "中文", "zh", "zh-cn", "zh-tw", "繁體中文", "简体中文"}:
        return "Chinese"
    if language.lower() in {"english", "en", "英文"}:
        return "English"
    return language or "English"
