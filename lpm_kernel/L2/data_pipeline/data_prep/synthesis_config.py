"""Concurrency shared by the L2 data synthesis generators."""
import os
from typing import Mapping, Optional


def get_synthesis_workers(environ: Optional[Mapping[str, str]] = None) -> int:
    """Use the Web parameter, with legacy lowercase support and default two."""
    environment = os.environ if environ is None else environ
    value = environment.get("CONCURRENCY_THREADS", environment.get("concurrency_threads", "2"))
    try:
        workers = int(value)
    except (TypeError, ValueError):
        raise ValueError("CONCURRENCY_THREADS must be a positive integer") from None
    if workers < 1:
        raise ValueError("CONCURRENCY_THREADS must be a positive integer")
    return workers
