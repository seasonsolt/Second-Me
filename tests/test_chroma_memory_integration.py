"""Real temporary Chroma/SQLite; deterministic embeddings; no private config.

Run the worker in a separate process so service import fixtures cannot pollute
other tests. Neither application .env nor configured embedding endpoints load.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_real_chroma_memory_lifecycle(tmp_path):
    import importlib.util
    if importlib.util.find_spec("chromadb") is None:
        pytest.skip("Chroma is not installed")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker", str(tmp_path)],
        cwd=ROOT, env={**os.environ, "ANONYMIZED_TELEMETRY": "False"},
        capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert report["checks"] == ["three_hits", "update", "delete", "other_document_preserved",
                               "model_switch", "endpoint_switch", "dimension_switch",
                               "legacy_preserved", "chat_provider_ignored"]


def worker(directory):
    import importlib.util
    import logging
    from contextlib import contextmanager
    from io import BytesIO
    import types
    from types import SimpleNamespace
    from unittest.mock import Mock

    directory = Path(directory)
    os.chdir(directory)  # Chroma Settings defaults to .env in cwd; never use the repository cwd.
    sys.path.insert(0, str(ROOT))
    import chromadb
    from chromadb.config import Settings
    import numpy as np
    from sqlalchemy import create_engine
    from sqlalchemy.orm import DeclarativeBase, sessionmaker

    os.environ["CHROMA_PERSIST_DIRECTORY"] = str(directory / "chroma")
    original_client = chromadb.PersistentClient
    chromadb.PersistentClient = lambda path: original_client(path=path, settings=Settings(anonymized_telemetry=False, _env_file=None))

    # Block accidental remote requests, including telemetry, in this worker.
    import requests
    def forbid_network(*args, **kwargs):
        raise AssertionError("Network is forbidden in the Chroma memory integration test")
    requests.sessions.Session.request = forbid_network

    def stub(name, **attributes):
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, ROOT / path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    package = stub("lpm_kernel.file_data")
    package.__path__ = [str(ROOT / "lpm_kernel/file_data")]
    logger = logging.getLogger("chroma-memory-integration")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    stub("lpm_kernel.configs.logging", get_train_process_logger=lambda: logger)
    stub("lpm_kernel.common.logging", logger=logger)
    configuration = SimpleNamespace(embedding_model_name="@cf/baai/bge-m3",
                                    embedding_endpoint="https://synthetic.invalid/embedding-a",
                                    provider_type="chat-only-provider")
    config_service = SimpleNamespace(get_available_llm=lambda: configuration)
    stub("lpm_kernel.api.services.user_llm_config_service", UserLLMConfigService=lambda: config_service)

    class FixedLLM:
        def __init__(self):
            self.user_llm_config_service = config_service

        def get_embedding(self, texts):
            dimension = 1536 if configuration.embedding_model_name == "text-embedding-3-small" else 1024
            vectors = []
            for value in texts:
                vector = np.zeros(dimension, dtype=np.float32)
                if "garden" in value or "unrelated" in value:
                    vector[2] = 1.
                else:
                    cosine = .9 if "Thursday" in value else .8 if "budget" in value else 1.
                    vector[0] = cosine
                    vector[1] = np.sqrt(1. - cosine ** 2)
                vectors.append(vector)
            return np.array(vectors)

    stub("lpm_kernel.common.llm", LLMClient=FixedLLM)
    class Base(DeclarativeBase):
        pass
    engine = create_engine(f"sqlite:///{directory / 'memory.sqlite'}")
    Session = sessionmaker(bind=engine)
    class DatabaseSession:
        _session_factory = Session
        @staticmethod
        @contextmanager
        def session():
            with Session() as current:
                try:
                    yield current
                    current.commit()
                except Exception:
                    current.rollback()
                    raise
    stub("lpm_kernel.common.repository.database_session", Base=Base, DatabaseSession=DatabaseSession)
    config = {"USER_RAW_CONTENT_DIR": str(directory / "uploads"), "LOCAL_BASE_DIR": str(directory),
              "DOCUMENT_CHUNK_SIZE": "40", "DOCUMENT_CHUNK_OVERLAP": "0"}
    stub("lpm_kernel.configs.config", Config=SimpleNamespace(from_env=lambda: config))

    load("lpm_kernel.file_data.process_status", "lpm_kernel/file_data/process_status.py")
    load("lpm_kernel.file_data.document_dto", "lpm_kernel/file_data/document_dto.py")
    load("lpm_kernel.common.repository.base_repository", "lpm_kernel/common/repository/base_repository.py")
    documents = load("lpm_kernel.file_data.document", "lpm_kernel/file_data/document.py")
    models = load("lpm_kernel.file_data.models", "lpm_kernel/file_data/models.py")
    memories = load("lpm_kernel.models.memory", "lpm_kernel/models/memory.py")
    load("lpm_kernel.L1.bio", "lpm_kernel/L1/bio.py")
    l1 = load("lpm_kernel.models.l1", "lpm_kernel/models/l1.py")
    status_bios = load("lpm_kernel.models.status_biography", "lpm_kernel/models/status_biography.py")
    Base.metadata.create_all(engine)
    models.Base.metadata.create_all(engine)
    load("lpm_kernel.file_data.document_repository", "lpm_kernel/file_data/document_repository.py")
    load("lpm_kernel.file_data.chroma_utils", "lpm_kernel/file_data/chroma_utils.py")
    load("lpm_kernel.file_data.embedding_service", "lpm_kernel/file_data/embedding_service.py")
    load("lpm_kernel.file_data.chunker", "lpm_kernel/file_data/chunker.py")
    stub("lpm_kernel.common.repository.vector_store_factory", VectorStoreFactory=SimpleNamespace(get_instance=lambda: None))
    stub("lpm_kernel.kernel.l0_base", InsightKernel=Mock(), SummaryKernel=Mock())

    def process_file(path):
        return SimpleNamespace(name=Path(path).name, mime_type="text/plain",
                               document_size=Path(path).stat().st_size,
                               raw_content=Path(path).read_text(),
                               extract_status=sys.modules["lpm_kernel.file_data.process_status"].ProcessStatus.SUCCESS)
    stub("lpm_kernel.file_data.process_factory", ProcessorFactory=SimpleNamespace(auto_detect_and_process=process_file))
    load("lpm_kernel.file_data.document_service", "lpm_kernel/file_data/document_service.py")
    storage_module = load("lpm_kernel.file_data.memory_service", "lpm_kernel/file_data/memory_service.py")
    storage = storage_module.StorageService(config)
    service = storage.document_service

    # Preserve an old incompatible collection; it must never be reset.
    legacy = service.embedding_service.client.get_or_create_collection("document_chunks", metadata={"hnsw:space": "cosine", "dimension": 1536})
    legacy.add(ids=["900"], documents=["legacy synthetic record"],
               embeddings=[[1.] + [0.] * 1535], metadatas=[{"document_id": "900"}])
    class File(BytesIO):
        def __init__(self, name, text):
            super().__init__(text.encode())
            self.filename = name
        def save(self, filename):
            Path(filename).write_bytes(self.getvalue())

    trip = "synthetic trip destination Alpha.\n\nsynthetic trip date Thursday.\n\nsynthetic trip budget 100."
    memory, document = storage.save_file(File("trip.txt", trip))
    other_memory, other_document = storage.save_file(File("garden.txt", "unrelated synthetic garden"))
    hits = service.embedding_service.search_similar_chunks("synthetic trip", 3)
    assert len(hits) == 3 and all(chunk.document_id == document.id for chunk, _ in hits)
    assert [score for _, score in hits] == pytest.approx([1., .9, .8], abs=1e-5)
    checks = ["three_hits"]
    original_collection = service.embedding_service.chunk_collection.name
    with Session() as current:
        current.add(l1.L1Version(version=1, status="SUCCESS"))
        current.add(status_bios.StatusBiography(content="c", content_third_view="c", summary="s",
                                                summary_third_view="s"))
        current.commit()
    service.refresh_document_index(document.id, "synthetic trip destination Beta.")
    hits = service.embedding_service.search_similar_chunks("synthetic trip", 10)
    assert any("Beta" in chunk.content for chunk, _ in hits)
    assert not any("Alpha" in chunk.content for chunk, _ in hits)
    with Session() as current:
        assert current.get(documents.Document, document.id).raw_content == "synthetic trip destination Beta."
        assert current.get(l1.L1Version, 1).status == "stale"
        assert current.query(status_bios.StatusBiography).count() == 1
    checks.append("update")

    # Same dimension, different model: singleton rebinding must use a new index.
    configuration.embedding_model_name = "mxbai-embed-large"
    assert service.embedding_service.search_similar_chunks("synthetic trip", 3) == []
    switched_collection = service.embedding_service.chunk_collection.name
    assert switched_collection != original_collection
    assert service.embedding_service.client.get_collection(original_collection).count() == 2
    service.refresh_document_index(document.id)
    service.refresh_document_index(other_document.id)
    assert any("Beta" in chunk.content for chunk, _ in service.embedding_service.search_similar_chunks("synthetic trip", 3))

    configuration.embedding_endpoint = "https://synthetic.invalid/embedding-b"
    assert service.embedding_service.search_similar_chunks("synthetic trip", 3) == []
    endpoint_collection = service.embedding_service.chunk_collection.name
    assert endpoint_collection != switched_collection
    service.refresh_document_index(document.id)
    service.refresh_document_index(other_document.id)
    configuration.provider_type = "different-chat-only-provider"
    service.embedding_service._ensure_current_model()
    assert service.embedding_service.chunk_collection.name == endpoint_collection
    configuration.embedding_model_name = "text-embedding-3-small"
    assert service.embedding_service.search_similar_chunks("synthetic trip", 3) == []
    assert service.embedding_service.dimension == 1536
    service.refresh_document_index(document.id)
    service.refresh_document_index(other_document.id)

    assert service.delete_file_by_name(memory.name)
    hits = service.embedding_service.search_similar_chunks("synthetic trip", 3)
    assert all(chunk.document_id != document.id for chunk, _ in hits)
    checks.append("delete")
    for entry in service.embedding_service.client.list_collections():
        name = entry.name if hasattr(entry, "name") else entry
        if name.startswith("document_chunks_"):
            records = service.embedding_service.client.get_collection(name).get()
            assert all(record.get("document_id") != str(document.id) for record in records["metadatas"])
    assert service._repository.find_one(other_document.id) is not None
    assert any(chunk.document_id == other_document.id for chunk, _ in hits)
    checks.extend(["other_document_preserved", "model_switch", "endpoint_switch", "dimension_switch"])
    assert legacy.get(ids=["900"])["documents"] == ["legacy synthetic record"]
    assert service.embedding_service.client.get_collection(original_collection).count() == 0
    checks.extend(["legacy_preserved", "chat_provider_ignored"])
    with Session() as current:
        assert current.query(memories.Memory).count() == 1
        assert current.query(models.ChunkModel).count() == 1
    print(json.dumps({"checks": checks, "chroma": chromadb.__version__, "embedding_requests": "fixed_local_vectors"}))


if __name__ == "__main__" and len(sys.argv) == 3 and sys.argv[1] == "--worker":
    worker(sys.argv[2])
