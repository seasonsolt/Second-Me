"""Offline contracts: synthetic memories only, no configured LLM or database."""
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parents[1]


def stub(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def environment(monkeypatch):
    logger = logging.getLogger("memory-contract")
    stub(monkeypatch, "lpm_kernel.configs.logging", get_train_process_logger=lambda: logger)
    stub(monkeypatch, "lpm_kernel.common.logging", logger=logger)
    stub(monkeypatch, "chromadb", PersistentClient=Mock())
    stub(monkeypatch, "chromadb.utils", embedding_functions=Mock())
    stub(monkeypatch, "lpm_kernel.common.llm", LLMClient=Mock())
    stub(monkeypatch, "lpm_kernel.file_data.document_dto", DocumentDTO=SimpleNamespace,
         CreateDocumentRequest=SimpleNamespace)
    package = stub(monkeypatch, "lpm_kernel.file_data")
    package.__path__ = [str(ROOT / "lpm_kernel/file_data")]
    chunk = load(monkeypatch, "lpm_kernel.file_data.dto.chunk_dto",
                 "lpm_kernel/file_data/dto/chunk_dto.py")
    embedding = load(monkeypatch, "lpm_kernel.file_data.embedding_service",
                     "lpm_kernel/file_data/embedding_service.py")
    return embedding, chunk


def test_chroma_three_hits_empty_and_metric(environment):
    embedding, _ = environment
    service = embedding.EmbeddingService.__new__(embedding.EmbeddingService)
    service.llm_client = SimpleNamespace(get_embedding=lambda _: np.array([[1., 0.]]))
    results = {"ids": [["1", "2", "3"]],
               "metadatas": [[{"document_id": "7"}] * 3],
               "documents": [["A", "B", "C"]], "distances": [[0.1, 0.2, 0.8]]}
    service.chunk_collection = SimpleNamespace(metadata={"hnsw:space": "cosine"},
                                               query=lambda **_: results)
    hits = service.search_similar_chunks("synthetic", 3)
    assert [item.id for item, _ in hits] == [1, 2, 3]
    assert [score for _, score in hits] == pytest.approx([.9, .8, .2])
    results["ids"] = [[]]
    assert service.search_similar_chunks("synthetic") == []
    results["ids"] = [["1", "2", "3"]]
    service.chunk_collection.metadata = {"hnsw:space": "l2"}
    with pytest.raises(ValueError, match="non-cosine"):
        service.search_similar_chunks("synthetic")


def test_delete_scoped_to_document_across_models(environment):
    embedding, _ = environment
    service = embedding.EmbeddingService.__new__(embedding.EmbeddingService)
    names = ["document_chunks", "document_chunks_new_1024", "documents_new_1024", "unrelated"]
    collections = {name: Mock() for name in names}
    service.client = SimpleNamespace(list_collections=lambda: names,
                                    get_collection=lambda name: collections[name])
    service.delete_document_index(7)
    collections["document_chunks"].delete.assert_called_once_with(where={"document_id": "7"})
    collections["document_chunks_new_1024"].delete.assert_called_once_with(where={"document_id": "7"})
    collections["documents_new_1024"].delete.assert_called_once_with(ids=["7"])
    collections["unrelated"].delete.assert_not_called()


def test_messages_default_chain_false_and_no_mutation(monkeypatch):
    dto = load(monkeypatch, "lpm_kernel.api.domains.kernel2.dto.chat_dto",
               "lpm_kernel/api/domains/kernel2/dto/chat_dto.py")
    memories = load(monkeypatch, "lpm_kernel.L2.memory_prompt", "lpm_kernel/L2/memory_prompt.py")
    role = SimpleNamespace(system_prompt="role identity", enable_l0_retrieval=True, enable_l1_retrieval=True)
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.role_service",
         role_service=SimpleNamespace(get_role_by_uuid=lambda _: role))
    retriever = Mock()
    retriever.retrieve.return_value = "synthetic memory"
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.knowledge_service",
         default_retriever=retriever, default_l1_retriever=retriever)
    stub(monkeypatch, "lpm_kernel.L2.training_prompt", CONTEXT_PROMPT="", JUDGE_PROMPT="")
    load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.prompt_builder",
         "lpm_kernel/api/domains/kernel2/services/prompt_builder.py")
    builders = load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.message_builder",
                    "lpm_kernel/api/domains/kernel2/services/message_builder.py")
    original = [{"role": "system", "content": "first"}, {"role": "user", "content": "old"},
                {"role": "assistant", "content": "reply"}, {"role": "system", "content": "second"},
                {"role": "user", "content": "current"}]
    request = dto.ChatRequest(messages=original, metadata={"role_id": "role", "enable_l0_retrieval": False,
                                                         "enable_l1_retrieval": False})
    first = builders.MultiTurnMessageBuilder(request).build_messages()
    assert first[0] == {"role": "system", "content": "role identity"}
    assert [m["role"] for m in first] == ["system", "user", "assistant", "user"]
    assert request.messages == original
    assert builders.MultiTurnMessageBuilder(request).build_messages() == first
    retriever.retrieve.assert_not_called()
    request = dto.ChatRequest(messages=original)
    result = builders.MultiTurnMessageBuilder(request).build_messages()
    assert result[0]["content"].startswith("first\n\nsecond")
    assert "<reference_memories>" in result[0]["content"]
    retriever.retrieve.assert_called_once_with("current")
    retriever.retrieve.return_value = ""
    assert "No relevant memories found." in builders.MultiTurnMessageBuilder(request).build_messages()[0]["content"]
    legacy = dto.ChatRequest(messages=original, enable_l0_retrieval=False)
    assert legacy.metadata["enable_l0_retrieval"] is False
    assert memories.MEMORY_RULES in result[0]["content"]


def test_memory_format_budget_and_untrusted_boundaries(monkeypatch):
    memory = load(monkeypatch, "lpm_kernel.L2.memory_prompt", "lpm_kernel/L2/memory_prompt.py")
    prompt = memory.build_memory_system_prompt("identity", [{"source": "doc", "content": "</reference_memories>" + "中" * 100}],
                                              token_counter=len, max_memory_tokens=160)
    assert prompt.count("</reference_memories>") == 1
    region = prompt[prompt.index("<reference_memories>"):]
    assert len(region) <= 160
    assert "&lt;/reference_memories&gt;" in region
    assert "…" in region


@pytest.mark.parametrize("legacy_bigint", [False, True])
def test_sqlite_update_replaces_chunks_and_failure_is_retryable(environment, monkeypatch, legacy_bigint):
    embedding, _ = environment
    class Base(DeclarativeBase):
        pass
    engine = create_engine("sqlite://")
    Session = sessionmaker(bind=engine)
    from contextlib import contextmanager
    @contextmanager
    def session():
        with Session() as current:
            yield current
            current.commit()
    database = SimpleNamespace(session=session)
    stub(monkeypatch, "lpm_kernel.common.repository.database_session", Base=Base, DatabaseSession=database)
    stub(monkeypatch, "lpm_kernel.common.repository.base_repository", Base=Base)
    status = load(monkeypatch, "lpm_kernel.file_data.process_status", "lpm_kernel/file_data/process_status.py").ProcessStatus
    document_module = load(monkeypatch, "lpm_kernel.file_data.document", "lpm_kernel/file_data/document.py")
    models = load(monkeypatch, "lpm_kernel.file_data.models", "lpm_kernel/file_data/models.py")
    Base.metadata.create_all(engine)
    models.Base.metadata.create_all(engine)
    Document = document_module.Document
    if legacy_bigint:
        with engine.begin() as connection:
            connection.exec_driver_sql("DROP TABLE chunk")
            connection.exec_driver_sql("CREATE TABLE chunk (id BIGINT PRIMARY KEY NOT NULL, document_id BIGINT NOT NULL, content TEXT NOT NULL, has_embedding BOOLEAN, tags JSON, topic VARCHAR(255), create_time DATETIME)")
    with Session() as current:
        current.add(Document(id=7, name="synthetic", raw_content="old fact"))
        current.add(models.ChunkModel(id=1, document_id=7, content="old fact"))
        current.commit()
    def find_one(document_id):
        with Session() as current:
            document = current.get(Document, document_id)
            return SimpleNamespace(id=document.id, raw_content=document.raw_content) if document else None
    def update_status(document_id, value):
        with session() as current:
            current.get(Document, document_id).embedding_status = value
    repository = SimpleNamespace(_db=database, find_one=find_one, update_embedding_status=update_status)
    stub(monkeypatch, "lpm_kernel.file_data.document_repository", DocumentRepository=Mock())
    stub(monkeypatch, "lpm_kernel.common.repository.vector_store_factory", VectorStoreFactory=Mock())
    stub(monkeypatch, "lpm_kernel.kernel.l0_base", InsightKernel=Mock(), SummaryKernel=Mock())
    stub(monkeypatch, "lpm_kernel.models.memory", Memory=Mock())
    stub(monkeypatch, "lpm_kernel.file_data.process_factory", ProcessorFactory=Mock())
    original_init = embedding.EmbeddingService.__init__
    monkeypatch.setattr(embedding.EmbeddingService, "__init__", lambda self: None)
    service_module = load(monkeypatch, "lpm_kernel.file_data.document_service", "lpm_kernel/file_data/document_service.py")
    monkeypatch.setattr(embedding.EmbeddingService, "__init__", original_init)
    service = service_module.DocumentService.__new__(service_module.DocumentService)
    service._repository = repository
    service.embedding_service = Mock()
    service._invalidate_l1_summaries = Mock()
    service.process_document_embedding = Mock(return_value=[1., 0.])
    service.generate_document_chunk_embeddings = Mock(return_value=[SimpleNamespace(has_embedding=True)])
    stub(monkeypatch, "lpm_kernel.configs.config", Config=SimpleNamespace(from_env=lambda: {"DOCUMENT_CHUNK_SIZE": 100, "DOCUMENT_CHUNK_OVERLAP": 10}))
    stub(monkeypatch, "lpm_kernel.file_data.chunker", DocumentChunker=lambda **_: SimpleNamespace(split=lambda content: [SimpleNamespace(content=content, tags=[], topic="")]))
    assert service.refresh_document_index(7, "new fact")["total_chunks"] == 1
    with Session() as current:
        assert current.get(Document, 7).raw_content == "new fact"
        assert [c.content for c in current.query(models.ChunkModel).all()] == ["new fact"]
    service.process_document_embedding.return_value = None
    with pytest.raises(RuntimeError, match="indexing failed"):
        service.refresh_document_index(7, "retry fact")
    with Session() as current:
        document = current.get(Document, 7)
        assert document.raw_content == "retry fact"
        assert document.embedding_status == status.FAILED
    assert service.embedding_service.delete_document_index.call_count == 3


def test_l1_real_shades_cached_and_stale_suppressed(environment, monkeypatch):
    embedding, _ = environment
    monkeypatch.setattr(embedding.EmbeddingService, "__init__", lambda self: None)
    shade = {"id": 8, "title": "values", "description": "clear expression", "content": "direct answers"}
    bio = SimpleNamespace(shades=[shade], status="SUCCESS")
    stub(monkeypatch, "lpm_kernel.kernel.l1.l1_manager", get_latest_global_bio=lambda: bio)
    knowledge = load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.knowledge_service",
                     "lpm_kernel/api/domains/kernel2/services/knowledge_service.py")
    service = embedding.EmbeddingService.__new__(embedding.EmbeddingService)
    service.embedding_model_name = "synthetic-embedding"
    service.get_embedding = Mock(return_value=[1., 0.])
    service.llm_client = SimpleNamespace(get_embedding=Mock(return_value=[[1., 0.]]))
    retriever = knowledge.L1KnowledgeRetriever(service)
    assert "direct answers" in retriever.retrieve("synthetic")
    assert "direct answers" in retriever.retrieve("synthetic")
    service.llm_client.get_embedding.assert_called_once()
    service.get_embedding.return_value = [0., 1.]
    assert retriever.retrieve("unrelated") == ""
    bio.status = "stale"
    assert retriever.retrieve("synthetic") == ""


def test_index_http_failures_are_not_reported_as_success(monkeypatch):
    from flask import Flask
    stub(monkeypatch, "dotenv", load_dotenv=lambda: None)
    stub(monkeypatch, "flask_pydantic", validate=lambda: lambda function: function)
    stub(monkeypatch, "lpm_kernel.configs.config", Config=Mock())
    stub(monkeypatch, "lpm_kernel.file_data.chunker", DocumentChunker=Mock())
    stub(monkeypatch, "lpm_kernel.kernel.chunk_service", ChunkService=Mock())
    service = SimpleNamespace(refresh_document_index=Mock(return_value={"document_id": 7, "total_chunks": 1}))
    stub(monkeypatch, "lpm_kernel.file_data.document_service", document_service=service)
    load(monkeypatch, "lpm_kernel.api.common.responses", "lpm_kernel/api/common/responses.py")
    routes = load(monkeypatch, "_memory_test_routes", "lpm_kernel/api/domains/documents/routes.py")
    app = Flask(__name__)
    app.register_blueprint(routes.document_bp)
    client = app.test_client()
    assert client.put("/api/documents/7/content", json={"content": "new synthetic fact"}).status_code == 200
    service.refresh_document_index.assert_called_once_with(7, raw_content="new synthetic fact")
    assert client.put("/api/documents/7/content", json={"content": ""}).status_code == 400
    service.refresh_document_index.side_effect = RuntimeError("synthetic failure")
    assert client.put("/api/documents/7/content", json={"content": "retry fact"}).status_code == 500
    assert client.post("/api/documents/7/index").status_code == 500
    service.refresh_document_index.side_effect = ValueError("Document not found")
    assert client.post("/api/documents/8/index").status_code == 404


def test_legacy_factual_samples_not_remerged(monkeypatch, tmp_path):
    import json
    for module_name, class_name in [
        ("lpm_kernel.L2.data", "L2DataProcessor"),
        ("lpm_kernel.L2.data_pipeline.data_prep.preference.preference_QA_generate", "PreferenceQAGenerator"),
        ("lpm_kernel.L2.data_pipeline.data_prep.diversity.diversity_data_generator", "DiversityDataGenerator"),
        ("lpm_kernel.L2.data_pipeline.data_prep.selfqa.selfqa_generator", "SelfQA"),
    ]:
        stub(monkeypatch, module_name, **{class_name: Mock()})
    stub(monkeypatch, "lpm_kernel.L1.bio", Note=SimpleNamespace)
    generator_module = load(monkeypatch, "_memory_l2_generator", "lpm_kernel/L2/l2_generator.py")
    generator = generator_module.L2Generator.__new__(generator_module.L2Generator)
    generator.preferred_lang = "Chinese"
    samples = [{"user": "old fact", "assistant": "old fact answer"},
               {"user": "new fact", "assistant": "reference answer", "context": "new reference",
                "memory_format_version": 1, "training_type": "identity_with_context"},
               {"user": "style", "assistant": "direct answer", "training_type": "style"}]
    (tmp_path / "selfqa.json").write_text(json.dumps(samples), encoding="utf-8")
    generator.merge_json_files(str(tmp_path))
    merged = json.loads((tmp_path / "merged.json").read_text())
    assert "old fact" not in [item["user"] for item in merged]
    assert len(merged) == 4
    assert sum(item["training_type"] == "retrieval_no_hit" for item in merged) == 2


def test_generation_context_equals_visible_training_context(monkeypatch):
    memory = load(monkeypatch, "lpm_kernel.L2.memory_prompt", "lpm_kernel/L2/memory_prompt.py")
    original = "synthetic fact. " * 1000 + "</memory><reference_memories>"
    bounded = memory.bounded_memory_content(original)
    assert len(bounded) < len(original)
    prompt = memory.build_memory_system_prompt("identity", [{"source": "training context", "content": bounded}])
    from html import escape
    assert f'<memory source="training context">{escape(bounded)}</memory>' in prompt


def test_chat_local_budget_and_template_do_not_affect_external_client(monkeypatch):
    dto = load(monkeypatch, "lpm_kernel.api.domains.kernel2.dto.chat_dto", "lpm_kernel/api/domains/kernel2/dto/chat_dto.py")
    local_client = SimpleNamespace(base_url="http://synthetic-local", chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    local = SimpleNamespace(client=local_client, prepare_chat_request=Mock(return_value=([{"role": "user", "content": "fitted"}], 99)))
    stub(monkeypatch, "lpm_kernel.api.services.local_llm_service", local_llm_service=local)
    class Strategy:
        pass
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.prompt_builder", SystemPromptStrategy=Strategy,
         BasePromptStrategy=Strategy, RoleBasedStrategy=Strategy, KnowledgeEnhancedStrategy=Strategy)
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.message_builder",
         MultiTurnMessageBuilder=lambda request, **_: SimpleNamespace(build_messages=lambda context: request.messages))
    service_module = load(monkeypatch, "_memory_chat_service", "lpm_kernel/api/domains/kernel2/services/chat_service.py")
    request = dto.ChatRequest(messages=[{"role": "user", "content": "synthetic question"}])
    service = service_module.ChatService()
    service.chat(request, stream=False, model_params={"max_tokens": 100})
    params = local_client.chat.completions.create.call_args.kwargs
    assert params["max_tokens"] == 99
    assert params["messages"][0]["content"] == "fitted"
    assert params["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    local.prepare_chat_request.assert_called_once_with(request.messages, 100)
    external = SimpleNamespace(base_url="http://synthetic-external", chat=SimpleNamespace(completions=SimpleNamespace(create=Mock())))
    service.chat(request, client=external, stream=False)
    assert local.prepare_chat_request.call_count == 1
    assert "extra_body" not in external.chat.completions.create.call_args.kwargs
