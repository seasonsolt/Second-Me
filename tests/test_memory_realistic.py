"""Realistic offline memory cases: production prompt path, mocked embeddings, no network or .env."""
import importlib.util
import logging
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ROOT = Path(__file__).resolve().parents[1]
FACT = "我的下一次牙医预约是十一月十二日上午九点半，在仁和口腔诊所。"


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


def realistic_chunk(prefix, fact):
    filler = f"{prefix}这是一段关于日常生活安排、工作计划和个人习惯的记录，内容较长且包含许多细节。"
    body = (filler * 40)[:1000 - len(fact)]
    return body + fact


@pytest.fixture
def prompt_stack(monkeypatch):
    monkeypatch.delenv("L0_SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("L1_SIMILARITY_THRESHOLD", raising=False)
    configuration = SimpleNamespace(embedding_model_name="@cf/baai/bge-m3")
    hits = []

    class FakeEmbeddingService:
        def __init__(self):
            self.llm_client = SimpleNamespace(
                user_llm_config_service=SimpleNamespace(get_available_llm=lambda: configuration))

        def search_similar_chunks(self, query, limit=5):
            return hits[:limit]

    stub(monkeypatch, "lpm_kernel.file_data.embedding_service",
         EmbeddingService=FakeEmbeddingService, ChunkDTO=SimpleNamespace)
    stub(monkeypatch, "lpm_kernel.kernel.l1.l1_manager", get_latest_global_bio=lambda: None)
    knowledge = load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.knowledge_service",
                     "lpm_kernel/api/domains/kernel2/services/knowledge_service.py")
    dto = load(monkeypatch, "lpm_kernel.api.domains.kernel2.dto.chat_dto",
               "lpm_kernel/api/domains/kernel2/dto/chat_dto.py")
    load(monkeypatch, "lpm_kernel.L2.memory_prompt", "lpm_kernel/L2/memory_prompt.py")
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.role_service",
         role_service=SimpleNamespace(get_role_by_uuid=lambda _: None))
    stub(monkeypatch, "lpm_kernel.L2.training_prompt", CONTEXT_PROMPT="", JUDGE_PROMPT="")
    load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.prompt_builder",
         "lpm_kernel/api/domains/kernel2/services/prompt_builder.py")
    builders = load(monkeypatch, "lpm_kernel.api.domains.kernel2.services.message_builder",
                    "lpm_kernel/api/domains/kernel2/services/message_builder.py")

    def system_prompt(question):
        request = dto.ChatRequest(messages=[{"role": "system", "content": "你是用户的第二个我。"},
                                            {"role": "user", "content": question}])
        return builders.MultiTurnMessageBuilder(request).build_messages()[0]["content"]

    return SimpleNamespace(knowledge=knowledge, configuration=configuration, hits=hits,
                           system_prompt=system_prompt)


def test_bge_m3_relevant_chunk_reaches_system_prompt(prompt_stack):
    chunks = [realistic_chunk("甲", FACT), realistic_chunk("乙", "另一件事：周末计划去爬山。"),
              realistic_chunk("丙", "无关内容：喜欢喝绿茶。")]
    assert all(len(chunk) == 1000 for chunk in chunks)
    prompt_stack.hits.extend([(SimpleNamespace(id=1, document_id=7, content=chunks[0]), 0.55),
                              (SimpleNamespace(id=2, document_id=7, content=chunks[1]), 0.52),
                              (SimpleNamespace(id=3, document_id=8, content=chunks[2]), 0.41)])
    prompt = prompt_stack.system_prompt("我下一次牙医预约是什么时候？")
    assert FACT in prompt
    assert chunks[0] in prompt and chunks[1] in prompt
    assert chunks[2] not in prompt
    assert "…" not in prompt


def test_openai_model_keeps_default_threshold(prompt_stack, monkeypatch):
    knowledge = prompt_stack.knowledge
    assert knowledge.default_similarity_threshold("text-embedding-3-small") == 0.7
    assert knowledge.default_similarity_threshold("@cf/baai/bge-m3") == 0.5
    prompt_stack.configuration.embedding_model_name = "text-embedding-3-small"
    prompt_stack.hits.append((SimpleNamespace(id=1, document_id=7, content=realistic_chunk("甲", FACT)), 0.55))
    prompt = prompt_stack.system_prompt("我下一次牙医预约是什么时候？")
    assert FACT not in prompt
    assert "No relevant memories found." in prompt
    prompt_stack.configuration.embedding_model_name = "@cf/baai/bge-m3"
    monkeypatch.setenv("L0_SIMILARITY_THRESHOLD", "0.6")
    assert FACT not in prompt_stack.system_prompt("我下一次牙医预约是什么时候？")


def test_space_strategy_chain_is_accepted(monkeypatch):
    dto = load(monkeypatch, "lpm_kernel.api.domains.kernel2.dto.chat_dto",
               "lpm_kernel/api/domains/kernel2/dto/chat_dto.py")
    stub(monkeypatch, "lpm_kernel.api.services.local_llm_service", local_llm_service=SimpleNamespace(client=None))

    class SystemPromptStrategy:
        pass

    class BasePromptStrategy(SystemPromptStrategy):
        pass

    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.prompt_builder",
         SystemPromptStrategy=SystemPromptStrategy, BasePromptStrategy=BasePromptStrategy,
         RoleBasedStrategy=SystemPromptStrategy, KnowledgeEnhancedStrategy=SystemPromptStrategy)
    stub(monkeypatch, "lpm_kernel.api.domains.kernel2.services.message_builder", MultiTurnMessageBuilder=Mock())
    service_module = load(monkeypatch, "_realistic_chat_service",
                          "lpm_kernel/api/domains/kernel2/services/chat_service.py")

    class HostSummaryStrategy(SystemPromptStrategy):
        """Mirrors SpaceBaseStrategy: a root strategy outside BasePromptStrategy."""

    service = service_module.ChatService()
    request = dto.ChatRequest(messages=[{"role": "user", "content": "Please summarize this discussion"}])
    assert service._get_strategy_chain(request, [HostSummaryStrategy]) == [HostSummaryStrategy]
    chain = [BasePromptStrategy, SystemPromptStrategy, HostSummaryStrategy]
    assert service._get_strategy_chain(request, chain) == chain
    with pytest.raises(ValueError):
        service._get_strategy_chain(request, [])
    with pytest.raises(ValueError):
        service._get_strategy_chain(request, [str])


def test_document_update_keeps_status_biography(monkeypatch):
    class Base(DeclarativeBase):
        pass
    engine = create_engine("sqlite://")
    Session = sessionmaker(bind=engine)

    @contextmanager
    def session():
        with Session() as current:
            yield current
            current.commit()

    logger = logging.getLogger("memory-realistic")
    stub(monkeypatch, "lpm_kernel.configs.logging", get_train_process_logger=lambda: logger)
    stub(monkeypatch, "lpm_kernel.common.repository.database_session", Base=Base,
         DatabaseSession=SimpleNamespace(session=session))
    package = stub(monkeypatch, "lpm_kernel.file_data")
    package.__path__ = [str(ROOT / "lpm_kernel/file_data")]
    for name, attributes in [
        ("lpm_kernel.common.repository.vector_store_factory", {"VectorStoreFactory": Mock()}),
        ("lpm_kernel.file_data.document_dto", {"DocumentDTO": SimpleNamespace, "CreateDocumentRequest": SimpleNamespace}),
        ("lpm_kernel.kernel.l0_base", {"InsightKernel": Mock(), "SummaryKernel": Mock()}),
        ("lpm_kernel.models.memory", {"Memory": Mock()}),
        ("lpm_kernel.file_data.document", {"Document": Mock()}),
        ("lpm_kernel.file_data.document_repository", {"DocumentRepository": Mock()}),
        ("lpm_kernel.file_data.embedding_service", {"EmbeddingService": Mock()}),
        ("lpm_kernel.file_data.process_factory", {"ProcessorFactory": Mock()}),
        ("lpm_kernel.file_data.models", {"ChunkModel": Mock()}),
        ("lpm_kernel.file_data.dto", {}),
        ("lpm_kernel.file_data.dto.chunk_dto", {"ChunkDTO": SimpleNamespace}),
        ("lpm_kernel.configs.config", {"Config": SimpleNamespace(from_env=lambda: {})}),
        ("lpm_kernel.file_data.chunker", {"DocumentChunker": lambda **_: SimpleNamespace(
            split=lambda content: [SimpleNamespace(content=content, tags=[], topic="")])}),
    ]:
        stub(monkeypatch, name, **attributes)
    load(monkeypatch, "lpm_kernel.L1.bio", "lpm_kernel/L1/bio.py")
    l1 = load(monkeypatch, "lpm_kernel.models.l1", "lpm_kernel/models/l1.py")
    bios = load(monkeypatch, "lpm_kernel.models.status_biography", "lpm_kernel/models/status_biography.py")
    Base.metadata.create_all(engine)
    with Session() as current:
        current.add(l1.L1Version(version=1, status="SUCCESS"))
        current.add(bios.StatusBiography(content="近况", content_third_view="近况", summary="摘要",
                                         summary_third_view="摘要"))
        current.commit()
    load(monkeypatch, "lpm_kernel.file_data.process_status", "lpm_kernel/file_data/process_status.py")
    service_module = load(monkeypatch, "lpm_kernel.file_data.document_service", "lpm_kernel/file_data/document_service.py")
    service = service_module.DocumentService.__new__(service_module.DocumentService)
    document_session = MagicMock()
    document_session.query.return_value.scalar.return_value = 0

    @contextmanager
    def document_db_session():
        yield document_session

    service._repository = SimpleNamespace(find_one=lambda _: SimpleNamespace(id=7, raw_content="旧事实"),
                                          _db=SimpleNamespace(session=document_db_session))
    service.embedding_service = Mock()
    service.process_document_embedding = Mock(return_value=[1., 0.])
    service.generate_document_chunk_embeddings = Mock(return_value=[SimpleNamespace(has_embedding=True)])
    assert service.refresh_document_index(7, FACT)["total_chunks"] == 1
    with Session() as current:
        assert current.get(l1.L1Version, 1).status == "stale"
        assert current.query(bios.StatusBiography).count() == 1


def test_reindex_all_documents_reports_counts(monkeypatch):
    from flask import Flask
    stub(monkeypatch, "dotenv", load_dotenv=lambda: None)
    stub(monkeypatch, "flask_pydantic", validate=lambda: lambda function: function)
    stub(monkeypatch, "lpm_kernel.configs.config", Config=Mock())
    stub(monkeypatch, "lpm_kernel.file_data.chunker", DocumentChunker=Mock())
    stub(monkeypatch, "lpm_kernel.kernel.chunk_service", ChunkService=Mock())
    service = SimpleNamespace(list_documents=lambda: [SimpleNamespace(id=1), SimpleNamespace(id=2)],
                              refresh_document_index=Mock(return_value={"total_chunks": 1}))
    stub(monkeypatch, "lpm_kernel.file_data.document_service", document_service=service)
    load(monkeypatch, "lpm_kernel.api.common.responses", "lpm_kernel/api/common/responses.py")
    routes = load(monkeypatch, "_realistic_document_routes", "lpm_kernel/api/domains/documents/routes.py")
    app = Flask(__name__)
    app.register_blueprint(routes.document_bp)
    client = app.test_client()
    response = client.post("/api/documents/reindex")
    assert response.status_code == 200
    assert response.json["data"] == {"total": 2, "succeeded": 2, "failed": 0, "failed_document_ids": []}
    service.refresh_document_index.side_effect = [None, RuntimeError("synthetic failure")]
    response = client.post("/api/documents/reindex")
    assert response.status_code == 500
    assert response.json["data"]["failed_document_ids"] == [2]
