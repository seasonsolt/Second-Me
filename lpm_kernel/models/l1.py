from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey
from sqlalchemy.orm import relationship
from lpm_kernel.common.repository.database_session import Base
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, TYPE_CHECKING
from lpm_kernel.L1.bio import Bio

if TYPE_CHECKING:
    from lpm_kernel.models.status_biography import StatusBiography


class L1Version(Base):
    __tablename__ = "l1_versions"

    version = Column(Integer, primary_key=True)
    create_time = Column(DateTime, nullable=False, default=datetime.now)
    status = Column(String(50), nullable=False)
    description = Column(String(500))

    # add relationship
    bios = relationship("L1Bio", back_populates="version_info")
    shades = relationship("L1Shade", back_populates="version_info")
    clusters = relationship("L1Cluster", back_populates="version_info")
    chunk_topics = relationship("L1ChunkTopic", back_populates="version_info")


class L1Bio(Base):
    __tablename__ = "l1_bios"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=False)
    content = Column(String(10000))
    content_third_view = Column(String(10000))
    summary = Column(String(2000))
    summary_third_view = Column(String(2000))
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    # add relationship
    version_info = relationship("L1Version", back_populates="bios")


class L1Shade(Base):
    __tablename__ = "l1_shades"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=False)
    name = Column(String(200))
    aspect = Column(String(200))
    icon = Column(String(100))
    desc_third_view = Column(String(1000))
    content_third_view = Column(String(2000))
    desc_second_view = Column(String(1000))
    content_second_view = Column(String(2000))
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    # add relationship
    version_info = relationship("L1Version", back_populates="shades")


class L1Cluster(Base):
    __tablename__ = "l1_clusters"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=False)
    cluster_id = Column(String(100))
    memory_ids = Column(JSON)
    cluster_center = Column(JSON)
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    # add relationship
    version_info = relationship("L1Version", back_populates="clusters")


class L1ChunkTopic(Base):
    __tablename__ = "l1_chunk_topics"

    id = Column(Integer, primary_key=True)
    version = Column(Integer, ForeignKey("l1_versions.version"), nullable=False)
    chunk_id = Column(String(100))
    topic = Column(String(500))
    tags = Column(JSON)
    create_time = Column(DateTime, nullable=False, default=datetime.now)

    # add relationship
    version_info = relationship("L1Version", back_populates="chunk_topics")


class L1VersionDTO:
    def __init__(
        self, version: int, status: str, description: str, create_time: datetime
    ):
        self.version = version
        self.status = status
        self.description = description
        self.create_time = create_time

    @classmethod
    def from_model(cls, model: "L1Version") -> "L1VersionDTO":
        return cls(
            version=model.version,
            status=model.status,
            description=model.description,
            create_time=model.create_time,
        )


@dataclass
class L1GenerationResult:
    """L1 generation result data class"""

    bio: Bio
    clusters: Dict[str, List]  # {"clusterList": [...]}
    chunk_topics: Dict[str, Dict]  # {cluster_id: {"indices": [], "docIds": [], ...}}
    generate_time: datetime = datetime.now()

    def to_dict(self) -> dict:
        """Convert to dictionary format"""
        return {
            "bio": self.bio,
            "clusters": self.clusters,
            "chunk_topics": self.chunk_topics,
            "generate_time": self.generate_time.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "L1GenerationResult":
        """Create instance from dictionary"""
        return cls(
            bio=data.get("bio"),
            clusters=data.get("clusters", {"clusterList": []}),
            chunk_topics=data.get("chunk_topics", {}),
            generate_time=datetime.fromisoformat(data["generate_time"])
            if "generate_time" in data
            else datetime.now(),
        )


@dataclass
class GlobalBioDTO:
    """Global biography data transfer object"""

    content: str
    content_third_view: str
    summary: str
    summary_third_view: str
    create_time: datetime
    shades: List[Dict] = None  # add shades field
    status: str = ""

    @classmethod
    def from_model(cls, model: "L1Bio", shades=None, status="") -> "GlobalBioDTO":
        """
        Create DTO from database model

        Args:
            model (L1Bio): database model object

        Returns:
            GlobalBioDTO: data transfer object
        """
        return cls(
            content=model.content,
            content_third_view=model.content_third_view,
            summary=model.summary,
            summary_third_view=model.summary_third_view,
            create_time=model.create_time,
            status=status,
            shades=[{
                "id": shade.id, "title": shade.name or "",
                "description": shade.desc_second_view or shade.desc_third_view or "",
                "content": shade.content_second_view or shade.content_third_view or "",
            } for shade in (shades or [])],
        )

    def to_dict(self) -> dict:
        """
        Convert to dictionary format

        Returns:
            dict: dictionary format data
        """
        return {
            "content": self.content,
            "content_third_view": self.content_third_view,
            "summary": self.summary,
            "summary_third_view": self.summary_third_view,
            "create_time": self.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            "shades": self.shades or [],  # ensure return list
        }


@dataclass
class StatusBioDTO:
    """Status biography data transfer object"""

    content: str
    content_third_view: str
    summary: str
    summary_third_view: str
    create_time: datetime
    update_time: datetime

    @classmethod
    def from_model(cls, model: "StatusBiography") -> "StatusBioDTO":
        """Create DTO from database model

        Args:
            model (StatusBiography): database model object

        Returns:
            StatusBioDTO: data transfer object
        """
        return cls(
            content=model.content,
            content_third_view=model.content_third_view,
            summary=model.summary,
            summary_third_view=model.summary_third_view,
            create_time=model.create_time,
            update_time=model.update_time,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary format

        Returns:
            dict: dictionary format data
        """
        return {
            "content": self.content,
            "content_third_view": self.content_third_view,
            "summary": self.summary,
            "summary_third_view": self.summary_third_view,
            "create_time": self.create_time.strftime("%Y-%m-%d %H:%M:%S"),
            "update_time": self.update_time.strftime("%Y-%m-%d %H:%M:%S"),
        }
