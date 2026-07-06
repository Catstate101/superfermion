"""
Model Registry — Publish, version, and load trained quantum models.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RegisteredModel:
    """A registered model in the registry.

    Attributes:
        name: Model name.
        version: Model version.
        description: Model description.
        metrics: Final training metrics.
        checkpoint_path: Path to the .sfm checkpoint.
        tags: Model tags.
        created_at: Registration timestamp.
        created_by: Creator identifier.
    """
    name: str
    version: str = "1.0"
    description: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "metrics": self.metrics,
            "checkpoint_path": self.checkpoint_path,
            "tags": self.tags,
            "created_at": self.created_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RegisteredModel:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def __repr__(self) -> str:
        return f"RegisteredModel('{self.name}', v{self.version})"


class ModelRegistry:
    """Local model registry for publishing and discovering models.

    Args:
        registry_dir: Directory for model storage.

    Examples:
        >>> registry = ModelRegistry()
        >>> registry.register("my_vqe", version="1.0", metrics={"energy": -1.14})
        >>> model = registry.get("my_vqe")
        >>> models = registry.list_models()
    """

    def __init__(self, registry_dir: Optional[str] = None) -> None:
        self._models: Dict[str, List[RegisteredModel]] = {}
        self._registry_dir = Path(registry_dir) if registry_dir else None
        if self._registry_dir:
            self._registry_dir.mkdir(parents=True, exist_ok=True)
            self._load_index()

    def _load_index(self) -> None:
        """Load registry index from disk."""
        if not self._registry_dir:
            return
        index_path = self._registry_dir / "index.json"
        if index_path.exists():
            try:
                with open(index_path, "r") as f:
                    data = json.load(f)
                for name, versions in data.items():
                    self._models[name] = [RegisteredModel.from_dict(v) for v in versions]
            except Exception:
                pass

    def _save_index(self) -> None:
        """Save registry index to disk."""
        if not self._registry_dir:
            return
        index_path = self._registry_dir / "index.json"
        data = {
            name: [m.to_dict() for m in versions]
            for name, versions in self._models.items()
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(
        self,
        name: str,
        version: str = "1.0",
        description: str = "",
        metrics: Optional[Dict[str, float]] = None,
        checkpoint_path: str = "",
        tags: Optional[Dict[str, str]] = None,
        created_by: str = "",
    ) -> RegisteredModel:
        """Register a new model.

        Args:
            name: Model name.
            version: Version string.
            description: Model description.
            metrics: Training metrics.
            checkpoint_path: Path to .sfm file.
            tags: Model tags.
            created_by: Creator.

        Returns:
            The registered model.
        """
        model = RegisteredModel(
            name=name,
            version=version,
            description=description,
            metrics=metrics or {},
            checkpoint_path=checkpoint_path,
            tags=tags or {},
            created_by=created_by,
        )

        if name not in self._models:
            self._models[name] = []
        self._models[name].append(model)
        self._save_index()
        return model

    def get(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[RegisteredModel]:
        """Get a registered model.

        Args:
            name: Model name.
            version: Specific version. None returns latest.

        Returns:
            RegisteredModel or None.
        """
        if name not in self._models:
            return None

        versions = self._models[name]
        if not versions:
            return None

        if version:
            for m in versions:
                if m.version == version:
                    return m
            return None

        return versions[-1]  # Latest

    def list_models(self) -> List[RegisteredModel]:
        """List all registered models (latest versions)."""
        return [versions[-1] for versions in self._models.values() if versions]

    def list_versions(self, name: str) -> List[str]:
        """List all versions of a model."""
        return [m.version for m in self._models.get(name, [])]

    def delete(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a model or specific version."""
        if name not in self._models:
            return False
        if version:
            self._models[name] = [m for m in self._models[name] if m.version != version]
        else:
            del self._models[name]
        self._save_index()
        return True

    def search(self, query: str) -> List[RegisteredModel]:
        """Search models by name substring."""
        results = []
        for versions in self._models.values():
            for m in versions:
                if query.lower() in m.name.lower() or query.lower() in m.description.lower():
                    results.append(m)
        return results

    @property
    def n_models(self) -> int:
        return len(self._models)

    def __repr__(self) -> str:
        return f"ModelRegistry(models={self.n_models})"
