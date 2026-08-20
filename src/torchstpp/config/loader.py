"""Read user configuration without constructing scientific objects."""

from __future__ import annotations

import copy
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TorchSTPPConfig:
    """A loaded TOML document with paths anchored to the TOML directory."""

    data: Mapping[str, Any]
    path: Path

    @property
    def base_dir(self) -> Path:
        return self.path.parent

    def section(self, name: str, *, required: bool = False) -> dict[str, Any]:
        value = self.data.get(name)
        if value is None:
            if required:
                raise ValueError(f"configuration is missing [{name}]")
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"[{name}] must be a TOML table")
        return copy.deepcopy(dict(value))

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.base_dir / path


def load_config(path: str | Path) -> TorchSTPPConfig:
    """Load a TOML file once; individual modules never read files themselves."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict):
        raise TypeError("the TOML root must be a table")
    return TorchSTPPConfig(data=data, path=config_path)
