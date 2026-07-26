"""YAML configuration loader for GeoPulse pipelines.

Uses Python :mod:`dataclasses` for the schema — deliberately NO pydantic /
hydra / dynaconf, to keep the core dependency list minimal (see Section 14 of
the handoff spec).

The top-level document must have a single ``geopulse:`` key. Example::

    geopulse:
      source:
        type: synthetic
        waveform: gaussian_pulse
        amplitude_nT: 500.0
        duration_s: 3600.0
        dt_s: 1.0
      earth:
        type: layered_1d
        model: quebec_7layer
      network:
        type: powergrid
        file: benchmarks/horton2012/network.json
      solver:
        type: lpm
      output:
        file: results.hdf5
        format: hdf5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from geopulse.exceptions import ConfigurationError

__all__ = [
    "SourceConfig",
    "EarthConfig",
    "NetworkConfig",
    "SolverConfig",
    "OutputConfig",
    "GeoPulseConfig",
    "load_config",
]


@dataclass(frozen=True)
class SourceConfig:
    """B-field source section."""

    type: str
    waveform: Optional[str] = None
    amplitude_nT: float = 0.0
    duration_s: float = 0.0
    dt_s: float = 1.0
    file: Optional[str] = None
    station_id: str = "SYN"


@dataclass(frozen=True)
class EarthConfig:
    """Earth-model section."""

    type: str
    model: Optional[str] = None
    file: Optional[str] = None


@dataclass(frozen=True)
class NetworkConfig:
    """Network section."""

    type: str
    file: Optional[str] = None


@dataclass(frozen=True)
class SolverConfig:
    """Solver section."""

    type: str = "lpm"
    options: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputConfig:
    """Output section."""

    file: str = "results.hdf5"
    format: str = "hdf5"


@dataclass(frozen=True)
class GeoPulseConfig:
    """Top-level GeoPulse configuration."""

    source: SourceConfig
    earth: EarthConfig
    network: NetworkConfig
    solver: SolverConfig
    output: OutputConfig

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeoPulseConfig":
        """Build a :class:`GeoPulseConfig` from a plain nested dict.

        Parameters
        ----------
        data : dict
            The parsed YAML content (either the root or its ``geopulse:``
            child, both accepted).

        Raises
        ------
        ConfigurationError
            If required top-level sections are missing.
        """
        root = data.get("geopulse", data)
        required = ("source", "earth", "network")
        missing = [k for k in required if k not in root]
        if missing:
            raise ConfigurationError(f"Config is missing required sections: {missing}")
        return cls(
            source=SourceConfig(**root["source"]),
            earth=EarthConfig(**root["earth"]),
            network=NetworkConfig(**root["network"]),
            solver=SolverConfig(**root.get("solver", {"type": "lpm"})),
            output=OutputConfig(**root.get("output", {})),
        )


def load_config(path: str | Path) -> GeoPulseConfig:
    """Load a YAML pipeline config from disk.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a ``.yaml`` / ``.yml`` file.

    Returns
    -------
    GeoPulseConfig
        Parsed and validated config.

    Raises
    ------
    ConfigurationError
        If the file does not exist or fails validation.
    """
    p = Path(path)
    if not p.is_file():
        raise ConfigurationError(f"Config file not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except yaml.YAMLError as e:
        raise ConfigurationError(f"YAML parse error in {p}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config root must be a mapping, got {type(data)}")
    return GeoPulseConfig.from_dict(data)
