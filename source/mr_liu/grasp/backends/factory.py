"""Configuration-driven construction of replaceable grasp backends."""

from __future__ import annotations

from typing import Any, Mapping

from mr_liu.grasp.backends.geometric import GeometricAntipodalBackend
from mr_liu.grasp.backends.graspgenx import (
    FallbackGraspBackend,
    GraspGenXBackend,
    GraspGenXConfig,
    GraspGenXTransport,
)


def create_grasp_backend(
    config: Mapping[str, Any],
    *,
    graspgenx_transport: GraspGenXTransport | None = None,
):
    """Build the selected backend while leaving the BusAgent node API stable."""

    name = str(config.get("backend", "geometric")).strip().lower()
    if name == "geometric":
        return GeometricAntipodalBackend()
    if name != "graspgenx":
        raise ValueError(f"Unknown fine-grasp backend {name!r}")

    section = config.get("backends", {}).get("graspgenx", {})
    if not isinstance(section, Mapping):
        raise ValueError("backends.graspgenx must be a mapping")
    primary = GraspGenXBackend(
        GraspGenXConfig.from_mapping(section),
        transport=graspgenx_transport,
    )
    fallback_name = section.get("fallback_backend", "geometric")
    if fallback_name in (None, False, "none", "disabled"):
        return primary
    if str(fallback_name).lower() != "geometric":
        raise ValueError("GraspGenX fallback_backend must be geometric or none")
    return FallbackGraspBackend(
        primary,
        GeometricAntipodalBackend(),
        fallback_on_empty=bool(section.get("fallback_on_empty", True)),
    )
