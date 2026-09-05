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
from mr_liu.grasp.backends.m2t2 import M2T2Backend, M2T2Config


def create_grasp_backend(
    config: Mapping[str, Any],
    *,
    graspgenx_transport: GraspGenXTransport | None = None,
    m2t2_transport=None,
):
    """Build the selected backend while leaving the BusAgent node API stable."""

    name = str(config.get("backend", "graspgenx")).strip().lower()
    if name == "geometric":
        return GeometricAntipodalBackend()
    if name == "m2t2":
        section = config.get("backends", {}).get("m2t2", {})
        if not isinstance(section, Mapping):
            raise ValueError("backends.m2t2 must be a mapping")
        primary = M2T2Backend(M2T2Config.from_mapping(section), transport=m2t2_transport)
        fallback_name = str(section.get("fallback_backend", "none")).strip().lower()
        if fallback_name in {"", "none", "disabled"}:
            return primary
        if fallback_name == "geometric":
            fallback = GeometricAntipodalBackend()
        elif fallback_name == "graspgenx":
            gx = config.get("backends", {}).get("graspgenx", {})
            if not isinstance(gx, Mapping):
                raise ValueError("backends.graspgenx must be a mapping")
            fallback = GraspGenXBackend(
                GraspGenXConfig.from_mapping(gx), transport=graspgenx_transport
            )
        else:
            raise ValueError("M2T2 fallback_backend must be geometric, graspgenx or none")
        return FallbackGraspBackend(
            primary,
            fallback,
            fallback_on_empty=bool(section.get("fallback_on_empty", True)),
        )
    if name != "graspgenx":
        raise ValueError(f"Unknown fine-grasp backend {name!r}")

    section = config.get("backends", {}).get("graspgenx", {})
    if not isinstance(section, Mapping):
        raise ValueError("backends.graspgenx must be a mapping")
    primary = GraspGenXBackend(
        GraspGenXConfig.from_mapping(section),
        transport=graspgenx_transport,
    )
    fallback_name = section.get("fallback_backend", "none")
    if fallback_name in (None, False, "none", "disabled"):
        return primary
    if str(fallback_name).lower() != "geometric":
        raise ValueError("GraspGenX fallback_backend must be geometric or none")
    return FallbackGraspBackend(
        primary,
        GeometricAntipodalBackend(),
        fallback_on_empty=bool(section.get("fallback_on_empty", True)),
    )
