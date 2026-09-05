"""Unseen-object Isaac benchmark case definitions and report aggregation.

This module deliberately keeps report parsing independent of Isaac Sim so the
benchmark harness and its failure classification can be unit tested quickly.
Isaac imports live inside :func:`spawn_benchmark_target`.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


DEFAULT_TARGET_XY_M = (0.324, -0.198)
TABLE_HEIGHT_M = 1.05
PHYSICAL_LIFT_THRESHOLD_M = 0.025


@dataclass(frozen=True)
class BenchmarkCase:
    """One reproducible physical grasp trial.

    ``dimensions_m`` is the XYZ bounding-box size. For a sphere, X is used as
    its diameter; for a cylinder, X is its diameter and Z its height.
    ``position_m`` is the object's centre in the robot base/world frame.
    """

    name: str
    shape: str
    dimensions_m: tuple[float, float, float]
    position_m: tuple[float, float, float]
    seed: int = 0
    yaw_deg: float = 0.0
    mass_kg: float = 0.035
    color_rgb: tuple[float, float, float] = (0.85, 0.12, 0.08)
    material: str = "matte"
    fragile: bool = False
    thin: bool = False
    reflective: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Benchmark case name is required")
        if self.shape not in {"cube", "sphere", "cylinder", "thin", "hammer", "mug"}:
            raise ValueError(f"Unsupported benchmark shape: {self.shape}")
        if len(self.dimensions_m) != 3 or any(value <= 0 for value in self.dimensions_m):
            raise ValueError("dimensions_m must contain three positive values")
        if len(self.position_m) != 3 or not all(math.isfinite(value) for value in self.position_m):
            raise ValueError("position_m must contain three finite values")
        if self.mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        if self.material not in {"matte", "metallic", "reflective"}:
            raise ValueError(f"Unsupported benchmark material: {self.material}")

    @property
    def semantic_label(self) -> str:
        descriptors = ["unseen", self.material, self.shape]
        if self.thin:
            descriptors.append("thin")
        return " ".join(dict.fromkeys(descriptors))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BenchmarkCase":
        data = dict(value)
        for key in ("dimensions_m", "position_m", "color_rgb"):
            if key in data:
                data[key] = tuple(float(item) for item in data[key])
        data["seed"] = int(data.get("seed", 0))
        return cls(**data)


def default_demo_case() -> BenchmarkCase:
    """Return the exact legacy cube scene used by run_fine_grasp_demo.py."""
    return BenchmarkCase(
        name="cube_default",
        shape="cube",
        dimensions_m=(0.036, 0.036, 0.036),
        position_m=(0.324, -0.198, 1.068),
    )


def default_unseen_cases(seed: int) -> list[BenchmarkCase]:
    """Build a small but physically distinct unseen-object regression suite."""
    # Keep perturbations inside the wrist camera's close-range field of view.
    # Each seed maps to a stable XY position and yaw for exact reproduction.
    import random

    randomizer = random.Random(seed)

    def varied(
        *,
        name: str,
        shape: str,
        dimensions: tuple[float, float, float],
        **kwargs: Any,
    ) -> BenchmarkCase:
        position = (
            DEFAULT_TARGET_XY_M[0] + randomizer.uniform(-0.003, 0.003),
            DEFAULT_TARGET_XY_M[1] + randomizer.uniform(-0.003, 0.003),
            TABLE_HEIGHT_M + dimensions[2] * 0.5,
        )
        return BenchmarkCase(
            name=name,
            shape=shape,
            dimensions_m=dimensions,
            position_m=position,
            seed=seed,
            yaw_deg=randomizer.uniform(-18.0, 18.0),
            **kwargs,
        )

    return [
        varied(name="cube", shape="cube", dimensions=(0.036, 0.036, 0.036)),
        varied(
            name="sphere",
            shape="sphere",
            dimensions=(0.034, 0.034, 0.034),
            color_rgb=(0.18, 0.62, 0.20),
        ),
        varied(
            name="cylinder",
            shape="cylinder",
            dimensions=(0.032, 0.032, 0.046),
            color_rgb=(0.15, 0.28, 0.78),
        ),
        varied(
            name="thin",
            shape="thin",
            dimensions=(0.050, 0.025, 0.007),
            color_rgb=(0.72, 0.55, 0.08),
            mass_kg=0.012,
            thin=True,
        ),
        varied(
            name="metallic_part",
            shape="cylinder",
            dimensions=(0.030, 0.030, 0.040),
            color_rgb=(0.48, 0.52, 0.56),
            material="reflective",
            reflective=True,
            mass_kg=0.060,
        ),
        varied(
            name="apple",
            shape="sphere",
            dimensions=(0.045, 0.045, 0.045),
            color_rgb=(0.72, 0.08, 0.06),
            mass_kg=0.105,
            fragile=True,
            metadata={"scenario": "fruit", "force_sensitive": True},
        ),
        varied(
            name="bottle",
            shape="cylinder",
            dimensions=(0.044, 0.044, 0.090),
            color_rgb=(0.12, 0.48, 0.62),
            mass_kg=0.120,
            metadata={"scenario": "container"},
        ),
        varied(
            name="hammer",
            shape="hammer",
            dimensions=(0.110, 0.050, 0.028),
            color_rgb=(0.18, 0.22, 0.25),
            mass_kg=0.150,
            metadata={"scenario": "industrial_tool", "preferred_region": "handle"},
        ),
        varied(
            name="coffee_mug",
            shape="mug",
            dimensions=(0.082, 0.056, 0.062),
            color_rgb=(0.82, 0.74, 0.60),
            mass_kg=0.125,
            fragile=True,
            metadata={"scenario": "service_container", "has_handle": True},
        ),
    ]


def write_case(path: str | Path, case: BenchmarkCase) -> None:
    Path(path).write_text(json.dumps(case.to_dict(), indent=2), encoding="utf-8")


def load_case(path: str | Path) -> BenchmarkCase:
    return BenchmarkCase.from_mapping(json.loads(Path(path).read_text(encoding="utf-8")))


def spawn_benchmark_target(case: BenchmarkCase, path: str):
    """Author a real rigid/collidable Isaac primitive for ``case``.

    The returned ``RigidPrim`` is used to measure ground-truth physical lift;
    this function must only be called after ``SimulationApp`` is initialized.
    """
    import omni.usd
    from isaacsim.core.experimental.objects import Cube, Cylinder, Sphere
    from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    position = [list(case.position_m)]
    color = tuple(case.color_rgb)
    yaw_rad = math.radians(case.yaw_deg)
    orientation = [[math.cos(yaw_rad * 0.5), 0.0, 0.0, math.sin(yaw_rad * 0.5)]]
    orientation_kwargs = {} if abs(yaw_rad) < 1e-12 else {"orientations": orientation}
    x_size, y_size, z_size = case.dimensions_m
    compound = case.shape in {"hammer", "mug"}
    if compound:
        stage = omni.usd.get_context().get_stage()
        root = UsdGeom.Xform.Define(stage, path)
        root_xform = UsdGeom.Xformable(root)
        root_xform.AddTranslateOp().Set(Gf.Vec3d(*case.position_m))
        if abs(case.yaw_deg) > 1e-12:
            root_xform.AddRotateZOp().Set(case.yaw_deg)

        def box(
            name: str,
            center: tuple[float, float, float],
            size: tuple[float, float, float],
        ) -> None:
            cube = UsdGeom.Cube.Define(stage, f"{path}/{name}")
            cube.CreateSizeAttr(1.0)
            cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xform = UsdGeom.Xformable(cube)
            xform.AddTranslateOp().Set(Gf.Vec3d(*center))
            xform.AddScaleOp().Set(Gf.Vec3d(*size))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

        def cylinder(
            name: str,
            center: tuple[float, float, float],
            radius: float,
            height: float,
        ) -> None:
            geom = UsdGeom.Cylinder.Define(stage, f"{path}/{name}")
            geom.CreateAxisAttr("Z")
            geom.CreateRadiusAttr(radius)
            geom.CreateHeightAttr(height)
            geom.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            UsdGeom.Xformable(geom).AddTranslateOp().Set(Gf.Vec3d(*center))
            UsdPhysics.CollisionAPI.Apply(geom.GetPrim())

        if case.shape == "hammer":
            # Horizontal handle plus transverse head.  The handle is thin
            # enough for SO-101 while the head tests region-aware proposals.
            box("Handle", (-0.010, 0.0, -0.006), (0.090, 0.014, 0.014))
            box("Head", (0.035, 0.0, 0.0), (0.026, 0.050, 0.028))
        else:
            # External mug body plus a collision-bearing C handle.  This proxy
            # exercises handle occlusion and fragile-object policy without a
            # category-specific grasp network.
            cylinder("Body", (-0.012, 0.0, 0.0), 0.026, 0.062)
            box("HandleTop", (0.023, 0.0, 0.020), (0.030, 0.012, 0.010))
            box("HandleBottom", (0.023, 0.0, -0.020), (0.030, 0.012, 0.010))
            box("HandleOuter", (0.038, 0.0, 0.0), (0.010, 0.012, 0.050))
    elif case.shape == "sphere":
        Sphere(path, positions=position, radii=x_size * 0.5, colors=color)
    elif case.shape == "cylinder":
        Cylinder(
            path,
            positions=position,
            **orientation_kwargs,
            radii=x_size * 0.5,
            heights=z_size,
            axes="Z",
            colors=color,
        )
    elif case.shape == "cube" and max(case.dimensions_m) - min(case.dimensions_m) < 1e-9:
        # Preserve the legacy demo's authored cube exactly when dimensions are
        # uniform instead of introducing a scale xform.
        Cube(path, positions=position, sizes=x_size, colors=color, **orientation_kwargs)
    else:
        Cube(
            path,
            positions=position,
            **orientation_kwargs,
            sizes=1.0,
            scales=[list(case.dimensions_m)],
            colors=color,
        )

    if not compound:
        GeomPrim(path, apply_collision_apis=True)
    rigid = RigidPrim(
        path,
        masses=[case.mass_kg],
        reset_xform_op_properties=compound,
    )

    if case.material != "matte":
        stage = omni.usd.get_context().get_stage()
        material_path = f"{path}_Material"
        material = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(1.0)
        roughness = 0.04 if case.material == "reflective" else 0.22
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        prim = stage.GetPrimAtPath(path)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    return rigid


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def classify_report(
    report: Mapping[str, Any] | None,
    *,
    process_returncode: int | None = None,
    process_error: str | None = None,
    lift_threshold_m: float = PHYSICAL_LIFT_THRESHOLD_M,
) -> dict[str, Any]:
    """Normalize one worker report and apply physical-success semantics."""
    if report is None:
        category = "process_error" if process_error or process_returncode else "missing_report"
        return {
            "success": False,
            "node_success": False,
            "physical_lift_success": False,
            "failure_category": category,
            "process_returncode": process_returncode,
            "process_error": process_error,
        }

    result = report.get("result") if isinstance(report.get("result"), Mapping) else {}
    metrics = result.get("metrics") if isinstance(result.get("metrics"), Mapping) else {}
    selected = (
        result.get("selected_grasp")
        if isinstance(result.get("selected_grasp"), Mapping)
        else {}
    )
    selected_metadata = (
        selected.get("metadata") if isinstance(selected.get("metadata"), Mapping) else {}
    )
    node_success = bool(result.get("success", False))
    actual_lift_m = _number(report.get("actual_target_lift_m"))
    physical_lift = actual_lift_m is not None and actual_lift_m >= lift_threshold_m
    failure = result.get("failure")
    if node_success and not physical_lift:
        category = "false_positive_no_physical_lift"
    elif not node_success and physical_lift:
        category = "verification_false_negative"
    elif not node_success:
        category = str(failure or "unclassified_failure")
    else:
        category = None
    alignment_m = _number(metrics.get("final_translation_error_m"))
    return {
        "success": bool(node_success and physical_lift),
        "node_success": node_success,
        "physical_lift_success": physical_lift,
        "failure_category": category,
        "actual_lift_m": actual_lift_m,
        "alignment_error_mm": alignment_m * 1000.0 if alignment_m is not None else None,
        "model_latency_ms": _number(metrics.get("model_latency_ms")),
        "cycle_time_s": _number(metrics.get("elapsed_s")),
        "backend": selected.get("backend") or metrics.get("backend"),
        "configured_backend": metrics.get("backend"),
        "fallback_reason": selected_metadata.get("fallback_reason"),
        "observations": metrics.get("observations"),
        "replans": metrics.get("replans"),
        "servo_steps": metrics.get("servo_steps"),
        "process_returncode": process_returncode,
        "process_error": process_error,
    }


def _metric_summary(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, float | None]:
    values = sorted(value for row in rows if (value := _number(row.get(key))) is not None)
    if not values:
        return {"mean": None, "median": None, "p95": None}
    p95_index = max(0, math.ceil(0.95 * len(values)) - 1)
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": values[p95_index],
    }


def aggregate_runs(runs: list[Mapping[str, Any]]) -> dict[str, Any]:
    successes = sum(bool(run.get("success")) for run in runs)
    failure_counts = Counter(
        str(run["failure_category"])
        for run in runs
        if run.get("failure_category") is not None
    )
    def grouped(field: str) -> dict[str, dict[str, int | float]]:
        result: dict[str, dict[str, int | float]] = {}
        for value in sorted({str(run.get(field, "unknown")) for run in runs}):
            selected = [run for run in runs if str(run.get(field, "unknown")) == value]
            selected_success = sum(bool(run.get("success")) for run in selected)
            result[value] = {
                "trials": len(selected),
                "successes": selected_success,
                "success_rate": selected_success / len(selected) if selected else 0.0,
            }
        return result
    return {
        "trials": len(runs),
        "successes": successes,
        "success_rate": successes / len(runs) if runs else 0.0,
        "node_success_rate": (
            sum(bool(run.get("node_success")) for run in runs) / len(runs) if runs else 0.0
        ),
        "physical_lift_rate": (
            sum(bool(run.get("physical_lift_success")) for run in runs) / len(runs)
            if runs
            else 0.0
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "by_shape": grouped("shape"),
        "by_material": grouped("material"),
        "metrics": {
            "alignment_error_mm": _metric_summary(runs, "alignment_error_mm"),
            "model_latency_ms": _metric_summary(runs, "model_latency_ms"),
            "cycle_time_s": _metric_summary(runs, "cycle_time_s"),
            "actual_lift_m": _metric_summary(runs, "actual_lift_m"),
        },
        "runs": runs,
    }


def render_markdown(summary: Mapping[str, Any]) -> str:
    trials = int(summary.get("trials", 0))
    successes = int(summary.get("successes", 0))
    lines = [
        "# FineGrasp unseen-object physical benchmark",
        "",
        f"Physical success: **{successes}/{trials} ({float(summary.get('success_rate', 0.0)):.1%})**",
        "",
        "A trial counts as success only when the FineGrasp node reports success and the",
        f"ground-truth rigid body rises at least {PHYSICAL_LIFT_THRESHOLD_M * 1000:.0f} mm.",
        "",
        "| Case | Seed | Shape | Material | Backend | Result | Failure | Lift (mm) | Align (mm) | Model (ms) | Cycle (s) |",
        "|---|---:|---|---|---|---|---|---:|---:|---:|---:|",
    ]
    for run in summary.get("runs", []):
        def formatted(key: str, multiplier: float = 1.0) -> str:
            value = _number(run.get(key))
            return "-" if value is None else f"{value * multiplier:.2f}"

        lines.append(
            "| {case} | {seed} | {shape} | {material} | {backend} | {result} | {failure} | {lift} | {align} | {model} | {cycle} |".format(
                case=run.get("case", "-"),
                seed=run.get("seed", "-"),
                shape=run.get("shape", "-"),
                material=run.get("material", "-"),
                backend=run.get("backend", "-"),
                result="PASS" if run.get("success") else "FAIL",
                failure=run.get("failure_category") or "-",
                lift=formatted("actual_lift_m", 1000.0),
                align=formatted("alignment_error_mm"),
                model=formatted("model_latency_ms"),
                cycle=formatted("cycle_time_s"),
            )
        )
    lines.extend(["", "## Failure classification", ""])
    failures = summary.get("failure_counts", {})
    if failures:
        lines.extend(f"- `{name}`: {count}" for name, count in failures.items())
    else:
        lines.append("- No failures")
    for title, key in (("Shape success", "by_shape"), ("Material success", "by_material")):
        lines.extend(["", f"## {title}", "", "| Group | Success | Rate |", "|---|---:|---:|"])
        for name, values in summary.get(key, {}).items():
            lines.append(
                f"| {name} | {values['successes']}/{values['trials']} | "
                f"{float(values['success_rate']):.1%} |"
            )
    lines.extend(["", "## Metrics", ""])
    metrics = summary.get("metrics", {})
    lines.append("| Metric | Mean | Median | P95 |")
    lines.append("|---|---:|---:|---:|")
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {_format_optional(values.get('mean'))} | "
            f"{_format_optional(values.get('median'))} | {_format_optional(values.get('p95'))} |"
        )
    return "\n".join(lines) + "\n"


def _format_optional(value: Any) -> str:
    number = _number(value)
    return "-" if number is None else f"{number:.3f}"
