"""Serve the official M2T2 checkpoint behind a small msgpack/ZMQ API.

This process owns torch, CUDA and the upstream M2T2 checkout.  Isaac Sim only
loads the torch-free client in ``mr_liu.grasp.backends.m2t2``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import random
import threading

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "_vendor" / "M2T2"


def _fixed_points(points, count):
    import torch
    value = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    value = value[np.isfinite(value).all(axis=1)]
    if len(value) == 0:
        raise ValueError("point cloud is empty")
    indices = np.arange(count, dtype=np.int64) % len(value)
    # A deterministic evenly spaced order avoids a changing target contact
    # mask when the same camera frame is retried.
    if len(value) > count:
        indices = np.linspace(0, len(value) - 1, count, dtype=np.int64)
    return torch.from_numpy(value[indices])


def _bottom_center(points, grid_resolution=0.01):
    """Estimate the support point expected by M2T2's placement decoder.

    M2T2 consumes the held object in the EE frame, but its placement decoder
    computes the placement offset from ``bottom_center`` in the scene frame.
    Keep this estimate in the scene/base frame before converting the object
    cloud to EE coordinates.
    """
    value = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    value = value[np.isfinite(value).all(axis=1)]
    if len(value) == 0:
        raise ValueError("object point cloud is empty")
    resolution = max(float(grid_resolution), 1e-4)
    z_min = float(value[:, 2].min())
    # A small band makes the estimate robust to depth quantisation while
    # retaining the lowest support surface used by the upstream dataset.
    near_bottom = value[value[:, 2] <= z_min + max(resolution, 0.005)]
    if len(near_bottom) == 0:
        near_bottom = value
    xy_grid = np.unique(np.round(near_bottom[:, :2] / resolution), axis=0) * resolution
    center_xy = xy_grid.mean(axis=0)
    return np.asarray([center_xy[0], center_xy[1], z_min], dtype=np.float32)


class M2T2Service:
    def __init__(self, checkpoint: Path, *, device: str = "cuda", config_path: Path | None = None):
        try:
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "M2T2 server dependencies are missing; run scripts/setup_m2t2.ps1 in a CUDA environment"
            ) from exc

        import sys
        sys.path.insert(0, str(VENDOR))
        try:
            from m2t2.m2t2 import M2T2
        except ImportError as exc:
            raise RuntimeError(
                "M2T2 upstream code or pointnet2_ops is unavailable; run scripts/setup_m2t2.ps1"
            ) from exc

        self.torch = torch
        requested_device = str(device).strip().lower()
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "M2T2 requires CUDA-enabled PyTorch; torch.cuda.is_available() is false"
            )
        self.device = torch.device(requested_device)
        cfg_path = config_path or VENDOR / "config.yaml"
        self.cfg = OmegaConf.load(str(cfg_path))
        self.eval_cfg = self.cfg.eval
        self.model = M2T2.from_config(self.cfg.m2t2)
        state = torch.load(str(checkpoint), map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()
        self.checkpoint = str(checkpoint)
        self._lock = threading.Lock()

    def _data(self, scene_points, object_points, *, task, ee_pose=None, bottom_center=None):
        torch = self.torch
        scene = _fixed_points(scene_points, int(self.cfg.data.num_points))
        obj = _fixed_points(object_points, int(self.cfg.data.num_object_points))
        scene_centered = scene - scene.mean(dim=0, keepdim=True)
        obj_centered = obj - obj.mean(dim=0, keepdim=True)
        if bottom_center is None:
            bottom_center = _bottom_center(obj.numpy())
        data = {
            "inputs": scene_centered.unsqueeze(0),
            "points": scene.unsqueeze(0),
            "seg": torch.zeros(1, scene.shape[0], dtype=torch.long),
            "object_inputs": obj_centered.unsqueeze(0),
            "cam_pose": torch.eye(4).unsqueeze(0),
            "ee_pose": torch.eye(4).unsqueeze(0) if ee_pose is None else torch.as_tensor(ee_pose, dtype=torch.float32).unsqueeze(0),
            "bottom_center": torch.as_tensor(bottom_center, dtype=torch.float32).reshape(1, 3),
            "task_is_pick": torch.tensor([task == "pick"]),
            "task_is_place": torch.tensor([task == "place"]),
        }
        return {key: value.to(self.device) if hasattr(value, "to") else value for key, value in data.items()}

    @staticmethod
    def _flatten(items):
        """Flatten M2T2's ``[batch][query][num_predictions,...]`` lists."""
        result = []
        groups = items[0] if items and isinstance(items[0], (list, tuple)) else items
        for group in groups:
            if isinstance(group, (list, tuple)):
                values = group
            else:
                values = [group]
            for value in values:
                if hasattr(value, "ndim") and value.ndim >= 1:
                    result.extend(value)
                else:
                    result.append(value)
        return result

    def infer_pick(self, request):
        torch = self.torch
        data = self._data(request["scene_points"], request["object_points"], task="pick")
        eval_cfg = deepcopy(self.eval_cfg)
        model_frame = str(request.get("model_frame", "base"))
        if model_frame not in {"base", "camera"}:
            raise ValueError(f"unsupported M2T2 model frame: {model_frame!r}")
        eval_cfg.world_coord = model_frame == "base"
        eval_cfg.mask_thresh = float(request.get("min_confidence", 0.0))
        with self._lock, torch.inference_mode():
            output = self.model.infer(data, eval_cfg)
        grasps = self._flatten(output.get("grasps", []))
        confidence = self._flatten(output.get("grasp_confidence", []))
        contacts = self._flatten(output.get("grasp_contacts", []))
        poses = [item.detach().cpu().numpy() for item in grasps]
        scores = [float(item.detach().cpu().item()) for item in confidence]
        ctrs = [item.detach().cpu().numpy() for item in contacts]
        limit = int(request.get("max_candidates", len(poses)))
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:limit]
        return {"grasps": [poses[i] for i in order], "confidences": [scores[i] for i in order],
                "contacts": [ctrs[i] for i in order] if len(ctrs) == len(poses) else [],
                "model_frame": model_frame, "checkpoint": self.checkpoint}

    def infer_place(self, request):
        torch = self.torch
        ee_pose = np.asarray(request["ee_pose"], dtype=np.float32).reshape(4, 4)
        object_base = np.asarray(request["object_points"], dtype=np.float32).reshape(-1, 3)
        bottom_center = _bottom_center(
            object_base,
            grid_resolution=float(getattr(self.cfg.data, "grid_resolution", 0.01)),
        )
        ee_inv = np.linalg.inv(ee_pose)
        object_ee = object_base @ ee_inv[:3, :3].T + ee_inv[:3, 3]
        data = self._data(
            request["scene_points"], object_ee, task="place", ee_pose=ee_pose,
            bottom_center=bottom_center,
        )
        eval_cfg = deepcopy(self.eval_cfg)
        model_frame = str(request.get("model_frame", "base"))
        if model_frame not in {"base", "camera"}:
            raise ValueError(f"unsupported M2T2 model frame: {model_frame!r}")
        eval_cfg.world_coord = model_frame == "base"
        eval_cfg.mask_thresh = float(request.get("min_confidence", 0.0))
        with self._lock, torch.inference_mode():
            output = self.model.infer(data, eval_cfg)
        placements = self._flatten(output.get("placements", []))
        confidence = self._flatten(output.get("placement_confidence", []))
        poses = [item.detach().cpu().numpy() for item in placements]
        scores = [float(item.detach().cpu().item()) for item in confidence]
        limit = int(request.get("max_candidates", len(poses)))
        order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")[:limit]
        return {"placements": [poses[i] for i in order], "confidences": [scores[i] for i in order],
                "model_frame": model_frame, "checkpoint": self.checkpoint}


class ZmqServer:
    def __init__(self, service: M2T2Service, host: str, port: int):
        import msgpack
        import msgpack_numpy
        import zmq
        msgpack_numpy.patch()
        self.service, self.msgpack, self.zmq = service, msgpack, zmq
        self.socket = zmq.Context.instance().socket(zmq.REP)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://{host}:{port}")

    def serve_forever(self):
        while True:
            raw = self.socket.recv()
            try:
                request = self.msgpack.unpackb(raw, raw=False)
                action = request.get("action")
                if action == "health":
                    response = {"status": "ready", "model": "M2T2", "checkpoint": self.service.checkpoint}
                elif action == "infer_pick":
                    seed = int(request.get("seed", 0)); random.seed(seed); np.random.seed(seed); self.service.torch.manual_seed(seed)
                    response = self.service.infer_pick(request)
                elif action == "infer_place":
                    seed = int(request.get("seed", 0)); random.seed(seed); np.random.seed(seed); self.service.torch.manual_seed(seed)
                    response = self.service.infer_place(request)
                else:
                    raise ValueError(f"unknown action: {action!r}")
            except Exception as exc:  # the client maps this to a model error
                response = {"error": f"{type(exc).__name__}: {exc}"}
            self.socket.send(self.msgpack.packb(response, use_bin_type=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5562)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "_models/m2t2/m2t2.pth")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--config", type=Path, default=VENDOR / "config.yaml")
    args = parser.parse_args()
    service = M2T2Service(args.checkpoint, device=args.device, config_path=args.config)
    print(f"M2T2 ready on tcp://{args.host}:{args.port} ({service.device})", flush=True)
    ZmqServer(service, args.host, args.port).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
