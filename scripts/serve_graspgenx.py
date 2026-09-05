"""Versioned BusAgent wrapper: seed each RPC, without editing upstream code."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch
from graspgenx.serving.zmq_server import GraspGenXZMQServer


class ReproducibleServer(GraspGenXZMQServer):
    def _segment(self, request):
        from PIL import Image
        rgb = np.asarray(request["rgb"], dtype=np.uint8)[:, :, :3]
        if not hasattr(self, "sam_predictor"):
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            checkpoint = Path(__file__).resolve().parents[1] / "_models/sam2/sam2.1_hiera_tiny.pt"
            self.sam_predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml",
                                                             str(checkpoint), device="cuda"))
        box, point = request.get("box"), request.get("point")
        grounding_used = False
        started = time.perf_counter()
        if box is None and point is None:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            if not hasattr(self, "grounding_model"):
                model_id = "IDEA-Research/grounding-dino-tiny"
                cache = str(Path(__file__).resolve().parents[1] / "_models/grounding_dino")
                self.grounding_processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache)
                self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                    model_id, cache_dir=cache).to("cuda").eval()
            inputs = self.grounding_processor(images=Image.fromarray(rgb),
                                              text=str(request.get("text", "object")).strip(" .") + ".",
                                              return_tensors="pt").to("cuda")
            with torch.inference_mode():
                output = self.grounding_model(**inputs)
            detection = self.grounding_processor.post_process_grounded_object_detection(
                output, inputs.input_ids, box_threshold=0.25, text_threshold=0.25,
                target_sizes=[rgb.shape[:2]])[0]
            # Multiple objects require upstream identity/position selection.
            if len(detection["boxes"]) != 1:
                return {"mask": None, "reason": "grounding_ambiguous_or_missing",
                        "detections": len(detection["boxes"])}
            box = detection["boxes"][0].cpu().numpy()
            grounding_used = True
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.sam_predictor.set_image(rgb)
            masks, scores, _ = self.sam_predictor.predict(
                box=None if box is None else np.asarray(box),
                point_coords=None if point is None else np.asarray([point]),
                point_labels=None if point is None else np.asarray([1]),
                multimask_output=True)
        index = int(np.argmax(scores))
        score = float(scores[index])
        return {"mask": masks[index].astype(bool) if score >= 0.60 else None,
                "segmentation_score": score, "grounding_used": grounding_used,
                "segmentation_ms": (time.perf_counter() - started) * 1000}

    def _dispatch(self, request: dict) -> dict:
        if request.get("action") == "segment":
            return self._segment(request)
        if str(request.get("action", "")).startswith("infer"):
            seed = int(request.get("seed", 0))
            if not 0 <= seed < 2**32:
                raise ValueError("seed must be an unsigned 32-bit integer")
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        result = super()._dispatch(request)
        if request.get("action") == "health":
            result["busagent_seed_protocol"] = 1
        if str(request.get("action", "")).startswith("infer"):
            result["seed"] = seed
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--assets_dir", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5556)
    args = parser.parse_args()
    # Deterministic RNG does not guarantee bitwise equality on different GPUs.
    torch.backends.cudnn.benchmark = False
    ReproducibleServer(config_path=args.config, assets_dir=args.assets_dir,
                       host=args.host, port=args.port).serve_forever()
