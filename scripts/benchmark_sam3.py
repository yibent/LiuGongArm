"""Measured SAM3 text segmentation on recorded RGB. No robot commands."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,default=Path("D:/sam3.pt"))
    parser.add_argument("--images",type=Path,nargs="+",required=True)
    parser.add_argument("--labels",nargs="+",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(kind="offline_rgb_segmentation_only",physical_success=None,labels=args.labels,
                checkpoint=str(args.checkpoint.resolve()),images=[],load_audit=[])
    try:
        import cv2
        import numpy as np
        import torch
        from ultralytics.models.sam import SAM3SemanticPredictor
        report.update(torch=torch.__version__,ultralytics=importlib.metadata.version("ultralytics"))
        with args.checkpoint.open("rb") as stream:
            report["checkpoint_sha256"]=hashlib.file_digest(stream,"sha256").hexdigest()
        original=torch.nn.Module.load_state_dict
        def audited_load(model,*a,**kw):
            result=original(model,*a,**kw)
            if type(model).__name__=="SAM3SemanticModel":
                report["load_audit"].append(dict(missing=list(result.missing_keys),unexpected=list(result.unexpected_keys)))
                if result.missing_keys:
                    raise RuntimeError("SAM3 checkpoint is missing model parameters")
            return result
        started=time.perf_counter()
        predictor=SAM3SemanticPredictor(overrides=dict(model=str(args.checkpoint.resolve()),device="0",
            task="segment",mode="predict",conf=.25,save=False,verbose=False,half=True))
        with patch.object(torch.nn.Module,"load_state_dict",audited_load):
            predictor.setup_model(model=None,verbose=False)
        if not report["load_audit"]:
            raise RuntimeError("Model parameter loading was not audited")
        torch.cuda.synchronize()
        report["model_load_s"]=time.perf_counter()-started
        for index,path in enumerate(args.images):
            rgb=cv2.imread(str(path))
            if rgb is None:
                raise ValueError(f"Unreadable image: {path}")
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started=time.perf_counter()
            results=predictor(source=rgb,text=args.labels,stream=False)
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-started
            result=results[0]
            row=dict(source=str(path.resolve()),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                     wall_s=elapsed,speed_ms=result.speed,cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                     detections=[])
            if result.masks is not None:
                masks=result.masks.data.cpu().numpy().astype(bool)
                np.savez_compressed(args.output/f"image_{index:02d}_masks.npz",masks=masks)
                for box,mask in zip(result.boxes,masks):
                    cls=int(box.cls.item())
                    row["detections"].append(dict(label=result.names[cls],confidence=float(box.conf.item()),
                        xyxy=box.xyxy[0].cpu().tolist(),mask_pixels=int(mask.sum())))
            cv2.imwrite(str(args.output/f"image_{index:02d}_overlay.jpg"),result.plot())
            report["images"].append(row)
            print(json.dumps(row),flush=True)
        report["status"]="passed"
    except Exception as exc:
        report.update(status="failed",error=repr(exc))
        raise
    finally:
        (args.output/"report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")


if __name__=="__main__":
    main()
