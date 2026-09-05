from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
WEIGHTS = ROOT / "yoloe-26x-seg.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Florence-2-large 慢环 FIND + YOLOE-26x / CV 快环 TRACK，带可改提示的 WebUI。",
    )
    parser.add_argument("--webui", action="store_true", help="启动 WebUI（默认：未指定 --source 时启动）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--source", type=str, help="本地图片/视频路径；提供则走命令行而不开 WebUI")
    parser.add_argument("--prompt", type=str, default="person", help="语言描述，多个目标用分号分隔")
    parser.add_argument("--fast", choices=["yoloe", "cv"], default="yoloe", help="快环后端")
    parser.add_argument("--interval", type=int, default=45, help="慢环间隔（帧），0 表示仅在丢跟踪/改提示时 FIND")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", type=str, help="标注结果输出路径")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS))
    parser.add_argument("--florence", type=str, help="Florence model directory or Hugging Face model ID")
    parser.add_argument("--memory-id", type=str, help="回放 runs/memories 中的对象记忆")
    parser.add_argument("--memory-view", choices=["scene", "wrist"], help="回放记忆时优先使用的相机视角")
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    from find_and_track.memory import ObjectMemoryStore
    from find_and_track.pipeline import FindTrackPipeline
    from find_and_track.types import RuntimeConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    pipe = FindTrackPipeline(
        yoloe_weights=args.weights,
        florence_id=args.florence,
        memory_store=ObjectMemoryStore(ROOT / "runs" / "memories"),
    )
    cfg = RuntimeConfig(
        prompt=args.prompt,
        fast_backend=args.fast,
        slow_interval=args.interval,
        conf=args.conf,
        imgsz=args.imgsz,
        memory_id=args.memory_id,
        memory_view=args.memory_view,
    )
    print(f"Loading models… prompt={args.prompt!r} fast={args.fast}", flush=True)
    pipe.load(cfg.fast_backend)
    n = 0
    for result in pipe.process_source(args.source, cfg, output=args.output):
        n += 1
        s = result.stats
        print(
            f"[{s.frame_idx:05d}] {s.path:5s} fps={s.fps:5.1f} "
            f"slow={s.slow_ms:7.0f}ms fast={s.fast_ms:6.1f}ms "
            f"find={s.n_slow} track={s.n_fast} {s.message}",
            flush=True,
        )
    if args.output:
        print(f"Wrote {args.output}")
    print(f"Done, {n} frame(s).")
    return 0


def main() -> int:
    args = parse_args()
    weights = Path(args.weights)
    if args.fast == "yoloe" and not weights.exists():
        print(f"YOLOE weights not found: {weights}", file=sys.stderr)
        return 1
    if args.source and not args.webui:
        return run_cli(args)
    from find_and_track.webui import serve
    from find_and_track.types import RuntimeConfig

    serve(
        host=args.host, port=args.port, weights=weights, florence_id=args.florence,
        config=RuntimeConfig(
            prompt=args.prompt, fast_backend=args.fast, slow_interval=args.interval,
            conf=args.conf, imgsz=args.imgsz,
            memory_id=args.memory_id, memory_view=args.memory_view,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
