"""One-command headless demo gallery, retaining successes and failures alike."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source"))
from mr_liu.grasp.benchmark import default_unseen_cases, write_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Reuse existing completed clips in this directory")
    args = parser.parse_args()
    import imageio_ffmpeg

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        parser.error("Output directory is not empty; choose a new directory or use --resume")
    output.mkdir(parents=True, exist_ok=True)
    cases = {case.name: case for case in default_unseen_cases(0)}
    sequence = [("01_cube_recovery", None, 0.04), ("02_apple", "apple", 0.),
                ("03_coffee_mug", "coffee_mug", 0.), ("04_hammer", "hammer", 0.)]
    clips, rows = [], []
    for name, case_name, shift in sequence:
        folder = output / name
        folder.mkdir(parents=True, exist_ok=True)
        clip, report_path = folder / "demo.mp4", folder / "report.json"
        if not (args.resume and clip.exists() and report_path.exists()
                and (folder / "demo.video.json").exists()):
            command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                       str(ROOT / "scripts/run_fine_grasp_demo.ps1"), "-Backend", "graspgenx",
                       "-Recovery", "active", "-SceneView", "oblique", "-RecordVideo",
                       "-TestTargetShiftM", str(shift), "-Output", str(folder)]
            if case_name:
                case_path = folder / "case.json"
                write_case(case_path, cases[case_name])
                command.extend(["-CaseJson", str(case_path)])
            (folder / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
            print(f"[gallery] Recording {name}", flush=True)
            with (folder / "console.log").open("w", encoding="utf-8") as log:
                run = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=300, check=False)
            # Failed grasps are valid demonstration outcomes, not reasons to omit a clip.
            if not (report_path.exists() and clip.exists()):
                raise RuntimeError(f"Recording failed ({run.returncode}): {folder}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        video = json.loads((folder / "demo.video.json").read_text(encoding="utf-8"))
        row = {"case": name, "success": report["result"]["success"],
               "failure": report["result"].get("failure"),
               "lift_m": report["actual_target_lift_m"], "duration_s": video["duration_s"],
               "recording_overhead": True, "source_report": str(report_path)}
        row["start_s"] = sum(item["duration_s"] for item in rows)
        rows.append(row)
        clips.append(clip)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        (output / "gallery.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    playlist = output / "concat.txt"
    # ffmpeg's concat file uses backslash escaping within single-quoted paths.
    playlist.write_text("\n".join("file '" + str(p).replace("\\", "/").replace("'", "'\\''") + "'"
                                  for p in clips), encoding="utf-8")
    combined = output / "BusAgent_grasp_gallery.mp4"
    chapters = output / "chapters.ffmeta"
    chapters.write_text(";FFMETADATA1\n" + "\n".join(
        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(row['start_s']*1000)}\n"
        f"END={round((row['start_s']+row['duration_s'])*1000)}\ntitle={row['case']}"
        for row in rows), encoding="utf-8")
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "concat",
                    "-safe", "0", "-i", str(playlist), "-i", str(chapters), "-map_metadata", "1",
                    "-c", "copy", "-movflags", "+faststart",
                    str(combined)], check=True, timeout=60)
    print(f"[gallery] Complete: {combined}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
