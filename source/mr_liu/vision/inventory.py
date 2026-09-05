"""Visible simulator objects, explicitly sourced from camera semantic pixels."""
import cv2
import numpy as np


def visible_inventory(bgr, semantic, observed_at):
    objects = []
    if not isinstance(semantic, dict):
        return None
    ids = np.asarray(semantic.get("data")).squeeze()
    if ids.shape != bgr.shape[:2]:
        return None
    labels = (semantic.get("info") or {}).get("idToLabels", {})
    for key, value in labels.items():
        category = value.get("class", "") if isinstance(value, dict) else ""
        if category not in {"block", "cube", "nut", "bolt", "wrench", "roller", "cup", "gear", "power_drill", "star"}:
            continue
        count, components, stats, _ = cv2.connectedComponentsWithStats((ids == int(key)).astype(np.uint8), 8)
        for i in range(1, count):
            if stats[i, cv2.CC_STAT_AREA] < 16:
                continue
            pixels = bgr[components == i]
            hue, sat, val = np.median(cv2.cvtColor(pixels.reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0], axis=0)
            color = ("black" if val < 65 else "white" if sat < 55 and val > 140 else
                     "red" if hue < 10 or hue >= 170 else "orange" if hue < 22 else
                     "yellow" if hue < 38 else "green" if hue < 90 else "blue" if hue < 130 else "purple")
            objects.append(dict(category=category, color=color, visible_pixels=int(len(pixels))))
    return dict(objects=objects, observed_at=observed_at, source="Isaac camera semantic pixels + RGB color",
                message="当前画面中可见物体候选：" + ("、".join(f"{o['color']} {o['category']}" for o in objects) or "未确认物体") + "。这是可见候选，不保证每个都可抓取。")
