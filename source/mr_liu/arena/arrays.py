"""Host snapshots of Isaac Lab Torch and Warp sensor arrays."""
import numpy as np
import ast

def numpy_data(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def semantic_selection(mask, labels, name):
    """Accept Sim 6 numeric IDs, packed RGBA IDs, and colorized masks."""
    shape = mask.shape[:2]
    selected = np.zeros(shape, dtype=bool)
    for key, value in labels.items():
        if name not in [label.strip() for label in str(value.get("class", "")).split(",")]:
            continue
        if str(key).isdigit():
            selected |= mask.squeeze() == int(key)
        else:
            rgba = np.asarray(ast.literal_eval(key), dtype=np.uint8)
            if rgba.shape != (4,):
                raise ValueError("Invalid semantic color mapping")
            if mask.ndim == 3 and mask.shape[-1] in {3, 4}:
                selected |= np.all(mask == rgba[:mask.shape[-1]], axis=-1)
            else:
                selected |= mask.squeeze() == int.from_bytes(rgba.tobytes(), "little")
    return selected
