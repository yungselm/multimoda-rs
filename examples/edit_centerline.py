"""
Interactive centerline editor.
- Left-click a point to remove it (highlighted in red, then deleted on next click elsewhere)
- Right-click to undo last removal
- Press 's' to save the cleaned centerline to a new CSV
- Press 'r' to reset to original
"""

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
cwd = Path.cwd()
if (cwd / "examples" / "data").exists():
    data_dir = cwd / "examples" / "data"
elif (cwd / "data").exists():
    data_dir = cwd / "data"
else:
    data_dir = cwd
os.chdir(data_dir)

CSV_IN = "../data/centerline_rca_short.csv"
CSV_OUT = "../data/centerline_rca_short_edited.csv"

rca_cl_raw = np.genfromtxt(CSV_IN, delimiter=",")
print(f"Loaded {len(rca_cl_raw)} points from {CSV_IN}")

# ---------------------------------------------------------------------------
# Editor state
# ---------------------------------------------------------------------------
points = rca_cl_raw.copy()  # working copy (Nx3 or Nx4 etc.)
removed_stack: list[np.ndarray] = []  # undo stack: each entry is (index, row)
PICK_RADIUS = 8  # pixels


def xyz(arr):
    """Return first three columns as x, y, z."""
    return arr[:, 0], arr[:, 1], arr[:, 2]


# ---------------------------------------------------------------------------
# Figure setup
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")
fig.suptitle(
    "Left-click: remove point  |  Right-click: undo  |  's': save  |  'r': reset",
    fontsize=9,
)

scatter = ax.scatter(
    *xyz(points),
    c="steelblue",
    s=18,
    picker=PICK_RADIUS,
    depthshade=False,
    label="centerline",
)
(line,) = ax.plot(*xyz(points), color="steelblue", linewidth=0.8, alpha=0.5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title(f"{len(points)} points remaining")


def redraw():
    x, y, z = xyz(points)
    scatter._offsets3d = (x, y, z)
    line.set_data_3d(x, y, z)
    ax.set_title(f"{len(points)} points remaining")
    fig.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------
def on_pick(event):
    """Left-click removes the picked point (first reported index)."""
    if event.mouseevent.button != 1:
        return
    if len(event.ind) == 0:
        return

    global points
    idx = event.ind[0]
    removed_stack.append((idx, points[idx].copy()))
    points = np.delete(points, idx, axis=0)
    print(f"Removed point #{idx}  |  {len(points)} remaining  |  right-click to undo")
    redraw()


def on_key(event):
    global points
    if event.key == "s":
        np.savetxt(CSV_OUT, points, delimiter=",", fmt="%.6f")
        print(f"Saved {len(points)} points → {CSV_OUT}")

    elif event.key == "r":
        points = rca_cl_raw.copy()
        removed_stack.clear()
        print("Reset to original")
        redraw()


def on_button_press(event):
    """Right-click = undo."""
    if event.button == 3 and removed_stack:
        idx, row = removed_stack.pop()
        global points
        points = np.insert(points, idx, row, axis=0)
        print(f"Restored point #{idx}  |  {len(points)} remaining")
        redraw()


fig.canvas.mpl_connect("pick_event", on_pick)
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_button_press)

plt.tight_layout()
plt.show()
