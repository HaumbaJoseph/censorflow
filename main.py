import time
import tkinter as tk
from collections import deque, defaultdict
from PIL import ImageGrab, ImageTk, ImageFilter
from nsfw_onnx import is_nsfw
import numpy as np
import keyboard  # pip install keyboard

# ===================== CONFIG =====================
TILE = 448           # tile size
STEP = TILE // 2     # overlap
BLUR_RADIUS = 35     # strong blur
SCAN_INTERVAL = 1500 # ms
NSFW_THRESHOLD = 0.25
VOTE_FRAMES = 3      # temporal voting frames
SKIN_WEIGHT = 0.1    # weight of skin-percentage heuristic
# ==================================================

print("🛡 Screen Shield starting…")
print("❌ Kill switches: ESC | Ctrl+Q | Ctrl+C")

# Tkinter fullscreen overlay
root = tk.Tk()
root.attributes("-fullscreen", True)
root.attributes("-topmost", True)
root.attributes("-alpha", 0.96)
root.overrideredirect(True)
canvas = tk.Canvas(root, highlightthickness=0)
canvas.pack(fill="both", expand=True)

# Temporal memory for each tile
vote_memory = defaultdict(lambda: deque(maxlen=VOTE_FRAMES))
blurred_zones = set()

def skin_percentage(tile):
    """Simple skin heuristic using RGB thresholds"""
    arr = np.array(tile)
    R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    skin = (R > 95) & (G > 40) & (G < 100) & (B > 20)
    return np.sum(skin) / (arr.shape[0]*arr.shape[1])

def scan_and_blur():
    global vote_memory, blurred_zones

    if keyboard.is_pressed("esc") or (keyboard.is_pressed("ctrl") and keyboard.is_pressed("q")):
        print("🛑 Kill switch pressed, exiting...")
        root.destroy()
        return

    screen = ImageGrab.grab()
    blurred = screen.copy()
    w, h = screen.size

    for y in range(0, h, STEP):
        for x in range(0, w, STEP):
            tile = screen.crop((x, y, x + TILE, y + TILE))
            
            # Dual model voting
            try:
                nsfw_flag, model_score = is_nsfw(tile, NSFW_THRESHOLD)
            except Exception as e:
                print(f"⚠ NSFW model error: {e}")
                nsfw_flag, model_score = False, 0.0

            skin_score = skin_percentage(tile)
            final_score = model_score * (1 - SKIN_WEIGHT) + skin_score * SKIN_WEIGHT
            vote_memory[(x, y)].append(final_score >= NSFW_THRESHOLD)

            # Only blur if majority of last frames flagged NSFW
            if sum(vote_memory[(x, y)]) >= (VOTE_FRAMES // 2 + 1):
                blur_tile = tile.filter(ImageFilter.GaussianBlur(BLUR_RADIUS))
                blurred.paste(blur_tile, (x, y))
                blurred_zones.add((x, y))
                if final_score >= NSFW_THRESHOLD:
                    print(f"⚠ NSFW detected at ({x},{y}) score={final_score:.2f}")

    tk_img = ImageTk.PhotoImage(blurred)
    canvas.delete("all")
    canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas.image = tk_img

    root.after(SCAN_INTERVAL, scan_and_blur)

# Start scanning
scan_and_blur()
root.mainloop()

