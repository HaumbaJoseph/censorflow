import numpy as np
import onnxruntime as ort
from PIL import Image

# ================= CONFIG =================
MODEL_1 = "models/nsfw_model.onnx"
MODEL_2 = None   # optional second model path
IMG_SIZE = 224
# =========================================

sessions = []

def load_model(path):
    return ort.InferenceSession(path, providers=["CPUExecutionProvider"])

sessions.append(load_model(MODEL_1))
if MODEL_2:
    sessions.append(load_model(MODEL_2))

INPUT_NAMES = [s.get_inputs()[0].name for s in sessions]
OUTPUT_NAMES = [s.get_outputs()[0].name for s in sessions]

print("ONNX models loaded:", len(sessions))


def run_model(session, input_name, output_name, img):
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img).astype(np.float32) / 255.0
    x = np.expand_dims(x, axis=0)

    out = session.run([output_name], {input_name: x})[0]
    out = np.array(out).squeeze()

    if out.ndim == 0:
        return float(out)
    if out.size == 1:
        return float(out[0])
    if out.size == 2:
        return float(out[1])  # assume [safe, nsfw]

    return float(np.max(out[1:]))


def skin_percentage(img):
    np_img = np.array(img.convert("RGB"))
    r, g, b = np_img[:,:,0], np_img[:,:,1], np_img[:,:,2]

    skin = (
        (r > 95) &
        (g > 40) &
        (b > 20) &
        ((r - g) > 15) &
        (r > g) &
        (g > b)
    )
    return np.mean(skin)


def is_nsfw(image, threshold=0.20):
    scores = []

    for s, inp, out in zip(sessions, INPUT_NAMES, OUTPUT_NAMES):
        scores.append(run_model(s, inp, out, image))

    model_score = max(scores)
    skin_score = skin_percentage(image)

    # combine signals
    final_score = max(model_score, skin_score)

    return final_score >= threshold, final_score

