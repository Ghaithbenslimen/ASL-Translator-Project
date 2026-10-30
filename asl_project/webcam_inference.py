"""
ASL webcam translator — final version (v3).
- Matches training distribution (no flip by default, wider padding)
- Can type the same letter twice: lift hand briefly between letters
- Keyboard: Q quit, C clear all, B backspace, S save, F toggle flip
- Hand gesture 'del' also deletes last letter (lower confidence threshold)
"""
import sys
import time
from collections import Counter, deque

import cv2
import mediapipe as mp
import numpy as np
import torch

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

sys.path.insert(0, ".")
from src.model import ASLNet
from src.dataset import CLASS_NAMES


MODEL_PATH               = "checkpoints/model_real.pt"
HAND_MODEL_PATH          = "hand_landmarker.task"
IMG_SIZE                 = 128
CONFIDENCE_THRESHOLD     = 0.65         # general threshold
DEL_CONFIDENCE_THRESHOLD = 0.45         # lower for 'del' gesture
STABLE_WINDOW            = 12
STABLE_CONFIRM           = 8
RESET_FRAMES             = 4            # frames without hand → allow same letter again
SPACE_SECONDS            = 2.5
CAMERA_INDEX             = 0
PAD                      = 0.15         # wider padding (matches training)
USE_FLIP                 = False        # no flip by default


# ================== ASL MODEL ==================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

ckpt = torch.load(MODEL_PATH, map_location=device)
model = ASLNet(num_classes=len(CLASS_NAMES), width=0.5).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"✓ ASL model loaded (val_acc={ckpt.get('val_acc', '?'):.2f}%)")
print(f"  Config: pad={PAD}  flip={USE_FLIP}")


# ================== HAND LANDMARKER ==================
base_options = BaseOptions(model_asset_path=HAND_MODEL_PATH)
hand_options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)
print("✓ Hand landmarker loaded")


# ================== STABILIZER & WORD BUILDER ==================
class LetterStabilizer:
    """
    Sliding-window vote with reset-on-empty.
    After RESET_FRAMES frames without a hand, 'current' is cleared,
    so you can sign the SAME letter again.
    """
    def __init__(self, window=STABLE_WINDOW, confirm=STABLE_CONFIRM,
                 reset_frames=RESET_FRAMES):
        self.window = deque(maxlen=window)
        self.confirm = confirm
        self.current = None
        self.empty_streak = 0
        self.reset_frames = reset_frames

    def update(self, letter, conf):
        threshold = DEL_CONFIDENCE_THRESHOLD if letter == "del" else CONFIDENCE_THRESHOLD

        # Invalid frame: no hand / low confidence / 'nothing' class
        if letter is None or letter == "nothing" or conf < threshold:
            self.empty_streak += 1
            if self.empty_streak >= self.reset_frames:
                self.current = None
                self.window.clear()
            return None

        # Valid frame
        self.empty_streak = 0
        self.window.append(letter)
        top, votes = Counter(self.window).most_common(1)[0]

        if votes >= self.confirm and top != self.current:
            self.current = top
            self.window.clear()
            return top
        return None

    def reset(self):
        self.window.clear()
        self.current = None
        self.empty_streak = 0


class WordBuilder:
    def __init__(self):
        self.text = ""
        self.last_letter_time = time.time()
        self.flash_until = 0.0
        self.flash_text = ""

    def add(self, letter):
        if letter == "del":
            self.backspace(flash=True)
            return
        elif letter == "space":
            if not self.text.endswith(" "):
                self.text += " "
        elif letter == "nothing":
            pass
        else:
            self.text += letter
        self.last_letter_time = time.time()

    def backspace(self, flash=False):
        if self.text:
            self.text = self.text[:-1]
        self.last_letter_time = time.time()
        if flash:
            self.flash_text = "DEL"
            self.flash_until = time.time() + 0.6

    def tick(self):
        if (time.time() - self.last_letter_time > SPACE_SECONDS
                and self.text and not self.text.endswith(" ")):
            self.text += " "
            self.last_letter_time = time.time()

    def clear(self):
        self.text = ""
        self.last_letter_time = time.time()
        self.flash_text = "CLEARED"
        self.flash_until = time.time() + 0.8


# ================== HELPERS ==================
def preprocess(hand_crop_bgr):
    img = cv2.resize(hand_crop_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def crop_hand(frame, hand_landmarks, pad=PAD):
    h, w = frame.shape[:2]
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x1 = max(0, int((min(xs) - pad) * w))
    y1 = max(0, int((min(ys) - pad) * h))
    x2 = min(w, int((max(xs) + pad) * w))
    y2 = min(h, int((max(ys) + pad) * h))
    if x2 <= x1 or y2 <= y1:
        return None, None

    side = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def draw_landmarks_simple(frame, hand_landmarks):
    h, w = frame.shape[:2]
    for lm in hand_landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)


# ================== MAIN LOOP ==================
def main():
    stabilizer = LetterStabilizer()
    builder = WordBuilder()
    fps_hist = deque(maxlen=30)

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"❌ Could not open camera {CAMERA_INDEX}")
        return

    print("✓ Camera open")
    print("  Q=quit  C=clear all  B=backspace  S=save  F=toggle flip")
    print("  Tip: lift your hand between letters to type the same letter twice")

    flip = USE_FLIP

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        if flip:
            frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = hand_detector.detect(mp_image)

        predicted, conf, bbox = None, 0.0, None

        if detection_result.hand_landmarks:
            hand_lms = detection_result.hand_landmarks[0]
            crop, bbox = crop_hand(frame, hand_lms)
            if crop is not None and crop.size > 0:
                tensor = preprocess(crop).to(device)
                with torch.no_grad():
                    probs = torch.softmax(model(tensor), dim=-1)[0]
                    conf, idx = probs.max(0)
                    conf = float(conf)
                    predicted = CLASS_NAMES[int(idx)]

                new_letter = stabilizer.update(predicted, conf)
                if new_letter:
                    builder.add(new_letter)
                    print(f"  + {new_letter}  ->  \"{builder.text}\"")

                draw_landmarks_simple(frame, hand_lms)
            else:
                stabilizer.update(None, 0.0)
        else:
            stabilizer.update(None, 0.0)

        builder.tick()

        fps = 1.0 / max(1e-6, time.time() - t0)
        fps_hist.append(fps)
        avg_fps = sum(fps_hist) / len(fps_hist)

        # ---------- DRAW ----------
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # top bar
        cv2.putText(frame,
                    f"Hand: {'YES' if bbox else 'no'}   FPS: {avg_fps:.1f}   flip={flip}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # current letter + confidence
        if predicted:
            color = (0, 255, 0) if conf > 0.8 else (0, 200, 255)
            cv2.putText(frame, f"{predicted} ({conf*100:.0f}%)",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3)

        # flash feedback for DEL / CLEARED
        if time.time() < builder.flash_until:
            cv2.putText(frame, builder.flash_text,
                        (frame.shape[1] // 2 - 100, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 4)

        # main text banner (bottom)
        banner = f">>> {builder.text}"
        if len(banner) > 60:
            banner = "..." + banner[-57:]
        cv2.putText(frame, banner,
                    (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        # shortcuts hint
        cv2.putText(frame, "Q:quit  C:clear  B:backspace  S:save  F:flip",
                    (10, frame.shape[0] - 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ASL Translator", frame)

        # ---------- KEY EVENTS ----------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            builder.clear()
            stabilizer.reset()
            print("  [cleared all]")
        elif key == ord('b'):
            builder.backspace(flash=True)
            print(f"  [backspace] -> \"{builder.text}\"")
        elif key == ord('s'):
            fname = f"screenshot_{int(time.time())}.png"
            cv2.imwrite(fname, frame)
            print(f"  [saved {fname}]")
        elif key == ord('f'):
            flip = not flip
            print(f"  flip = {flip}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nFinal text: \"{builder.text}\"")


if __name__ == "__main__":
    main()