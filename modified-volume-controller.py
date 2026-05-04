import argparse
import logging
import math
import os
import time
import statistics

import cv2
import mediapipe as mp
import pyautogui

# Optional PulseAudio
try:
    import pulsectl
    PULSE_AVAILABLE = True
except ImportError:
    PULSE_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#  CONFIG 
class Config:
    MODEL_PATH = 'hand_landmarker.task'
    VOL_MIN_LEN = 25
    VOL_MAX_LEN = 220
    PINCH_THRESH = 40
    CLICK_CD = 0.25
    VOL_SMOOTH = 0.8
    FPS_ALPHA = 0.9

#  VOLUME 
class VolumeController:
    def __init__(self):
        self.last_vol = 50
        if PULSE_AVAILABLE:
            self.pulse = pulsectl.Pulse('gesture-volume')
        else:
            self.pulse = None

    def set(self, vol):
        vol = max(0, min(100, vol))

        # Avoid unnecessary updates
        if abs(vol - self.last_vol) < 2:
            return

        self.last_vol = vol

        if self.pulse:
            try:
                sinks = self.pulse.sink_list()
                if sinks:
                    self.pulse.volume_set_all_chans(sinks[0], vol / 100)
                return
            except Exception:
                pass

        os.system(f"amixer set Master {int(vol)}% >/dev/null 2>&1")

#  UTILS 
def distance(lm1, lm2, w, h):
    x1, y1 = int(lm1.x * w), int(lm1.y * h)
    x2, y2 = int(lm2.x * w), int(lm2.y * h)
    return math.hypot(x2 - x1, y2 - y1), (x1, y1), (x2, y2)

def fingers_up(hand):
    fingers = [False] * 5

    # Thumb (orientation aware)
    is_right = hand[0].x > 0.5
    fingers[0] = hand[4].x > hand[3].x if is_right else hand[4].x < hand[3].x

    # Other fingers
    for i, (tip, pip) in enumerate([(8,6), (12,10), (16,14), (20,18)]):
        fingers[i+1] = hand[tip].y < hand[pip].y

    return fingers

#  MAIN 
def main(model_path, camera_id):

    # MediaPipe setup
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandOptions = mp.tasks.vision.HandLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = HandOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        num_hands=1
    )

    detector = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(camera_id)
    screen_w, screen_h = pyautogui.size()

    volume = VolumeController()

    mode = "IDLE"
    smooth_vol = 50
    last_click = 0

    prev_time = time.time()
    fps_avg = 0

    logger.info("Gesture system started")

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        results = detector.detect_for_video(mp_img, int(time.time() * 1000))

        if results.hand_landmarks:
            hand = results.hand_landmarks[0]

            fingers = fingers_up(hand)
            count = sum(fingers)

            # ===== MODE =====
            if count == 0:
                mode = "IDLE"

            elif count == 1:
                mode = "MOUSE"

                idx = hand[8]
                x = int(idx.x * screen_w)
                y = int(idx.y * screen_h)

                pyautogui.moveTo(x, y)

                # Click
                dist_val, _, _ = distance(hand[4], hand[8], w, h)
                now = time.time()

                if dist_val < Config.PINCH_THRESH and (now - last_click) > Config.CLICK_CD:
                    pyautogui.click()
                    last_click = now

            elif count == 2:
                mode = "VOLUME"

                dist_val, p1, p2 = distance(hand[4], hand[8], w, h)

                vol = (dist_val - Config.VOL_MIN_LEN) * 100 / (Config.VOL_MAX_LEN - Config.VOL_MIN_LEN)
                vol = max(0, min(100, vol))

                smooth_vol = Config.VOL_SMOOTH * smooth_vol + (1 - Config.VOL_SMOOTH) * vol
                volume.set(smooth_vol)

                cv2.line(img, p1, p2, (0, 255, 0), 2)

        # ===== FPS =====
        now = time.time()
        fps = 1 / (now - prev_time) if now != prev_time else 0
        prev_time = now
        fps_avg = Config.FPS_ALPHA * fps_avg + (1 - Config.FPS_ALPHA) * fps

        # ===== UI =====
        cv2.putText(img, f'Mode: {mode}', (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(img, f'FPS: {int(fps_avg)}', (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(img, f'Vol: {int(smooth_vol)}%', (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)

        cv2.imshow("Gesture Controller", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

#  ENTRY 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=Config.MODEL_PATH)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    main(args.model, args.camera)