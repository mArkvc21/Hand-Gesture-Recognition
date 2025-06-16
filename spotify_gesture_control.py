import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import spotipy
from spotipy.oauth2 import SpotifyOAuth

# --- Spotify Setup ---
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="62beb5f2d08a46b381b757140a2c3509",              # Replace this
    client_secret="fc2491f667924c5a80e79a4c5328bcea",      # Replace this
    redirect_uri="http://127.0.0.1:8888/callback",
    scope="user-modify-playback-state user-read-playback-state"
))

devices = sp.devices()
if not devices['devices']:
    print("⚠️  Start playing something on Spotify to activate device.")

# --- MediaPipe setup ---
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def set_volume(percent):
    percent = int(np.clip(percent, 0, 100))
    # MacOS volume change
    pyautogui.press('volumeup') if percent > 50 else pyautogui.press('volumedown')

def get_finger_states(lmList):
    fingers = []
    tipIds = [4, 8, 12, 16, 20]
    if lmList[tipIds[0]][0] > lmList[tipIds[0]-1][0]:
        fingers.append(1)  # Thumb
    else:
        fingers.append(0)
    for id in range(1, 5):
        if lmList[tipIds[id]][1] < lmList[tipIds[id]-2][1]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    lmList = []

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            h, w, _ = img.shape
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

        if len(lmList) >= 21:
            fingers = get_finger_states(lmList)

            # Volume control (thumb and index)
            x1, y1 = lmList[4]
            x2, y2 = lmList[8]
            length = math.hypot(x2 - x1, y2 - y1)
            vol_percent = np.interp(length, [30, 200], [0, 100])
            set_volume(vol_percent)
            cv2.putText(img, f'Vol: {int(vol_percent)}%', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # Gesture-based media control
            if fingers == [0, 1, 0, 0, 0]:  # Only index finger
                sp.pause_playback() if sp.current_playback()['is_playing'] else sp.start_playback()
                cv2.putText(img, "Play/Pause", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            elif fingers == [0, 1, 1, 0, 0]:  # Index + Middle
                sp.next_track()
                cv2.putText(img, "Next Track", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            elif fingers == [1, 0, 0, 0, 0]:  # Only thumb
                sp.previous_track()
                cv2.putText(img, "Previous Track", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("🎵 Gesture Control (Spotify + Volume)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
