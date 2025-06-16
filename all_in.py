import cv2
import mediapipe as mp
import numpy as np
import math
import os
import time
import pyautogui

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def set_volume(percent):
    percent = int(np.clip(percent, 0, 100))
    os.system(f"osascript -e 'set volume output volume {percent}'")

def detect_media_gesture(landmarks):
    finger_tips = [8, 12, 16, 20]
    fingers = []

    if landmarks[4][0] < landmarks[3][0]:
        fingers.append(1)
    else:
        fingers.append(0)


    for tip in finger_tips:
        if landmarks[tip][1] < landmarks[tip - 2][1]:
            fingers.append(1)
        else:
            fingers.append(0)

    if fingers == [0, 1, 0, 0, 0]:
        return "Play/Pause"
    elif fingers == [0, 1, 1, 0, 0]:
        return "Next"
    elif fingers == [1, 0, 0, 0, 0]:
        return "Previous"
    elif fingers == [0, 0, 0, 0, 0]:
        return "Stop"
    else:
        return "None"

prev_gesture = None
last_time = 0

while True:
    success, img = cap.read()
    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            lmList = []
            h, w, _ = img.shape
            for lm in handLms.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if len(lmList) >= 9:
                x1, y1 = lmList[4] 
                x2, y2 = lmList[8]  

                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

                length = math.hypot(x2 - x1, y2 - y1)
                vol_percent = np.interp(length, [30, 200], [0, 100])
                set_volume(vol_percent)

                vol_bar = np.interp(length, [30, 200], [400, 150])
                cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 2)
                cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)
                cv2.putText(img, f'{int(vol_percent)} %', (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 0), 3)

            gesture = detect_media_gesture(lmList)
            if gesture != prev_gesture or time.time() - last_time > 1.5:
                if gesture == "Play/Pause":
                    pyautogui.press("space")
                    print("Media: Play/Pause")
                elif gesture == "Next":
                    pyautogui.hotkey("ctrl", "right")
                    print("Media: Next Track")
                elif gesture == "Previous":
                    pyautogui.hotkey("ctrl", "left")
                    print("Media: Previous Track")
                elif gesture == "Stop":
                    pyautogui.press("space")
                    print("Media: Stop")

                prev_gesture = gesture
                last_time = time.time()

            cv2.putText(img, f'Gesture: {gesture}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Volume + Media Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
