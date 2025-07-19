import cv2
import numpy as np
import mediapipe as mp


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0)
canvas = None
prev_x, prev_y = 0, 0

def fingers_up(hand_landmarks):
    fingers = []

    if hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x:
        fingers.append(1)
    else:
        fingers.append(0)

    for id in [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP]:
        if hand_landmarks.landmark[id].y < hand_landmarks.landmark[id - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = hand_landmarks.landmark
            fingers = fingers_up(hand_landmarks)
            x = int(lm_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            y = int(lm_list[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            if fingers[1] == 1 and all(f == 0 for i, f in enumerate(fingers) if i != 1):
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)
                prev_x, prev_y = x, y
            elif all(f == 0 for f in fingers):
                cv2.circle(frame, (x, y), 20, (0, 0, 255), -1)
                cv2.circle(canvas, (x, y), 20, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0
            else:
                prev_x, prev_y = 0, 0
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    frame = cv2.add(frame, canvas)
    cv2.imshow("Virtual Whiteboard", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
