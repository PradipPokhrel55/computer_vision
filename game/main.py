import cv2
import tkinter as tk
from PIL import Image, ImageTk
import pyautogui
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Setup Tkinter window
root = tk.Tk()
root.overrideredirect(True)  # Remove window borders
root.wm_attributes("-topmost", True)  # Always on top
root.wm_attributes("-transparent", True)  # macOS compatible transparency
root.config(bg='systemTransparent')      # Transparent background on macOS
# If 'systemTransparent' doesn't work, comment above line and uncomment below:
# root.attributes('-alpha', 0.7)  # Semi-transparent window (70% opacity)
# root.config(bg='black')

canvas = tk.Canvas(root, width=320, height=240, highlightthickness=0, bg='black')
canvas.pack()
root.geometry("+100+100")  # Position window on screen

last_press_time = time.time()
delay = 0.5  # Delay between key presses

def is_palm_open(hand_landmarks, img_height):
    """
    Simple heuristic to check if palm is open or closed.
    Counts extended fingers (index, middle, ring, pinky).
    Returns True if palm open (3 or more fingers extended), else False.
    """
    tips_ids = [8, 12, 16, 20]
    mcps_ids = [5, 9, 13, 17]

    extended_count = 0
    for tip_id, mcp_id in zip(tips_ids, mcps_ids):
        tip = hand_landmarks.landmark[tip_id]
        mcp = hand_landmarks.landmark[mcp_id]

        tip_y = tip.y * img_height
        mcp_y = mcp.y * img_height

        if tip_y < mcp_y:  # finger extended (tip above mcp)
            extended_count += 1

    return extended_count >= 3

def update_frame():
    global last_press_time

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            label = handedness.classification[0].label  # 'Left' or 'Right'

            current_time = time.time()
            if current_time - last_press_time > delay:
                palm_open = is_palm_open(hand_landmarks, h)

                if label == 'Right':
                    if palm_open:
                        pyautogui.press("up")
                        print("Right hand palm open → Up")
                    else:
                        pyautogui.press("down")
                        print("Right hand palm closed → Down")
                elif label == 'Left':
                    if palm_open:
                        pyautogui.press("left")
                        print("Left hand palm open → Left")
                    else:
                        pyautogui.press("right")
                        print("Left hand palm closed → Right")

                last_press_time = current_time

    # Convert OpenCV frame to Tkinter-compatible image
    frame = cv2.resize(frame, (320, 240))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    imgtk = ImageTk.PhotoImage(image=img)

    canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
    canvas.imgtk = imgtk  # Keep reference

    root.after(10, update_frame)

# Start the update loop
update_frame()
root.mainloop()

# Release resources on exit
cap.release()
