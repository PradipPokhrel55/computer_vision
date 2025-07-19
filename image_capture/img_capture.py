import cv2
import os

capture = cv2.VideoCapture(0)
save_path = "dataset/your_name/"
os.makedirs(save_path, exist_ok=True)

count = 0
while count < 100:  
    ret, frame = capture.read()
    if not ret:
        break
    cv2.imshow("Capturing", frame)
    cv2.imwrite(f"{save_path}/{count}.jpg", frame)
    count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
