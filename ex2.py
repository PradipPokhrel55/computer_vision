import cv2
import face_recognition
import mediapipe as mp


known_images = {
    "Pradip": "pradip.jpg",
    "Rohan": "rohan.jpg",
    "Deep": "deep.jpg"
}

known_encodings = []
known_names = []

for name, filepath in known_images.items():
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        known_encodings.append(encodings[0])
        known_names.append(name)

mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            top = max(y, 0)
            right = min(x + bw, w)
            bottom = min(y + bh, h)
            left = max(x, 0)

            face_location = [(top, right, bottom, left)]
            encodings = face_recognition.face_encodings(rgb, face_location)

            if encodings:
                matches = face_recognition.compare_faces(known_encodings, encodings[0])
                name = "Unknown"

                if True in matches:
                    match_index = matches.index(True)
                    name = known_names[match_index]

                cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
