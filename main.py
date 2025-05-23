import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from CNN import FaceRecognitionCNN

# Define transformation (updated for RGB normalization)
transform = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Updated for 3 channels
])

# Define the number of classes
class_names = ['Deep', 'Rohan', 'Pradip']
num_classes = len(class_names)  # Ensuring num_classes matches the length of class_names

# Set device for model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load trained weights
model = FaceRecognitionCNN(num_classes).to(device)
model.load_state_dict(torch.load("face_recognition_cnn.pth", map_location=torch.device('cpu')))
model.eval()

# Load the pre-trained face cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the webcam capture
capture = cv2.VideoCapture(0)

# Start the loop for real-time face detection and recognition
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    # Loop through all detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # Convert the face to PIL image and apply transformations
        face = Image.fromarray(face)
        face = transform(face)
        face = face.unsqueeze(0)  # Add batch dimension

        # Perform the prediction with no gradient computation
        with torch.no_grad():
            prediction = model(face.to(device))
            predicted_class_idx = torch.argmax(prediction, dim=1).item()
            
            # Get the class label (and handle out-of-range predictions)
            if predicted_class_idx < len(class_names):
                label = class_names[predicted_class_idx]
            else:
                label = "Unknown"

            # Optionally show confidence score
            confidence = torch.softmax(prediction, dim=1)[0][predicted_class_idx].item()
            label = f"{label} ({confidence:.2f})"

        # Draw rectangle around face and put the label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with face recognition
    cv2.imshow("Face Recognition", frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any open windows
capture.release()
cv2.destroyAllWindows()