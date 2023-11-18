import cv2
import numpy as np
from PIL import Image

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the dress filter image
dress_filter = Image.open()

# Initialize the camera
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Adjust the dress size to fit the entire body
        dress_resized = dress_filter.resize((w, h))

        # Convert the dress filter image to OpenCV format
        dress_np = cv2.cvtColor(np.array(dress_resized), cv2.COLOR_RGBA2BGRA)

        # Blend the dress filter with the body
        alpha = 0.5  # Adjust transparency as needed
        overlay = frame.copy()
        cv2.addWeighted(dress_np, alpha, overlay[y:y+h, x:x+w], 1 - alpha, 0, overlay[y:y+h, x:x+w])
        frame[y:y+h, x:x+w] = overlay[y:y+h, x:x+w]

    # Display the frame
    cv2.imshow("Dress Filter", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
