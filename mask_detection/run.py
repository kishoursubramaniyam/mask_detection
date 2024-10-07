from keras.models import load_model # TensorFlow is required for Keras to work
import cv2 # Install opencv-python.
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Load the pre-trained face detection model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

while True:
    # Grab the webcamera's image
    ret, image = camera.read()

    # Convert to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Loop through each detected face
    for (x, y, w, h) in faces:
        # Extract the face region for prediction
        face_region = image[y:y+h, x:x+w]
        face_resized = cv2.resize(face_region, (224, 224), interpolation=cv2.INTER_AREA)

        # Make the image a numpy array and reshape it to the model's input shape
        face_array = np.asarray(face_resized, dtype=np.float32).reshape(1, 224, 224, 3)
        # Normalize the image array
        face_array = (face_array / 127.5) - 1

        # Predict using the model
        prediction = model.predict(face_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Remove trailing whitespace or newline
        confidence_score = prediction[0][index]

        # Check the class and print corresponding message
        if index == 0:
            print("Mask detected")
            color = (0, 255, 0)  # Green for mask detected
            label = f"Mask: {confidence_score * 100:.2f}%"
        else:
            print("Mask not detected")
            color = (0, 0, 255)  # Red for no mask detected
            label = f"No Mask: {confidence_score * 100:.2f}%"

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)

        # Put label on top of the rectangle
        cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Show the image with the rectangles
    cv2.imshow("Webcam Image", image)

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII code for the 'esc' key on your keyboard
    if keyboard_input == 27:
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()
