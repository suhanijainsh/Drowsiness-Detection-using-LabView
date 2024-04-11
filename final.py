from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import pandas as pd
import os

# Function to calculate the eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

labout = ""  # Initialize labout as an empty string

# Load face detector and landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/suhanijain/Desktop/suhani/shape_predictor_68_face_landmarks.dat")

# Define facial landmarks indexes for eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize a list to store the results for each image
results = []

# Set the number of consecutive frames for drowsiness detection
CONSECUTIVE_FRAMES_THRESHOLD = 1
EYE_AR_THRESH = 0.3

# Specify the directory containing the images
image_directory = "/Users/suhanijain/Desktop/suhani/"

# Iterate through a range of numbers (adjust as needed)
for i in range(1, 25):
    # Construct the image path dynamically
    image_path = os.path.join(image_directory, f"temp{i}.png")

    # Check if the image exists
    if os.path.exists(image_path):
        # Reset the consecutive frames counter for each image
        consecutive_frames = 0

        # Read the image frame
        frame = cv2.imread(image_path)
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        rects = detector(gray, 0)

        # Loop over the face detections
        for rect in rects:
            # Determine facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Extract left and right eye coordinates
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]

            # Calculate eye aspect ratio for left and right eyes
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # Average the eye aspect ratio for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # Check if eyes are closed
            if ear < EYE_AR_THRESH:
                consecutive_frames += 1
                # Check if consecutive frames threshold is reached
                if consecutive_frames >= CONSECUTIVE_FRAMES_THRESHOLD:
                    labout = "drowsy"
                    break  # Exit the loop if eyes are drowsy for consecutive frames
            else:
                # Reset the consecutive frames counter if eyes are open
                consecutive_frames = 0

        # If loop completes without assigning "drowsy", set labout to "okay"
        if consecutive_frames < CONSECUTIVE_FRAMES_THRESHOLD:
            labout = "okay"
        results.append(labout)

def eyestat():
    return labout

# Concatenate the results list into a single string separated by commas
labout = ", ".join(results)

# Print the final result
print("Results:", labout)
