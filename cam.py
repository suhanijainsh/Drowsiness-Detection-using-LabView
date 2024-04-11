import cv2
import time
import pandas as pd

interval = 300
cap = cv2.VideoCapture(0)

# Create a list to store the file paths
file_paths = []

for i in range(25):
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    filename = f"temp{i}.png"
    cv2.imwrite(filename, frame)
    file_paths.append(filename)
    time.sleep(interval / 1000.0)

# Create a Pandas DataFrame and save it to an XLSX file
df = pd.DataFrame({'File Path': file_paths})
df.to_excel('image_paths.xlsx', index=False)