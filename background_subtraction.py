# Background Subtraction from non-changing pixels

import cv2
import numpy as np
from scipy import stats

# Path to your video file
video_path = 'Tower_Deploy_450x_color.mp4'
cap = cv2.VideoCapture(video_path)

# Number of frames to build background
N = 40
frames = []

print("Capturing background from first {} frames...".format(N))
for _ in range(N):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Stack into 4D array: (N, H, W, 3)
stack = np.stack(frames, axis=0)

# Use scipy mode to compute static background
reshaped = stack.reshape(N, -1, 3)
mode, _ = stats.mode(reshaped, axis=0, keepdims=False)
background = mode.reshape(stack.shape[1:]).astype(np.uint8)

# Reset video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Define morphological kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

print("Starting background subtraction...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: slight blur to suppress noise before subtraction
    # blurred_frame = cv2.GaussianBlur(frame, (13, 13), 0)
    # blurred_bg = cv2.GaussianBlur(background, (13, 13), 0)

    # diff = cv2.absdiff(blurred_frame, blurred_bg)
    diff = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Increase threshold to suppress weak differences
    _, fg_mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)

    # Morphological operations to remove small noise
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    # Extract cleaned-up foreground
    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Foreground Only', foreground)
    cv2.imshow('Original', frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()