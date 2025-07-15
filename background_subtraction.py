# Background Subtraction from non-changing pixels

import cv2
import numpy as np

# Path to your video file
video_path = 'Tower_Deploy_450x_color.mp4'

# Open video file
cap = cv2.VideoCapture(video_path)

# Capture first N frames to estimate static background
N = 30
frames = []

print("Capturing background from first {} frames...".format(N))
for _ in range(N):
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# Stack into numpy array and compute median background (in color)
stack = np.stack(frames, axis=0)
background = np.median(stack, axis=0).astype(np.uint8)

# Reset video to the beginning
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Starting background subtraction...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Compute absolute difference from background
    diff = cv2.absdiff(frame, background)

    # Convert to grayscale then threshold
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

    # Optional: apply mask to the frame to highlight moving regions
    foreground = cv2.bitwise_and(frame, frame, mask=fg_mask)

    # Show result
    cv2.imshow('Foreground Mask', fg_mask)
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Only', foreground)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

