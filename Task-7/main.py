import cv2
from tracker import *  # Import the tracker module

# Create tracker object
tracker = EuclideanDistTracker()

cap = cv2.VideoCapture("highway.mp4")  # Initialize video capture

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

while True:
   ret, frame = cap.read()  # Read the frame from the video capture

   height, width, _ = frame.shape  # Get frame dimensions

   # Extract Region of interest
   roi = frame[340: 720, 500: 800]  # Crop the region of interest

   # 1. Object Detection
   mask = object_detector.apply(roi)  # Apply background subtraction to ROI
   _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)  # Threshold the mask
   contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

   detections = []  # Initialize detections list
   for cnt in contours:
       # Calculate area and remove small elements
       area = cv2.contourArea(cnt)
       if area > 100:
           # Find bounding box for the contour
           x, y, w, h = cv2.boundingRect(cnt)

           detections.append([x, y, w, h])  # Add bounding box to detections

   # 2. Object Tracking
   boxes_ids = tracker.update(detections)  # Update the tracker with new detections

   for box_id in boxes_ids:
       x, y, w, h, id = box_id  # Unpack box_id

       # Draw bounding box and id on the roi
       cv2.putText(roi, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
       cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 3)

   cv2.imshow("roi", roi)  # Display the roi
   cv2.imshow("Frame", frame)  # Display the frame
   cv2.imshow("Mask", mask)  # Display the mask

   key = cv2.waitKey(30)
   if key == 27:
       break

cap.release()  # Release the video capture
cv2.destroyAllWindows()  # Close all windows