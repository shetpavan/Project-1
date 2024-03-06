import cv2
import numpy as np

# Initialize the video capture object to capture video from the default camera (0)
cap = cv2.VideoCapture(0)

# Read the first frame from the camera and flip it horizontally
_, prev = cap.read()
prev = cv2.flip(prev, 1)

# Read the second frame from the camera and flip it horizontally
_, new = cap.read()
new = cv2.flip(new, 1)

# Create an infinite loop to continuously process and display frames
while True:
   # Calculate the absolute difference between the previous and current frames
   diff = cv2.absdiff(prev, new)
   
   # Convert the difference frame from BGR to grayscale
   diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

   # Apply Gaussian blur to the difference frame to reduce noise
   diff = cv2.blur(diff, (5, 5))

   # Apply a binary threshold to the blurred difference frame
   _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)

   # Dilate the thresholded frame to connect adjacent contours
   thresh = cv2.dilate(thresh, None, 3)

   # Erode the thresholded frame to remove small noise and holes
   thresh = cv2.erode(thresh, np.ones((4, 4)), 1)

   # Find contours in the thresholded frame
   contour, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   # Draw a red circle on the previous frame to mark the center of the moving object
   cv2.circle(prev, (20, 200), 5, (0, 0, 255), -1)

   # Iterate through the found contours
   for contour in contour:
       # Calculate the area of the contour
       if cv2.contourArea(contour) > 30000:
           # Calculate the bounding rectangle and minimum enclosing circle of the contour
           (x, y, w, h) = cv2.boundingRect(contour)
           (x1, y1), rad = cv2.minEnclosingCircle(contour)

           # Convert the coordinates to integers
           x1 = int(x1)
           y1 = int(y1)

           # Draw a blue line from the center of the frame to the center of the moving object
           cv2.line(prev, (20, 200), (x1, y1), (255, 0, 0), 4)

           # Calculate and display the distance between the center of the frame and the center of the moving object
           cv2.putText(prev, "{}".format(int(np.sqrt((x1 - 20) ** 2 + (y1 - 200) ** 2))), (100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

           # Draw a green rectangle around the moving object
           cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 255, 0), 2)

           # Draw a red circle at the center of the moving object
           cv2.circle(prev, (x1, y1), 5, (0, 0, 255), -1)

   # Display the previous frame with the added visualizations
   cv2.imshow("orig", prev)

   # Update the previous frame with the current frame
   prev = new

   # Read the next frame from the camera
   _, new = cap.read()
   new = cv2.flip(new, 1)

   # Check if the user pressed the 'Esc' key
   if cv2.waitKey(1) == 27:
       # If the 'Esc' key is pressed, release the video capture object and close all windows
       break

# Release the
