import cv2
import math

cap = cv2.VideoCapture("test_2.mp4")  # Change 0 to a video file path if needed

pointsList = []
slow_motion_factor = 0.1  # Adjust this value to control playback speed (0.5 for half speed)


def mousePoints(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        size = len(pointsList)
        if size != 0 and size % 3 != 0:
            # Draw line and circle only if 2 points have been selected
            cv2.line(img, tuple(pointsList[round((size - 1) / 3) * 3]), (x, y), (0, 0, 255), 2)
            cv2.circle(img, (x, y), 5, (0, 0, 255), cv2.FILLED)
            pointsList.append([x, y])


def gradient(pt1, pt2):
    return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0])


def getAngle(pointsList):
    pt1, pt2, pt3 = pointsList[-3:]
    m1 = gradient(pt1, pt2)
    m2 = gradient(pt1, pt3)
    angR = math.atan((m2 - m1) / (1 + (m2 * m1)))
    angD = round(math.degrees(angR))
    cv2.putText(img, str(angD), (pt1[0] - 40, pt1[1] - 20), cv2.FONT_HERSHEY_COMPLEX,
                1.5, (0, 0, 255), 2)


while True:
    ret, img = cap.read()  # Capture frame from video
    if not ret:
        break

    # Slow-motion playback:
    # Reduce the number of frames displayed per second based on slow_motion_factor
    if cv2.waitKey(int(1 / slow_motion_factor)) & 0xFF == ord('q'):
        break

    if len(pointsList) % 3 == 0 and len(pointsList) != 0:
        getAngle(pointsList)

    cv2.imshow('Image', img)
    cv2.setMouseCallback('Image', mousePoints)

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()




