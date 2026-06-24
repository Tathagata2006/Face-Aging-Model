#Corner Detection

import numpy as np
import cv2

img = cv2.imread('New folder/chessboard.png')
img = cv2.resize(img, (0, 0), fx = 0.75, fy = 0.75)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#goodFeaturesToTrack(image, number of corners, minimum quality(0-1), min euclidean distance)
corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
corners = np.int16(corners)
for corner in corners :
    x, y = corner.ravel() #ravel() flattens an np.ndarray
    cv2.circle(img, (x, y), 5, (255, 0, 0), -1)

for i in range(len(corners)-1) :
    for j in range(i+1, len(corners)) :
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        color = tuple(map(lambda x : int(x), np.random.randint(0, 255, size = 3)))
        cv2.line(img, corner1, corner2, color, 1)

cv2.imshow('frame', img)
cv2.waitKey(0)
cv2. destroyAllWindows()