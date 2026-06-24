import cv2
import random
import numpy as np

img = cv2.imread('New folder/bird.jpg', -1)

print(type(img))
print(img)
##print(img.shape)

#Accessing pixels
print(img[200][45:48])

#Changing colours
for i in range(100) :
    for j in range(int(img.shape[1]/2)) :
        img[i][j] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
cv2.imshow('changed', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Copy & paste
tag = img[200:350, 600:950]
img[0:150, 0:350] = tag
cv2.imshow('changed', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
