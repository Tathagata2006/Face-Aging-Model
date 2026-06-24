import cv2


#Reading an image

img = cv2.imread('New folder/eye.jpg', -1)

# imread_color = -1
# imread_grayscale = 0
# imread_unchanged  = 1


#Showing/Displaying images

cv2.imshow('EYE', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#resize

img = cv2.resize(img, (1200, 1000))
img1 = img
img = cv2.resize(img, (0, 0), fx = 1.1, fy = 0.05)
cv2.imshow('Eye', img)
cv2.imshow('Eye1', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

#ROTATE

#(size adjustment and loading)
img2 = cv2.imread('New folder/elephant.jpg', 1)
img2 = cv2.resize(img2, (1000, 1000))
cv2.imshow('original', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('EleRotated', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Write / save a image

cv2.imwrite('New folder/new_elephant.png', img2)