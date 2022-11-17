import cv2
import numpy as np


img_colour = cv2.imread('D:/GitHub/recognise_uno_cards/images/screws.png')  # open the saved image in colour
img = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)   # convert to B/W
img_sm = cv2.blur(img, (5, 5))         # smoothing
thr_value, img_th = cv2.threshold(img_sm, 0, 255, cv2.THRESH_OTSU)   # binarisation
kernel = np.ones((5, 5), np.uint8)
img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection
contours, hierarchy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny
cv2.drawContours(img_colour, contours, -1, (0, 255, 0), 1)         # paint contours on top of original coloured mage
cv2.imshow('picture', img_colour)
cv2.imshow('edges', img_canny)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

print(len(contours))