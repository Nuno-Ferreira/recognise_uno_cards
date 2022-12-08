import cv2
import numpy as np
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Users\nunom\AppData\Local\Programs\Python\Python310\Lib\site-packages\pytesseract\pytesseract.py"

folder_path = r'D:/GitHub/recognise_uno_cards/images/'

card = 'b0'
card_colors = ['b', 'g', 'y', 'r']
#implement for loop to go through all the images, use it to go through colors and numbers | implement wait time if needed to change cards
#for c in card_colors:
 # for i in range(9):
#    card[c][i]


img_colour = cv2.imread(folder_path + card + '.jpg')  # open the saved image in colour
img = cv2.cvtColor(img_colour, cv2.COLOR_BGR2GRAY)   # convert to B/W
#img_sm = cv2.blur(img, (1, 1))         # smoothing
thr_value, img_th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)   # binarisation
kernel = np.ones((5, 5), np.uint8)
img_close = cv2.morphologyEx(img_th, cv2.MORPH_OPEN, kernel)      # morphology correction
img_canny = cv2.Canny(img_close, 50, 100)                          # edge detection
estimatedThreshold, thresholdImage=cv2.threshold(img,160,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(thresholdImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   # extract contours on binarised image, not on canny

data = []

for i, c in enumerate(contours):         # loop through all the found contours
    if hierarchy[0,i,3] != -1 and hierarchy[0,i,3] != 0:
      print(i, ':', hierarchy[0, i])          # display contour hierarchy
      print('length: ', len(c))               # display numbr of points in contour c
      perimeter = cv2.arcLength(c, True)     # perimeter of contour c (curved length)
      print('perimeter: ', perimeter)               
      epsilon = 0.02*perimeter    # parameter of polygon approximation: smaller values provide more vertices
      vertex_approx = len(cv2.approxPolyDP(c, epsilon, True))     # approximate with polygon
      print('approx corners: ', vertex_approx, '\n')                    # number of vertices
      if len(contours[i]) >= 5:
        ellipse = cv2.fitEllipse(c)
      (xc,yc),(d1,d2),angle = ellipse

      axes_ratio = round(d1/d2, 3)
      print('axes ratio: ', axes_ratio, '\n')
      cv2.drawContours(img_colour, [c], 0, (0, 255, 0), 2)   # paint contour c
      cv2.putText(img_colour, str(i), (c[0, 0, 0]+20, c[0, 0, 1]+30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))    # identify contour c
      #[x,y,w,h] = cv2.boundingRect(c)
    # cv2.rectangle(img_colour, (x,y), (x+w,y+h), (255, 0, 0), 2)
      cv2.ellipse(img_colour, ellipse, (255, 0, 0), 2)

    # add a function/if statement to get rid of all the other ellipses
    #if hierarchy[3] != -1:


    
    #sample = [] #add the features here so that I can use data to extract all the features to do machine learning 
    #data.append(sample)
  
#text = pytesseract.image_to_string(thresholdImage) # config="--psm 6"
#tessedit_char_whitelist=0123456789
#print(text)

cv2.namedWindow('picture', cv2.WINDOW_NORMAL)
cv2.imshow('picture',img_colour)
#<cv2.imshow('gray', img)
#cv2.imshow('Image', thresholdImage)
key = cv2.waitKey(0)
cv2.destroyAllWindows()

#print(data)
# label = [ 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9] #add each number times 4 for each colour
# then add labels to each cards within data
# can use filename[0] to add label
