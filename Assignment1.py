#Jayneel Pratap Jain#
#101983237#
#jpjain@myseneca.ca#
#ASSIGNMENT-1#

import cv2
import numpy as np


image = np.ones((600, 600, 3), dtype=np.uint8) * 255
H, W = image.shape[:2]



cv2.ellipse(image, (300, 180), (90, 90 ), 120, 0 ,300, (0, 0, 255), -1) 
cv2.ellipse(image, (300,180), (30, 30), 120, 0, 300, (255, 255, 255), -1)


cv2.ellipse(image, (200, 360), (90, 90 ), 0, 0 ,300, (0, 255, 0), -1) 
cv2.ellipse(image, (200,360), (30, 30), 0, 0, 300, (255, 255, 255), -1)


cv2.ellipse(image, (400, 360), (90, 90 ), 300, 0 ,300, (255, 0, 0), -1) 
cv2.ellipse(image, (400,360), (30, 30), 300, 0, 300, (255, 255, 255), -1)

text = "OpenCV"
font = cv2.FONT_HERSHEY_COMPLEX
font_scale = 1.5
thickness = 6

(text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
x = (W - text_w) // 2
y = H - 20  

cv2.putText(image, text, (x, y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
