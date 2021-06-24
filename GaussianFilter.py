import cv2 as cv
import numpy as np


def getLogImage(img):
    c = 255 / np.log(1 + np.max(img))
    log_image = c * (np.log(img + 1))
    log_image = np.array(log_image, np.uint8)
    return log_image

def resize(img, scale_pct) :
    ht = int(img.shape[1]*scale_pct)
    wt = int(img.shape[0]*scale_pct)
    I = cv.resize(img, (wt, ht), interpolation=cv.INTER_AREA)
    return I


I = cv.imread("./images/pic.jpg")

# apply Gaussian filter on resized image
Ig = cv.GaussianBlur(I, (49, 49), cv.BORDER_DEFAULT)
Ie = getLogImage(I) - getLogImage(Ig)

print(Ie.shape, I.shape)

cv.imshow("I", I)
cv.imshow("Ig", Ig)
cv.imshow("Ie", Ie)
cv.waitKey(0)
