import numpy as np
import cv2 as cv


img = cv.imread('/home/eskr/Documents/missing data summer school DTU/MissingDataChallenge_team13/MissingDataOpenData/originals/00000001_000.jpg')

mask = cv.imread('/home/eskr/Documents/missing data summer school DTU/MissingDataChallenge_team13/MissingDataOpenData/masked/00000001_000_stroke_masked.png', cv.IMREAD_GRAYSCALE)

dst = cv.inpaint(img, mask, 3, cv.INPAINT_NS)

cv.imshow('dst', dst)
cv.waitKey(0)
cv.destroyAllWindows()
