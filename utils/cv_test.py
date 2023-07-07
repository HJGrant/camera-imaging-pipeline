import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import rawpy as rp

path = r'C:\\Users\\Hamish\Documents\\E4E\\Fishsense\\raw-jpg-framework\\data\\flat_port_check.dng'

img1 = cv.imread(path)
img2 = rp.imread(path).raw_image

plt.imshow(img2)

plt.show()