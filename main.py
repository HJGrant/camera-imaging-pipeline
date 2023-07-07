import matplotlib.pyplot as plt
from classes.image_processing import imageProcessing
import cv2 as cv


#path = r'C:\\Users\\Hamish\Documents\\E4E\\Fishsense\\raw-jpg-framework\\data\\flat_port_check.dng'
path = r'C:\\Users\\Hamish\\Documents\\E4E\\Fishsense\\P6300001.ORF'
img1 = imageProcessing(path, True)


img1.linearization()
img1.demosaic()
img1.denoising()
img1.colorSpace()
img1.whiteBalance()
img1.exposureCompensation()

img1.imageResize()


cv.imshow("urer", img1.getLastImage())
k = cv.waitKey(0)
cv.destroyAllWindows()

#fig, axs = plt.subplots(1,2)
#axs[0].imshow(img1.getFirstImage())
#axs[1].imshow(img1.getLastImage())


#print(raw_img.shape)