import cv2
import rawpy
import numpy as np

with rawpy.imread(r'C:\\Users\\Hamish\\Documents\\E4E\\Fishsense\\P6300001.ORF') as raw:

    raw_img = raw.raw_image.copy()
    #raw_img = raw_img / (2**12-1) * 255
    #raw_norm = cv2.normalize(raw_img, None, 0, 255,
        #cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    raw_norm = raw_img.astype(np.uint8)
    #raw_img_eq = cv2.equalizeHist(raw_img.astype(np.uint8))

    #pattern = raw.raw_pattern()

    # colour = cv2.cvtColor(raw_img, cv2.COLOR_BAYER_BG2BGR)

    #resized = cv2.resize(raw_img_eq, (1200,900))

    # cv2.imshow("burer", resized)

    colour = cv2.cvtColor(raw_norm, cv2.COLOR_BayerGB2GRAY)


    resized_eq = cv2.resize(colour,(1508,2020))
    cv2.imshow("urer", colour)

    
    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

    print(np.shape(raw_img))
    #print(pattern)