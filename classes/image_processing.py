import numpy as np
import rawpy
import matplotlib.pyplot as plt
import cv2 as cv


class imageProcessing():

    def __init__(self, path, colour):
        self.raw = rawpy.imread(path)
        self.img = [self.raw.raw_image.copy()]
        self.colour = colour

    def getImages(self):
        return self.img
    
    def getFirstImage(self):
        return self.img[0]

    def getLastImage(self):
        return self.img[-1]
    
    def imageResize(self):
        scale_percent = 20 # percent of original size
        width = int(self.img[-1].shape[1] * (scale_percent / 100))
        height = int(self.img[-1].shape[0] * (scale_percent / 100))
        dim = (width, height)   

        resized = cv.resize(self.img[-1], dim, interpolation = cv.INTER_AREA)

        self.img.append(resized)

        return self.img[-1]

    def linearization(self):
        buf = ((self.img[-1] - self.img[-1].min()) * (1/(self.img[-1].max() - self.img[-1].min()) * 255)).astype('uint8')
        self.img.append(buf)
        return self.img[-1]

    def demosaic(self):
        buf = cv.demosaicing(self.img[-1], cv.COLOR_BayerGB2BGR) 
        self.img.append(buf)
        return self.img[-1]

    def lens_correction(self):
        h, w = self.img[-1].shape[:2]
        mtx = np.load('camera_matrix.npy')
        dist = np.load('distortion_coefficient.npy')
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        # undistort
        dst = cv.undistort(self.img[-1], mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]

        return dst

    def denoising(self):
        buf = cv.fastNlMeansDenoisingColored(self.img[-1], 100, 10, 7, 21)
        self.img.append(buf)
        return self.img[-1]
    
    def colorSpace(self):
        #buf = cv.normalize(self.img[-1], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        #buf = cv.convertTo(self.img[-1], cv.CV_8U, 1.0/255)
        #buf = cv.equalizeHist(buf)
        buf = self.img[-1].astype(np.uint8)
        print(buf.shape)

        if self.colour == True:
            buf = cv.cvtColor(buf, cv.COLOR_BGR2RGB)
        else:
            buf = cv.cvtColor(buf, cv.COLOR_BGR2GRAY)

        self.img.append(buf)
        return self.img[-1]
    
    def whiteBalance(self):
        current_image = self.img[-1]

        b, g, r = cv.split(current_image)
        r_avg = cv.mean(r)[0]
        g_avg = cv.mean(g)[0]
        b_avg = cv.mean(b)[0]

        k = (r_avg + g_avg + b_avg) / 3
        kr = k / r_avg
        kg = k / g_avg
        kb = k / b_avg  

        r = cv.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
        g = cv.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
        b = cv.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)

        balance_img = cv.merge([b, g, r])

        self.img.append(balance_img)

        return self.img[-1]
    
    def exposureCompensation(self):
        rows, cols, c = self.img[-1].shape
        mask1 = np.ones((rows, cols))

        exp1 = cv.detail.ExposureCompensator()
        exp1.apply(0, (0,0), self.img[-1], 1)  