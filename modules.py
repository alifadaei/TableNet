# Tesseract image segmentation modes: https://pyimagesearch.com/2021/11/15/tesseract-page-segmentation-modes-psms-explained-how-to-improve-your-ocr-accuracy/

# Phase 1: Extract table from image

# Relating projects and solutions:
# https://nanonets.com/blog/table-extraction-deep-learning/

# Part 1: design a model for character distribution in image
# Spoting colomns:
# I developed a code based on tesseract to calculate the character density of uniformly divided vertical segments in order to spot the columns.
# This model divides image into N vertical segments. So N is an arguable threshold that must be suitable.


from matplotlib import pyplot as plt
import numpy as np
import pytesseract


class TableImage:
    def __init__(self, IMG, N, Threshold) -> None:
        self.IMG = IMG
        self.THRESHOLD = Threshold
        self.IMAGEWIDTH = IMG.shape[1]
        self.IMAGEHEIGHT = IMG.shape[0]
        self.R = int(0.5 * self.IMAGEWIDTH // N)
        # x domain test
        self.X = np.linspace(0, [self.IMAGEWIDTH], num=N+1, dtype='int')[1:]
        self.calculateDensities()
        self.extractColumns()

    def calculateDensities(self):
        self.Y = self.getYArray()

    def getCharacterCount(self, x):
        cropped_img = self.IMG[:, x-2*self.R:x]
        recognized_text = pytesseract.image_to_string(
            cropped_img, lang='eng', config='--psm 6').replace(' ', '').replace('\n', '')
        return len(recognized_text)

    def getYArray(self):
        arr = []
        for x in self.X:
            characters_count = self.getCharacterCount(x[0])
            arr.append(characters_count)
        return np.array(arr)

    def printfn(self, showImage, showSegments, showDensities, showColumns):
        # plotting segments with corresponding char densities
        if showColumns:
            X = self.X
            for x in self.cols:
                plt.axvline(x=X[x]-self.R, color='green')
            plt.axvline(x=0, color='green')
            plt.axvline(x=self.IMAGEWIDTH, color='green')
        if showSegments:
            for x in self.X:
                plt.axvline(x=x[0], color='red')
        if showImage:
            plt.imshow(self.IMG)

        if showDensities:
            X = self.X
            Y = self.Y
            R = self.R
            MAX_Y = self.Y.max()
            if showImage:
                IMAGEHEIGHT = self.IMAGEHEIGHT
                plt.plot(X-R, IMAGEHEIGHT - Y / MAX_Y * 2/3*IMAGEHEIGHT)
            else:
                plt.plot(X-R, Y)
            plt.show()

    def extractColumns(self):
        Y = self.Y
        # for zero densities
        zeros = []
        for i in range(1, len(Y)-1):
            if Y[i] < Y[i-1] and Y[i] < self.THRESHOLD:
                zeros.append(i)
        self.cols = zeros
