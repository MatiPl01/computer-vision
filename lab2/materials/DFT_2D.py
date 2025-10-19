import numpy as np
import matplotlib.pyplot as plt
import cmath

import cv2


def DFT2D(image):
    data = np.asarray(image)
    # M, N = image.size # (img x, img y)
    M, N = image.shape[:2]
    dft2d = np.zeros((M, N), dtype=complex)
    for k in range(M):
        for l in range(N):
            sum_matrix = 0.0
            for m in range(M):
                for n in range(N):
                    e = cmath.exp(-2j * np.pi * ((k * m) / M + (l * n) / N))
                    sum_matrix += data[m, n] * e
            dft2d[k, l] = sum_matrix
    return dft2d


# img = PIL.Image.open("flowers_01.jpg")
# img2 = img.resize((50,50))

# read an image
img = cv2.imread("materials/Lab_02_students/data/messi5.jpg")

# create cropped grayscale image from the original image
img2_ = cv2.cvtColor(img[50:100, 50:100], cv2.COLOR_BGR2GRAY)

img2 = img2_[:, :, np.newaxis]

print("img2.shape={}".format(img2.shape))

plt.imshow(img2, cmap="gray")
plt.show()

dft = DFT2D(img2)

dft.real[0, 0] = 0
plt.imshow(dft.real)
plt.show()

print(dft.real[:8, :8])
