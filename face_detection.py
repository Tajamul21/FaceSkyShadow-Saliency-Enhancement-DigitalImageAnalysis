# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 21:47:08 2022
@author: Tajamul Ashraf
@author: Krishnapriya S
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema, argrelmax

# Load the cascade
face_cascade = cv2.CascadeClassifier(
    r'C:\Users\Krishnapriya S\Desktop\PHD\IITD\Course work\DIA\opencv-master\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread(r'C:\Users\Krishnapriya S\Desktop\PHD\IITD\Course work\DIA\pic4.jpg')
# Display
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.imwrite('inputimage.png', img)

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
# Draw rectangle around the faces
for (x, y, w, h) in faces:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img_crop = img[y:y + h, x:x + h]
    cv2.imshow("img_crop", img_crop)
    cv2.waitKey(0)
    cv2.imwrite('imgcrop.png', img_crop)
    break

# SKIN DETECTION

# Convert to LAB image
lab_img = cv2.cvtColor(img_crop, cv2.COLOR_BGR2LAB)
L, A, B = cv2.split(lab_img)

# Ellipse to threshold skin
A_arr = (A > 143) * (A - 143) + (A < 143) * (143 - A)
B_arr = (B > 148) * (B - 148) + (B < 148) * (148 - B)
thresh = (A_arr / 6.5) * 2 + (B_arr / 12) * 2

N = np.zeros(img_crop.shape[0:2], dtype='uint8')

# Normalise
thresh = (thresh - np.min(thresh)) / (np.max(thresh) - np.min(thresh))

norm_thresh = thresh * 255

# Threshold to create mask
for i in range(300):
    for j in range(300):
        if norm_thresh[i][j] <= 137:
            N[i][j] = 255
        else:
            N[i][j] = 0

cv2.imshow('Skin mask', N)
cv2.waitKey(0)
cv2.imwrite('skinmask.png', N)
# Get skin area
final_result = cv2.bitwise_and(img_crop, img_crop, mask=N)
cv2.imshow('Skin', final_result)

cv2.waitKey(0)
cv2.imwrite('skinimage.png', final_result)

# Histogram of skin
n, bins, p = plt.hist(final_result.ravel(), bins=256, range=[1, 256], fc='k', ec='k')  # calculating histogram

# %%
# Finding the terms b,d,m

x = np.linspace(np.min(final_result), np.max(final_result), num=255)

# Finding the maxima's in histogram
ind_max = argrelmax(n)
x_max = x[ind_max]
y_max = n[ind_max]
index_first_max = np.argmax(y_max)
maximum_y = y_max[index_first_max]

# dark mode intensity
d = x_max[index_first_max]

y_maxima = []
# %%
for i in range(index_first_max, len(y_max) - 1):
    if ((y_max[i] - y_max[i + 1]) < 0) and ((y_max[i + 1] - y_max[i + 2]) < 0):
        y_maxima.append(y_max[i])

# %%

local_min = []
l_min = []
for i in range(index_first_max, len(y_max) - 1):
    if ((y_max[i] - y_max[i + 1]) > 0):
        local_min.append(y_max[i + 1])
for i in range(len(local_min) - 1):
    if ((local_min[i] - local_min[i + 1]) < 0) and ((local_min[i + 1] - local_min[i + 2]) < 0) and (
            (local_min[i + 2] - local_min[i + 3]) < 0):
        l_min.append(local_min[i + 1])
        print(l_min)
        break

index_l_min = np.where(y_max == l_min)
m = x_max[index_l_min]

Second_mode = []

for i in range(index_l_min[0][0], len(y_max) - 1):
    Second_mode.append(y_max[i])
Second_mode = np.array(Second_mode)
index_Second_mode_max = np.where(y_max == Second_mode.max())
b = x_max[index_Second_mode_max]

# Convert and split into H,S,V components

final_result_1 = cv2.cvtColor(final_result, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(final_result_1)

# multiplicative factor
f = (b - d) / (m - d)
for i in range(v.shape[0]):
    for j in range(v.shape[1]):
        if v[i][j] < m:
            v[i][j] = v[i][j] * f[0]

final_result_hsv = cv2.merge([h, s, v])

y_maxima = np.array(y_maxima)
# pl.sh

# Corrected image
final_result_rgb = cv2.cvtColor(final_result_hsv, cv2.COLOR_HSV2RGB)
final_result_rgb = cv2.cvtColor(final_result_rgb, cv2.COLOR_RGB2BGR)

cv2.waitKey(0)
cv2.imwrite('enhancedrgb.png', final_result_rgb)

# Histogram of corrected image
plt.hist(final_result_rgb.ravel(), bins=256, range=[1, 256], fc='k', ec='k')  # calculating histogram

# exposure_corrected=np.uint8(final_result_rgb[:,:,:]*0.8)
# cv2.imshow('c',exposure_corrected)
# cv2.waitKey(0)

edge_preserved = cv2.bilateralFilter(final_result_rgb, 9, 75, 75)
cv2.imshow('edge_preservedskin', edge_preserved)
cv2.waitKey(0)

# Mask for non skin area
bgnd = cv2.bitwise_and(img_crop, img_crop, mask=255 - N)

# Combining background and skin
x1 = edge_preserved | bgnd
cv2.imshow('f', x1)
cv2.waitKey(0)
cv2.imwrite('edge_preservedimage.png', x1)

# Fix the skin back to image
for (x, y, w, h) in faces:
    # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    img[y:y + h, x:x + h] = x1
    cv2.imshow("Finalimage", img)
    cv2.waitKey(0)
    cv2.imwrite('final.png', img)
    break
