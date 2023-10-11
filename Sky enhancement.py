# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 22:15:12 2022

@author: Krishnapriya S
@author: Tajamul Ashraf
"""
import cv2
import math
import seaborn as sns
import skimage
from skimage import exposure
# Load the cascade

import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms

# INPUT IMAGE
img_ref = cv2.imread("blue-sky-bright-clouds-1254571.jpg")
cv2.imshow('img', img_ref)
cv2.imwrite('img_ref.png', img_ref)
cv2.waitKey(0)

img_ref_rgb = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
hsv_img_ref = cv2.cvtColor(img_ref_rgb, cv2.COLOR_RGB2HSV)

# MASK RANGE
light_sky = (80, 175, 104)
dark_sky = (115, 240, 230)
light_cloud = (75, 2, 155)
dark_cloud = (115, 125, 255)
mask1_ref = cv2.inRange(hsv_img_ref, light_sky, dark_sky)
mask2_ref = cv2.inRange(hsv_img_ref, light_cloud, dark_cloud)

mask_ref = mask1_ref + mask2_ref

result_sky_ref = cv2.bitwise_and(img_ref, img_ref, mask=mask1_ref)
cv2.imshow("1", result_sky_ref)
cv2.imwrite('skyref.png', result_sky_ref)
cv2.waitKey(0)

plt.hist(result_sky_ref.ravel(), bins=256, range=[1, 256])

result_cloud_ref = cv2.bitwise_and(img_ref, img_ref, mask=mask2_ref)
plt.hist(result_cloud_ref.ravel(), bins=256, range=[1, 256])
cv2.imshow("2", result_cloud_ref)
cv2.imwrite('cloudref.png', result_cloud_ref)
cv2.waitKey(0)


# HISTOGRAM EQUALIZATION
eq_sky_ref = cv2.equalizeHist(cv2.cvtColor(result_sky_ref, cv2.COLOR_BGR2GRAY))
plt.hist(eq_sky_ref.ravel(), bins=256, range=[1, 256])

eq_cloud_ref = cv2.equalizeHist(cv2.cvtColor(result_cloud_ref, cv2.COLOR_BGR2GRAY))
plt.hist(eq_cloud_ref.ravel(), bins=256, range=[1, 256])

img_target = cv2.imread("pic87.jpg")
img_target_rgb = cv2.cvtColor(img_target, cv2.COLOR_GRAY2RGB)
hsv_img_target = cv2.cvtColor(img_target_rgb, cv2.COLOR_RGB2HSV)

cv2.imshow('img', img_target)
cv2.imwrite('img_tar.png', img_target)

# FOR CLOUDS

light_sky = (104, 49, 130)
dark_sky = (110, 125, 170)
light_cloud = (100, 13, 145)
dark_cloud = (125, 60, 180)
mask1_target = cv2.inRange(hsv_img_target, light_sky, dark_sky)
mask2_target = cv2.inRange(hsv_img_target, light_cloud, dark_cloud)
cv2.imshow("mask1_tar", mask1_target)
cv2.waitKey(0)
cv2.imwrite('mask1_tar.png', mask1_target)

cv2.imshow("mask2_tar", mask2_target)
cv2.waitKey(0)
cv2.imwrite('mask2_tar.png', mask2_target)
mask_target = mask1_target + mask2_target

mask_rest = 255 - mask_target

result_sky_tar = cv2.bitwise_and(img_target, img_target, mask=mask1_target)
plt.hist(result_sky_tar.ravel(), bins=256, range=[1, 256])
# plt.show()

result_cloud_tar = cv2.bitwise_and(img_target, img_target, mask=mask2_target)
plt.hist(result_cloud_tar.ravel(), bins=256, range=[1, 256])

cv2.imshow("skyresult_tar", result_sky_tar)
cv2.waitKey(0)
cv2.imwrite('skyresult_tar.png', result_sky_tar)

cv2.imshow("cloudresult_tar", result_cloud_tar)
cv2.waitKey(0)
cv2.imwrite('cloudresult_tar.png', result_cloud_tar)
eq_sky_tar = cv2.equalizeHist(cv2.cvtColor(result_sky_tar, cv2.COLOR_BGR2GRAY))
plt.hist(eq_sky_tar.ravel(), bins=256, range=[1, 256])

eq_cloud_tar = cv2.equalizeHist(cv2.cvtColor(result_cloud_tar, cv2.COLOR_BGR2GRAY))
cv2.imshow('cloud', eq_cloud_tar)
cv2.imwrite('eq_cld.png', eq_cloud_tar)
cv2.waitKey(0)
plt.hist(eq_cloud_tar.ravel(), bins=256, range=[1, 256])

cv2.imshow("b", eq_cloud_tar)
cv2.waitKey(0)
cv2.imwrite('b.png', eq_cloud_tar)

multi = True if result_sky_tar.shape[-1] > 1 else False
matched = match_histograms(result_sky_tar, result_sky_ref, multichannel=True)

multi1 = True if result_cloud_tar.shape[-1] > 1 else False
matched1 = match_histograms(result_cloud_tar, result_cloud_ref, multichannel=True)

a = cv2.bitwise_and(matched, matched, mask=mask1_target)
cv2.imshow("a", a)
cv2.waitKey(0)
cv2.imwrite('a.png', a)
b = cv2.bitwise_and(matched1, matched1, mask=mask2_target)

c = cv2.bitwise_and(img_target, img_target, mask=mask_rest)

eq_cloud_tar = (cv2.cvtColor(eq_cloud_tar, cv2.COLOR_GRAY2RGB))
d = (a + eq_cloud_tar + c) | result_cloud_tar
cv2.imshow("d", d)
cv2.imwrite('d.png', d)
cv2.waitKey(0)

# FILTER
enhanced_gray = cv2.cvtColor(d, cv2.COLOR_RGB2GRAY)
filtered = enhanced_gray
ngbr = np.array([3, 3])
for i in range(1, (enhanced_gray.shape[0] - 1)):
    for j in range(1, (enhanced_gray.shape[1] - 1)):
        high = 0
        for m in range(-1, 2):
            for n in range(-1, 2):
                ngbr = enhanced_gray[i + m][j + n]
                high = max(high, ngbr)
        filtered[i][j] = high

final = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)

final = cv2.bilateralFilter(d, 9, 75, 75)
cv2.imshow('final', final)
cv2.waitKey(0)
