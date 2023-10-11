
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve



from google.colab.patches import cv2_imshow

import cv2

# SALIENCY MAP

img = cv2.imread('/content/2.png')
saliencyMap = calc_energy(img)
cv2_imshow(saliencyMap)

blur = cv2.blur(saliencyMap, (3, 3))
cv2_imshow(blur)
saliencyMap = ((saliencyMap - np.min(saliencyMap)) / (np.max(saliencyMap) - np.min(saliencyMap)))




def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])


    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map


saliencyMap = calc_energy(img)

import numpy as np

img = cv2.imread('2.png')
imgLab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
dark_img = 255 * np.ones([len(img[0]), len(img[1])], dtype=object)
bright_img = np.zeros([len(img[0]), len(img[1])], dtype=object)


# THRESHOLDING RANGE

light_low = (51)
light_high = (255)
dark_low = (0)
dark_high = (50)
mask1_ref = cv2.inRange(imgLab[:, :, 0], light_low, light_high)
mask2_ref = cv2.inRange(imgLab[:, :, 0], dark_low, dark_high)

print(len(img))

n_dark_pix = np.sum(mask1_ref == 255)
print('Number of dark pixels:', n_dark_pix)

pt_dark = (0.95) * n_dark_pix

print(pt_dark)

n_light_pix = np.sum(mask1_ref == 0)
print('Number of white pixels:', n_light_pix)

pt_light = (0.35) * n_light_pix

print(pt_light)

ratio = pt_light / pt_dark

if ratio <= 2:
    F_sal = ratio
else:
    F_sal = 2

print("F_sal = ", F_sal)

F_sal = min(2.0, 1.0 * np.percentile(n_light_pix, 35) / np.percentile(n_dark_pix, 95))

cv2_imshow(mask1_ref)
cv2_imshow(mask2_ref)

result_dark = cv2.bitwise_and(imgLab[:, :, 0], imgLab[:, :, 0], mask=mask2_ref)
result_light = cv2.bitwise_and(imgLab[:, :, 0], imgLab[:, :, 0], mask=mask1_ref)

cv2_imshow(result_dark)
cv2_imshow(result_light)

L, A, B = cv2.split(imgLab)

Bnew = L

# EQUATION

for i in range(imgLab.shape[0]):
    for j in range(imgLab.shape[1]):
        # print(i, j)

        Bnew[i][j] = (F_sal * saliencyMap[i][j] * L[i][j]) + ((1 - saliencyMap[i][j]) * L[i][j])
BNEW = cv2.merge([Bnew, A, B])
cv2_imshow(BNEW)


imgrgb = cv2.cvtColor(BNEW, cv2.COLOR_LAB2BGR)
imgrgb1 = cv2.cvtColor(BNEW, cv2.COLOR_LAB2RGB)
cv2_imshow(img)

cv2_imshow(imgrgb)

e = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
cv2_imshow(e)
bb = cv2.cvtColor(e, cv2.COLOR_GRAY2RGB)
cv2_imshow(bb)


