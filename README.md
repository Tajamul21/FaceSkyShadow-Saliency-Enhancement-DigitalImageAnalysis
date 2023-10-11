# -Face-with-sky-and-shadow-saliency-enhancement-using-Digital-Image-Analysis
Image enhancement in python for face, sky, and shadowed saliency.
# Image Enhancement Project

Enhance your images with advanced techniques including face, sky, and shadowed saliency enhancements. This project includes tools to improve image quality and clarity.

## Table of Contents
1. [Face Enhancement](#face-enhancement)
   1.1 [Face Detection](#face-detection)
   1.2 [Skin Detection](#skin-detection)
   1.3 [Face Enhancement](#face-enhancement)
   1.4 [Bilateral Filtering](#bilateral-filtering)
   1.5 [Restore the Enhanced Skin into Image](#restore-the-enhanced-skin-into-image)
   1.6 [Result Analysis](#result-analysis)
2. [Sky Enhancement](#sky-enhancement)
   2.1 [Sky and Cloud Detection](#sky-and-cloud-detection)
   2.2 [Histogram Matching](#histogram-matching)
3. [Shadowed Saliency Enhancement](#shadowed-saliency-enhancement)
   3.1 [Computing Saliency Map](#computing-saliency-map)
   3.2 [Computing Fsal](#computing-fsal)
   3.3 [Updation of Luminance](#updation-of-luminance)
4. [Contributors](#contributors)
5. [License](#license)

## Face Enhancement

### Face Detection
The project begins by detecting faces in the input image using a Haar Cascade classifier.

### Skin Detection
The system applies skin detection by converting the image to the LAB color space and creating a mask based on predefined color thresholds.

### Face Enhancement
Enhancement of facial regions is achieved through techniques like histogram equalization, and the results are blended back into the image.

### Bilateral Filtering
Bilateral filtering is applied to improve skin texture while preserving edges and details.

### Restore the Enhanced Skin into Image
The enhanced skin regions are combined with the original image to produce an image with improved facial appearance.

### Result Analysis
The code analyzes the results and provides insights into the image enhancement process.

## Sky Enhancement

### Sky and Cloud Detection
The project includes methods for detecting the sky and clouds in the image by using color masks in the HSV color space.

### Histogram Matching
Histogram matching is applied to improve the consistency of colors and contrast in the sky regions of the image.

## Shadowed Saliency Enhancement

### Computing Saliency Map
The project computes a saliency map to identify regions of interest in the image.

### Computing Fsal
The code calculates Fsal, which represents the saliency of shadowed areas in the image.

### Updation of Luminance
The luminance values of the image are updated to enhance the shadowed saliency regions.

## Contributors

- [Krishnapriya S](link-to-profile)
- [Tajamul Ashraf](link-to-profile)

## License

This project is licensed under the XYZ License - see the [LICENSE.md](LICENSE.md) file for details.

## How to Use

1. Ensure you have the required Python libraries, such as OpenCV, NumPy, and other dependencies, installed.
2. Execute the main script to apply the chosen enhancement. For example, for face enhancement:

```bash
python face_enhancement.py
