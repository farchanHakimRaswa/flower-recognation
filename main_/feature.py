import numpy as np
import mahotas
import cv2

# bins for histogram
bins = 8
# train_test_split size
test_size = 0.10
# seed for reproducing same results
seed = 9

# Compute the hog descriptor for an image
def fd_hog_descriptor(image, n_bins = 16):
    # We get the derivatives of the image
    dx = cv2.Sobel(image, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_32F, 0, 1)
    # Calculate the magnitude and the angle
    magnitude, angle = cv2.cartToPolar(dx, dy)
    # Quantizing binvalues in (0..n_bins)
    binvalues = np.int32(n_bins*angle/(2*np.pi))
    # Divide the image on 4 squares and compute
    # the histogram
    magn_cells = magnitude[:10, :10], magnitude[10:, :10], magnitude[:10, 10:], magnitude[10:, 10:]
    bin_cells = binvalues[:10, :10], binvalues[10:, :10], binvalues[:10, 10:], binvalues[10:, 10:]
    # With "bincount" we can count the number of occurrences of a
    # flat array to create the histogram. Those flats arrays we can
    # create it with the NumPy function "ravel"
    histogram = [np.bincount(bin_cell.ravel(), magn.ravel(), n_bins)
                    for bin_cell, magn in zip(bin_cells, magn_cells)]
    # And return an array with the histogram
    return np.hstack(histogram)

# feature-descriptor-1: Hu Moments
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

def create_bag_of_words(images, detector_type, k_size = 10):
    # Create an empty vocabulary with BOWKMeans
    vocabulary = cv2.BOWKMeansTrainer(clusterCount=k_size)

    if detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create()

    elif detector_type == 'SIFT':
        detector = cv2.xfeatures2d.SIFT_create()

    elif detector_type == 'KAZE':
        detector = cv2.KAZE_create()

    else:
        raise ValueError('Not a suitable detector')

    print("Creating the unclustered geometric vocabulary")

    descriptors, keypoints = [], []

    for img in images:
        # Detect the keypoints on the image and
        # compute the descriptor for those keypoints
        kp, descriptor = detector.detectAndCompute(img, None)
        descriptors.append(descriptor)
        keypoints.append(kp)
        vocabulary.add(descriptor)

    print("DONE!!")
    print("Creating the clusters with K-means")
    # K-Means clustering
    BOW = vocabulary.cluster()
    print("DONE!!")
    BOW = BOW.astype(np.float32)

    return BOW, keypoints, descriptors
