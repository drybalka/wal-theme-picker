#!/usr/bin/env python3

from argparse import ArgumentParser
from PIL import Image, ImageCms
import numpy as np

MAX_FIT_ITERATIONS = 200
CLUSTERS = 14


def color_distance(diff):
    return (diff**2).sum(axis=1)


# Weighted k-means algorithm. Assumes that data is scaled to a [0, 1] cube.
# Returns the best centroids in the form (position, importance) and fit errror
# If no weights are provided all data points are assumed to be equal
# If no metric is provided, the standard Euclidian metric is used
def wkmean(k, data, weights=None, metric=None):
    MAX_UPDATE_ITERATIONS = 1000
    # Dimension of each data point
    dim = np.array(data[1]).shape[0]

    def euclidian_metric(coords):
        return (coords**2).sum(axis=1)

    if weights is None:
        weights = np.array([1 for _ in data])
    if metric is None:
        metric = euclidian_metric

    # Initial guess for a centroid position
    def random_point():
        return np.random.rand(dim)

    # Stop criteria for k-means update
    def should_stop(oldCentroids, centroids, iterations):
        if iterations > MAX_UPDATE_ITERATIONS:
            return True
        return (oldCentroids == centroids).all()

    # Returns array of indexes of closest centroid for each point
    def assign_centroid(data):
        distances = [0 for _ in range(k)]
        for ii in range(k):
            diff = data - centroids[ii]
            distances[ii] = metric(diff)
        distances = np.array(distances)
        closest_centroid = distances.argmin(axis=0)
        return closest_centroid

    # New centroid positions, their importance and error
    def update_centroids(data, weights, labels):
        neighborhood = np.array([labels == jj for jj in range(k)])
        neighborhood = neighborhood * weights
        # Sets unnormalized cumulative centroid
        centroids = neighborhood.dot(data)

        importances = np.array([0 for _ in range(k)]).astype(float)
        error = 0
        for jj in range(k):
            s = sum(neighborhood[jj])
            if s == 0:
                centroids[jj] = random_point()
            else:
                centroids[jj] = centroids[jj] / s
                diff = data - centroids[jj]
                distances = metric(diff)
                error += distances.dot(neighborhood[jj])
                importances[jj] = s
        return np.array(centroids), importances, error

    # Rescale data
    rweights = weights / sum(weights)
    mu = rweights.dot(data)
    sigma = np.sqrt(rweights.dot(metric(data - mu)))
    rdata = (data - mu) / sigma

    # Initialize clusters
    centroids = np.array([random_point() for _ in range(k)])
    importances = [0 for _ in range(k)]
    error = 10**15

    # Initialize book keeping vars
    iterations = 0
    oldCentroids = None

    # Repeat until centroids converge to a local minima
    while not should_stop(oldCentroids, centroids, iterations):
        oldCentroids = centroids
        iterations += 1

        # Find the closest centroid for each point
        labels = assign_centroid(rdata)

        # Update centroid positions
        centroids, importances, error = update_centroids(rdata,
                                                         rweights, labels)

    # Rescale centroids back
    centroids = centroids * sigma + mu

    return centroids, importances, error


# Loads an image and extracts colors and their frequencies from an image
def get_image_colors():
    parser = ArgumentParser(
            description='Tries to pick the best color palette for a given image \
                    from a set of hand-picked syntax-highlighting palettes.')
    parser.add_argument('image_path', metavar='image_path', type=str)
    args = parser.parse_args()

    im = Image.open(args.image_path).convert('RGB')

    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    rgb2lab_transform = ImageCms.buildTransformFromOpenProfiles(
            srgb_profile, lab_profile, "RGB", "LAB")
    im = ImageCms.applyTransform(im, rgb2lab_transform)

    tally = im.getcolors()
    counts = np.array([el[0] for el in tally])
    colors = np.array([el[1] for el in tally])

    # Rescale the colors into Wiki LAB space
    # colors = colors.dot(np.diag([100/256, 1, 1])).astype(int)
    # colors = colors - np.array([0, 128, 128])
    colors = colors / 256

    return counts, colors


# Uses K-means algorithm to find the best fitting palette
# of palette_size lengths for the image
def compute_image_palette(colors, counts, palette_size):

    # Repeats search multiple times to find the best fit
    best_palette = None
    best_importances = None
    best_error = 10**15
    for ii in range(MAX_FIT_ITERATIONS):
        palette, importances, error = wkmean(palette_size, colors,
                                             weights=counts,
                                             metric=color_distance)
        if error < best_error:
            best_error = error
            best_palette = palette
            best_importances = importances

    return best_palette, best_importances


def pick_closest_palette(palette, importances):
    inds = importances.argsort()
    sorted_importances = importances[inds[::-1]]
    sorted_palette = palette[inds[::-1]]
    print(sorted_palette)
    print(sorted_importances)
    # sorted_palette = sorted_palette.dot(np.diag([256/100, 1, 1])).astype(int)
    # sorted_palette += [0, 128, 128]
    sorted_palette *= 256
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(
            lab_profile, srgb_profile, "LAB", "RGB")
    array = [sorted_palette[ii // 2500] for ii in range(CLUSTERS*50*50)]
    array = np.reshape(array, (CLUSTERS*50, 50, 3))
    array = array.astype(np.uint8)
    im = Image.fromarray(array)
    im = ImageCms.applyTransform(im, lab2rgb_transform)
    im.show()


if __name__ == '__main__':
    counts, colors = get_image_colors()

    inds = counts.argsort()
    sorted_counts = counts[inds[::-1]] / sum(counts)
    sorted_colors = colors[inds[::-1]]
    print(sorted_colors)
    print(sorted_counts)

    palette, importances = compute_image_palette(colors, counts, CLUSTERS)
    pick_closest_palette(palette, importances)
