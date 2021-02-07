#!/usr/bin/env python3

from argparse import ArgumentParser
from PIL import Image, ImageCms
import numpy as np
from kmean import wkmean

MAX_FIT_ITERATIONS = 100
CLUSTERS = 10


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
    # colors = colors / 256

    return counts, colors


# Uses K-means algorithm to find the best fitting palette
# of palette_size lengths for the image
def compute_image_palette(colors, counts, palette_size, method='k++_pdf'):

    # Repeats search multiple times to find the best fit
    best_palette = None
    best_importances = None
    best_error = 10**15
    for ii in range(MAX_FIT_ITERATIONS):
        palette, importances, error = wkmean(palette_size, colors,
                                             weights=counts,
                                             method=method)
        if error < best_error:
            best_error = error
            best_palette = palette
            best_importances = importances

    return best_palette, best_importances


def pick_closest_palette(palette, importances, colors):
    inds = importances.argsort()
    sorted_importances = importances[inds[::-1]]
    sorted_palette = palette[inds[::-1]]
    print(sorted_palette)
    print(sorted_importances)
    # sorted_palette = sorted_palette.dot(np.diag([256/100, 1, 1])).astype(int)
    # sorted_palette += [0, 128, 128]
    # sorted_palette *= 256
    srgb_profile = ImageCms.createProfile("sRGB")
    lab_profile = ImageCms.createProfile("LAB")
    lab2rgb_transform = ImageCms.buildTransformFromOpenProfiles(
            lab_profile, srgb_profile, "LAB", "RGB")

    L = 50
    def colorize(ii):
        mod = ii // (2*L**2)
        if ii % (2*L) < L:
            if mod < len(colors):
                return colors[mod]
            else:
                return [0, 128, 128]
        else:
            return sorted_palette[mod]

    array = np.array([colorize(ii) for ii in range(CLUSTERS*L*L*2)])
    array = np.reshape(array, (CLUSTERS*L, 2*L, 3))
    array = array.astype(np.uint8)
    im = Image.fromarray(array)
    im = ImageCms.applyTransform(im, lab2rgb_transform)
    im.show()


if __name__ == '__main__':
    counts, colors = get_image_colors()

    inds = counts.argsort()
    sorted_counts = counts[inds[::-1]] / sum(counts)
    sorted_colors = colors[inds[::-1]]
    colors = sorted_colors
    counts = sorted_counts
    print(sorted_colors)
    print(sorted_counts)

    palette, importances = compute_image_palette(colors, counts, CLUSTERS,
                                                 method='k++_pdf')
    pick_closest_palette(palette, importances, colors)
