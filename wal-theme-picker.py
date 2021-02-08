#!/usr/bin/env python3

from argparse import ArgumentParser
from PIL import Image, ImageCms
import numpy as np
from kmean import wkmean
from colordiff import rgb, rgb2lab, lab2rgb, rgb_dist

MAX_FIT_ITERATIONS = 100
CLUSTERS = 10


# Loads an image and extracts colors and their frequencies from an image
def get_image_colors(args):
    im = Image.open(args.image_path).convert('RGB')
    tally = im.getcolors()
    counts = np.array([el[0] for el in tally])
    colors = np.array([rgb2lab(el[1]) for el in tally])

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
    # palette = [lab2rgb(col) for col in palette]
    # print_palette(palette, importances)

    pass


def print_palette(palette, importances):
    L = 50
    inds = importances.argsort()
    sorted_palette = np.array(palette)[inds[::-1]]
    array = np.array([sorted_palette[ii // (L**2)]
                      for ii in range(CLUSTERS*L*L)])
    array = np.reshape(array, (CLUSTERS*L, L, 3))
    array = array.astype(np.uint8)
    im = Image.fromarray(array)
    im.show()


if __name__ == '__main__':
    parser = ArgumentParser(
            description='Tries to pick the best color palette for a given image \
                    from a set of hand-picked syntax-highlighting palettes.')
    parser.add_argument('image_path', metavar='image_path', type=str)
    args = parser.parse_args()
    counts, colors = get_image_colors(args)

    palette, importances = compute_image_palette(colors, counts, CLUSTERS,
                                                 method='k++_pdf')
    pick_closest_palette(palette, importances, colors)
