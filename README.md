# wal-theme-picker

A helpful script to pick the best suited predefined themes for `wal` (https://github.com/dylanaraps/pywal) based on the colors in the image.

`wal` (pywal) is a great tool to generate a color palette from the dominant colors in an image and then apply it system-wide. However, sometimes it generates a palette that is too plane or bland, especially for monochromatic images. Such a color scheme is often less useful for syntax-highlighting compared to hand-picked built-in pywal themes. This is the main motivation for the `wal-theme-picker`.

Under the hood `wal-theme-picker` uses k-means clustering to extract the dominant colors in the image, then compares them with themes in `wal`, assigns each theme a rating based on a semi-empirical color-distance formula, and outputs the best-scoring themes.

Note, that the notion of "the best" theme is very subjective and relies heavily on the personal taste. Therefore, `wal-theme-picker` also proposes an interactive menu to try out the best-scoring themes with an option to revert the changes. There is also a possibility to print out the dominant colors and the palette for visual comparison.

### Dependencies
The only dependency is the installed `wal`.

### Usage
```sh
usage: wal-theme-picker [-h] [-n N] [-c C] [-p] [-i] image_path

Tries to pick the best color palette for a given image from a set of hand-picked
syntax-highlighting palettes.

positional arguments:
  image_path

optional arguments:
  -h, --help  show this help message and exit
  -n N        number of themes to print
  -c C        number of dominating colors in image
  -p          print image palette (first column) and n best themes in feh
  -i          call interactive menu to install one of the suggested themes using wal
```
For example, `wal-theme-picker -n 5 -c 3 -p -i ~/wallpaper.png` will output the names of the 5 best-scoring themes based on the 3 dominant colors in `wallpaper.png`, display the palettes in `feh` (or another default image viewer), and start an interactive menu to apply the themes using `wal`.
