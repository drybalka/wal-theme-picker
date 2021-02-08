#!/usr/bin/env python

from math import sqrt


def rgb_dist(rgb1, rgb2):
    lab1 = rgb2lab(rgb1)
    lab2 = rgb2lab(rgb2)
    return cie94(lab1, lab2)


def rgb(x):
    """Convert #[AA]RRGGBB color in integer or string to (r,g,b) tuple

    Alpha (AA) component is simply ignored.

    rgb(0xff0000ff)
    >>> (0, 0, 255)
    rgb('#ff0000')
    >>> (255, 0, 0)
    """

    if isinstance(x, str) and x[0] == '#':
        x = int(x[1:], 16)
    return ((x >> 16) & 0xff, (x >> 8) & 0xff, (x) & 0xff)


def cie94(L1_a1_b1, L2_a2_b2):
    """Calculate color difference by using CIE94 formulae

    See http://en.wikipedia.org/wiki/Color_difference or
    http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html.

    cie94(rgb2lab((255, 255, 255)), rgb2lab((0, 0, 0)))
    >>> 58.0
    cie94(rgb2lab(rgb(0xff0000)), rgb2lab(rgb('#ff0000')))
    >>> 0.0
    """

    L1, a1, b1 = L1_a1_b1
    L2, a2, b2 = L2_a2_b2

    C1 = sqrt(_square(a1) + _square(b1))
    C2 = sqrt(_square(a2) + _square(b2))
    delta_L = L1 - L2
    delta_C = C1 - C2
    delta_a = a1 - a2
    delta_b = b1 - b2
    delta_H_square = _square(delta_a) + _square(delta_b) - _square(delta_C)
    return (sqrt(_square(delta_L)
            + _square(delta_C) / _square(1.0 + 0.045 * C1)
            + delta_H_square / _square(1.0 + 0.015 * C1)))


def rgb2lab(R_G_B):
    """Convert RGB colorspace to Lab

    Adapted from http://www.easyrgb.com/index.php?X=MATH.
    """

    R, G, B = R_G_B

    # Convert RGB to XYZ

    var_R = (R / 255.0)        # R from 0 to 255
    var_G = (G / 255.0)        # G from 0 to 255
    var_B = (B / 255.0)        # B from 0 to 255

    if (var_R > 0.04045):   var_R = ((var_R + 0.055) / 1.055) ** 2.4
    else:                   var_R = var_R / 12.92
    if (var_G > 0.04045):   var_G = ((var_G + 0.055) / 1.055) ** 2.4
    else:                   var_G = var_G / 12.92
    if (var_B > 0.04045):   var_B = ((var_B + 0.055) / 1.055) ** 2.4
    else:                   var_B = var_B / 12.92

    var_R = var_R * 100.0
    var_G = var_G * 100.0
    var_B = var_B * 100.0

    # Observer. = 2°, Illuminant = D65
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505

    # Convert XYZ to L*a*b*

    var_X = X / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
    var_Y = Y / 100.000        # ref_Y = 100.000
    var_Z = Z / 108.883        # ref_Z = 108.883

    if (var_X > 0.008856):  var_X = var_X ** (1/3.0)
    else:                   var_X = (7.787 * var_X) + (16.0 / 116.0)
    if (var_Y > 0.008856):  var_Y = var_Y ** (1/3.0)
    else:                   var_Y = (7.787 * var_Y) + (16.0 / 116.0)
    if (var_Z > 0.008856):  var_Z = var_Z ** (1/3.0)
    else:                   var_Z = (7.787 * var_Z) + (16.0 / 116.0)

    CIE_L = (116.0 * var_Y) - 16.0
    CIE_a = 500.0 * (var_X - var_Y)
    CIE_b = 200.0 * (var_Y - var_Z)
    return (CIE_L, CIE_a, CIE_b)


def lab2rgb(L_a_b):
    """Convert Lab colorspace to RGB

    Adapted from http://www.easyrgb.com/index.php?X=MATH.
    """
    L, a, b = L_a_b

    # Convert L*a*b* to XYZ

    var_Y = (L + 16) / 116
    var_X = a / 500 + var_Y
    var_Z = var_Y - b / 200

    if (var_Y**3 > 0.008856):   var_Y = var_Y**3
    else:                       var_Y = (var_Y - 16 / 116) / 7.787
    if (var_X**3 > 0.008856):   var_X = var_X**3
    else:                       var_X = (var_X - 16 / 116) / 7.787
    if (var_Z**3 > 0.008856):   var_Z = var_Z**3
    else:                       var_Z = (var_Z - 16 / 116) / 7.787

    X = var_X * 95.047      # ref_X = 95.047    Observer= 2°, Illuminant= D65
    Y = var_Y * 100.000     # ref_Y = 100.000
    Z = var_Z * 108.883     # ref_Z = 108.883

    # Convert XYZ to RGB

    var_X = X / 100
    var_Y = Y / 100
    var_Z = Z / 100

    var_R = var_X *  3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y *  1.8758 + var_Z *  0.0415
    var_B = var_X *  0.0557 + var_Y * -0.2040 + var_Z *  1.0570

    if (var_R > 0.0031308):     var_R = 1.055 * (var_R ** (1 / 2.4)) - 0.055
    else:                       var_R = 12.92 * var_R
    if (var_G > 0.0031308):     var_G = 1.055 * (var_G ** (1 / 2.4)) - 0.055
    else:                       var_G = 12.92 * var_G
    if (var_B > 0.0031308):     var_B = 1.055 * (var_B ** (1 / 2.4)) - 0.055
    else:                       var_B = 12.92 * var_B

    sR = var_R * 255
    sG = var_G * 255
    sB = var_B * 255

    return (sR, sG, sB)



def _old_rgb2lab(R_G_B):
    """Old implementation of rgb2lab, the result is strange :D

    Adapted from http://www.f4.fhtw-berlin.de/~barthel/ImageJ/ColorInspector//HTMLHelp/farbraumJava.htm.
    """

    R, G, B = R_G_B

    # http://www.brucelindbloom.com

    # float: r, g, b, X, Y, Z, fx, fy, fz, xr, yr, zr
    # float: Ls, as_, bs
    eps = 216.0/24389.0
    k = 24389.0/27.0

    Xr = 0.964221  # reference white D50
    Yr = 1.0
    Zr = 0.825211

    # RGB to XYZ
    r = R/255.0 #R 0..1
    g = G/255.0 #G 0..1
    b = B/255.0 #B 0..1

    if not (0<=r<=1 and 0<=g<=1 and 0<=b<=1):
        raise ValueError('RGB out of 0..255 range')

    # assuming sRGB (D65)
    if r <= 0.04045:
        r = r/12
    else:
        r = ((r+0.055)/1.055) ** 2.4

    if g <= 0.04045:
        g = g/12
    else:
        g = ((g+0.055)/1.055) ** 2.4

    if b <= 0.04045:
        b = b/12
    else:
        b = ((b+0.055)/1.055) ** 2.4

    X =  0.436052025*r     + 0.385081593*g + 0.143087414 *b
    Y =  0.222491598*r     + 0.71688606 *g + 0.060621486 *b
    Z =  0.013929122*r     + 0.097097002*g + 0.71418547  *b

    # XYZ to Lab
    xr = X/Xr
    yr = Y/Yr
    zr = Z/Zr

    if xr > eps:
        fx =  xr ** (1.0/3.0)
    else:
        fx = (k * xr + 16.0) / 116.0

    if yr > eps:
        fy =  yr ** (1.0/3.0)
    else:
        fy = (k * yr + 16.0) / 116.0

    if zr > eps:
        fz =  zr ** (1.0/3.0)
    else:
        fz = (k * zr + 16.0) / 116.0

    Ls = ( 116 * fy ) - 16
    as_ = 500*(fx-fy)
    bs = 200*(fy-fz)

    return (int(2.55*Ls + 0.5), # L
            int(as_ + 0.5),     # a
            int(bs + 0.5))      # b


def _square(x):
    return x * x
