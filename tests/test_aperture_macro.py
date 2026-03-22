#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2026 Jan Sebastian Götte <gerbonara@jaseg.de>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on https://github.com/tracespace/tracespace
#

import math
from contextlib import contextmanager

from PIL import Image
import pytest

from gerbonara.rs274x import GerberFile
from gerbonara.graphic_objects import Line, Arc, Flash, Region
from gerbonara.apertures import *
from gerbonara.cam import FileSettings
from gerbonara.utils import MM, Inch

from .image_support import svg_soup
from .utils import *


@contextmanager
def run_aperture_macro_test(tmpfile, img_support, inst: ApertureMacroInstance, epsilon=1e-3):
    gbr = GerberFile()

    inst_rot_90 = inst.rotated(math.pi/2)
    inst_rot_45 = inst.rotated(math.pi/4)
    inst_rot_neg90 = inst.rotated(-math.pi/2)
    for x, y in [(0, 0), (0, 10), (10, 0), (10, 10)]:
        gbr.objects.append(Flash(x=x, y=y, aperture=inst, unit=MM))
        gbr.objects.append(Flash(x=x, y=20+y, aperture=inst_rot_90, unit=MM))
        gbr.objects.append(Flash(x=20+x, y=y, aperture=inst_rot_neg90, unit=MM))
        gbr.objects.append(Flash(x=20+x, y=20+y, aperture=inst_rot_45, unit=MM))

    # inches, to pixel align our SVG output with gerbv's!
    bounds = (-.5, -.5), (2.0, 2.0) # bottom left, top right

    # The below code is mostly copy-pasted from test_rs274x.py.

    out_svg = tmpfile('SVG Output', '.svg')
    with open(out_svg, 'w') as f:
        # Use inch units here to make sure we and gerbv agree on the exact pixel size of the output since both calculate
        # it from the DPI setting.
        f.write(str(gbr.to_svg(force_bounds=bounds, arg_unit='inch', fg='black', bg='white')))

    # Reference export via gerber through GerbV
    out_gbr = tmpfile('GBR Output', '.gbr')
    gbr.save(out_gbr)

    # NOTE: Instead of having gerbv directly export a PNG, we ask gerbv to output SVG which we then rasterize using
    # resvg. We have to do this since gerbv's built-in cairo-based PNG export has severe aliasing issues. In contrast,
    # using resvg for both allows an apples-to-apples comparison of both results.
    ref_svg = tmpfile('Reference export', '.svg')
    w, h = bounds[1][0] - bounds[0][0], bounds[1][1] - bounds[0][1]
    img_support.gerbv_export(out_gbr, ref_svg, origin=bounds[0], size=(w, h), fg='#000000', bg='#ffffff')
    with svg_soup(ref_svg) as soup:
        img_support.cleanup_gerbv_svg(soup)

    ref_png = tmpfile('Reference render', '.png')
    img_support.svg_to_png(ref_svg, ref_png, dpi=300, bg='white')

    out_png = tmpfile('Output render', '.png')
    img_support.svg_to_png(out_svg, out_png, dpi=300, bg='white')

    mean, _max, hist = img_support.image_difference(ref_png, out_png, diff_out=tmpfile('Difference', '.png'))
    assert hist[9] < 1
    assert mean < epsilon
    assert hist[3:].sum() < epsilon*hist.size


@pytest.mark.parametrize('aperture_type', [
    lambda: CircleAperture(4.0, unit=MM),
    lambda: CircleAperture(4.0, hole_dia=1.5, unit=MM),
    lambda: RectangleAperture(4.0, 3.0, unit=MM),
    lambda: ObroundAperture(4.0, 2.5, unit=MM),
    lambda: PolygonAperture(4.0, 6, unit=MM),
])
def test_macro_conversions(tmpfile, img_support, aperture_type):
    ap = aperture_type()
    inst = ap.to_macro()
    run_aperture_macro_test(tmpfile, img_support, inst)


