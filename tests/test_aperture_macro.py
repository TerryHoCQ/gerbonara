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
import operator as op
from contextlib import contextmanager

from PIL import Image
import pytest

from gerbonara.rs274x import GerberFile
from gerbonara.graphic_objects import Line, Arc, Flash, Region
from gerbonara.apertures import *
from gerbonara import aperture_macros as am
from gerbonara.aperture_macros import (
    ConstantExpression, ParameterExpression, OperatorExpression,
    NegatedExpression, VariableExpression, UnitExpression,
)
from gerbonara.aperture_macros.expression import expr
from gerbonara.aperture_macros.parse import _parse_expression
from gerbonara.cam import FileSettings
from gerbonara.utils import MM, Inch, MILLIMETERS_PER_INCH

from .image_support import svg_soup
from .utils import *

# Short aliases used throughout expression tests
C = ConstantExpression
P = ParameterExpression


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


@pytest.mark.parametrize('params', [(10, 0), (7, 0), (10, 5)])
def test_generic_macro_circle(tmpfile, img_support, params):
    ap = am.GenericMacros.circle(*params)
    # epsilon changed since gerbv approximates circles with cubic splines which ends up pretty wrong at this scale
    run_aperture_macro_test(tmpfile, img_support, ap, epsilon=1e-2)

@pytest.mark.parametrize('params', [
    (10, 10, 0, 0),
    (10,  5, 0, 0),
    ( 5, 10, 0, 0),
    (10, 10, 5, 0),
    (10,  7, 3, 0),
    (10, 10, 0, math.pi/2),
    (10, 10, 0, math.pi/3),
    (10,  5, 0, math.pi/3),
    ( 7, 10, 3, math.pi/3)])
def test_generic_macro_rect(tmpfile, img_support, params):
    ap = am.GenericMacros.rect(*params)
    run_aperture_macro_test(tmpfile, img_support, ap, epsilon=1e-2)

@pytest.mark.parametrize('params', [
    (10, 10, 0, 0, 0),
    (10,  5, 0, 0, 0),
    ( 5, 10, 0, 0, 0),
    (10, 10, 0, 5, 0),
    (10,  7, 0, 3, 0),
    (10, 10, 0, 0, math.pi/2),
    (10, 10, 0, 0, math.pi/3),
    (10,  5, 0, 0, math.pi/3),
    ( 7, 10, 0, 3, math.pi/3),
    (10, 10, 2, 0, 0),
    (10,  5, 2, 0, 0),
    ( 5, 10, 2, 0, 0),
    (10, 10, 2, 5, 0),
    (10,  7, 2, 3, 0),
    (10, 10, 2, 0, math.pi/2),
    (10, 10, 2, 0, math.pi/3),
    (10,  5, 2, 0, math.pi/3),
    ( 7, 10, 2, 3, math.pi/3),
    ])
def test_generic_macro_rounded_rect(tmpfile, img_support, params):
    ap = am.GenericMacros.rounded_rect(*params)
    run_aperture_macro_test(tmpfile, img_support, ap, epsilon=1e-2)

@pytest.mark.parametrize('params', [
    (10,  8, 2, 0, 0),
    (10,  8, 4, 0, 0),
    ( 8, 10, 2, 0, 0),
    (10,  8, 2, 3, 0),
    (10,  8, 2, 0, math.pi/2),
    (10,  8, 2, 0, math.pi/3),
    (10,  8, 2, 3, math.pi/3),
    (10,  8, 0, 0, 0), # d=0: degenerate case (rectangle)
    ])
def test_generic_macro_isosceles_trapezoid(tmpfile, img_support, params):
    ap = am.GenericMacros.isosceles_trapezoid(*params)
    run_aperture_macro_test(tmpfile, img_support, ap)

@pytest.mark.parametrize('params', [
    # (w, h, d, margin, hole_dia, rotation)
    (10, 8, 2, 1, 0, 0),
    (10, 8, 4, 1, 0, 0),
    (10, 8, 2, 1, 3, 0),
    (10, 8, 2, 1, 0, math.pi/2),
    (10, 8, 2, 1, 0, math.pi/3),
    (10, 8, 2, 1, 3, math.pi/3),
    ])
def test_generic_macro_rounded_isosceles_trapezoid(tmpfile, img_support, params):
    ap = am.GenericMacros.rounded_isosceles_trapezoid(*params)
    run_aperture_macro_test(tmpfile, img_support, ap)

@pytest.mark.parametrize('params', [
    # (w, h, hole_dia, rotation), w >= h required
    (10, 5, 0, 0),
    ( 8, 4, 0, 0),
    (10, 5, 2, 0),
    ( 7, 7, 0, 0), # w == h: circle
    (10, 5, 0, math.pi/2),
    (10, 5, 0, math.pi/3),
    (10, 5, 2, math.pi/3),
    ])
def test_generic_macro_obround(tmpfile, img_support, params):
    ap = am.GenericMacros.obround(*params)
    run_aperture_macro_test(tmpfile, img_support, ap, epsilon=1e-2)

@pytest.mark.parametrize('params', [
    # (n, diameter, hole_dia, rotation)
    (3, 10, 0, 0),
    (4, 10, 0, 0),
    (5, 10, 0, 0),
    (6, 10, 0, 0),
    (6, 10, 3, 0),
    (6, 10, 0, math.pi/6),
    (5, 10, 0, math.pi/4),
    (5, 10, 3, math.pi/4),
    (3, 10, 3, math.pi/3),
    ])
def test_generic_macro_polygon(tmpfile, img_support, params):
    ap = am.GenericMacros.polygon(*params)
    run_aperture_macro_test(tmpfile, img_support, ap)

@pytest.mark.parametrize('abc', [(2.0, 1.6, 2.3), (2.2, 1.6, 2.3), (2.1, 1.7, 2.4)])
def test_macro_formulas(tmpfile, img_support, abc):
    @am.ApertureMacro.map()
    class test_macro:
        a: float
        b: float
        c: float
        
        def draw(self):
            d = 1.3
            yield am.Circle('mm', 0, d, 0, 0)
            yield am.Circle('mm', 0, d, 2, 0)
            yield am.Circle('mm', 0, d, 4, 0)
            yield am.Circle('mm', 0, d, 2, self.a)
            yield am.Circle('mm', 0, d, 2, self.a+self.b)
            yield am.Circle('mm', 0, d, 2, self.a+self.b+self.c)
            yield am.Circle('mm', 0, d, 4, self.a * 1.1)
            yield am.Circle('mm', 0, d, 4, self.b * 1.9)
            yield am.Circle('mm', 0, d, 4, self.c * 2.2)
            yield am.Circle('mm', 0, d, 6, 2 * self.a / self.b)
            yield am.Circle('mm', 0, d, 6, 4 * self.b / self.c)
            yield am.Circle('mm', 0, d, 6, 6 * self.c / self.a)
            yield am.Circle('mm', 0, d, 8, self.a - self.b * self.a / self.c)
            yield am.Circle('mm', 0, d, 8, 2 + self.a - self.b * self.a / self.c)
            yield am.Circle('mm', 0, d, 8, self.a - 2 * self.b * self.a / self.c)
    
    inst = test_macro(*abc)
    run_aperture_macro_test(tmpfile, img_support, inst)


# =============================================================================
# Expression language unit tests
# =============================================================================

class TestConstantExpression:
    def test_value_stored(self):
        assert C(5).value == 5

    def test_float_conversion(self):
        assert float(C(3.14)) == pytest.approx(3.14)

    def test_calculate_no_binding(self):
        assert C(42.0).calculate() == pytest.approx(42.0)

    def test_calculate_ignores_binding(self):
        assert C(7.0).calculate({1: 99.0}) == pytest.approx(7.0)

    def test_to_gerber_integer(self):
        assert C(5).to_gerber() == '5'

    def test_to_gerber_float(self):
        assert C(1.5).to_gerber() == '1.5'

    def test_to_gerber_trailing_zeros_stripped(self):
        assert C(1.500000).to_gerber() == '1.5'
        assert C(2.0).to_gerber() == '2'

    def test_to_gerber_zero(self):
        assert C(0).to_gerber() == '0'

    def test_to_gerber_negative_zero_avoided(self):
        # -0.0 must not serialize as '-0'
        assert C(-0.0).to_gerber() == '0'

    def test_equality_exact(self):
        assert C(3.0) == C(3.0)

    def test_equality_within_tolerance(self):
        assert C(1.0) == C(1.0 + 1e-10)

    def test_inequality_outside_tolerance(self):
        assert not (C(1.0) == C(2.0))

    def test_equality_with_plain_number(self):
        assert C(0) == 0
        assert C(1) == 1
        assert C(-1) == -1

    def test_parameters_empty(self):
        assert list(C(5).parameters()) == []


class TestParameterExpression:
    def test_to_gerber(self):
        assert P(1).to_gerber() == '$1'
        assert P(3).to_gerber() == '$3'
        assert P(42).to_gerber() == '$42'

    def test_calculate_with_binding(self):
        assert P(1).calculate({1: 5.0}) == pytest.approx(5.0)
        assert P(2).calculate({1: 10.0, 2: 20.0}) == pytest.approx(20.0)

    def test_calculate_unresolved_raises(self):
        with pytest.raises(IndexError):
            P(1).calculate({})

    def test_calculate_missing_param_raises(self):
        with pytest.raises(IndexError):
            P(2).calculate({1: 5.0})

    def test_parameters_yields_self(self):
        p = P(1)
        assert list(p.parameters()) == [p]

    def test_optimized_with_binding_resolves(self):
        assert P(1).optimized({1: 7.5}) == C(7.5)

    def test_optimized_without_binding_is_identity(self):
        p = P(1)
        assert p.optimized({}) is p


class TestArithmeticOperators:
    @pytest.mark.parametrize('a,b', [(3.0, 7.0), (-1.5, 4.2), (0.5, 0.25), (0.0, 5.0)])
    def test_add(self, a, b):
        assert (P(1) + P(2)).calculate({1: a, 2: b}) == pytest.approx(a + b)

    @pytest.mark.parametrize('a,b', [(3.0, 7.0), (-1.5, 4.2), (0.5, 0.25), (5.0, 5.0)])
    def test_sub(self, a, b):
        assert (P(1) - P(2)).calculate({1: a, 2: b}) == pytest.approx(a - b)

    @pytest.mark.parametrize('a,b', [(3.0, 7.0), (-1.5, 4.2), (0.5, 0.25), (0.0, 5.0)])
    def test_mul(self, a, b):
        assert (P(1) * P(2)).calculate({1: a, 2: b}) == pytest.approx(a * b)

    @pytest.mark.parametrize('a,b', [(6.0, 3.0), (-4.5, 1.5), (1.0, 4.0)])
    def test_div(self, a, b):
        assert (P(1) / P(2)).calculate({1: a, 2: b}) == pytest.approx(a / b)

    def test_radd(self):
        assert (5.0 + P(1)).calculate({1: 3.0}) == pytest.approx(8.0)

    def test_rsub(self):
        assert (10.0 - P(1)).calculate({1: 3.0}) == pytest.approx(7.0)

    def test_rmul(self):
        assert (2.0 * P(1)).calculate({1: 4.0}) == pytest.approx(8.0)

    def test_rdiv(self):
        assert (10.0 / P(1)).calculate({1: 2.0}) == pytest.approx(5.0)

    def test_neg(self):
        assert (-P(1)).calculate({1: 5.0}) == pytest.approx(-5.0)

    def test_pos_is_identity(self):
        p = P(1)
        assert +p is p


# Cross-check expression evaluation against Python's own arithmetic.
# A single lambda serves both roles: called with P() objects it builds an expression tree;
# called with plain numbers it computes the Python reference value (ints/floats auto-convert).
@pytest.mark.parametrize('f,binding', [
    (lambda p1, p2, p3: p1 + p2,            {1: 3,  2: 7,  3: 0}),
    (lambda p1, p2, p3: p1 - p2,            {1: 10, 2: 3,  3: 0}),
    (lambda p1, p2, p3: p1 * p2,            {1: 3,  2: 4,  3: 0}),
    (lambda p1, p2, p3: p1 / p2,            {1: 9,  2: 3,  3: 0}),
    (lambda p1, p2, p3: p1 + p2 + p3,       {1: 1,  2: 2,  3: 3}),
    (lambda p1, p2, p3: p1 * p2 + p3,       {1: 2,  2: 3,  3: 4}),
    (lambda p1, p2, p3: p1 + p2 * p3,       {1: 2,  2: 3,  3: 4}),
    (lambda p1, p2, p3: (p1 + p2) * p3,     {1: 2,  2: 3,  3: 4}),
    (lambda p1, p2, p3: p1 / p2 + p3,       {1: 6,  2: 3,  3: 1}),
    (lambda p1, p2, p3: p1 - p2 * p3,       {1: 10, 2: 2,  3: 3}),
    (lambda p1, p2, p3: (p1 + p2) / p3,     {1: 3,  2: 5,  3: 4}),
    (lambda p1, p2, p3: p1 * (p2 - p3),     {1: 3,  2: 7,  3: 2}),
    (lambda p1, p2, p3: p1 * 2 + 3,         {1: 5,  2: 0,  3: 0}),
    (lambda p1, p2, p3: 10 - p1 * p2,       {1: 2,  2: 3,  3: 0}),
    (lambda p1, p2, p3: p1 / 2 + p2,        {1: 6,  2: 1,  3: 0}),
    (lambda p1, p2, p3: -p1 + p2,           {1: 3,  2: 7,  3: 0}),
    (lambda p1, p2, p3: p1 + (-p2),         {1: 10, 2: 3,  3: 0}),
    (lambda p1, p2, p3: p1 * (-p2),         {1: 3,  2: 4,  3: 0}),
    (lambda p1, p2, p3: (-p1) * (-p2),      {1: 3,  2: 4,  3: 0}),
    (lambda p1, p2, p3: (-p1) / (-p2),      {1: 6,  2: 3,  3: 0}),
    (lambda p1, p2, p3: p1 - (-p2),         {1: 5,  2: 3,  3: 0}),
    (lambda p1, p2, p3: (p1+p2) * (p1-p3),  {1: 5,  2: 3,  3: 2}),
    (lambda p1, p2, p3: p1 / p2 * p3,       {1: 6,  2: 2,  3: 5}),
])
def test_expression_against_python(f, binding):
    """Build a gerbonara expression and compare its result to Python's evaluation."""
    a, b, c = binding.get(1, 0), binding.get(2, 0), binding.get(3, 0)
    assert f(P(1), P(2), P(3)).calculate(binding) == pytest.approx(f(a, b, c), rel=1e-9, abs=1e-12)


class TestConstantFolding:
    """Operations on two ConstantExpressions must immediately produce a ConstantExpression."""

    def test_add(self):
        result = C(3) + C(4)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(7)

    def test_sub(self):
        result = C(10) - C(4)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(6)

    def test_mul(self):
        result = C(3) * C(4)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(12)

    def test_div(self):
        result = C(10) / C(4)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(2.5)

    def test_neg_of_constant(self):
        result = -C(5)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(-5)

    def test_nested(self):
        result = (C(3) + C(4)) * C(2)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(14)

    def test_deeply_nested(self):
        result = C(2) * C(3) + C(4) * C(5)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(26)


class TestAlgebraicOptimizations:
    """Each algebraic simplification rule in OperatorExpression.optimized()."""

    def test_zero_plus_x(self):
        assert C(0) + P(1) == P(1)

    def test_x_plus_zero(self):
        assert P(1) + C(0) == P(1)

    def test_zero_times_x(self):
        assert C(0) * P(1) == C(0)

    def test_x_times_zero(self):
        assert P(1) * C(0) == C(0)

    def test_one_times_x(self):
        assert C(1) * P(1) == P(1)

    def test_x_times_one(self):
        assert P(1) * C(1) == P(1)

    def test_x_times_neg_one_negates(self):
        assert (P(1) * C(-1)).calculate({1: 5.0}) == pytest.approx(-5.0)

    def test_neg_one_times_x_negates(self):
        assert (C(-1) * P(1)).calculate({1: 5.0}) == pytest.approx(-5.0)

    def test_x_minus_zero(self):
        assert P(1) - C(0) == P(1)

    def test_zero_minus_x_is_neg_x(self):
        assert (C(0) - P(1)).calculate({1: 5.0}) == pytest.approx(-5.0)

    def test_x_minus_x_is_zero(self):
        p = P(1)
        assert p - p == C(0)

    def test_x_minus_neg_y_is_x_plus_y(self):
        assert (P(1) - (-P(2))).calculate({1: 3.0, 2: 4.0}) == pytest.approx(7.0)

    def test_x_div_one(self):
        assert P(1) / C(1) == P(1)

    def test_x_div_neg_one_negates(self):
        assert (P(1) / C(-1)).calculate({1: 5.0}) == pytest.approx(-5.0)

    def test_x_div_x_is_one(self):
        p = P(1)
        assert p / p == C(1)

    def test_neg_x_times_neg_y_cancels(self):
        assert ((-P(1)) * (-P(2))).calculate({1: 3.0, 2: 4.0}) == pytest.approx(12.0)

    def test_neg_x_div_neg_y_cancels(self):
        assert ((-P(1)) / (-P(2))).calculate({1: 6.0, 2: 3.0}) == pytest.approx(2.0)

    def test_x_plus_neg_y_becomes_subtraction(self):
        assert (P(1) + (-P(2))).calculate({1: 10.0, 2: 3.0}) == pytest.approx(7.0)

    def test_neg_x_plus_y_reverses_subtraction(self):
        assert ((-P(1)) + P(2)).calculate({1: 3.0, 2: 10.0}) == pytest.approx(7.0)

    def test_x_mul_neg_y_pulls_negation_out(self):
        e = P(1) * (-P(2))
        assert isinstance(e, NegatedExpression)
        assert e.calculate({1: 3.0, 2: 4.0}) == pytest.approx(-12.0)

    def test_neg_x_mul_y_pulls_negation_out(self):
        e = (-P(1)) * P(2)
        assert isinstance(e, NegatedExpression)
        assert e.calculate({1: 3.0, 2: 4.0}) == pytest.approx(-12.0)

    def test_x_div_neg_y_pulls_negation_out(self):
        e = P(1) / (-P(2))
        assert isinstance(e, NegatedExpression)
        assert e.calculate({1: 6.0, 2: 3.0}) == pytest.approx(-2.0)

    def test_neg_x_div_y_pulls_negation_out(self):
        e = (-P(1)) / P(2)
        assert isinstance(e, NegatedExpression)
        assert e.calculate({1: 6.0, 2: 3.0}) == pytest.approx(-2.0)


class TestNegatedExpression:
    def test_double_negation(self):
        p = P(1)
        assert -(-p) == p

    def test_double_negation_evaluates_correctly(self):
        assert (-(-P(1))).calculate({1: 5.0}) == pytest.approx(5.0)

    def test_negation_of_constant_folds(self):
        assert -C(5) == C(-5)

    def test_negation_of_subtraction_flips_operands(self):
        # -(a - b) == b - a
        assert (-(P(1) - P(2))).calculate({1: 3.0, 2: 7.0}) == pytest.approx(4.0)

    def test_negation_of_zero_is_zero(self):
        assert -C(0) == C(0)

    def test_to_gerber_parameter_no_parens(self):
        assert NegatedExpression(P(1)).to_gerber() == '-$1'

    def test_to_gerber_operator_uses_parens(self):
        inner = OperatorExpression(op.add, P(1), P(2))
        assert NegatedExpression(inner).to_gerber() == '-($1+$2)'


class TestToGerber:
    def test_constant_integer(self):
        assert C(5).to_gerber() == '5'

    def test_constant_float(self):
        assert C(1.5).to_gerber() == '1.5'

    def test_parameter(self):
        assert P(1).to_gerber() == '$1'
        assert P(99).to_gerber() == '$99'

    def test_add_operator(self):
        assert OperatorExpression(op.add, P(1), P(2)).to_gerber() == '$1+$2'

    def test_sub_operator(self):
        assert OperatorExpression(op.sub, P(1), P(2)).to_gerber() == '$1-$2'

    def test_mul_uses_x(self):
        # Gerber spec uses 'x' for multiplication, not '*'
        assert OperatorExpression(op.mul, P(1), P(2)).to_gerber() == '$1x$2'

    def test_div_operator(self):
        assert OperatorExpression(op.truediv, P(1), P(2)).to_gerber() == '$1/$2'

    def test_lhs_operator_gets_parens(self):
        lhs = OperatorExpression(op.add, P(1), P(2))
        e = OperatorExpression(op.mul, lhs, P(3))
        assert e.to_gerber() == '($1+$2)x$3'

    def test_rhs_operator_gets_parens(self):
        rhs = OperatorExpression(op.add, P(2), P(3))
        e = OperatorExpression(op.mul, P(1), rhs)
        assert e.to_gerber() == '$1x($2+$3)'

    def test_nested_lhs_and_rhs_parens(self):
        lhs = OperatorExpression(op.add, P(1), P(2))
        outer = OperatorExpression(op.add, lhs, P(3))
        assert outer.to_gerber() == '($1+$2)+$3'

    def test_negated_mul_to_gerber(self):
        # P(1) * (-P(2)) optimises to -(P(1)*P(2)); NegatedExpression wraps the product
        assert (P(1) * (-P(2))).to_gerber() == '-($1x$2)'

    def test_negated_div_to_gerber(self):
        assert (P(1) / (-P(2))).to_gerber() == '-($1/$2)'

    def test_negative_constant(self):
        assert C(-5).to_gerber() == '-5'


class TestParsing:
    def test_constant_integer(self):
        assert _parse_expression('5', {}, set()) == C(5)

    def test_constant_float(self):
        assert _parse_expression('1.5', {}, set()) == C(1.5)

    def test_parameter_reference(self):
        params = set()
        assert _parse_expression('$1', {}, params) == P(1)
        assert 1 in params

    def test_multiple_parameters_tracked(self):
        params = set()
        _parse_expression('$1+$3', {}, params)
        assert params == {1, 3}

    def test_add(self):
        assert _parse_expression('$1+$2', {}, set()).calculate({1: 3, 2: 4}) == pytest.approx(7)

    def test_sub(self):
        assert _parse_expression('$1-$2', {}, set()).calculate({1: 10, 2: 4}) == pytest.approx(6)

    def test_mul_gerber_x_syntax(self):
        assert _parse_expression('$1x$2', {}, set()).calculate({1: 3, 2: 4}) == pytest.approx(12)

    def test_mul_uppercase_x(self):
        assert _parse_expression('$1X$2', {}, set()).calculate({1: 3, 2: 4}) == pytest.approx(12)

    def test_div(self):
        assert _parse_expression('$1/$2', {}, set()).calculate({1: 10, 2: 4}) == pytest.approx(2.5)

    def test_negation(self):
        assert _parse_expression('-$1', {}, set()).calculate({1: 5}) == pytest.approx(-5)

    def test_parenthesized(self):
        assert _parse_expression('($1+$2)x$3', {}, set()).calculate({1: 3, 2: 4, 3: 2}) == pytest.approx(14)

    def test_known_variable_becomes_variable_expression(self):
        e = _parse_expression('$1', {1: C(10)}, set())
        assert isinstance(e, VariableExpression)

    @pytest.mark.parametrize('gerber_str,py_str,binding', [
        ('$1+$2',         'a+b',     {1: 5,  2: 3         }),
        ('$1-$2',         'a-b',     {1: 5,  2: 3         }),
        ('$1x$2',         'a*b',     {1: 5,  2: 3         }),
        ('$1/$2',         'a/b',     {1: 6,  2: 3         }),
        ('($1+$2)x$3',    '(a+b)*c', {1: 2,  2: 3,  3: 4 }),
        ('$1x$2+$3',      'a*b+c',   {1: 2,  2: 3,  3: 4 }),
        ('-$1+$2',        '-a+b',    {1: 2,  2: 7         }),
        ('$1/$2+$1x$2',   'a/b+a*b', {1: 6,  2: 2         }),
    ])
    def test_parse_and_evaluate(self, gerber_str, py_str, binding):
        e = _parse_expression(gerber_str, {}, set())
        a = binding.get(1, 0)
        b = binding.get(2, 0)
        c = binding.get(3, 0)
        expected = eval(py_str)  # noqa: S307 – controlled test literals only
        assert e.calculate(binding) == pytest.approx(expected)

    @pytest.mark.parametrize('make_expr,binding', [
        (lambda p1, p2, p3: p1 + p2,                {1: 3,  2: 7,  3: 0}),
        (lambda p1, p2, p3: p1 - p2,                {1: 10, 2: 3,  3: 0}),
        (lambda p1, p2, p3: p1 * p2,                {1: 3,  2: 4,  3: 0}),
        (lambda p1, p2, p3: p1 / p2,                {1: 9,  2: 3,  3: 0}),
        (lambda p1, p2, p3: (p1 + p2) * p3,         {1: 2,  2: 3,  3: 4}),
        (lambda p1, p2, p3: p1 * p2 - p3,           {1: 5,  2: 2,  3: 3}),
        (lambda p1, p2, p3: p1 / p2 + p3,           {1: 6,  2: 3,  3: 1}),
        (lambda p1, p2, p3: -p1 + p2,               {1: 3,  2: 7,  3: 0}),
        (lambda p1, p2, p3: p1 * (-p2),             {1: 3,  2: 4,  3: 0}),
        (lambda p1, p2, p3: (-p1) * p2,             {1: 3,  2: 4,  3: 0}),
        (lambda p1, p2, p3: p1 / (-p2),             {1: 9,  2: 3,  3: 0}),
        (lambda p1, p2, p3: (-p1) / p2,             {1: 9,  2: 3,  3: 0}),
        (lambda p1, p2, p3: (p1 + p2) / p3 - p1,   {1: 3,  2: 5,  3: 4}),
        (lambda p1, p2, p3: p1 * p2 + p3 / p1,     {1: 3,  2: 4,  3: 6}),
        (lambda p1, p2, p3: -(p1 + p2) * p3,        {1: 2,  2: 3,  3: 4}),
    ])
    def test_to_gerber_round_trip(self, make_expr, binding):
        """to_gerber() followed by _parse_expression() must preserve the evaluated value."""
        original = make_expr(P(1), P(2), P(3))
        gerber = original.to_gerber()
        parsed = _parse_expression(gerber, {}, set())
        assert parsed.calculate(binding) == pytest.approx(original.calculate(binding))


class TestUnitExpression:
    def test_mm_to_mm_unchanged(self):
        assert UnitExpression(C(25.4), MM).calculate(unit=MM) == pytest.approx(25.4)

    def test_inch_to_mm(self):
        assert UnitExpression(C(1.0), Inch).calculate(unit=MM) == pytest.approx(MILLIMETERS_PER_INCH)

    def test_mm_to_inch(self):
        assert UnitExpression(C(25.4), MM).calculate(unit=Inch) == pytest.approx(25.4 / MILLIMETERS_PER_INCH)

    def test_inch_to_inch_unchanged(self):
        assert UnitExpression(C(2.0), Inch).calculate(unit=Inch) == pytest.approx(2.0)

    def test_none_unit_passes_through(self):
        assert UnitExpression(C(5.0), None).calculate(unit=MM) == pytest.approx(5.0)

    def test_negation_preserves_unit(self):
        neg = -UnitExpression(C(5.0), MM)
        assert isinstance(neg, UnitExpression) and neg.unit == MM
        assert neg.calculate(unit=MM) == pytest.approx(-5.0)

    def test_add_same_unit(self):
        result = UnitExpression(C(3.0), MM) + UnitExpression(C(4.0), MM)
        assert isinstance(result, UnitExpression)
        assert result.calculate(unit=MM) == pytest.approx(7.0)

    def test_add_mixed_units_converts(self):
        # 1 inch + 1 mm, result held in Inch
        result = UnitExpression(C(1.0), Inch) + UnitExpression(C(1.0), MM)
        assert result.calculate(unit=Inch) == pytest.approx(1.0 + 1.0 / MILLIMETERS_PER_INCH)

    def test_add_scalar_raises(self):
        with pytest.raises(ValueError):
            UnitExpression(C(5.0), MM) + C(3.0)

    def test_radd_scalar_raises(self):
        # BUG: asymmetric unit safety — C(3.0) + UnitExpression(...) does NOT raise because
        # Python dispatches to Expression.__add__ first, which has no unit awareness.
        # Only plain Python scalars (not Expression subclasses) trigger __radd__ on UnitExpression.
        # There is no really nice fix for this, so we just leave it in for now.
        with pytest.raises(ValueError):
            5.0 + UnitExpression(C(5.0), MM)

    def test_mul_by_scalar(self):
        result = UnitExpression(C(3.0), MM) * C(2)
        assert isinstance(result, UnitExpression)
        assert result.calculate(unit=MM) == pytest.approx(6.0)

    def test_div_by_scalar(self):
        result = UnitExpression(C(6.0), MM) / C(2)
        assert isinstance(result, UnitExpression)
        assert result.calculate(unit=MM) == pytest.approx(3.0)

    def test_nested_unit_expression_flattens(self):
        # Wrapping a UnitExpression in another converts rather than double-wrapping
        inner = UnitExpression(C(1.0), Inch)
        outer = UnitExpression(inner, MM)
        assert not isinstance(outer.expr, UnitExpression)
        assert outer.calculate(unit=MM) == pytest.approx(MILLIMETERS_PER_INCH)

    def test_parameters_forwarded(self):
        assert list(UnitExpression(P(1), MM).parameters()) == [P(1)]


class TestExprHelper:
    def test_passthrough_expression(self):
        p = P(1)
        assert expr(p) is p

    def test_wraps_int(self):
        result = expr(5)
        assert isinstance(result, ConstantExpression) and result.value == 5

    def test_wraps_float(self):
        result = expr(3.14)
        assert isinstance(result, ConstantExpression) and result.value == pytest.approx(3.14)


class TestVariableExpression:
    def test_optimized_non_operator_unwraps(self):
        # A VariableExpression wrapping something that simplifies to a non-OperatorExpression
        # should unwrap and return the simplified value directly.
        result = VariableExpression(C(5)).optimized()
        assert result == C(5)

    def test_optimized_keeps_operator_expression(self):
        ve = VariableExpression(OperatorExpression(op.add, P(1), P(2)))
        assert isinstance(ve.optimized(), VariableExpression)

    def test_to_gerber_without_register_uses_inner(self):
        assert VariableExpression(C(42)).to_gerber(register_variable=None) == '42'

    def test_to_gerber_with_register_allocates_dollar_variable(self):
        allocated = {}

        def register(e):
            key = e.to_gerber()
            if key not in allocated:
                allocated[key] = len(allocated) + 1
            return allocated[key]

        inner = OperatorExpression(op.add, P(1), P(2))
        result = VariableExpression(inner).to_gerber(register_variable=register)
        assert result.startswith('$') and int(result[1:]) >= 1
