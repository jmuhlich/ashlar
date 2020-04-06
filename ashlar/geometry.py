from __future__ import division
import numbers
import attr
import numpy as np
from .util import attrib, cached_property


@attr.s(frozen=True)
class Vector(object):
    """Geometric vector in 2-D, with floating-point coordinates."""

    y = attrib(converter=float, doc="Y coordinate.")
    x = attrib(converter=float, doc="X coordinate.")

    @classmethod
    def from_ndarray(cls, a):
        """Construct Vector from numpy array [Y, X]."""
        if a.shape != (2,):
            raise ValueError("array shape must be (2,)")
        return cls(*a)

    def min(self):
        return min(self.y, self.x)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.y + other.y, self.x + other.x)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Vector(self.y - other.y, self.x - other.x)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Vector(self.y * other, self.x * other)

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Vector(self.y / other, self.x / other)

    __div__ = __truediv__

    def __neg__(self):
        return Vector(-self.y, -self.x)

    def __array__(self):
        return np.array([self.y, self.x])


@attr.s(frozen=True)
class Rectangle(object):
    """Axis-aligned rectangle in 2-D."""

    vector1 = attrib(
        validator=attr.validators.instance_of(Vector),
        doc="Lower-left corner."
    )
    vector2 = attrib(
        validator=attr.validators.instance_of(Vector),
        doc="Upper-right corner."
    )

    @classmethod
    def from_shape(cls, vector, shape):
        """Construct Rectangle from a Vector point and Vector shape."""
        return cls(vector, vector + shape)

    @classmethod
    def rmin(cls, v1, v2):
        """Return Vector at "rectangle minimum" of v1 and v2."""
        return Vector(x=min(v1.x, v2.x), y=min(v1.y, v2.y))

    @classmethod
    def rmax(cls, v1, v2):
        """Return Vector at "rectangle maximum" of v1 and v2."""
        return Vector(x=max(v1.x, v2.x), y=max(v1.y, v2.y))

    def __attrs_post_init__(self):
        # Normalize to make vector1 the lower corner and vector2 the upper.
        v1 = self.rmin(self.vector1, self.vector2)
        v2 = self.rmax(self.vector1, self.vector2)
        object.__setattr__(self, 'vector1', v1)
        object.__setattr__(self, 'vector2', v2)

    def __add__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Rectangle(self.vector1 + other, self.vector2 + other)

    def __sub__(self, other):
        if not isinstance(other, Vector):
            return NotImplemented
        return Rectangle(self.vector1 - other, self.vector2 - other)

    def __mul__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Rectangle(self.vector1 * other, self.vector2 * other)

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            return NotImplemented
        return Rectangle(self.vector1 / other, self.vector2 / other)

    __div__ = __truediv__

    @cached_property
    def shape(self):
        """A Vector created from the rectangle's width and height."""
        return self.vector2 - self.vector1

    @cached_property
    def area(self):
        """Product of rectangle's width and height."""
        s = self.shape
        return s.x * s.y

    @cached_property
    def center(self):
        """Vector at the center of the rectangle."""
        return (self.vector1 + self.vector2) / 2

    @cached_property
    def as_slice(self):
        """Representation of rectangle, rounded, as a numpy array slice."""
        ys = self.vector1.y, self.vector2.y
        xs = self.vector1.x, self.vector2.x
        indices = [np.round(v).astype(int) for v in (ys, xs)]
        slices = tuple(slice(*v) for v in indices)
        return slices

    def inflate(self, d):
        """Return a Rectangle that's `d` units bigger on all sides.

        Each side of the rectangle is pushed out from its center by `d`
        units. Negative values for `d` are also allowed, to pull the sides
        in. Attempting to shrink a rectangle by more than half of its width or
        height will result in a rectangle with both corners at the center of the
        original rectangle and therefore zero area.

        """
        d_v = Vector(d, d)
        v1 = self.vector1 - d_v
        v2 = self.vector2 + d_v
        center = self.center
        if v1.x > center.x or v1.y > center.y:
            v1 = v2 = center
        return Rectangle(v1, v2)

    def intersection(self, other, min_overlap=0):
        """Return the intersection of self and `other` as a Rectangle.

        If self and `other` don't overlap or merely touch, the returned
        Rectangle will always have an area of zero. Its two corners represent
        the points along one edge of self closest to the corners of `other`,
        using the city block metric.

        """
        p1 = self.nearest_point(other.vector1)
        p2 = self.nearest_point(other.vector2)
        rect = Rectangle(p1, p2)
        padding = min_overlap - rect.shape.min()
        if padding > 0:
            rect = self.intersection(rect.inflate(padding))
        return rect

    def nearest_point(self, other):
        """Return the point inside self that's nearest to `other`.

        The city block distance metric is used, NOT Euclidean!

        """
        y = np.clip(other.y, self.vector1.y, self.vector2.y)
        x = np.clip(other.x, self.vector1.x, self.vector2.x)
        return Vector(y, x)
