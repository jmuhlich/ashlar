import os
import itertools
import concurrent.futures
import attr
import attr.validators as av
import numpy as np
from . import metadata, geometry, util, align, plot
from .util import attrib, cached_property


@attr.s(frozen=True, kw_only=True)
class RegistrationProcess(object):
    """Container for parameters and algorithms for registration.

    The default `neighbor_overlap_cutoff` percentile of 50 was chosen to reject
    diagonally adjacent tiles in a regular grid. As there are slightly more
    up-down and left-right neighbors than diagonal neighbors in a grid, the 50th
    percentile will correspond to an up-down or left-right neighbor intersection
    area. Unusual tile position collections may require tuning of this
    parameter.

    The default `neighbor_overlap_bias` value of 0 will only consider strictly
    overlapping tiles. Increasing this parameter will also consider touching or
    even disjoint tiles. The typical use case for increasing `bias` is for data
    sets where neighboring stage positions just touch, but due to stage position
    error the imaged regions do have some actual overlap that can be
    registered. Specifying a small bias value will include these touching
    neighbors in the neighbors graph.

    For reproducible random number generation, `random_seed` may be set to a
    fixed value. For further control, a numpy RandomState instance may be passed
    via `random_state`. Use one or the other of these arguments, not both.

    """

    tileset = attrib(
        validator=av.instance_of(metadata.TileSet),
        doc="TileSet to register."
    )
    channel_number = attrib(
        converter=int,
        doc="Index of imaging channel to use for registration."
    )
    neighbor_overlap_cutoff = attrib(
        default=50, converter=float,
        validator=util.validate_range(0, 100),
        doc="Percentile cutoff for determining which tiles are neighbors."
    )
    neighbor_overlap_bias = attrib(
        default=0, converter=float,
        doc="Distance to expand/contract tile bounds before neighbor testing."
    )
    overlap_minimum_size = attrib(
        default=0, converter=float,
        doc="Neighbor overlap windows will be expanded to at least this size."
    )
    num_permutations = attrib(
        default=1000, converter=int,
        doc="Number of permutations used to sample the error distribution."
    )
    random_seed = attrib(
        default=None, validator=av.optional(av.instance_of(int)),
        doc="Seed for the pseudo-random number generator."
    )
    random_state = attrib(
        default=None,
        validator=av.optional(av.instance_of(np.random.RandomState)),
        doc="A numpy RandomState, constructed using `random_seed` by default."
    )

    def __attrs_post_init__(self):
        if self.random_seed is not None and self.random_state is not None:
            raise ValueError(
                "Can only specify random_seed or random_state, not both."
            )
        if self.random_state is None:
            rand = np.random.RandomState(self.random_seed)
            object.__setattr__(self, 'random_state', rand)

    @cached_property
    def graph(self):
        """Neighbors graph of the tileset."""
        return self.tileset.build_neighbors_graph(
            self.neighbor_overlap_cutoff, self.neighbor_overlap_bias
        )

    @cached_property
    def plot(self):
        """Plotter utility object (see plot.RegistrationProcessPlotter)."""
        return plot.RegistrationProcessPlotter(self)

    def random_tile_pair_index(self):
        return self.random_state.randint(len(self.tileset), size=2)

    def tile_random_neighbor_index(self, i):
        neighbors = list(self.graph.neighbors(i))
        return self.random_state.choice(neighbors)

    def get_tile(self, i):
        return self.tileset.get_tile(i, self.channel_number)

    def neighbor_permutation_tasks(self):
        for i in range(self.num_permutations):
            while True:
                a, b = self.random_tile_pair_index()
                if a != b and (a, b) not in self.graph.edges:
                    break
            a_neighbor = self.tile_random_neighbor_index(a)
            new_b_bounds = self.tileset.rectangles[a_neighbor]
            yield a, b, new_b_bounds

    def compute_neighbor_permutation(self, a, b, new_b_bounds):
        plane1 = self.get_tile(a).plane
        plane2 = self.get_tile(b).plane
        plane2 = attr.evolve(plane2, bounds=new_b_bounds)
        intersection1 = plane1.intersection(plane2, self.overlap_minimum_size)
        intersection2 = plane2.intersection(plane1, self.overlap_minimum_size)
        alignment = align.register_planes(intersection1, intersection2)
        return alignment.error

    def neighbor_alignment_tasks(self):
        return self.graph.edges

    def compute_neighbor_alignment(self, a, b):
        plane1 = self.get_tile(a).plane
        plane2 = self.get_tile(b).plane
        intersection1 = plane1.intersection(plane2, self.overlap_minimum_size)
        intersection2 = plane2.intersection(plane1, self.overlap_minimum_size)
        alignment = align.register_planes(intersection1, intersection2)
        return align.EdgeTileAlignment(alignment, a, b)
