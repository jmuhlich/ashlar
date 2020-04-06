import os
import concurrent.futures
import attr
import attr.validators as av
import numpy as np
import networkx as nx
from . import metadata, geometry, util, align, plot
from .util import attrib, cached_property
from .geometry import Vector, Rectangle


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
    filter_sigma = attrib(
        default=0.0, converter=float,
        validator=util.validate_nonnegative(),
        doc="Width of Gaussian filter to apply to images before alignment."
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
    max_permutations = attrib(
        default=1000, converter=int,
        doc="Maximum permutations used to sample the error distribution."
    )
    error_threshold_percentile = attrib(
        default=1, converter=float,
        validator=util.validate_range(0, 100),
        doc="Percentile of error distribution to use as max allowable error.",
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

    def permutation_error_tasks(self):
        # Compute error threshold for rejecting aligments. We generate a
        # distribution of error scores for many known non-overlapping image
        # regions and take a certain percentile as the maximum allowable error.
        # The percentile becomes our accepted false-positive ratio.
        edges = self.graph.edges
        num_tiles = len(self.tileset)
        # If not enough overlapping tiles to matter, skip this whole thing.
        if len(edges) <= 1:
            return
        rects = self.tileset.rectangles
        widths = np.array([
            rects[t1].intersection(
                rects[t2], min_overlap=self.overlap_minimum_size
            ).shape.min()
            for t1, t2 in edges
        ])
        w = widths.max()
        # Strip template rectangle -- horizontal, across the entire image width.
        srect = Rectangle.from_shape(
            Vector(0, 0), Vector(w, self.tileset.tile_shape[1])
        )
        max_offset = self.tileset.tile_shape[0] - w
        # Number of possible pairs minus number of actual neighbor pairs.
        num_distant_pairs = num_tiles * (num_tiles - 1) // 2 - len(edges)
        # Reduce permutation count for small datasets -- there are fewer
        # possible truly distinct strips with fewer tiles. The calculation here
        # is just a heuristic, not rigorously derived.
        num_permutations = self.max_permutations
        if num_distant_pairs <= 8:
            num_permutations = (num_distant_pairs + 1) * 10
        # Generate n random non-overlapping image strips.
        max_tries = 100
        for i in range(num_permutations):
            # Limit tries to avoid infinite loop in pathological cases.
            for current_try in range(max_tries):
                t1, t2 = self.random_tile_pair_index()
                o1, o2 = self.random_state.randint(max_offset, size=2)
                #o1, o2 = self.random_state.random(2) * max_offset
                r1 = srect + rects[t1].vector1 + Vector(o1, 0)
                r2 = srect + rects[t2].vector1 + Vector(o2, 0)
                # Check for non-overlapping strips and abort the retry loop.
                if r1.intersection(r2).area == 0:
                    break
            else:
                # Retries exhausted. This should be very rare.
                warn_data(
                    "Could not find non-overlapping strips in {max_tries} tries"
                )
            yield t1, t2, r1, r2

    def compute_permutation_error(self, a, b, a_region, b_region):
        img1 = self.get_tile(a).plane.crop(a_region).image
        img2 = self.get_tile(b).plane.crop(b_region).image
        min_shape = np.minimum(img1.shape, img2.shape)
        img1 = img1[:min_shape[0], :min_shape[1]]
        img2 = img2[:min_shape[0], :min_shape[1]]
        _, error = align.register(img1, img2, self.filter_sigma)
        return error

    def compute_error_threshold(self, neighbor_permutation_errors):
        threshold = np.percentile(
            neighbor_permutation_errors,
            self.error_threshold_percentile
        )
        return threshold

    def neighbor_alignment_tasks(self):
        return self.graph.edges

    def compute_neighbor_alignment(self, a, b):
        plane1 = self.get_tile(a).plane
        plane2 = self.get_tile(b).plane
        intersection1 = plane1.intersection(plane2, self.overlap_minimum_size)
        intersection2 = plane2.intersection(plane1, self.overlap_minimum_size)
        alignment = align.register_planes(
            intersection1, intersection2, self.filter_sigma
        )
        return align.EdgeTileAlignment(alignment, a, b)

    def compute_spanning_tree(self, alignments, error_threshold):
        alignments = (a for a in alignments if a.error < error_threshold)
        edges = ((a.tile_index_1, a.tile_index_2, a.error) for a in alignments)
        wg = nx.Graph()
        wg.add_nodes_from(self.graph)
        wg.add_weighted_edges_from(edges)
        spanning_tree = nx.Graph()
        spanning_tree.add_nodes_from(wg)
        for c in nx.connected_components(wg):
            cc = wg.subgraph(c)
            center = nx.center(cc)[0]
            paths = nx.single_source_dijkstra_path(cc, center).values()
            for path in paths:
                nx.add_path(spanning_tree, path)
        return spanning_tree


@attr.s(kw_only=True)
class RegistrationProcessExecutor(object):

    process = attrib(
        validator=av.instance_of(RegistrationProcess),
        doc="RegistrationProcess to execute."
    )
    pool = attrib(
        factory=util.SerialExecutor,
        validator=av.optional(av.instance_of(concurrent.futures.Executor)),
        doc="Concurrent executor for parallel execution."
    )
    verbose = attrib(
        default=False, converter=bool,
        doc="Whether to display progress messages.",
    )
    permutation_errors_ = attrib(default=None)
    error_threshold_ = attrib(default=None)
    neighbor_alignments_ = attrib(default=None)
    spanning_tree_ = attrib(default=None)
    positions_local_ = attrib(default=None)
    linear_model_ = attrib(default=None)
    positions_ = attrib(default=None)

    @cached_property
    def plot(self):
        """Plotter utility object (see plot.RegistrationProcessExecutorPlotter)."""
        return plot.RegistrationProcessExecutorPlotter(self)

    def run(self):
        self.permutation_errors()
        self.error_threshold()
        self.neighbor_alignments()
        self.spanning_tree()
        #self.positions()
        #self.linear_model()

    def permutation_errors(self):
        self.permutation_errors_ = self._execute(
            self.process.compute_permutation_error,
            self.process.permutation_error_tasks()
        )

    def error_threshold(self):
        self.error_threshold_ = self.process.compute_error_threshold(
            self.permutation_errors_
        )

    def neighbor_alignments(self):
        self.neighbor_alignments_ = self._execute(
            self.process.compute_neighbor_alignment,
            self.process.neighbor_alignment_tasks()
        )

    def spanning_tree(self):
        self.spanning_tree_ = self.process.compute_spanning_tree(
            self.neighbor_alignments_, self.error_threshold_
        )

    def positions_local(self):
        self.positions_local_ = self.process.compute_positions_local(
            self.spanning_tree_, self.neighbor_alignments_
        )

    def linear_model(self):
        self.linear_model_ = self.process.compute_linear_model(
            self.spanning_tree_, self.positions_local_
        )

    def positions(self):
        self.positions_ = self.process.compute_positions(
            self.spanning_tree_, self.positions_local_, self.linear_model_
        )

    def _execute(self, fn, task_args):
        progress = util.future_progress if self.verbose else util.future_results
        futures = util.executor_submit(self.pool, fn, task_args)
        results = progress(futures)
        return results
