import numpy as np
from functools import (wraps, reduce)
from scipy.spatial import (ConvexHull)
from dataclasses import (dataclass)
from matplotlib import pyplot as plt
from matplotlib import collections


def memoize(method):
    @wraps(method)
    def memoized_method(self):
        cache_name = "_{}_memo".format(method.__name__)
        if not hasattr(self, cache_name):
            setattr(self, cache_name, method(self))
        return getattr(self, cache_name)
    return memoized_method


@dataclass
class Box:
    N: int       # logical size
    L: float     # physical size

    @property
    def dim(self):
        return 2

    @property
    def res(self):
        return self.L/self.N

    @property
    @memoize
    def fftfreq(self):
        return np.fft.fftfreq(self.N, self.L/(self.N*2*np.pi))

    @property
    @memoize
    def k_abs(self):
        k = self.fftfreq
        return np.sqrt(k[None,:]**2 + k[:,None]**2)

    @property
    @memoize
    def grid(self):
        return np.indices(self.shape) * (self.L / self.N) - (self.L / 2)
    
    @property
    @memoize
    def grid_bounds(self):
        return np.indices(i + 1 for i in self.shape) * (self.L / self.N) - (self.L / 2)

    @property
    def points(self):
        return self.grid.transpose([1, 2, 0]).reshape(-1, 2)

    @property
    def shape(self):
        return (self.N, self.N)

    @property
    def size(self):
        return reduce(operator.mul, self.shape, initial=1)


def random_density(box, power_spectrum):
    noise = np.random.normal(0.0, 1.0, size=box.shape)
    with np.errstate(all='ignore'):
        delta_f = np.fft.fftn(noise) * np.sqrt(power_spectrum(box.k_abs))
    delta_f[0,0] = 0.0
    delta = np.fft.ifftn(delta_f).real
    delta /= delta.std()
    return delta


def measure_power_spectrum(box, field):
    p = np.argsort(k_abs, axis=None).reshape(box.shape)
    f = np.fft.fftn(field)
    power = (f * f.conj()).real
    kspec = box.k_abs.flat[p].mean(axis=1)
    pspec = power.flat[p].mean(axis=1) / box.size
    return kspec, pspec


def compute_potential(box, density):
    with np.errstate(all='ignore'):
        f = -np.fft.fftn(density) / box.k_abs**2
    f[0,0] = 0.0
    return np.fft.ifftn(f).real


def delaunay_areas(box, ch, selection):
    """Compute areas of selected simplices.

    :param ch: Convex Hull
    :param selection: indices of selected simplices.
    :return: ndarray of areas, having same shape as `selection`.
    """
    a, b, c = ch.points[ch.simplices[selection]][:,:,:box.dim].transpose([1,0,2])
    return np.abs(np.cross(a - b, c - b) / 2)


def delaunay_edges(box, ch, selection, valid):
    edge_simpl = edges(box, ch, valid)
    return np.array([np.intersect1d(x[0], x[1]) for x in ch.simplices[edge_simpl]])


def delaunay_class(box, ch, selection, threshold):
    """Compute the classification of each simplex using the given threshold.

    :param ch: Convex hull
    :param selection: indices of selected simplices.
    :return: Number of edges with square length longer than the threshold for
    each simplex. Array of integer of same shape as selection.

    ======  =========
    number  class
    ======  =========
    0       void
    1       curto-parabolic point
    2       filament
    3       node
    ======  =========
    """
    pts = ch.points[ch.simplices[selection]][:,:,:box.dim]
    dists2 = ((pts - np.roll(pts, 1, axis=1))**2).sum(axis=2)
    return np.where(dists2 > (threshold * box.res)**2, 1, 0).sum(axis=1)


def voronoi_points(ch, selection):
    """Compute the dual vertices of the selected simplices."""
    return - ch.equations[selection][:,:2] / ch.equations[selection][:,2][:,None] / 2


def edges(box, ch, valid):
    """List all edges in the convex hull."""
    nb = np.zeros(shape=(len(valid), 2*(box.dim+1)), dtype=int)
    nb[:,1::2] = ch.neighbors[valid]  # neighbours index into simplices and equations
    nb[:,0::2] = valid[:,None]        # so does `valid`, we intersperse them to create pairs
    return np.unique(np.sort(nb.reshape([-1, 2]), axis=1), axis=0)


def edge_points(ch, edges):
    """Compute the dual vertices for all edges."""
    save = np.seterr(invalid = 'ignore', divide = 'ignore')
    pts = - ch.equations[:,:2] / ch.equations[:,2][:,None] / 2
    np.seterr(**save)
    return pts[edges]


def edge_length(ch, edges):
    """Get the length of each edge (in the Delaunay triangulation)."""
    # find the points common to both simplices, should always be two points
    # this operation performs a loop in Python TODO: find numpy expression
    edge_verts = np.array([np.intersect1d(x[0], x[1]) for x in ch.simplices[edges]])
    return np.sqrt(np.sum((ch.points[edge_verts][:,1,:2] - ch.points[edge_verts][:,0,:2])**2, axis=1))


def plot_structure(box, ch, xlim, ylim, ax=None, point_scale=10, line_scale=1.0, plot_grid=True, filament_color='maroon'):
    """Plot the power diagram."""
    selection = np.where(np.dot(ch.equations[:,0:3], [0, 0, -1]) > 0.00001)[0]
    valid = selection[np.where(np.all(np.isin(ch.neighbors[selection], selection), axis=1))[0]]

    m_edges = edges(box, ch, valid)
    m_edge_lengths = edge_length(ch, m_edges)
    m_edge_points = edge_points(ch, m_edges)

    edge_sel = np.where(m_edge_lengths > np.sqrt(2)*box.res)[0]

    if plot_grid:
        lc_grid = collections.LineCollection(m_edge_points, linewidths=1.0, color='#888888')

    lc = collections.LineCollection(m_edge_points[edge_sel], linewidths=line_scale*m_edge_lengths[edge_sel], color=filament_color)
    lc.set_capstyle('round')

    X = voronoi_points(ch, valid)
    mass = delaunay_areas(box, ch, valid) / box.res**2
    big_points = np.where(delaunay_class(box, ch, valid, threshold=1.0) > 2)

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if plot_grid:
        ax.add_collection(lc_grid)

    ax.add_collection(lc)
    ax.set_aspect('equal')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.scatter(X[big_points,0], X[big_points,1], s=mass[big_points] * point_scale * 1.25, zorder=2, c='black', alpha=0.5)
    ax.scatter(X[big_points,0], X[big_points,1], s=mass[big_points] * point_scale, zorder=4, alpha=0.5)


def run_model(box, pot, time):
    return ConvexHull(np.c_[box.points, ((box.grid**2).sum(axis=0) - 2*time * pot).flat])
