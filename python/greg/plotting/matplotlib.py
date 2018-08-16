import scipy.sparse as sps
import matplotlib.pyplot as plt

def plot_abline(intercept, slope, ax=None, *args, **kwargs):
    """ Plot a line given its slope and intercept """
    if ax:
        get_xlim = ax.get_xlim
        get_ylim = ax.get_ylim
        plot = ax.plot
    else:
        get_xlim = plt.xlim
        get_ylim = plt.ylim
        plot = plt.plot

    x_min, x_max = get_xlim()
    y_min, y_max = get_ylim()

    x0, x1 = x_min, x_max
    y0 = max(y_min, intercept + slope * x0)
    y1 = min(y_max, intercept + slope * x1)

    plot([x0, x1], [y0, y1], *args, **kwargs)


def sparse_hexbin(m, ax=None):
    """ View a sparse matrix """
    m_coo = sps.coo_matrix(m)
    hexbin = ax.hexbin if ax is not None else plt.hexbin
    return hexbin(m_coo.col, m_coo.row)
