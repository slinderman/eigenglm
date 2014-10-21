"""
Build a set of plotting classes. These classes should be specific
to particular components or state variables. For example, we might
have a plotting class for the network. The classes should be initialized
with a model to determine how they should plot the results.

The classes should be able to:
 - plot either a single sample or the mean of a sequence of samples
   along with error bars.

 - take in an axis (or a figure) or create a new figure if not specified
 - take in a color or colormap for plotting


"""
import numpy as np
import matplotlib.pyplot as plt
from hips.plotting.colormaps import gradient_cmap

rwb_cmap = gradient_cmap([[1,0,0],
                          [1,1,1],
                          [0,0,0]])

class PlotProvider(object):
    """
    Abstract class for plotting a sample or a sequence of samples
    """
    def __init__(self, figsize=None):
        """
        Create a figure for this plot
        """
        self.fig = plt.figure(figsize=figsize)

    def plot(self, sample, ax=None, update=True):
        """
        Plot the sample or sequence of samples
        """
        raise NotImplementedError()

class NetworkPlotProvider(PlotProvider):
    """
    Class to plot the connectivity network
    """

    def __init__(self,
                 figsize=None,
                 A_true=None,
                 W_true=None,
                 W_min=None,
                 W_max=None,
                 cmap=rwb_cmap,
                 px_per_node=10):
        super(NetworkPlotProvider, self).__init__(figsize)

        self.A_true = A_true
        self.W_true = W_true
        self.W_min = W_min
        self.W_max = W_max
        self.cmap = cmap
        self.px_per_node = px_per_node

        # Create axes
        if self.A_true is not None and self.W_true is not None:
            self.ax = self.fig.add_subplot(121, aspect='equal')
            self.ax_true = self.fig.add_subplot(122, aspect='equal')

            # If the bounds are not given, use the true bounds (plus some slack
            if self.W_max is None and self.W_min is None:
                self.W_max = 1.33 * np.amax(abs(self.A_true * self.W_true))
                self.W_min = -self.W_max

            # Plot the true network
            self.plot((self.A_true, self.W_true), ax=self.ax_true, title='True Network')
            plt.show()
        else:
            self.ax = self.fig.add_subplot(111, aspect='equal')

    def plot(self, samples, ax=None, update=True, title="Inferred Network"):
        # Get the axis
        if ax is None:
            ax = self.ax

        if update:
            ax.cla()

        # Ensure sample is a list
        if not isinstance(samples, list):
            samples = [samples]

        # Extract A and W
        A_samples = np.array([s[0] for s in samples])
        W_samples = np.array([s[1] for s in samples])
        AW_samples = A_samples * W_samples

        D,N,_ = AW_samples.shape
        assert AW_samples.shape[2] == N

        # Plot the mean network
        AW = AW_samples.mean(axis=0)


        # Make sure bounds are set
        W_max = np.amax(abs(AW)) if self.W_max is None else self.W_max
        W_min = -W_max if self.W_min is None else self.W_min

        # Plot the network as an image
        im = ax.imshow(np.kron(AW, np.ones((self.px_per_node, self.px_per_node))),
                       vmin=W_min, vmax=W_max,
                       extent=[0,1,0,1],
                       interpolation='nearest',
                       cmap=self.cmap)

        # Set ticks
        ticks = 0.5/N + np.arange(N, dtype=np.float)/N
        labels = 1 + np.arange(N)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlabel('Incoming')
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels[::-1])
        ax.set_ylabel('Outgoing')

        if update:
            plt.pause(0.0001)

        if title is not None:
            ax.set_title(title)


class LocationPlotProvider(PlotProvider):
    """
    Plot the latent locations of the neurons
    """
    def plot(self, xs, ax=None, name='location_provider', color='k'):
        """
        Plot a histogram of the inferred locations for each neuron
        """
        # Ensure sample is a list
        if not isinstance(xs, list):
            xs = [xs]

        if name not in xs[0]['latent']:
            return

        # Get the locations
        loccomp = self.population.latent.latentdict[name]
        locprior = loccomp.location_prior
        locvars = loccomp.get_variables()
        Ls = np.array([seval(loccomp.Lmatrix,
                            locvars, x['latent'][name])
                       for x in xs])
        [N_smpls, N, D] = Ls.shape

        for n in range(N):
            # plt.subplot(1,N,n+1, aspect=1.0)
            # plt.title('N: %d' % n)

            if N_smpls == 1:
                if D == 1:
                    plt.plot([Ls[0,n,0], Ls[0,n,0]],
                             [0,2], color=color, lw=2)
                elif D == 2:
                    ax.plot(Ls[0,n,1], Ls[0,n,0], 's',
                             color=color, markerfacecolor=color)
                    ax.text(Ls[0,n,1]+0.25, Ls[0,n,0]+0.25, '%d' % n,
                             color=color)

                    # Set the limits
                    ax.set_xlim((locprior.min0-0.5, locprior.max0+0.5))
                    ax.set_ylim((locprior.max1+0.5, locprior.min1-0.5))
                else:
                    raise Exception("Only plotting locs of dim <= 2")
            else:
                # Plot a histogram of samples
                if D == 1:
                    ax.hist(Ls[:,n,0], bins=20, normed=True, color=color)
                elif D == 2:
                    ax.hist2d(Ls[:,n,1], Ls[:,n,0], bins=np.arange(-0.5,5), cmap='Reds', alpha=0.5, normed=True)

                    # Set the limits
                    ax.set_xlim((locprior.min0-0.5, locprior.max0+0.5))
                    ax.set_ylim((locprior.max1+0.5, locprior.min1-0.5))

                    # ax.colorbar()
                else:
                    raise Exception("Only plotting locs of dim <= 2")



