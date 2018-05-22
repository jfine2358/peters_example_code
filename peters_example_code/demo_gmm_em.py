import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal


"""
A simple demo of using EM to train a gaussian mixture model.  
"""


def get_rng(rng):
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


def generate_data(means, covariances, priors, n_samples, rng=None):
    """
    Generate data from a gaussian mixture model
    :param Array(n_components, n_dims) means: The means for each gaussian
    :param Array(n_components, n_dims, n_dims) covariances: The covariance matrix for each gaussian
    :param Array(n_components) priors: The prior on each gaussian
    :param int n_samples: Number of samples
    :param rng: Random number generator or seed
    :return Array(n_samples, n_dims): Generated data
    """
    rng = get_rng(rng)
    assert len(means)==len(covariances)
    distros = [multivariate_normal(mean = mu, cov=cov) for mu, cov in zip(means, covariances)]
    identity = rng.choice(a=len(priors), p=priors, size=n_samples)
    samples = np.array([distros[this_id].rvs(random_state=rng) for this_id in identity])
    return samples


def e_step(x, means, covs, priors):
    """
    Compute the Posterior, i.e. the "responsibility" of each gaussian for each data point.
    :param Array(n_samples, n_dims) x: The data to fit
    :param Array(n_components, n_dims) means: The means for each gaussian
    :param Array(n_components, n_dims, n_dims) covs: The covariance matrix for each gaussian
    :param Array(n_components) priors: The prior on each gaussian
    :return Array(n_samples, n_components): The "responsibilities" of each gaussian for each data point (ie the posterior)
    """
    assert len(means)==len(covs)==len(priors)
    q = np.array([multivariate_normal(mean = mu, cov=cov).pdf(x) * pi for mu, cov, pi in zip(means, covs, priors)]).T  # (n_samples, n_components)
    return q/q.sum(axis=1, keepdims=True)


def mean_log_likelihood(x, means, covs, priors):
    """
    Return the mean log-likelihood of datapoints under the model.
    :param Array(n_samples, n_dims) x: The data to fit
    :param Array(n_components, n_dims) means: The means for each gaussian
    :param Array(n_components, n_dims, n_dims) covs: The covariance matrix for each gaussian
    :param Array(n_components) priors: The prior on each gaussian
    :return float: The mean log-likelihood over samples
    """
    q = np.array([multivariate_normal(mean = mu, cov=cov).pdf(x) * pi for mu, cov, pi in zip(means, covs, priors)]).sum(axis=0)  # (n_samples,)
    return np.mean(np.log(q))


def m_step(x, q, means, covs, priors):
    """
    Maximize the parameters given the current posterior estimate.

    :param Array(n_samples, n_dims) x: The data to fit
    :param Array(n_samples, n_components) q: The "responsibilities" of each gaussian for each data point (ie the posterior)
    :param Array(n_components, n_dims) means: The means for each gaussian
    :param Array(n_components, n_dims, n_dims) covs: The covariance matrix for each gaussian
    :param Array(n_components) priors: The prior on each gaussian
    :return Tuple[Array(n_components, n_dims), Array(n_components, n_dims, n_dims), Array(n_components)]: The updated parameters
    """
    assert len(means)==len(covs)==len(priors)

    # M step .. We want to maximize  V = sum_i( E_{q(t_i) log p(p_x, t_i, | theta))
    new_means = (q[:, :, None] * x[:, None, :]).sum(axis=0) / q[:, :, None].sum(axis=0)  # (n_components, n_dims)

    # And setting dV/d_sigma = 0
    new_covs = [np.einsum('ni,nj->ij', delta, delta) / q[:, c].sum(axis=0) for c, mu in enumerate(new_means) for delta in [q[:, [c]]*(x - mu)]]  # (n_components, n_dims, n_dims)

    # And dV/d_pi
    new_priors = q.mean(axis=0)  # (n_dims, )

    return new_means, new_covs, new_priors


class FunPlot2D(object):
    """
    Plots a 2D Function using a contour plot, and updates the plot every time it is called.
    """

    def __init__(self, xlims=None, ylims=None, n_points=128):

        self.x_lims = xlims
        self.y_lims = ylims
        self.n_points = n_points
        self.h = None
        self.p = None

    def __call__(self, func):
        if self.h is not None:
            for coll in self.h.collections:
                plt.gca().collections.remove(coll)
        if self.x_lims is None:
            self.x_lims = plt.gca().get_xlim()
        if self.y_lims is None:
            self.y_lims = plt.gca().get_ylim()
        x_pts, y_pts = np.meshgrid(np.linspace(self.x_lims[0], self.x_lims[1], self.n_points), np.linspace(self.y_lims[0], self.y_lims[1], self.n_points))
        self.p = np.concatenate([x_pts[..., None], y_pts[..., None]], axis=-1).reshape(-1, 2)
        self.shape = x_pts.shape
        self.h = plt.contour(x_pts, y_pts, func(self.p).reshape(self.shape))


def demo_gaussian_mixture_em(n_samples = 1000, pause=0.3, seed=None, data_seed = 1234, n_steps=100, n_model_components = 3):
    """
    Generate data from pre-defined gaussian mixture, and then infer the parameters of this mixture again.

    :param n_samples: Number of data samples
    :param pause: Time to pause plot (in seconds) between iterations
    :param seed: Random Seed, a number or None to randomize.  1234 shows good behaviour, 1237 shows pathalogical behaviour
    :param n_steps: Number of iterations to run for
    :param n_model_components: Number of components to use in model (generating model always has 3)
    """

    true_means = [(2, 2), (-1, 0), (2, -3)]
    true_covs = [[0.5, -0.3], [-0.3, 0.5]], [[0.5, 0], [0, 0.5]], [[1, .6], [.6, 1]]
    true_prior = .3, .5, .2
    x = generate_data(means=true_means, covariances=true_covs, priors= true_prior, n_samples = n_samples, rng=data_seed)

    n_dims = x.shape[1]

    rng = np.random.RandomState(seed)
    means = rng.randn(n_model_components, n_dims)
    covs = [np.identity(n_dims) for _ in range(n_model_components)]
    prior = np.ones(n_model_components)/float(n_model_components)

    # Plot the
    plt.scatter(x[:, 0], x[:, 1])
    plt.grid()
    plotter = FunPlot2D()

    for i in range(n_steps):
        plotter(lambda pts: sum(pi*multivariate_normal(mean=mu, cov=cov).pdf(pts) for pi, mu, cov in zip(prior, means, covs)))
        logp = mean_log_likelihood(x, means, covs, prior)
        plt.title('Step {}.  Mean Log Likelihood = {:.4g}'.format(i, logp))

        plt.pause(pause)
        q = e_step(x, means, covs, prior)
        means, covs, prior = m_step(x, q, means, covs, prior)


if __name__ == '__main__':
    demo_gaussian_mixture_em()
