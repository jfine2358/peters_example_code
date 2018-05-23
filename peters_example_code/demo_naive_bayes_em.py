import numpy as np

"""
This example demonstrates the use of Expectation Maximization to discover latent categories of a Naive Bayes model on
randomly generated bernoulli data.

It implements the Algorithm on page 14 of:
    The Naive Bayes Model, Maximum-Likelihood Estimation, and the EM Algorithm
    Michael Collins
    http://www.cs.columbia.edu/~mcollins/em.pdf
"""


def get_rng(rng):
    return rng if isinstance(rng, np.random.RandomState) else np.random.RandomState(rng)


def bernoulli_pmf(p, x):
    assert np.all((x==0)|(x==1))
    return np.where(x==1, p, 1.-p)


def normalize_to_unit_sum(arr, axis = None):
    arr = np.array(arr, copy=False, dtype=np.float)
    assert arr.ndim>0, 'This makes no sense on scalars'
    if axis is None:
        assert arr.ndim==1, 'If array dim is not 1 you must specify axis'
        axis=0
    assert np.all(arr>=0)
    return arr / np.sum(arr, axis=axis, keepdims=True)


def generate_true_parameters(n_dims, class_weights = (0.5, 0.5), x_dist ='bernoilli', rng = 1234):
    """
    :param n_dims:
    :param class_weights:
    :param x_dist:
    :param rng:
    :return: (p_y, p_x_given_y)  Where
        p_y is a shape (n_classes, ) array of class probabilities
        p_x_given_y is a shape (n_classes, n_dims) array of probabilities such that p_x_given_y[i, j] is the probability that x[i]==1 given x[j]==j
    """
    assert x_dist=='bernoilli', 'Only bernoilli implemented for now.'
    rng = get_rng(rng)
    p_y = normalize_to_unit_sum(class_weights)  # (n_classes, )
    p_x_given_y = rng.uniform(0, 1, size=(len(class_weights), n_dims))  # (n_classes, n_dims)
    return p_y, p_x_given_y


def simulate_data(n_samples, p_y, p_x_given_y, rng = None):

    assert p_y.ndim==1
    assert p_x_given_y.ndim==2
    assert p_x_given_y.shape[0]==p_y.shape[0]

    rng = get_rng(rng)
    y = rng.choice(range(len(p_y)), p=p_y, size=(n_samples, ))  # (n_samples, )
    p_x_given_ysamples = p_x_given_y[y]  # (n_samples, n_dims)
    x = rng.binomial(n=1, p=p_x_given_ysamples)
    return y, x


def do_em(n_steps, x, n_classes, rng = None):
    """
    Do n_steps of expectation maximization on a randomly initialized naive bayes model with categorical hidden and
    bernoulli visible variables.

    :param n_steps: Number of steps of EM to do.
    :param x: A (n_samples, n_dims) array of visible data
    :param n_classes: Number of classes to discover
    :param rng: Random number generator or seed
    :return: q_y, q_x_given_y     ... where:
        q_y is a (n_classes, ) array of prior probabilities for each class
        q_x_given_y is a (n_classes, n_dims) array representing the probability of each class

    Note: This is numberically unstable, because we don't implement our products as log-sums (may have numerical problems
    when d is large or so...)
    """
    # Reference page 14 of http://www.cs.columbia.edu/~mcollins/em.pdf
    n_samples, n_dims = x.shape

    q_y = normalize_to_unit_sum(rng.uniform(0, 1, size=n_classes), axis=0)  # (n_classes, )
    q_x_given_y = rng.uniform(0, 1, size=(n_classes, n_dims))  # (n_classes, n_dims)

    for t in range(n_steps):

        print('Step {} of {}...'.format(t, n_steps))

        # Note: Danger here, for numerical stability we should really be doing logsums
        delta_unnorm = q_y[None, :] * np.prod(bernoulli_pmf(q_x_given_y[None, :, :], x[:, None, :]), axis=2)    # (n_samples, n_classes)
        delta = normalize_to_unit_sum(delta_unnorm, axis=1)  # (n_samples, n_classes)
        assert np.all((0<=delta) & (delta<=1))
        q_y = delta.mean(axis=0)  # (n_classes, )
        q_x_given_y = (delta[:, :, None]*x[:, None, :]).sum(axis=0) / delta.sum(axis=0)[:, None]  # (n_classes, n_dims)
        assert np.all((0<=q_x_given_y) & (q_x_given_y<=1))

    print('EM Complete')
    return q_y, q_x_given_y


def run_naive_bayes_em(n_samples=10000, n_dims=8, n_steps=100, class_weights = (3, 1), rng=None):
    """
    This function:
    - Randomly generates a Naive Bayes model, defined by p(y) and p(x|y)
    - Generates data (x, y) from this model.
    - Runs EM given the visibile data (x) to infer the distributions q(y) and q(x|y)
    - Displays plots comparing the parameters, samples, of the true (p) and inferred (q) models.

    With the default paremters, we should see that EM always discovers the latent classes (though of course their
    class-indices may be randomly swapped)

    :param n_samples: Number of data points
    :param n_dims: Dimensionality of input data
    :param n_steps: Number of EM steps to run
    :param class_weights: Unnormalized class probabilities for generating data.  Length corresponds to number of latent classes
    :param rng: A random number generator or seed.
    """

    rng = get_rng(rng)
    n_classes = len(class_weights)

    # Create "true" parameters for the model
    p_y, p_x_given_y = generate_true_parameters(n_dims=n_dims, class_weights=class_weights, rng=rng)

    # Generate data from true parameters
    y_true, x_true = simulate_data(n_samples=n_samples, p_y = p_y, p_x_given_y=p_x_given_y, rng=rng)

    # Do EM to estimate model parameters from data
    q_y, q_x_given_y = do_em(n_steps=n_steps, x=x_true, n_classes=n_classes, rng=rng)

    # Generate Data from the learned model.
    y_gen, x_gen = simulate_data(n_samples=n_samples, p_y=q_y, p_x_given_y=q_x_given_y, rng=rng)

    # Plot:
    plot_sizes = (2, 3)

    from matplotlib import pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.subplots_adjust(hspace=0.3, wspace=0.3)

    plt.subplot2grid(plot_sizes, (0, 0))
    plt.bar(np.arange(n_classes), p_y, label='p(y)')
    plt.xlabel('y')
    plt.ylabel('p(y)')

    plt.subplot2grid(plot_sizes, (0, 1))
    for k in range(n_classes):
        plt.plot(np.arange(n_dims), p_x_given_y[k, :], label = 'p(x|y={})'.format(k))
    plt.xlabel('x-dimension')
    plt.legend()

    plt.subplot2grid(plot_sizes, (0, 2))
    ixs = np.argsort(y_true)
    plt.imshow(np.concatenate([y_true[ixs, None], x_true[ixs]], axis=1), aspect='auto', cmap='gray')
    plt.xlabel('$y_{true}$ (left col), $x_{true}$ (other cols)')
    plt.ylabel('sample #')
    plt.title('Samples from p(x), sorted by class')

    plt.subplot2grid(plot_sizes, (1, 0))
    plt.bar(np.arange(n_classes), q_y, label='q(y)')
    plt.xlabel('y')
    plt.ylabel('q(y)')
    plt.ylabel('q(y)')
    plt.xlabel('y')

    plt.subplot2grid(plot_sizes, (1, 1))
    for k in range(n_classes):
        plt.plot(np.arange(n_dims), q_x_given_y[k, :], label = 'q(x|y={})'.format(k))
    plt.xlabel('x-dimension')
    plt.legend()

    plt.subplot2grid(plot_sizes, (1, 2))
    ixs = np.argsort(y_gen)
    plt.imshow(np.concatenate([y_gen[ixs, None], x_gen[ixs]], axis=1), aspect='auto', cmap='gray')
    plt.xlabel('$y_{gen}$ (left col), $x_{gen}$ (other cols)')
    plt.ylabel('sample #')
    plt.title('Samples from q(x), sorted by class')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    run_naive_bayes_em()
