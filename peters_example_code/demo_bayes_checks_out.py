import numpy as np

Nx = 2  # Number of possible values which each element of x can take
Nd = 4  # Number of dimensions of x
Nc = 3  # Number of latent categories for non-fraudulent x
Nf = 2  # Number of fraud states (generally 2 for 0=NoFraud, 1=Fraud)

assert Nx == 2, "We currently assumen Nx==2, because we use this to generate all possible combinations of x"
assert Nf == 2, "We currently assumen Nf==2, because p_x_given_c_f considers f to be boolean"


def binary_permutations(n_bits):
    """
    Given some number of bits, return a shape (2**n_bits, n_bits) boolean array containing every permoutation
    of those bits as a row.
    :param n_bits: An integer number of bits
    :return: A shape (2**n_bits, n_bits) boolean array containing every permoutation
        of those bits as a row.
    """
    return np.right_shift(np.arange(2**n_bits)[:, None], np.arange(n_bits-1, -1, -1)[None, :]) & 1


def generate_random_params(shape, norm_axis=-1):
    """
    Generate a random array of parameters normalized along the given axis.
    """
    params = np.random.rand(*shape) if isinstance(shape, tuple) else np.random.rand(shape)
    params = params/params.sum(axis=norm_axis, keepdims=True)
    return params


theta_c = generate_random_params(Nc)  # A one-of-Nc probability distribution
theta_f = generate_random_params(Nf)  # A one-of-Nf probability distribution (Nf=2 for 2 fraud states)

p_c = lambda c: theta_c[c]
p_f = lambda f: theta_f[f]

p_c_f = lambda c, f: p_c(c)*p_f(f)   # Here we code in our marginal-indepenence assumption (C -> X <- F)

theta_xfraud = generate_random_params((Nx, Nd), norm_axis=0)
p_xfraud = lambda x: np.prod(theta_xfraud[x, np.arange(Nd)])  # Here we code in the conditional-independence assumption (probability factorizes of dimensions, i.e. X[i] is conditionally independent of X[j] given F, C, and i!=j)

theta_xnonfraud = generate_random_params((Nx, Nd, Nc), norm_axis=0)  # (Nx, Nd, Nc)
p_xnonfraud = lambda x, c: np.prod(theta_xnonfraud[x, np.arange(Nd), c])  # Also codes in conditional-independence assumption

# Now, define our conditional:
p_x_given_c_f = lambda x, c, f: (p_xfraud(x) if f==1 else p_xnonfraud(x, c))  # This function is valid so long as for a given c, f, it sums to 1 for all possible xs

# Note, we could also have defined this differently... We could have defined p_xifraud(x[i], i), p_xinonfraud([x[i], i]), p_xi_given_c_f(x[i], i, c, f).
# Then we could have instead summed over (all_possible_is, all_possible_xis, all_possible_cs, all_possible_fs).  This way when we sum the probs up we would take account of the conditional-independence assuption

# Now, assert that our probabilities add up as they should.
assert Nx == 2
all_possible_xs = binary_permutations(Nd)  # (Nx**Nd, Nd)
all_possible_cs = np.arange(Nc)  # (Nc, )
all_possible_fs = np.arange(Nf)  # (Nf, )

# First, assert that the marginal distributions of F, C, add to 1 (they should, they were defined that way)
print 'sum(p(F)): {}'.format(np.sum([p_f(f) for f in all_possible_fs]))
print 'sum(p(C)): {}'.format(np.sum([p_c(c) for c in all_possible_cs]))
print 'sum(p(F, C)): {}'.format(np.sum([[p_c_f(c, f) for f in all_possible_fs] for c in all_possible_cs]))

# First, assert that the conditional probability formulation is correct, by checking that for every f, c, the probabilities of x sum to 1.
table_p_x_given_c_f = np.array([[[p_x_given_c_f(x, c, f) for f in all_possible_fs] for c in all_possible_cs] for x in all_possible_xs])  # (|X|, Nc, Nd)
print 'sum(p(X|C, F)) for each C in {}, F in {}: \n{}'.format(all_possible_cs, all_possible_fs, table_p_x_given_c_f.sum(axis=0))

# Next, assert that the total joint probability sums to 1
table_p_x_f_c = np.array([[[p_x_given_c_f(x, c, f) * p_c_f(c, f) for x in all_possible_xs] for c in all_possible_cs] for f in all_possible_fs])
print 'sum(p(X, C, F))={}'.format(table_p_x_f_c.sum())
