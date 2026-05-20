import numpy as np


# -------------------------------------------------
# Question 1: Joint Gaussian PDF and Marginals
# -------------------------------------------------

def joint_gaussian_pdf(
    x,
    y,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6
):
    """
    Return the bivariate Gaussian PDF f_XY(x,y).
    """

    q = (
        ((x - mu_x) ** 2) / (sigma_x ** 2)
        - 2 * rho * ((x - mu_x) * (y - mu_y)) / (sigma_x * sigma_y)
        + ((y - mu_y) ** 2) / (sigma_y ** 2)
    )

    denominator = (
        2
        * np.pi
        * sigma_x
        * sigma_y
        * np.sqrt(1 - rho ** 2)
    )

    exponent = -q / (2 * (1 - rho ** 2))

    return (1 / denominator) * np.exp(exponent)


def marginal_pdf_x(x, mu_x=1, sigma_x=2):
    """
    Return marginal Gaussian PDF of X.
    """

    return (
        1
        / (np.sqrt(2 * np.pi) * sigma_x)
    ) * np.exp(
        -((x - mu_x) ** 2) / (2 * sigma_x ** 2)
    )


def marginal_pdf_y(y, mu_y=-2, sigma_y=3):
    """
    Return marginal Gaussian PDF of Y.
    """

    return (
        1
        / (np.sqrt(2 * np.pi) * sigma_y)
    ) * np.exp(
        -((y - mu_y) ** 2) / (2 * sigma_y ** 2)
    )


def covariance_matrix(sigma_x=2, sigma_y=3, rho=0.6):
    """
    Return covariance matrix.
    """

    return np.array([
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2]
    ])


def joint_pdf_grid_integral(
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    n=250
):
    """
    Numerically approximate integral of joint Gaussian PDF.
    """

    x_min = mu_x - 4 * sigma_x
    x_max = mu_x + 4 * sigma_x

    y_min = mu_y - 4 * sigma_y
    y_max = mu_y + 4 * sigma_y

    x_vals = np.linspace(x_min, x_max, n)
    y_vals = np.linspace(y_min, y_max, n)

    dx = x_vals[1] - x_vals[0]
    dy = y_vals[1] - y_vals[0]

    total = 0.0

    for x in x_vals:
        for y in y_vals:
            total += (
                joint_gaussian_pdf(
                    x,
                    y,
                    mu_x,
                    mu_y,
                    sigma_x,
                    sigma_y,
                    rho
                )
                * dx
                * dy
            )

    return total


# -------------------------------------------------
# Question 2: Simulation and Independence
# -------------------------------------------------

def generate_joint_gaussian_samples(
    n=100000,
    mu_x=1,
    mu_y=-2,
    sigma_x=2,
    sigma_y=3,
    rho=0.6,
    seed=0
):
    """
    Generate jointly Gaussian samples.
    """

    np.random.seed(seed)

    mean = [mu_x, mu_y]

    cov = [
        [sigma_x ** 2, rho * sigma_x * sigma_y],
        [rho * sigma_x * sigma_y, sigma_y ** 2]
    ]

    samples = np.random.multivariate_normal(
        mean,
        cov,
        size=n
    )

    x_samples = samples[:, 0]
    y_samples = samples[:, 1]

    return x_samples, y_samples


def sample_means(x_samples, y_samples):
    """
    Return sample means of X and Y.
    """

    return np.mean(x_samples), np.mean(y_samples)


def sample_covariance_matrix(x_samples, y_samples):
    """
    Return sample covariance matrix.
    """

    return np.cov(x_samples, y_samples, ddof=1)


def sample_correlation(x_samples, y_samples):
    """
    Return sample correlation coefficient.
    """

    return np.corrcoef(x_samples, y_samples)[0, 1]


def gaussian_independence_check(rho):
    """
    For jointly Gaussian variables:

    rho = 0  -> independent
    rho != 0 -> dependent
    """

    return rho == 0


def zero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0 and verify covariance is near zero.
    """

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=0,
        seed=0
    )

    cov_matrix = sample_covariance_matrix(x, y)

    covariance = cov_matrix[0, 1]

    return abs(covariance) < 0.05


def nonzero_rho_covariance_check(n=100000):
    """
    Generate samples with rho=0.6 and verify covariance is nonzero.
    """

    x, y = generate_joint_gaussian_samples(
        n=n,
        rho=0.6,
        seed=0
    )

    cov_matrix = sample_covariance_matrix(x, y)

    covariance = cov_matrix[0, 1]

    return abs(covariance - 3.6) < 0.15
