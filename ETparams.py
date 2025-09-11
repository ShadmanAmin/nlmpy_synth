import numpy as np
import scipy.stats as stats
from scipy.stats import qmc, rankdata, pearsonr
import pandas as pd
import concurrent.futures
class ETParameter:
    """
    A class to represent a parameter for the ET model with mixture distribution bounds.
    
    Attributes:
        name (str): The name of the parameter (e.g., "Tr", "NDVI").
        mu1_bounds (tuple): Lower and upper bounds for the first mean (mu1).
        mu2_bounds (tuple): Lower and upper bounds for the second mean (mu2).
        w1_bounds (tuple): Lower and upper bounds for the mixture weight (w1).
        sigma1_bounds (tuple): Lower and upper bounds for the first standard deviation (sigma1).
        sigma2_bounds (tuple): Lower and upper bounds for the second standard deviation (sigma2).
    """
    
    def __init__(self, name: str,
                 mu1_bounds: tuple[float, float],
                 mu2_bounds: tuple[float, float],
                 w1_bounds: tuple[float, float],
                 sigma1_bounds: tuple[float, float],
                 sigma2_bounds: tuple[float, float]):
        self.name = name
        self.mu1_bounds = mu1_bounds
        self.mu2_bounds = mu2_bounds
        self.w1_bounds = w1_bounds
        self.sigma1_bounds = sigma1_bounds
        self.sigma2_bounds = sigma2_bounds
        self.mu1 = None
        self.mu2 = None
        self.w1 = None
        self.sigma1 = None
        self.sigma2 = None

    def lhs_sample(self, N_samples: int = 1):
        """
        Draws N_samples Latin‐Hypercube points in (mu1, mu2, w1, sigma1, sigma2),
        scales them to your bounds, and stores them in self.mu1, etc.
        """
        sampler = qmc.LatinHypercube(d=5)
        unit = sampler.random(n=N_samples)

        lows  = [self.mu1_bounds[0], self.mu2_bounds[0],
                 self.w1_bounds[0],  self.sigma1_bounds[0],
                 self.sigma2_bounds[0]]
        highs = [self.mu1_bounds[1], self.mu2_bounds[1],
                 self.w1_bounds[1],  self.sigma1_bounds[1],
                 self.sigma2_bounds[1]]
        sample = qmc.scale(unit, lows, highs)

        # unpack columns
        self.mu1    = sample[:, 0]
        self.mu2    = sample[:, 1]
        self.w1     = sample[:, 2]
        self.sigma1 = sample[:, 3]
        self.sigma2 = sample[:, 4]
        self.w2     = 1.0 - self.w1

        return sample

    def __repr__(self):
        return (f"<ETParameter {self.name}: "
                f"μ1∈{self.mu1_bounds}, μ2∈{self.mu2_bounds}, "
                f"w1∈{self.w1_bounds}, σ1∈{self.sigma1_bounds}, σ2∈{self.sigma2_bounds}>")

class MixtureETParameter(ETParameter):
    """
    A subclass of ETParameter that creates a mixture distribution object
    from the sampled parameters.
    """
    
    def create_dist(self, dist_type: str, sample_index: int = 0):
        """
        Create a mixture distribution using the sampled parameters for a given index.
        
        Parameters:
        sample_index (int): The index of the sampled parameters to use (default is 0).
        
        Returns:
            mixture (scipy.stats._distribution_infrastructure.Mixture): 
            the mixture distribution created from two normal distributions.
                
        Raises:
            ValueError: If the sampling method has not been called yet.
        """
        # Ensure that the sampling has been done.
        if (self.mu1 is None or self.mu2 is None or
            self.sigma1 is None or self.sigma2 is None or
            self.w1 is None or self.w2 is None):
            self.lhs_sample(N_samples=1)
        
        # Extract the values for the given sample index.
        mu1_val = self.mu1[sample_index]
        mu2_val = self.mu2[sample_index]
        sigma1_val = self.sigma1[sample_index]
        sigma2_val = self.sigma2[sample_index]
        w1_val = self.w1[sample_index]
        w2_val = self.w2[sample_index]
        
        # Create the two normal distributions.
        population_1 = stats.Normal(mu=mu1_val, sigma=sigma1_val)
        population_2 = stats.Normal(mu=mu2_val, sigma=sigma2_val)
        
        # Create the mixture distribution using the populations and the corresponding weights.
        if dist_type not in ("normal", "mixture"):
            raise ValueError("dist_type must be 'normal' or 'mixture'")
        if dist_type == 'mixture':
            mixture = stats.Mixture([population_1, population_2], weights=[w1_val, w2_val])
        else:
            parent = stats.Mixture([population_1, population_2], weights=[w1_val, w2_val])
            std_dev = parent.standard_deviation()
            mean = parent.mean()
            mixture = stats.Normal(mu=mean, sigma=std_dev)
        return mixture
def create_ET_parameters():
    """
    Create instances of ETParameter and MixtureETParameter for different parameters.
    
    Returns:
        tuple: A tuple containing the created ETParameter and MixtureETParameter instances.
    """
    tr_param = MixtureETParameter(
        name="Tr",
        mu1_bounds=(280, 300),
        mu2_bounds=(300, 320),
        w1_bounds=(0.3, 0.7),
        sigma1_bounds=(1, 10),
        sigma2_bounds=(1, 15)
    )

    alb_param = MixtureETParameter(
        name="Alb",
        mu1_bounds=(0.1, 0.5),
        mu2_bounds=(0.5, 0.9),
        w1_bounds=(0.3, 0.7),
        sigma1_bounds=(0.01, 0.03),
        sigma2_bounds=(0.01, 0.03)
    )

    ndvi_param = MixtureETParameter(
        name="NDVI",
        mu1_bounds=(0.1, 0.5),
        mu2_bounds=(0.5, 0.9),
        w1_bounds=(0.2, 0.8),
        sigma1_bounds=(0.02, 0.05),
        sigma2_bounds=(0.02, 0.05)
    )

    p_param = MixtureETParameter(
        name="P",
        mu1_bounds=(90000, 96000),
        mu2_bounds=(96000, 110000),
        w1_bounds=(0.3, 0.7),
        sigma1_bounds=(1000, 3000),
        sigma2_bounds=(1000, 3000)
    )

    ta_param = MixtureETParameter(
        name="Ta",
        mu1_bounds=(270, 290),
        mu2_bounds=(290, 310),
        w1_bounds=(0.3, 0.7),
        sigma1_bounds=(1, 5),
        sigma2_bounds=(1, 5)
    )

    sdn_param = MixtureETParameter(
        name="Sdn",
        mu1_bounds=(200, 600),
        mu2_bounds=(600, 1000),
        w1_bounds=(0.4, 0.6),
        sigma1_bounds=(10, 30),
        sigma2_bounds=(10, 30)
    )
    ldn_param = MixtureETParameter(
        name="Ldn",
        mu1_bounds=(100, 250),
        mu2_bounds=(250, 500),
        w1_bounds=(0.3, 0.7),
        sigma1_bounds=(5, 15),
        sigma2_bounds=(5, 15)
    )
    phys_params = {
        'tr_param': tr_param,
        'alb_param': alb_param,
        'ndvi_param': ndvi_param,
        'p_param': p_param,
        'ta_param': ta_param,
        'sdn_param': sdn_param,
        'ldn_param': ldn_param
    }
    return tr_param, alb_param, ndvi_param, p_param, ta_param, sdn_param, ldn_param, phys_params

def gsmax_mmol_to_ms(g_mmol,T_air,P_air):
    R = 8.314472 # J/(mol*K)
    g_mol = g_mmol/1000 # convert to mol/m²/s
    g_ms  = g_mol * (R * T_air / P_air)    # m s⁻¹
    return g_ms  


import matplotlib.pyplot as plt
def plot_PRCC(prcc_results):
    plt.figure(constrained_layout=True)
    plt.bar(prcc_results['Parameter'], prcc_results['PRCC'],
             color='skyblue', edgecolor='k')
    plt.title('Partial Rank Correlation Coefficients (PRCC)')
    plt.ylabel('PRCC')
    plt.show()
    

def extract_gaussian(data):
    from sklearn.mixture import GaussianMixture
    import scipy.stats as stats
    data = data[~np.isnan(data)]
    data = data.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type='diag')
    gmm = gmm.fit(data)

    means = gmm.means_.flatten()
    covariances = gmm.covariances_.flatten()
    weights = gmm.weights_.flatten()

    population1 = stats.Normal(mu=means[0], sigma=np.sqrt(covariances[0]))
    population2 = stats.Normal(mu=means[1], sigma=np.sqrt(covariances[1]))

    mixture = stats.Mixture([population1, population2], weights=weights)
    skewness = mixture.skewness()
    

    return means, covariances, weights, skewness


def find_gaussian_instances(
    df,
    means,
    covariances,
    weights,
    skewness,
    tol=1e-6
):
    """
    Return all rows in df whose (mu1, mu2, sigma1, sigma2, w1, w2, skewness)
    match the provided GMM parameters within an absolute tolerance.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['mu1','mu2','sigma1','sigma2','w1','w2','skewness'].
    means : array_like of length 2
        [mu1, mu2] from your GMM fit.
    covariances : array_like of length 2
        [var1, var2] = GMM covariances. We compare sqrt(var) to df['sigma*'].
    weights : array_like of length 2
        [w1, w2] from your GMM fit.
    skewness : float
        Mixture skewness to match.
    tol : float
        Absolute tolerance for np.isclose.

    Returns
    -------
    pandas.DataFrame
        Subset of df matching those parameters.
    """
    mu1, mu2 = means
    var1, var2 = covariances
    w1, w2   = weights

    sigma1 = np.sqrt(var1)
    sigma2 = np.sqrt(var2)

    mask = (
        np.isclose(df['mu1'], mu1, atol=tol) &
        np.isclose(df['mu2'], mu2, atol=tol) &
        np.isclose(df['sigma1'], sigma1, atol=tol) &
        np.isclose(df['sigma2'], sigma2, atol=tol) &
        np.isclose(df['w1'], w1, atol=tol) &
        np.isclose(df['w2'], w2, atol=tol) &
        np.isclose(df['skewness'], skewness, atol=tol)
    )

    return df[mask]

def find_instances(df,mean,std,tol=1):
    mask = (
        np.isclose(df['mean'], mean, atol=tol) &
        np.isclose(df['std'], std, atol=tol)
    )

    return df[mask]