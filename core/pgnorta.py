from http.client import UnimplementedFileMode
import progressbar
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy.stats import gamma
from scipy.stats import spearmanr
import numpy as np
import matplotlib.pyplot as plt
import os
from colored import fg, attr

class PGnorta():
    def __init__(self, base_intensity, cov, alpha):
        """Initialize a PGnorta dataset loader.

        Args:
            base_intensity (np.array): A list containing the mean of arrival count in each time step.
            cov (np.array): Covariance matrix of the underlying normal copula.
            alpha (np.array): A list containing the parameter of gamma distribution in each time step.
        """
        assert len(base_intensity) == len(alpha) and len(alpha) == np.shape(
            cov)[0] and np.shape(cov)[0] == np.shape(cov)[1]
        assert min(base_intensity) > 0, 'only accept nonnegative intensity'
        self.base_intensity = base_intensity
        self.p = len(base_intensity)  # the sequence length
        self.cov = cov
        self.alpha = alpha
        self.seq_len = len(base_intensity)

    def z_to_lam(self, Z, first=True):
        """Convert Z to intensity.

        Args:
            Z (np.array): The value of the normal copula for the first or last severl time steps or the whole sequence.
                          For one sample, it can be either a one dimension list with length q, or a 1 * q array.
                          For multiple samples, it should be a n * q array, n is the number of samples.
            first (bool, optional): Whether the given Z is for the first several time steps or not. Defaults to True.

        Returns:
            intensity (np.array): The value of intensity suggested by Z. An array of the same shape as Z.
        """
        if Z.ndim == 1:
            n_step = len(Z)
        else:
            n_step = np.shape(Z)[1]
        U = norm.cdf(Z)
        if first:
            B = gamma.ppf(q=U, a=self.alpha[:n_step],
                          scale=1/self.alpha[:n_step])
            intensity = B * self.base_intensity[:n_step]
        else:
            B = gamma.ppf(q=U, a=self.alpha[-n_step:],
                          scale=1/self.alpha[-n_step:])
            intensity = B * self.base_intensity[-n_step:]
        return intensity
    
    def sample_intensity(self, n_sample):
        """Sample arrival intensity from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            intensity (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        z = multivariate_normal.rvs(np.zeros(self.p), self.cov, n_sample)
        intensity = self.z_to_lam(z)
        return intensity
        

    def sample_count(self, n_sample):
        """Sample arrival count from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            count (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        intensity = self.sample_intensity(n_sample)
        count = np.random.poisson(intensity)
        return count
    
    def sample_both(self, n_sample):
        """Sample both arrival count and intensity from PGnorta model.

        Args:
            n_sample (int): Number of samples to generate.

        Returns:
            count (np.array): An array of size (n_sample * seq_len), each row is one sample.
            intensity (np.array): An array of size (n_sample * seq_len), each row is one sample.
        """
        intensity = self.sample_intensity(n_sample)
        count = np.random.poisson(intensity)
        return intensity, count



def estimate_PGnorta(count_mat, zeta=9/16, max_T=1000, M=100, img_dir_name=None, rho_mat_dir_name=None):
    p = np.shape(count_mat)[1]
    lam = np.mean(count_mat, axis=0)
    var_X = np.var(count_mat, axis=0)
    alpha = lam ** 2 / (var_X - lam)

    if np.min(alpha) < 0:
        print('The arrival count of the {}-th time interval does not satisfy variance >= mean'.format(np.where(alpha < 0)[0]))
        
    alpha[alpha < 0] = 10000 # alpha 越大，则生成的arrival count的mean和variance越接近

    kappa_t = lambda t : 0.1 * t ** (- zeta)
    rho_jk_record = np.zeros((p,p,max_T))


    # if tile rho_mat_dir_name exist, read it directly
    if os.path.exists(rho_mat_dir_name):
        print(fg('blue') + 'Loading rho_matrix directly.' + attr('reset'))
        rho_jk_record = np.load(rho_mat_dir_name)
    else:
        print(fg('blue') + 'No existing rho_matrix file. Estimate the model now.' + attr('reset'))
        with progressbar.ProgressBar(max_value=p ** 2) as bar:
            n_estimated = 0
            for j in range(p):
                for k in range(p):
                    if j == k:
                        rho_jk_record[j,k,:] = 1
                        continue
                    rho_jk = 0
                    hat_r_jk_X = spearmanr(count_mat[:,j], count_mat[:,k])[0]
                    for t in range(1, max_T):
                        # for m = 1 to M do
                        Z = multivariate_normal.rvs(np.zeros(2), [[1,rho_jk],[rho_jk,1]], M)
                        U = norm.cdf(Z)
                        B_j = gamma.ppf(q=U[:,0], a=alpha[j], scale=1/alpha[j])
                        B_k = gamma.ppf(q=U[:,1], a=alpha[k], scale=1/alpha[k])
                        T_j, T_k = lam[j] * B_j, lam[k] * B_k
                        X_j, X_k = np.random.poisson(T_j), np.random.poisson(T_k)
                        # end for

                        tilde_r_jk_X = spearmanr(X_j, X_k)[0]

                        rho_jk += kappa_t(t) * (hat_r_jk_X - tilde_r_jk_X)
                        rho_jk_record[j,k,t] = rho_jk
                    # plt.figure()
                    # plt.plot(rho_jk_record[j,k,:])
                    # plt.show()
                    # plt.close()
                    n_estimated += 1
                    if img_dir_name is not None:
                        n_plot = 0
                        plt.figure()
                        for j_ in range(p):
                            for k_ in range(p):
                                if rho_jk_record[j_, k_, -1] != 0: 
                                    plt.plot(rho_jk_record[j_,k_,:])
                                    n_plot += 1
                                    if n_plot == 50:
                                        break
                        plt.title('rho estimation trajectory')
                        plt.savefig(img_dir_name)
                        plt.close()
                    bar.update(n_estimated)
        if rho_mat_dir_name is not None:
            np.save(rho_mat_dir_name, rho_jk_record)
            print(fg('blue') + 'rho_matrix saved to file.' + attr('reset'))
    norta = PGnorta(base_intensity=lam, cov=rho_jk_record[:,:,-1], alpha=alpha)
    return norta



def sample_PGnorta_marginal(base_intensity_t, alpha_t, n_sample):
    z = np.random.normal(0, 1, n_sample)
    U = norm.cdf(z)
    B = gamma.ppf(q=U, a=alpha_t,
                  scale=1/alpha_t)
    intensity = B * base_intensity_t
    count = np.random.poisson(intensity)
    return intensity, count