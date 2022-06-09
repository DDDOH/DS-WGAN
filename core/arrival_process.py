import numpy as np
import numpy as np
import progressbar
from scipy.interpolate import interp1d
from scipy.stats import gamma


class ArrivalProcess():
    # general arrival process
    def __init__(self, T, arrival_ls):
        self.arrival_ls = np.sort(arrival_ls)
        self.T = T
        self.n_arrival = len(arrival_ls)

    def run_through():
        raise NotImplemented

    def set_service_time(self, service_ls):
        self.service_ls = service_ls

    def get_count_vector(self, interval_len):
        # group arrivals into intervals of length interval_len, and count the number of arrivals in each interval
        return np.histogram(self.arrival_ls, bins=np.arange(0, self.T + interval_len, interval_len))[0]

    def set_wait_time(self, wait_ls):
        self.wait_ls = wait_ls

    def get_wait_summary(self, vis_interval_ls, quantile):
        # get the summary of wait time in each interval
        wait_summary = {'mean': np.zeros(len(vis_interval_ls) - 1),
                        'var': np.zeros(len(vis_interval_ls) - 1),
                        'quantile': np.zeros(len(vis_interval_ls) - 1)}
        for i in range(len(vis_interval_ls)-1):
            interval_start = vis_interval_ls[i]
            interval_end = vis_interval_ls[i + 1]
            index = np.searchsorted(
                self.arrival_ls, [interval_start, interval_end])

            if index[0] != index[1]:
                assert np.min(
                    self.arrival_ls[index[0]:index[1]]) >= interval_start
                assert np.max(
                    self.arrival_ls[index[0]:index[1]]) <= interval_end

            wait_ls_within_interval = self.wait_ls[index[0]:index[1]]
            wait_summary['mean'][i] = np.mean(self.wait_ls[index[0]:index[1]])
            wait_summary['var'][i] = np.var(self.wait_ls[index[0]:index[1]])
            wait_summary['quantile'][i] = np.quantile(
                self.wait_ls[index[0]:index[1]], quantile) if index[0] != index[1] else np.nan
        return wait_summary


class BatchArrivalProcess():
    def __init__(self, T, arrival_ls_ls):
        self.arrival_process_ls = [ArrivalProcess(
            T, arrival_ls) for arrival_ls in arrival_ls_ls]

        self.n_arrival_process = len(self.arrival_process_ls)

    def set_service_time(self, service_sampler):
        for i in range(self.n_arrival_process):
            self.arrival_process_ls[i].set_service_time(
                service_sampler(self.arrival_process_ls[i].n_arrival))

    def set_wait_time(self, i, wait_ls):
        self.arrival_process_ls[i].set_wait_time(wait_ls)

    def get_batch_wait_summary(self, vis_interval_ls, quantile):
        batch_wait_summary = {'mean': np.zeros((self.n_arrival_process, len(vis_interval_ls) - 1)),
                              'var': np.zeros((self.n_arrival_process, len(vis_interval_ls) - 1)),
                              'quantile': np.zeros((self.n_arrival_process, len(vis_interval_ls) - 1))}
        for i in range(self.n_arrival_process):
            wait_summary = self.arrival_process_ls[i].get_wait_summary(
                vis_interval_ls, quantile)
            batch_wait_summary['mean'][i, :] = wait_summary['mean']
            batch_wait_summary['var'][i, :] = wait_summary['var']
            batch_wait_summary['quantile'][i, :] = wait_summary['quantile']
        return batch_wait_summary


class NHPP(ArrivalProcess):
    # non-homogeneous poisson process
    def __init__(self, T, lam_t):
        self.T = T
        self.lam_t = lam_t
        arrival_ls = self.simulate_arrival_ls()
        super().__init__(T, arrival_ls)

    def simulate_arrival_ls(self):
        # self.lam_t = self.simulate_lam_t()
        max_lam = np.max(self.lam_t)
        N_arrival = np.random.poisson(max_lam * self.T)
        unfiltered_arrival_time = np.sort(
            np.random.uniform(0, self.T, size=N_arrival))

        dt = self.T / len(self.lam_t)

        # for each t in unfiltered_arrival_time, which segment it belongs to
        unfiltered_arrival_time_index = (
            unfiltered_arrival_time/dt).astype(int)
        keep_prob = self.lam_t[unfiltered_arrival_time_index] / max_lam
        whether_keep = np.random.rand(len(keep_prob)) <= keep_prob

        filtered_arrival_time = unfiltered_arrival_time[whether_keep]
        return filtered_arrival_time


class BatchCIR():
    def __init__(self, base_lam, interval_len):
        # base_lam: shape of R_t
        # P: num of intervals
        # interval_len: length of one interval
        # n_CIR: num of CIR for train

        self.P = len(base_lam)
        self.T = self.P * interval_len
        self.interval_len = interval_len
        x = np.linspace(interval_len/2, self.T-interval_len/2, self.P)
        y = base_lam
        R_t = interp1d(x, y, kind='quadratic', fill_value='extrapolate')
        R_t = np.vectorize(R_t)

        # parameters for CIR
        # infinite server queue
        # kappa = 0.2
        # sigma = 0.4
        # alpha = 0.3

        # multi server queue
        self.kappa = 3
        # sigma = 0.4
        self.sigma = 2
        self.alpha = 0.3

        # how many steps to discretize CIR
        self.N = 5000

        # CIR update rule:
        # $\mathrm{d} \lambda(t)=\kappa(R(t)-\lambda(t)) \mathrm{d} t+\sigma R(t)^{\alpha} \lambda(t)^{1 / 2} \mathrm{~d} B(t)$
        self.T_ls = np.linspace(0, self.T, self.N, endpoint=False)
        self.R_t_on_T_ls = R_t(self.T_ls)
        self.R_t_exp_alpha_on_T_ls = self.R_t_on_T_ls ** self.alpha
        self.dt = self.T/self.N

    def simulate_batch_CIR(self, n_CIR):
        # CIR_ls = np.ndarray((n_CIR,), dtype=np.object)
        count_mat = np.zeros((n_CIR, self.P))
        arrival_epoch_batch = np.ndarray((n_CIR,), dtype=np.object)
        for i in progressbar.progressbar(range(n_CIR)):
            CIR_i = NHPP(
                T=self.T, lam_t=self.simulate_lam_t())
            arrival_epoch_batch[i] = CIR_i.arrival_ls
            count_mat[i, :] = CIR_i.get_count_vector(
                self.interval_len)
        return count_mat, arrival_epoch_batch

    def simulate_lam_t(self):
        sqrt_dt = np.sqrt(self.dt)
        beta = 100
        B_t = np.random.normal(size=self.N)

        Z_t_0 = self.R_t_on_T_ls[0] * gamma.rvs(a=beta, scale=1/beta)
        Z_ls = np.zeros(self.N)
        Z_ls[0] = Z_t_0
        for i in range(self.N - 1):
            t = self.T_ls[i]
            Z_t = Z_ls[i]
            R_t_val = self.R_t_on_T_ls[i]
            R_t_exp_alpha_val = self.R_t_exp_alpha_on_T_ls[i]

            d_Z_t = self.kappa * (R_t_val - Z_t) * self.dt + self.sigma * \
                R_t_exp_alpha_val * (Z_t**0.5) * sqrt_dt * B_t[i]
            Z_ls[i+1] = Z_t + d_Z_t
        return Z_ls

    # def simulate_CIR_arrival(self):
    #     Z_ls = self.simulate_lam_t()
    #     max_lam = np.max(Z_ls)
    #     N_arrival = np.random.poisson(max_lam * self.T)
    #     unfiltered_arrival_time = np.sort(
    #         np.random.uniform(0, self.T, size=N_arrival))

    #     keep_prob = np.array([Z_ls[int(t/self.dt)]
    #                          for t in unfiltered_arrival_time])/max_lam
    #     whether_keep = np.random.rand(len(keep_prob)) <= keep_prob

    #     filtered_arrival_time = unfiltered_arrival_time[whether_keep]
        # return filtered_arrival_time


if __name__ == '__main__':
    P = 22
    interval_len = 0.5
    T = P * interval_len

    base_lam = np.array([124, 175, 239, 263, 285,
                         299, 292, 276, 249, 257,
                         274, 273, 268, 259, 251,
                         252, 244, 219, 176, 156,
                         135, 120])
    x = np.linspace(interval_len/2, T-interval_len/2, len(base_lam))
    y = base_lam
    f2 = interp1d(x, y, kind='quadratic', fill_value='extrapolate')

    def R_t(t):
        assert t >= 0
        assert t <= T
        return f2(t)
    R_t = np.vectorize(R_t)

    # 这里在干嘛 我不理解
    R_i = []
    for i in range(P):
        interval_start = i * interval_len
        interval_end = (i + 1) * interval_len
        t_ls = np.arange(interval_start, interval_end, 0.002)
        r_ls = R_t(t_ls)
        R_i.append(np.mean(r_ls))
    R_i = np.array(R_i)

    # infinite server queue
    # kappa = 0.2
    # sigma = 0.4
    # alpha = 0.3

    # multi server queue
    kappa = 3
    # sigma = 0.4
    sigma = 2
    alpha = 0.3

    # 计算 R(t) 在每个离散的仿真点上的数值，所有CIR都用这个
    N = 5000
    T_ls = np.linspace(0, T, N, endpoint=False)
    R_t_on_T_ls = R_t(T_ls)
    R_t_exp_alpha_on_T_ls = R_t_on_T_ls ** alpha
    dt = T/N

    def simulate_lam_t():
        sqrt_dt = np.sqrt(dt)
        beta = 100
        B_t = np.random.normal(size=N)

        Z_t_0 = R_t_on_T_ls[0] * gamma.rvs(a=beta, scale=1/beta)
        Z_ls = np.zeros(N)
        Z_ls[0] = Z_t_0
        for i in range(N - 1):
            t = T_ls[i]
            Z_t = Z_ls[i]
            R_t_val = R_t_on_T_ls[i]
            R_t_exp_alpha_val = R_t_exp_alpha_on_T_ls[i]

            d_Z_t = kappa * (R_t_val - Z_t) * dt + sigma * \
                R_t_exp_alpha_val * (Z_t**0.5) * sqrt_dt * B_t[i]
            Z_ls[i+1] = Z_t + d_Z_t
        return Z_ls

    def simulate_CIR_arrival():
        Z_ls = simulate_lam_t()
        max_lam = np.max(Z_ls)
        N_arrival = np.random.poisson(max_lam * T)
        unfiltered_arrival_time = np.sort(
            np.random.uniform(0, T, size=N_arrival))

        keep_prob = np.array([Z_ls[int(t/dt)]
                             for t in unfiltered_arrival_time])/max_lam
        whether_keep = np.random.rand(len(keep_prob)) <= keep_prob

        filtered_arrival_time = unfiltered_arrival_time[whether_keep]
        return filtered_arrival_time

    # max_arrival = 0
    # get real CIR arrival process
    real_CIR_size = 6000
    real_CIR_ls = np.ndarray((real_CIR_size,), dtype=np.object)
    real_count_mat = np.zeros((real_CIR_size, P))
    for i in progressbar.progressbar(range(real_CIR_size)):
        real_CIR_ls[i] = ArrivalProcess(T=T, arrival_ls=simulate_CIR_arrival())
        real_count_mat[i, :] = real_CIR_ls[i].get_count_vector(interval_len)
