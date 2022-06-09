import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch
from .des.des_py.des import *
from .arrival_process import BatchArrivalProcess
from .utils.arrival_epoch_simulator import arrival_epoch_simulator
from scipy import stats


def evaluate_marginal(count_WGAN, count_PGnorta, count_train, result_dir, iteration):

    assert np.shape(count_WGAN)[1] == np.shape(
        count_PGnorta)[1] == np.shape(count_train)[1]

    wasserstein_PG_train_rec = []
    wasserstein_WGAN_train_rec = []
    p = np.shape(count_WGAN)[1]
    for interval in range(p):
        wasserstein_PG_train_rec.append(scipy.stats.wasserstein_distance(
            count_PGnorta[:, interval], count_train[:, interval]))
        wasserstein_WGAN_train_rec.append(scipy.stats.wasserstein_distance(
            count_WGAN[:, interval], count_train[:, interval]))

    plt.figure()
    plt.plot(wasserstein_PG_train_rec, label='PGnorta & Train')
    plt.plot(wasserstein_WGAN_train_rec, label='DS-WGAN & Train')
    plt.title('Wasserstein distance')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'figures',
                'marginal_w_distance_{}.png').format(iteration))
    plt.close()

    # marginal ecdf
    n_column = 5
    n_row = np.ceil(p/n_column).astype(int) * 2
    # figsize = (width, height)
    fig, axes = plt.subplots(n_row, n_column, figsize=(
        n_column * 6, n_row * 3))

    for row in range(n_row):
        for column in range(n_column):
            interval = column + row//2 * n_column

            if interval == p:
                break

            data_dict = {'DS-WGAN': count_WGAN[:, interval],
                         'PGnorta': count_PGnorta[:, interval],
                         'Training set': count_train[:, interval]}

            if row % 2 == 0:
                ecdf_plot(axes[row, column], data_dict,
                          alpha=0.5, xlim=None, legend=(column == n_column-1) or (interval == p - 1), title='interval {} ecdf'.format(interval+1))
            else:
                kde_plot(axes[row, column], data_dict,
                         alpha=0.5, xlim=None, legend=(column == n_column-1) or (interval == p - 1), title='interval {} kde'.format(interval+1))

    plt.tight_layout()
    plt.savefig(os.path.join(result_dir, 'figures',
                'marginal_ecdf_kde_{}.png').format(iteration))
    plt.close('all')
    return {'PG': wasserstein_PG_train_rec, 'WGAN': wasserstein_WGAN_train_rec}


def ecdf_plot(ax, data_dict, alpha, xlim=None, legend=False, title=''):
    # iterate over data_dict
    for key, value in data_dict.items():
        ax.plot(np.sort(value), np.linspace(
            0, 1, len(value), endpoint=False), label=key, alpha=alpha)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    # ax.axes.xaxis.set_ticklabels([])
    if legend:
        ax.legend()
    ax.set_title(title)


def kde_plot(ax, data_dict, alpha, xlim=None, legend=False, title=''):
    color_idx = 0
    for key, value in data_dict.items():
        try:
            kernel = stats.gaussian_kde(value)
            x = np.linspace(value.min(), value.max(), 1000)
            ax.plot(x, kernel(x), label=key, alpha=alpha, c='C'+str(color_idx))
        except:
            ax.text(0.5, 0.5, 'Singular matrix',
                    fontsize=20, c='C'+str(color_idx))
            print('Singular matrix encountered')
        color_idx += 1

    if xlim is not None:
        ax.set_xlim(xlim)
    if legend:
        ax.legend()
    ax.set_title(title)
    # raise NotImplemented


def evaluate_joint(count_WGAN, count_PGnorta, count_train, result_dir, iteration):
    assert np.shape(count_WGAN)[1] == np.shape(
        count_PGnorta)[1] == np.shape(count_train)[1]

    past_future_corr_WGAN = []
    past_future_corr_PG = []
    past_future_corr_train = []

    cumsum_WGAN = np.cumsum(count_WGAN, axis=1)
    cumsum_PG = np.cumsum(count_PGnorta, axis=1)
    cumsum_train = np.cumsum(count_train, axis=1)

    sum_WGAN = np.sum(count_WGAN, axis=1)
    sum_PG = np.sum(count_PGnorta, axis=1)
    sum_train = np.sum(count_train, axis=1)

    p = np.shape(count_WGAN)[1]
    for interval in range(p - 1):
        past_future_corr_WGAN.append(scipy.stats.pearsonr(
            cumsum_WGAN[:, interval], sum_WGAN - cumsum_WGAN[:, interval])[0])
        past_future_corr_PG.append(scipy.stats.pearsonr(
            cumsum_PG[:, interval], sum_PG - cumsum_PG[:, interval])[0])
        past_future_corr_train.append(scipy.stats.pearsonr(
            cumsum_train[:, interval], sum_train - cumsum_train[:, interval])[0])

    plt.figure()
    plt.plot(past_future_corr_WGAN, label='WGAN')
    plt.plot(past_future_corr_PG, label='PGnorta')
    plt.plot(past_future_corr_train, label='Train')
    plt.title('Past-Future correlation')
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'figures',
                'past_future_correlation_{}.png').format(iteration))
    plt.close()

    return {'PG': past_future_corr_PG, 'WGAN': past_future_corr_WGAN, 'TRAIN': past_future_corr_train}


def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, save=False, result_dir=None):
    """Visualize and compare the real and fake.
    """
    # if real is torch tensor, convert to numpy
    if isinstance(real, torch.Tensor):
        real = real.detach().cpu().numpy()
    if isinstance(fake, torch.Tensor):
        fake = fake.detach().cpu().numpy()
    real_size = np.shape(real)[0]
    fake_size = np.shape(fake)[0]

    P = np.shape(real)[1]

    max_intensity = max(np.max(fake), np.max(real))
    plt.figure(figsize=(16, 3))
    plt.subplot(141)
    plt.plot(np.mean(fake, axis=0), label='fake')
    plt.plot(np.mean(real, axis=0), label='real')
    plt.plot(PGnorta_mean, label='PGnorta')
    plt.xlabel('Time interval')
    plt.ylabel('Intensity or count')
    plt.title('Mean')
    plt.legend()

    plt.subplot(142)
    plt.plot(np.var(fake, axis=0), label='fake')
    plt.plot(np.var(real, axis=0), label='real')
    plt.plot(PGnorta_var, label='PGnorta')
    plt.xlabel('Time interval')
    plt.title('Std')
    plt.legend()

    plt.subplot(143)
    plt.scatter(np.tile(np.arange(P), real_size).reshape(
        P, real_size), real, alpha=0.003)
    plt.ylim(0, max_intensity * 1.2)
    plt.xlabel('Time interval')
    plt.title('Scatter plot of real')

    plt.subplot(144)
    plt.scatter(np.tile(np.arange(P), fake_size).reshape(
        P, fake_size), fake, alpha=0.003)
    plt.ylim(0, max_intensity * 1.2)
    plt.xlabel('Time interval')
    plt.title('Scatter plot of fake')
    if save:
        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, 'figures', msg + '.png'))
    else:
        plt.show()

    plt.close()


def scatter_plot(interval_1_real, interval_2_real, interval_1_fake, interval_2_fake, msg, save=False, result_dir=None):
    plt.figure()
    plt.scatter(interval_1_real, interval_2_real, label='True', alpha=0.2)
    plt.scatter(interval_1_fake, interval_2_fake, label='Fake', alpha=0.2)
    plt.legend()
    if save:
        plt.tight_layout()
        plt.savefig(
            os.path.join(result_dir, 'figures', msg + '.png'))
    plt.close()


def eval_infinite_server_queue(count_WGAN, exp_data, fig_dir_name):
    P = count_WGAN.shape[0]
    wgan_arrival_epoch_ls = arrival_epoch_simulator(
        count_WGAN, exp_data['interval_len'])
    # [ ] save results for exp_data['test_arrival_epoch_ls'] to save time
    # [ ] confidence interval?
    # set service time
    # for fake count, use arrival_epoch_simulator, get arrival epoch
    # for CIR, use arrival epoch directly
    # run through queue and compare

    # lognormal distribution with
    # mean 206.44 and variance 23,667 (in seconds) as estimated from the data. Each waiting call
    lognormal_var = 206.44/3600
    lognormal_mean = 23667/3600**2
    # lognormal_var = 0.1
    # lognormal_mean = 0.1
    lognormal_var = 0.1
    lognormal_mean = 0.1
    normal_sigma = (
        np.log(lognormal_var / lognormal_mean ** 2 + 1))**0.5
    normal_mean = np.log(lognormal_mean) - \
        normal_sigma ** 2 / 2

    # service_rate = 1
    # sampler = lambda size: np.random.exponential(1/service_rate, size=size)
    def sampler(size): return np.random.lognormal(
        mean=normal_mean, sigma=normal_sigma, size=size)

    service_rate = 1/np.mean(sampler(1000))
    # print('service_rate:', service_rate)
    # _ = plt.hist(sampler(1000),bins=50,range=(0,1))
    # _ = plt.hist(sampler(1000),bins=50)

    para = {}
    para['TRAIN_SIZE'] = 300
    n_rep = 20

    # real_count_mat = exp_data['train'].numpy()
    # wgan_arrival_epoch_ls

    eval_t_ls = np.arange(0, 11, 0.05)

    T = P * exp_data['interval_len']

    fake_batch_arrival_process = BatchArrivalProcess(
        T, wgan_arrival_epoch_ls)
    real_batch_arrival_process = BatchArrivalProcess(
        T, exp_data['test_arrival_epoch_ls'])

    real_batch_arrival_process.set_service_time(sampler)
    fake_batch_arrival_process.set_service_time(sampler)

    fake_occupied_mat = infinite_server_queue_batch(
        fake_batch_arrival_process, eval_t_ls)
    real_occupied_mat = infinite_server_queue_batch(
        real_batch_arrival_process, eval_t_ls)

    fake_mean_occupied = np.mean(fake_occupied_mat, axis=0)
    real_mean_occupied = np.mean(real_occupied_mat, axis=0)
    fake_var_occupied = np.var(fake_occupied_mat, axis=0)
    real_var_occupied = np.var(real_occupied_mat, axis=0)

    plt.figure(figsize=(14, 5))
    plt.subplot(121)
    plt.plot(eval_t_ls, fake_mean_occupied, label='DS-WGAN')
    plt.plot(eval_t_ls, real_mean_occupied, label='Real CIR data')
    plt.title('mean_occupied')
    plt.subplot(122)
    plt.plot(eval_t_ls, fake_var_occupied, label='DS-WGAN')
    plt.plot(eval_t_ls, real_var_occupied, label='Real CIR data')
    plt.legend()
    plt.title('var_occupied')
    plt.savefig(fig_dir_name)
    plt.close('all')

    # fake_count_mat = exp_data['train'].numpy()
    # fake_PC_size = len(fake_count_mat)
    # fake_PC_ls = np.ndarray((fake_PC_size,), dtype=np.object)
    # T = exp_data['interval_len'] * P

    # # for i in progressbar.progressbar(range(fake_PC_size)):
    # #     fake_PC_ls[i] = ArrivalProcess(T=T,arrival_ls=arrival_epoch_sampler(fake_count_mat[i,:]))
    # eval_t_ls = np.arange(0, 11, 0.05)

    # fake_PC_n_occupied = np.zeros(
    #     (fake_PC_size, len(eval_t_ls)))
    # for i in progressbar.progressbar(range(fake_PC_size)):
    #     fake_PC_n_occupied[i, :] = infinite_server_queue_batch(
    #         fake_PC_ls[i].arrival_ls, sampler, eval_t_ls)

    # fake_n_occupied_mean_mat = np.zeros(
    #     (n_rep, len(eval_t_ls)))
    # fake_n_occupied_var_mat = np.zeros((n_rep, len(eval_t_ls)))

    # for i in range(n_rep):
    #     start_id = i * para['TRAIN_SIZE']
    #     end_id = (i + 1) * para['TRAIN_SIZE']
    #     fake_n_occupied_mean_one_rep = np.mean(
    #         fake_PC_n_occupied[start_id:end_id, :], axis=0)
    #     fake_n_occupied_var_one_rep = np.var(
    #         fake_PC_n_occupied[start_id:end_id, :], axis=0)
    #     fake_n_occupied_mean_mat[i,
    #                              :] = fake_n_occupied_mean_one_rep
    #     fake_n_occupied_var_mat[i,
    #                             :] = fake_n_occupied_var_one_rep

    # fake_mean_lb, fake_mean_ub = get_CI(
    #     fake_n_occupied_mean_mat)
    # fake_var_lb, fake_var_ub = get_CI(fake_n_occupied_var_mat)


def eval_multi_server_queue(count_WGAN, exp_data, file_dir_name, iteration, backend='cpp'):
    if not hasattr(eval_multi_server_queue, "real_wait_ls"):
        print('Run multi server queue for real CIR data')
        P = count_WGAN.shape[1]

        n_vis_interval = 44

        # total length
        eval_multi_server_queue.T = P * exp_data['interval_len']
        # the time point that number of server will change
        eval_multi_server_queue.change_point = np.linspace(
            0, eval_multi_server_queue.T, P+1)
        eval_multi_server_queue.n_server_ls = [20] * P

        vis_interval_ls = np.linspace(
            0, eval_multi_server_queue.T, n_vis_interval+1)

        eval_infinite_server_queue.vis_interval_mid = (
            vis_interval_ls[1:] + vis_interval_ls[:-1])/2

        # lognormal distribution with
        # mean 206.44 and variance 23,667 (in seconds) as estimated from the data. Each waiting call
        lognormal_var = 206.44/3600
        lognormal_mean = 23667/3600**2
        # lognormal_var = 0.1
        # lognormal_mean = 0.1
        lognormal_var = 0.1
        lognormal_mean = 0.1
        normal_sigma = (
            np.log(lognormal_var / lognormal_mean ** 2 + 1))**0.5
        normal_mean = np.log(lognormal_mean) - \
            normal_sigma ** 2 / 2

        def sampler(size): return np.random.lognormal(
            mean=normal_mean, sigma=normal_sigma, size=size)
        eval_multi_server_queue.sampler = sampler

        real_batch_arrival_process = BatchArrivalProcess(
            eval_multi_server_queue.T, exp_data['test_arrival_epoch_ls'])
        real_batch_arrival_process.set_service_time(
            eval_multi_server_queue.sampler)

        def get_CI(input_mat):
            lb = np.percentile(input_mat, 2.5, axis=0)
            ub = np.percentile(input_mat, 97.5, axis=0)
            return lb, ub

        if backend in ['cpp', 'both']:
            import des_cpp

            # eval_multi_server_queue.real_wait_ls_cpp = np.ndarray(
            #     (real_batch_arrival_process.n_arrival_process,), dtype=np.object)

            for i in progressbar.progressbar(range(real_batch_arrival_process.n_arrival_process)):
                a = real_batch_arrival_process.arrival_process_ls[i].arrival_ls
                b = real_batch_arrival_process.arrival_process_ls[i].service_ls
                c = eval_multi_server_queue.change_point.tolist()
                wait_ls = des_cpp.multi_server_queue(
                    a, b, c, eval_multi_server_queue.n_server_ls, False)

                # set -1 to nan
                wait_ls = np.array(wait_ls)
                wait_ls[wait_ls == -1] = np.nan
                # exit_ls = wait_ls + \
                #     real_batch_arrival_process.arrival_process_ls[i].service_ls
                # eval_multi_server_queue.real_wait_ls_cpp[i] = wait_ls
                real_batch_arrival_process.set_wait_time(i, wait_ls)
                # plt.plot(a, wait_ls)
            # plt.xlim(0, eval_multi_server_queue.T)
            # plt.show()

            eval_multi_server_queue.real_summary = real_batch_arrival_process.get_batch_wait_summary(
                vis_interval_ls, quantile=0.8)

        if backend in ['python', 'both']:
            server_ls = get_server_ls(
                eval_multi_server_queue.change_point, eval_multi_server_queue.n_server_ls)
            eval_multi_server_queue.servers = ChangingServerCluster(server_ls)
            eval_multi_server_queue.real_wait_ls_py = batch_multi_server_queue(
                real_batch_arrival_process, eval_multi_server_queue.servers)

        if backend == 'both':
            # [ ] compare the two results and fix some potential bugs
            plt.figure(figsize=(18, 5))
            plt.subplot(121)
            for i in range(len(eval_multi_server_queue.real_wait_ls_cpp)):
                plt.plot(
                    real_batch_arrival_process.arrival_process_ls[i].arrival_ls, eval_multi_server_queue.real_wait_ls_cpp[i])
            plt.xlim(0, eval_multi_server_queue.T)
            plt.subplot(122)
            for i in range(len(eval_multi_server_queue.real_wait_ls_py)):
                plt.plot(
                    real_batch_arrival_process.arrival_process_ls[i].arrival_ls, eval_multi_server_queue.real_wait_ls_py[i][0])
            plt.xlim(0, eval_multi_server_queue.T)
            plt.show()

    print('Run multi server queue for fake CIR data')
    if backend == 'cpp':
        wgan_arrival_epoch_ls = arrival_epoch_simulator(
            count_WGAN, exp_data['interval_len'])
        fake_batch_arrival_process = BatchArrivalProcess(
            eval_multi_server_queue.T, wgan_arrival_epoch_ls)
        fake_batch_arrival_process.set_service_time(
            eval_multi_server_queue.sampler)

        # fake_wait_ls = np.ndarray(
        #         (fake_batch_arrival_process.n_arrival_process,), dtype=np.object)

        for i in progressbar.progressbar(range(fake_batch_arrival_process.n_arrival_process)):
            a = fake_batch_arrival_process.arrival_process_ls[i].arrival_ls
            b = fake_batch_arrival_process.arrival_process_ls[i].service_ls
            c = eval_multi_server_queue.change_point.tolist()
            wait_ls = des_cpp.multi_server_queue(
                a, b, c, eval_multi_server_queue.n_server_ls, False)

            # set -1 to nan
            wait_ls = np.array(wait_ls)
            wait_ls[wait_ls == -1] = np.nan
            # exit_ls = wait_ls + \
            #     real_batch_arrival_process.arrival_process_ls[i].service_ls
            # fake_wait_ls[i] = wait_ls
            fake_batch_arrival_process.set_wait_time(i, wait_ls)

    elif backend == 'python':
        # service_rate = 1
        # sampler = lambda size: np.random.exponential(1/service_rate, size=size)
        count_WGAN = count_WGAN[:20, :]
        wgan_arrival_epoch_ls = arrival_epoch_simulator(
            count_WGAN, exp_data['interval_len'])
        fake_batch_arrival_process = BatchArrivalProcess(
            eval_multi_server_queue.T, wgan_arrival_epoch_ls)
        fake_batch_arrival_process.set_service_time(
            eval_multi_server_queue.sampler)
        fake_wait_ls = batch_multi_server_queue(
            fake_batch_arrival_process, eval_multi_server_queue.servers)

    fake_summary = fake_batch_arrival_process.get_batch_wait_summary(vis_interval_ls=vis_interval_ls,
                                                                     quantile=0.8)
    if True:
        # visualize fake_wait_ls & real_wait_ls
        plt.figure()
        mean_real_lb, mean_real_ub = get_CI(
            eval_multi_server_queue.real_summary['mean'])
        mean_fake_lb, mean_fake_ub = get_CI(fake_summary['mean'])
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 mean_real_lb, 'r--', label='real_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 mean_real_ub, 'r--', label='real_ub')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 mean_fake_lb, 'b--', label='fake_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 mean_fake_ub, 'b--', label='fake_ub')
        plt.savefig(os.path.join(file_dir_name,
                    'multi_mean_{}.png'.format(iteration)))

        plt.figure()
        var_real_lb, var_real_ub = get_CI(
            eval_multi_server_queue.real_summary['var'])
        var_fake_lb, var_fake_ub = get_CI(fake_summary['var'])
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 var_real_lb, 'r--', label='real_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 var_real_ub, 'r--', label='real_ub')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 var_fake_lb, 'b--', label='fake_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 var_fake_ub, 'b--', label='fake_ub')
        plt.savefig(os.path.join(file_dir_name,
                    'multi_var_{}.png'.format(iteration)))

        plt.figure()
        quantile_real_lb, quantile_real_ub = get_CI(
            eval_multi_server_queue.real_summary['quantile'])
        quantile_fake_lb, quantile_fake_ub = get_CI(fake_summary['quantile'])
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 quantile_real_lb, 'r--', label='real_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 quantile_real_ub, 'r--', label='real_ub')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 quantile_fake_lb, 'b--', label='fake_lb')
        plt.plot(eval_infinite_server_queue.vis_interval_mid,
                 quantile_fake_ub, 'b--', label='fake_ub')
        plt.savefig(os.path.join(file_dir_name,
                    'multi_quantile_{}.png'.format(iteration)))

    plt.figure()
    plt.subplot(121)
    plt.pcolor(eval_multi_server_queue.real_summary['mean'])
    plt.colorbar()
    plt.subplot(122)
    plt.pcolor(fake_summary['mean'])
    plt.colorbar()
    plt.savefig(os.path.join(file_dir_name,
                'multi_pcolor_{}.png'.format(iteration)))
