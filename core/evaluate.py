import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import torch
from .utils.des.des import *
from .arrival_process import BatchCIR, BatchArrivalProcess
from .utils.arrival_epoch_simulator import arrival_epoch_simulator


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
    return {'PG': wasserstein_PG_train_rec, 'WGAN': wasserstein_WGAN_train_rec}


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


# def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, result_dir, save=False):
#     """Visualize and compare the real and fake.
#     """
#     real_size = np.shape(real)[0]
#     fake_size = np.shape(fake)[0]

#     P = np.shape(real)[1]

#     assert np.shape(real)[1] == np.shape(
#         fake)[1] == len(PGnorta_mean) == len(PGnorta_var)

#     max_intensity = max(np.max(fake), np.max(real))
#     plt.figure(figsize=(16, 3))
#     plt.subplot(141)
#     plt.plot(np.mean(fake, axis=0), label='fake')
#     plt.plot(np.mean(real, axis=0), label='real')
#     plt.plot(PGnorta_mean, label='PGnorta')
#     plt.xlabel('Time interval')
#     plt.ylabel('Intensity or count')
#     plt.title('Mean')
#     plt.legend()

#     plt.subplot(142)
#     plt.plot(np.var(fake, axis=0), label='fake')
#     plt.plot(np.var(real, axis=0), label='real')
#     plt.plot(PGnorta_var, label='PGnorta')
#     plt.xlabel('Time interval')
#     plt.title('Std')
#     plt.legend()

#     plt.subplot(143)
#     plt.scatter(np.tile(np.arange(P), real_size).reshape(
#         P, real_size), real, alpha=0.003)
#     plt.ylim(0, max_intensity * 1.2)
#     plt.xlabel('Time interval')
#     plt.title('Scatter plot of real')

#     plt.subplot(144)
#     plt.scatter(np.tile(np.arange(P), fake_size).reshape(
#         P, fake_size), fake, alpha=0.003)
#     plt.ylim(0, max_intensity * 1.2)
#     plt.xlabel('Time interval')
#     plt.title('Scatter plot of fake')
#     if save:
#         plt.savefig(
#             result_dir + msg + '.png')
#     else:
#         plt.show()
#     plt.close()


def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, save=False, result_dir=None):
    """Visualize and compare the real and fake.
    """
    # if real is torch tensor, convert to numpy
    if isinstance(real, torch.Tensor):
        real = real.detach().numpy()
    if isinstance(fake, torch.Tensor):
        fake = fake.detach().numpy()
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


def marginal_ecdf():
    # TODO update this function to make the ecdf plot
    # get the latest net G:
    netG_ls = glob.glob(result_dir + 'netG_*.pth')
    netG_iter_ls = [int(netG_ls[i].split('_')[-1].split('.')[0])
                    for i in range(len(netG_ls))]
    max_iter = max(netG_iter_ls)

    netG = Generator()
    netG.load_state_dict(torch.load(
        result_dir + 'netG_{}.pth'.format(max_iter)))
    netG.eval()

    color = fg('blue')

    # marginal mean & var, past & future correlation, W-distance for PGnorta & DS-WGAN, with CI
    n_rep_CI = 100  # how many replications to get the CI
    n_sample = N_TRAIN  # how many samples generated for each replication

    # get count_PGnorta_mat
    if DEBUG:
        norta = estimate_PGnorta(train, zeta=7/16, max_T=MAX_T, M=100, img_dir_name=result_dir +
                                 'sumamry_PGnorta_rho_estimation_record.jpg', rho_mat_dir_name=result_dir+'rho_mat.npy')
    else:
        norta = estimate_PGnorta(train, zeta=7/16, max_T=MAX_T, M=100, img_dir_name=result_dir +
                                 'sumamry_PGnorta_rho_estimation_record.jpg', rho_mat_dir_name=result_dir+'rho_mat.npy')

    marginal_mean_PGnorta_rec = np.zeros((n_rep_CI, P))
    marginal_var_PGnorta_rec = np.zeros((n_rep_CI, P))
    marginal_mean_DSWGAN_rec = np.zeros((n_rep_CI, P))
    marginal_var_DSWGAN_rec = np.zeros((n_rep_CI, P))
    past_future_corr_PGnorta_rec = np.zeros((n_rep_CI, P-1))
    past_future_corr_DSWGAN_rec = np.zeros((n_rep_CI, P-1))
    past_future_corr_train = None
    W_distance_PGnorta_rec = np.zeros((n_rep_CI, P))
    W_distance_DSWGAN_rec = np.zeros((n_rep_CI, P))

    # print(color + 'Now computing CI' + attr('reset'))
    for i in progressbar.progressbar(range(n_rep_CI)):
        # get count_WGAN_mat
        noise = torch.randn(n_sample, args.seed_dim)

        G_Z = netG(noise)
        intensity_val = G_Z.cpu().data.numpy()
        sign = np.sign(intensity_val)
        intensity_val = intensity_val * sign
        count_val = np.random.poisson(intensity_val)
        count_WGAN_mat = count_val * sign

        marginal_mean_DSWGAN_rec[i, :], marginal_var_DSWGAN_rec[i, :] = np.mean(
            count_WGAN_mat, axis=0), np.var(count_WGAN_mat, axis=0)

        count_PGnorta_mat = norta.sample_count(n_sample)
        marginal_mean_PGnorta_rec[i, :], marginal_var_PGnorta_rec[i, :] = np.mean(
            count_PGnorta_mat, axis=0), np.var(count_PGnorta_mat, axis=0)

        w_dist = evaluate_marginal(
            count_WGAN_mat, count_PGnorta_mat, train, result_dir)
        past_future_corr = evaluate_joint(
            count_WGAN_mat, count_PGnorta_mat, train, result_dir)

        W_distance_PGnorta_rec[i, :] = w_dist['PG']
        W_distance_DSWGAN_rec[i, :] = w_dist['WGAN']
        past_future_corr_PGnorta_rec[i, :] = past_future_corr['PG']
        past_future_corr_DSWGAN_rec[i, :] = past_future_corr['WGAN']
        past_future_corr_train = past_future_corr['TRAIN']

    fill_between_alpha = 0.3
    plt.figure()
    plt.fill_between(np.arange(
        P), marginal_mean_PGnorta_CI['low'], marginal_mean_PGnorta_CI['up'], label='PGnorta', alpha=fill_between_alpha)
    plt.fill_between(np.arange(
        P), marginal_mean_DSWGAN_CI['low'], marginal_mean_DSWGAN_CI['up'], label='DS-WGAN', alpha=fill_between_alpha)
    plt.plot(np.mean(train, axis=0), label='Training set', c='C2')
    plt.legend()
    plt.xlabel('Time Interval')
    plt.ylabel('Mean of arrival count')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    # plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
    plt.tight_layout()
    plt.savefig(result_dir + 'figures/{}_compare_mean.pdf'.format(args.dataset))

    plt.figure()
    plt.fill_between(np.arange(
        P), marginal_var_PGnorta_CI['low'], marginal_var_PGnorta_CI['up'], label='PGnorta', alpha=fill_between_alpha)
    plt.fill_between(np.arange(
        P), marginal_var_DSWGAN_CI['low'], marginal_var_DSWGAN_CI['up'], label='DS-WGAN', alpha=fill_between_alpha)
    plt.plot(np.var(train, axis=0), label='Training set', c='C2')
    plt.legend()
    plt.xlabel('Time interval')
    plt.ylabel('Variance of arrival count')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    plt.tight_layout()
    plt.savefig(result_dir + 'figures/{}_compare_var.pdf'.format(args.dataset))

    plt.figure()
    plt.fill_between(np.arange(
        P-1), past_future_corr_PGnorta_CI['low'], past_future_corr_PGnorta_CI['up'], label='PGnorta', alpha=fill_between_alpha)
    plt.fill_between(np.arange(
        P-1), past_future_corr_DSWGAN_CI['low'], past_future_corr_DSWGAN_CI['up'], label='DS-WGAN', alpha=fill_between_alpha)
    plt.plot(past_future_corr_train, label='Training set', c='C2')
    plt.legend()
    plt.xlabel('$j$')
    plt.ylabel(
        r'$\operatorname{Corr}\left(\mathbf{Y}_{1: j}, \mathbf{Y}_{j+1: p}\right)$')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    # plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
    plt.tight_layout()
    plt.savefig(
        result_dir + 'figures/{}_compare_past_future_corr.pdf'.format(args.dataset))
    plt.close('all')

    plt.figure()
    plt.fill_between(np.arange(
        P), W_distance_PGnorta_CI['low'], W_distance_PGnorta_CI['up'], label=r'$D_{j}^{(P)}$', alpha=fill_between_alpha)
    plt.fill_between(np.arange(
        P), W_distance_DSWGAN_CI['low'], W_distance_DSWGAN_CI['up'], label=r'$D_{j}^{(D)}$', alpha=fill_between_alpha)
    plt.legend()
    plt.xlabel('Time Interval $j$')
    plt.ylabel('Wasserstein distance')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    # plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
    plt.tight_layout()
    plt.savefig(
        result_dir + 'figures/{}_compare_w_dist.pdf'.format(args.dataset))
    plt.close('all')

    # arrival count mat for ecdf and histogram
    n_sample = 10000
    # get count_WGAN_mat

    noise = torch.randn(n_sample, args.seed_dim)

    G_Z = netG(noise)

    intensity_val = G_Z.cpu().data.numpy()
    sign = np.sign(intensity_val)
    intensity_val = intensity_val * sign
    count_val = np.random.poisson(intensity_val)
    count_WGAN_mat = count_val * sign

    count_PGnorta_mat = norta.sample_count(n_sample)

    if not os.path.exists(result_dir + 'figures/appendix/'):
        os.mkdir(result_dir + 'figures/appendix/')

    # plot cdf & ecdf, calculate statistics for marginal distribution
    for interval in progressbar.progressbar(range(p)):
        plt.figure()
        ecdf_count_PGnorta = sns.ecdfplot(
            data=count_PGnorta_mat[:, interval], alpha=0.3, label='PGnorta')
        ecdf_count_WGAN = sns.ecdfplot(
            data=count_WGAN_mat[:, interval], alpha=0.3, label='DS-WGAN')
        ecdf_train = sns.ecdfplot(
            data=train[:, interval], alpha=0.3, label='Training set')
        plt.legend()
        plt.xlabel('Arrival count')
        if interval == 0:
            plt.title(
                'Empirical c.d.f of marginal arrival count for 1-st time interval'.format(interval+1))
        if interval == 1:
            plt.title(
                'Empirical c.d.f of marginal arrival count for 2-nd time interval'.format(interval+1))
        if interval == 2:
            plt.title(
                'Empirical c.d.f of marginal arrival count for 3-rd time interval'.format(interval+1))
        else:
            plt.title(
                'Empirical c.d.f of marginal arrival count for {}-th  time interval'.format(interval+1))
        plt.tight_layout()

        plt.savefig(result_dir + 'figures/appendix/' + '{}_marginal_count_ecdf'.format(args.dataset) +
                    str(interval) + '.pdf')
        plt.close()

        plt.figure()
        bins = 50
        fig_alpha = 0.2
        plt.hist(count_PGnorta_mat[:, interval], bins=bins, alpha=fig_alpha,
                 label='PGnorta', density=True)
        plt.hist(count_WGAN_mat[:, interval], bins=bins, alpha=fig_alpha,
                 label='DS-WGAN', density=True)
        plt.hist(train[:, interval], bins=bins, alpha=fig_alpha,
                 label='Training set', density=True)
        plt.xlabel('Arrival count')
        if interval == 0:
            plt.title(
                'Histogram of marginal arrival count for 1-st time interval'.format(interval+1))
        if interval == 1:
            plt.title(
                'Histogram of marginal arrival count for 2-nd time interval'.format(interval+1))
        if interval == 2:
            plt.title(
                'Histogram of marginal arrival count for 3-rd time interval'.format(interval+1))
        else:
            plt.title(
                'Histogram of marginal arrival count for {}-th time interval'.format(interval+1))
        sns_data = {'PGnorta': count_PGnorta_mat[:, interval],
                    'Training': train[:, interval], 'DS-WGAN': count_WGAN_mat[:, interval]}
        sns.kdeplot(data=sns_data, common_norm=False)
        plt.legend()
        plt.tight_layout()
        plt.savefig(result_dir + 'figures/appendix/' + '{}_marginal_count_hist'.format(args.dataset) +
                    str(interval) + '.pdf')
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
    real_batch_arrival_proess = BatchArrivalProcess(
        T, exp_data['test_arrival_epoch_ls'])

    real_batch_arrival_proess.set_service_time(sampler)
    fake_batch_arrival_process.set_service_time(sampler)

    fake_occupied_mat = infinite_server_queue_batch(
        fake_batch_arrival_process, eval_t_ls)
    real_occupied_mat = infinite_server_queue_batch(
        real_batch_arrival_proess, eval_t_ls)

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


def eval_multi_server_queue(count_WGAN, exp_data, file_dir_name):
    if not hasattr(eval_multi_server_queue, "real_wait_ls"):
        print('this message should appear only once')
        P = count_WGAN.shape[1]
        count_WGAN = count_WGAN[:20, :]
        eval_multi_server_queue.T = P * exp_data['interval_len']
        server_ls = get_server_ls(np.linspace(
            0, eval_multi_server_queue.T, P+1), [100]*P)
        # server_ls = [ScheduledServer(torch.tensor([0.0, 9.0, 10.0], requires_grad=True), ['idle', 'home', 'busy']),
        #              ScheduledServer(torch.tensor(
        #                  [0.0, 10.0], requires_grad=True), ['idle', 'busy']),
        #              ScheduledServer(torch.tensor(
        #                  [0.0, 3.0, 4.0, 8.0], requires_grad=True), ['idle', 'busy', 'idle', 'busy']),
        #              ScheduledServer(torch.tensor([2.0, 5.0, 7.0, 9.0], requires_grad=True), ['idle', 'busy', 'idle', 'busy'])]
        eval_multi_server_queue.servers = ChangingServerCluster(server_ls)

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
        real_batch_arrival_proess = BatchArrivalProcess(
            eval_multi_server_queue.T, exp_data['test_arrival_epoch_ls'])
        real_batch_arrival_proess.set_service_time(
            eval_multi_server_queue.sampler)
        eval_multi_server_queue.real_wait_ls = batch_multi_server_queue(
            real_batch_arrival_proess, eval_multi_server_queue.servers)

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

    a = 1

    # visualize fake_wait_ls & real_wait_ls

    # group real_wait_ls and fake_wait_ls by their arrivaling time

    # raise NotImplementedError('queue type not implemented')
