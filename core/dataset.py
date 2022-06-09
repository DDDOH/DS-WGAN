from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gamma, multivariate_normal, norm, uniform
import os
import torch
from .arrival_process import BatchCIR
from .pgnorta import get_PGnorata_from_img

# from core.dataset import get_real_dataset, get_synthetic_dataset, visualize

# from . import get_real_dataset, get_synthetic_dataset, visualize


def prepare_data(args, result_dir=''):
    dataset = args.dataset
    # new_data = args.resample

    return_val = {}

    if dataset in ['bikeshare', 'callcenter']:
        train, test = get_real_dataset(dataset)
    elif dataset == 'cir':
        # set CIR parameters
        # get CIR arrival count, arrival epoch
        P = 22
        interval_len = 0.5
        T = P * interval_len

        base_lam = np.array([124, 175, 239, 263, 285,
                            299, 292, 276, 249, 257,
                            274, 273, 268, 259, 251,
                            252, 244, 219, 176, 156,
                            135, 120])

        cir = BatchCIR(base_lam, interval_len)
        train, _ = cir.simulate_batch_CIR(n_CIR=args.n_train)
        test, return_val['test_arrival_epoch_ls'] = cir.simulate_batch_CIR(
            n_CIR=args.n_test)
        return_val['interval_len'] = interval_len
        # train --[dswgan]--> trained model ----> fake arrival count --[arrival epoch simulator]--> fake arrival epoch
        # arrival epoch --[run through queue]--> waiting time

        # raise NotImplemented
    elif dataset in ['uniform', 'bimodal', 'pgnorta']:
        data_dir = 'assets/{}/'.format(dataset)
        intensity, train = get_synthetic_dataset(
            args.n_interval, args.n_train, dataset, data_dir)
        # raise error with message 'dataset not supported'
    else:
        raise NotImplemented('Dataset not supported')
    visualize(dataset, os.path.join(result_dir, 'figures'), train)

    return_val['train'] = torch.tensor(train, dtype=torch.float)
    return return_val


# set marginal mean, marginal variance for intensity.
# set corr_mat for the underlying multi-normal distribution
CC_test = np.load(
    'assets/test_callcenter.npy')
CC_train = np.load(
    'assets/train_callcenter.npy')

data = CC_train
p = np.shape(data)[1]
corr_mat = np.corrcoef(data, rowvar=False)

marginal_mean = np.mean(data, axis=0)
marginal_var = np.var(data, axis=0)


def get_synthetic_dataset(P, N_TRAIN, DATASET, DATA_DIR=None):

    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)

    # print(
    #         "Data directory already exists, remove exisiting DATA_DIR ({}) ()? [y/n]".format(DATA_DIR))
    #     if input() == 'y':
    #         shutil.rmtree(DATA_DIR)
    #     elif input() == 'n':
    #         print("Abort")
    #         exit()
    #     else:
    #         print("Invalid input")
    #         exit()
    # else:
    data_dir_name = os.path.join(
        DATA_DIR, 'n_interval_{}_n_sample_{}.npz'.format(P, N_TRAIN))
    if not os.path.exists(data_dir_name):
        # print(
        #     "Data directory already exists, remove exisiting file ({})? [y/n]".format(data_dir_name))
        # if input() == 'y':
        #     shutil.rmtree(data_dir_name)
        # elif input() == 'n':
        #     print("Abort")
        #     exit()
        # else:
        #     print("Invalid input")
        #     exit()

        print('Sample new dataset and save to {}'.format(data_dir_name))

        if DATASET not in ['uniform', 'bimodal', 'pgnorta']:
            raise ValueError('Dataset not supported')
        if DATASET == 'uniform':
            intensity, train = sample_uniform(P, N_TRAIN)
        if DATASET == 'bimodal':
            intensity, train = sample_bimodal(P, N_TRAIN)
        if DATASET == 'pgnorta':
            intensity, train = sample_pgnorta(N_TRAIN)
        np.savez(data_dir_name, train=train, intensity=intensity)
    else:
        print(
            "Data directory already exists, load existing file ({})".format(data_dir_name))
        npzfile = np.load(data_dir_name)
        train = npzfile['train']
        intensity = npzfile['intensity']

    return intensity, train
    # np.save(DATA_DIR + 'train_{}.npy'.format(DATASET), train)
    # np.save(DATA_DIR + 'intensity_{}.npy'.format(DATASET), intensity)

# else:
#     # if NEW_DATA is False, then DATA_DIR must be a valid directory
#     assert os.path.isdir(
#         DATA_DIR), 'DATA_DIR must be a valid directory if NEW_DATA is False'
#     print('Load from exisitng files in {}'.format(DATA_DIR))
#     intensity, train = np.load(DATA_DIR + 'intensity_{}.npy'.format(
#         DATASET)), np.load(DATA_DIR + 'train_{}.npy'.format(DATASET))
#     return intensity, train


def get_real_dataset(DATASET):
    # if DATA_DIR != '':  # if DATA_DIR is '', which means evaluating in the result folder, no need to check the data path
    #     assert os.path.isdir(DATA_DIR)
    # print('Load from exisitng files in {}'.format(DATA_DIR))
    if DATASET in ['callcenter', 'bikeshare']:
        train, test = np.load('core/dataset/train_{}.npy'.format(DATASET)
                              ), np.load('core/dataset/test_{}.npy'.format(DATASET))
        return train, test
    else:
        raise ValueError('Dataset not supported')


def sample_uniform(P, N_TRAIN):
    assert P <= p
    uniform_start = marginal_mean[:P] * 15
    uniform_end = marginal_mean[:P] * (15 + np.random.rand() * 20)

    z = multivariate_normal.rvs(np.zeros(P), corr_mat[:P, :P], N_TRAIN)
    U = norm.cdf(z)
    intensity = np.empty_like(U)
    for i in range(P):
        intensity[:, i] = uniform.ppf(
            q=U[:, i], loc=uniform_start[i], scale=uniform_end[i] - uniform_start[i])
    train = np.random.poisson(intensity)

    return intensity, train


def sample_pgnorta(N_TRAIN):
    pgnorta = get_PGnorata_from_img()
    return pgnorta.sample_both(N_TRAIN)


def sample_bimodal(P, N_TRAIN):
    assert P <= p
    modal1_mean = marginal_mean[:P] * 30
    modal2_mean = marginal_mean[:P] * 30 + \
        marginal_mean[:P] * 15 * np.random.uniform(-1, 1.5, P)

    # a randomly generated psd matrix
    A = np.random.rand(P, P)
    B = np.dot(A, A.transpose())

    modal1_cov = np.cov(data, rowvar=False)
    modal2_cov = B * marginal_var * np.random.uniform(0.8, 1.2, size=(P))

    prob_modal_1 = 0.7

    u = np.random.rand(N_TRAIN)

    intensity_modal_1 = multivariate_normal.rvs(
        mean=modal1_mean, cov=modal1_cov, size=N_TRAIN)
    intensity_modal_2 = multivariate_normal.rvs(
        mean=modal2_mean, cov=modal2_cov, size=N_TRAIN)

    intensity = np.empty_like(intensity_modal_1)
    intensity[u < prob_modal_1] = intensity_modal_1[u < prob_modal_1]
    intensity[u >= prob_modal_1] = intensity_modal_2[u >= prob_modal_1]

    train = np.random.poisson(intensity)

    return intensity, train


def visualize(dataset, fig_dir, train):
    P = np.shape(train)[1]
    fig, ax = plt.subplots()
    plt.plot(np.mean(train, axis=0))
    plt.scatter(np.tile(np.arange(P), np.shape(train)[0]).reshape(
        P, np.shape(train)[0]), train, alpha=0.02, c='C1', s=40, edgecolors='none', label=r'$(j,X_{i,j})$')
    plt.xlabel('Time Interval')
    plt.ylabel('Arrival Count')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    plt.ylim(np.min(train)*0.9, np.max(train)*1.1)
    legend_elements = [Line2D([0], [0], color='C0', lw=1, label='Marginal Mean'),
                       Line2D([0], [0], marker='o', lw=0, color='w', label=r'$(j,X_{i,j})$',
                              markerfacecolor='C1', markersize=7, alpha=0.6)]
    ax.legend(handles=legend_elements)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '{}_mean_scatter.pdf').format(dataset))
    plt.close('all')

    plt.figure()
    plt.plot(np.var(train, axis=0))
    plt.xlabel('Time Interval')
    plt.ylabel('Marginal Variance of Arrival Count')
    plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, '{}_var.pdf').format(dataset))
    plt.close('all')
