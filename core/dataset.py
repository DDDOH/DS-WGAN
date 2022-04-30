import os
from scipy.stats import gamma, multivariate_normal, norm, uniform
import numpy as np
import shutil


# set marginal mean, marginal variance for intensity.
# set corr_mat for the underlying multi-normal distribution
CC_test = np.load(
    '/Users/hector/Desktop/Doubly_Stochastic_WGAN/Dataset/oakland_callcenter_dswgan/test_callcenter.npy')
CC_train = np.load(
    '/Users/hector/Desktop/Doubly_Stochastic_WGAN/Dataset/oakland_callcenter_dswgan/train_callcenter.npy')

data = CC_train
p = np.shape(data)[1]
corr_mat = np.corrcoef(data, rowvar=False)

marginal_mean = np.mean(data, axis=0)
marginal_var = np.var(data, axis=0)


def get_synthetic_dataset(P, N_TRAIN, DATASET, NEW_DATA, DATA_DIR=None):
    if NEW_DATA:
        if os.path.exists(DATA_DIR):
            print(
                "Data directory already exists, remove exisiting DATA_DIR ({}) ()? [y/n]".format(DATA_DIR))
            if input() == 'y':
                shutil.rmtree(DATA_DIR)
            elif input() == 'n':
                print("Abort")
                exit()
            else:
                print("Invalid input")
                exit()
        print('Sample new dataset and save to {}'.format(DATA_DIR))
        if DATASET not in ['uniform', 'bimodal']:
            raise ValueError('Dataset not supported')
        if DATASET == 'uniform':
            intensity, train = sample_uniform(P, N_TRAIN)
        if DATASET == 'bimodal':
            intensity, train = sample_bimodal(P, N_TRAIN)
        os.mkdir(DATA_DIR)
        np.save(DATA_DIR + 'train_{}.npy'.format(DATASET), train)
        np.save(DATA_DIR + 'intensity_{}.npy'.format(DATASET), intensity)

    else:
        # if NEW_DATA is False, then DATA_DIR must be a valid directory
        assert os.path.isdir(
            DATA_DIR), 'DATA_DIR must be a valid directory if NEW_DATA is False'
        print('Load from exisitng files in {}'.format(DATA_DIR))
        intensity, train = np.load(DATA_DIR + 'intensity_{}.npy'.format(
            DATASET)), np.load(DATA_DIR + 'train_{}.npy'.format(DATASET))
    return intensity, train


def get_real_dataset(DATASET, DATA_DIR):
    if DATA_DIR != '':  # if DATA_DIR is '', which means evaluating in the result folder, no need to check the data path
        assert os.path.isdir(DATA_DIR)
    print('Load from exisitng files in {}'.format(DATA_DIR))
    if DATASET in ['callcenter', 'bikeshare']:
        train, test = np.load(DATA_DIR + 'train_{}.npy'.format(DATASET)
                              ), np.load(DATA_DIR + 'test_{}.npy'.format(DATASET))
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
