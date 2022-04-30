import numpy as np
import scipy
import matplotlib.pyplot as plt


def evaluate_marginal(count_WGAN, count_PGnorta, count_train, result_dir):
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

    return {'PG': wasserstein_PG_train_rec, 'WGAN': wasserstein_WGAN_train_rec}

    # plt.figure()
    # plt.plot(wasserstein_PG_train_rec, label='PGnorta & Train')
    # plt.plot(wasserstein_WGAN_train_rec, label='DS-WGAN & Train')
    # plt.title('Wasserstein distance')
    # plt.legend()
    # plt.savefig(result_dir + 'summary_wasserstein_distance.png')
    # plt.close()


def evaluate_joint(count_WGAN, count_PGnorta, count_train, result_dir):
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

    return {'PG': past_future_corr_PG, 'WGAN': past_future_corr_WGAN, 'TRAIN': past_future_corr_train}

    # plt.figure()
    # plt.plot(past_future_corr_WGAN, label='WGAN')
    # plt.plot(past_future_corr_PG, label='PGnorta')
    # plt.plot(past_future_corr_train, label='Train')
    # plt.title('Past-Future correlation')
    # plt.legend()
    # plt.savefig(result_dir + 'summary_past_future_correlation.png')
    # plt.close()


def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, result_dir, save=False):
    """Visualize and compare the real and fake.
    """
    real_size = np.shape(real)[0]
    fake_size = np.shape(fake)[0]

    P = np.shape(real)[1]

    assert np.shape(real)[1] == np.shape(
        fake)[1] == len(PGnorta_mean) == len(PGnorta_var)

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
        plt.savefig(
            result_dir + msg + '.png')
    else:
        plt.show()
    plt.close()
