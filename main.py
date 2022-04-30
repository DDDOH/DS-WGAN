import glob
import os
import shutil
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import progressbar
import seaborn as sns
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from colored import attr, fg
from geomloss import SamplesLoss
from matplotlib.lines import Line2D
from scipy.stats import gamma, norm

from core.dataset import get_real_dataset, get_synthetic_dataset
from core.evaluate import evaluate_joint, evaluate_marginal
from core.pgnorta import estimate_PGnorta

today = date.today().strftime("%y_%m_%d")


DATASET = 'callcenter'  # 'uniform' or 'bimodal' or 'bikeshare' or 'callcenter'
NEW_DATA = False

TRAIN = True
P = 16  # works only when TRAIN == True and DATASE == 'bikeshare' or 'callcenter'
SEED_DIM = 4  # works only when TRAIN == True
NOISE = 'normal'  # uniform or normal
ONLY_MLP = True
lr_final = 1e-4
lr_initial = 0.001
DIM = [512, 512, 512, 512, 256]
N_LAYER = len(DIM)
N_TRAIN = 700
# RELEASE = True  # whether modify the CI to match with previous result


REAL_DATA = DATASET in ['bikeshare', 'callcenter']

result_dir = '/Users/hector/Offline Documents/Doubly_Stochastic_WGAN/tmp_results/' + \
    today + '_' + '{}__P_{}__seed_dim_{}/'.format(DATASET, P, SEED_DIM)
if DATASET == 'bikeshare':
    data_dir = 'Dataset/bikeshare_dswgan_real/'
elif DATASET == 'callcenter':
    data_dir = 'Dataset/oakland_callcenter_dswgan/'
else:
    data_dir = 'Dataset/{}__P_{}/'.format(DATASET, P)

DEBUG = True
if DEBUG:
    ITERS = 300
    SAVE_FREQ = 100
    MAX_T = 10
else:
    ITERS = 30000
    SAVE_FREQ = 1000
    MAX_T = 4000


EVAL = True  # whether evaluate immediately after training

EVAL_IN_RESULT_DIR = True
if EVAL_IN_RESULT_DIR:
    NEW_DATA = False
    TRAIN = False
    EVAL = True


if not EVAL_IN_RESULT_DIR:
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)

    shutil.copyfile(__file__,
                    result_dir + 'source.py')
    if os.path.exists(result_dir + 'core'):
        shutil.rmtree(result_dir + 'core')
    shutil.copytree("core",
                    result_dir + 'core')


if EVAL_IN_RESULT_DIR:
    result_dir = ''
    data_dir = ''

if NEW_DATA:
    if DATASET in ['bikeshare', 'callcenter']:
        train, test = get_real_dataset(DATASET, data_dir)
    else:
        intensity, train = get_synthetic_dataset(
            P, N_TRAIN, DATASET, NEW_DATA, data_dir)
else:
    train = np.load(data_dir + 'train_{}.npy'.format(MARGIN))
    intensity = np.load(data_dir + 'intensity_{}.npy'.format(MARGIN))


if not EVAL_IN_RESULT_DIR:
    shutil.copyfile(data_dir + 'train_{}.npy'.format(DATASET),
                    result_dir + 'train_{}.npy'.format(DATASET))
    if REAL_DATA:
        shutil.copyfile(data_dir + 'test_{}.npy'.format(DATASET),
                        result_dir + 'test_{}.npy'.format(DATASET))
    else:
        shutil.copyfile(data_dir + 'intensity_{}.npy'.format(DATASET),
                        result_dir + 'intensity_{}.npy'.format(DATASET))


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

# Create the figure
# plt.subplots()
ax.legend(handles=legend_elements)
# plt.legend()
plt.tight_layout()
plt.savefig(result_dir + 'figures/{}_train_mean_scatter.pdf'.format(DATASET))
plt.close('all')

plt.figure()
plt.plot(np.var(train, axis=0))
plt.xlabel('Time Interval')
plt.ylabel('Marginal Variance of Arrival Count')
plt.xticks(ticks=np.arange(0, P), labels=np.arange(0, P)+1)
plt.tight_layout()
plt.savefig(result_dir + 'figures/{}_train_var.pdf'.format(DATASET))
plt.close('all')

# # Estimate PGnorta Gamma marginal distribution
p = np.shape(train)[1]
B = np.mean(train, axis=0)
var_X = np.var(train, axis=0)
alpha = B ** 2 / (var_X - B)

if np.min(alpha) < 0:
    print(
        'The arrival count of the {}-th time interval does not satisfy variance >= mean'.format(np.where(alpha < 0)[0]))

alpha[alpha < 0] = 10000  # alpha 越大，则生成的arrival count的mean和variance越接近


def sample_PGnorta_marginal(base_intensity_t, alpha_t, n_sample):
    z = np.random.normal(0, 1, n_sample)
    U = norm.cdf(z)
    B = gamma.ppf(q=U, a=alpha_t,
                  scale=1/alpha_t)
    intensity = B * base_intensity_t
    count = np.random.poisson(intensity)
    return intensity, count


training_set = torch.tensor(train, dtype=torch.float)

P = np.shape(training_set)[1]
TRAIN_SIZE = np.shape(training_set)[0]


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.normal = torch.distributions.normal.Normal(
            loc=0, scale=1, validate_args=None)

        nonlinear_layers = [
            nn.Linear(SEED_DIM, DIM[0]), nn.LeakyReLU(0.1, True)]
        for i in range(N_LAYER-1):
            nonlinear_layers.append(nn.Linear(DIM[i], DIM[i+1]))
            nonlinear_layers.append(nn.LeakyReLU(0.1, True))
        nonlinear_layers.append(nn.Linear(DIM[-1], P))

        self.nonlinear = nn.Sequential(*nonlinear_layers)
        self.linear = nn.Linear(SEED_DIM, P)

    def forward(self, noise):
        if ONLY_MLP:
            return self.nonlinear(noise)
        else:
            linear_output = self.linear(noise)
            nonlinear_output = self.nonlinear(noise)
            output = self.normal.cdf(linear_output + nonlinear_output)
        return output


def compare_plot(real, fake, PGnorta_mean, PGnorta_var, msg, save=False):
    """Visualize and compare the real and fake.
    """
    real_size = np.shape(real)[0]
    fake_size = np.shape(fake)[0]

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


def scatter_plot(interval_1_real, interval_2_real, interval_1_fake, interval_2_fake, msg, save=False):
    plt.figure()
    plt.scatter(interval_1_real, interval_2_real, label='True', alpha=0.2)
    plt.scatter(interval_1_fake, interval_2_fake, label='Fake', alpha=0.2)
    plt.legend()
    if save:
        plt.savefig(result_dir + msg + '.png')
    plt.close()


# get marginal mean & var of PGnorta
n_sample = 100000  # number of samples to visualize
lam = B.numpy() if torch.is_tensor(B) else B
PGnorta_count_mean = np.zeros_like(lam)
PGnorta_count_var = np.zeros_like(lam)
PGnorta_intensity_mean = np.zeros_like(lam)
PGnorta_intensity_var = np.zeros_like(lam)
for interval in progressbar.progressbar(range(p)):
    base_intensity_t = lam[interval]
    alpha_t = alpha[interval]
    intensity_PGnorta, count_PGnorta = sample_PGnorta_marginal(
        base_intensity_t, alpha_t, n_sample)

    PGnorta_count_mean[interval] = np.mean(count_PGnorta)
    PGnorta_count_var[interval] = np.var(count_PGnorta)
    PGnorta_intensity_mean[interval] = np.mean(intensity_PGnorta)
    PGnorta_intensity_var[interval] = np.var(intensity_PGnorta)

if TRAIN:
    # Build the generator network.
    netG = Generator()
    # We'll compute the sinkhorn distance between the real samples and fake samples.
    # Sinkhorn distance is an approximated wassestein distance.
    # For details, see https://www.kernel-operations.io/geomloss/
    # This is an alternative version of GAN.
    # On the current experiment instance, sinkhorn distance runs much more faster than the full GAN version and
    # thus is used for this demo.
    sinkorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.5)
    B = torch.tensor(B, dtype=torch.float)

    # may consider smaller learning rate
    gamma_G = (lr_final/lr_initial)**(1/ITERS)
    optimizerG = optim.Adam(netG.parameters(), lr=lr_initial, betas=(0.5, 0.9))
    optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(
        optimizerG, gamma=gamma_G, last_epoch=-1)

    G_cost_record = []
    lr_record = []

    for iteration in progressbar.progressbar(range(ITERS), redirect_stdout=True):
        if NOISE == 'normal':
            noise = torch.randn(TRAIN_SIZE, SEED_DIM)
        if NOISE == 'uniform':
            noise = torch.rand(TRAIN_SIZE, SEED_DIM)
        G_Z = netG(noise)

        if ONLY_MLP:
            intensity_pred = G_Z
        else:
            ppf_G_Z = torch.empty_like(G_Z)
            for i in range(p):
                ppf_G_Z[:, i] = GammaPPF_torch(G_Z[:, i], alpha[i], 1/alpha[i])

            intensity_pred = ppf_G_Z * B

        intensity_val = intensity_pred.cpu().data.numpy()
        sign = np.sign(intensity_val)
        intensity_val = intensity_val * sign
        count_val = np.random.poisson(intensity_val)
        count_val = count_val * sign

        w_mid = 1 + (count_val - intensity_val)/(2 * intensity_val)
        w_mid = np.maximum(w_mid, 0.5)
        w_mid = np.minimum(w_mid, 1.5)
        b_mid = count_val - w_mid * intensity_val

        # TODO fix warning here
        # UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor.
        w_mid_tensor = autograd.Variable(torch.Tensor([w_mid]))
        b_mid_tensor = autograd.Variable(torch.Tensor([b_mid]))

        pred_fake = (intensity_pred * w_mid_tensor + b_mid_tensor).squeeze()

        G_cost = sinkorn_loss(pred_fake, training_set)
        netG.zero_grad()
        G_cost.backward()
        G_cost_record.append(G_cost.detach().numpy())
        optimizerG.step()
        optimizerG_lrdecay.step()

        lr_record.append(optimizerG_lrdecay.get_last_lr())

        if iteration % 100 == 0:
            # Visualize the real and fake intensity every 100 training iterations.
            intensity_pred_np = intensity_pred.detach().numpy()
            print(
                'Generator loss in {}-th iteration: {}'.format(iteration, G_cost.item()))

            if not REAL_DATA:
                compare_plot(real=intensity, fake=intensity_pred_np, msg='intensity'+str(iteration),
                             PGnorta_mean=PGnorta_intensity_mean, PGnorta_var=PGnorta_intensity_var, save=True)
            compare_plot(real=train, fake=count_val, msg='count'+str(iteration),
                         PGnorta_mean=PGnorta_count_mean, PGnorta_var=PGnorta_count_var, save=True)

            if not REAL_DATA:
                scatter_plot(interval_1_real=intensity[:, 0], interval_2_real=intensity[:, 1],
                             interval_1_fake=intensity_pred_np[:,
                                                               0], interval_2_fake=intensity_pred_np[:, 1],
                             msg='scatter_intensity_'+str(iteration), save=True)
            scatter_plot(interval_1_real=train[:, 0], interval_2_real=train[:, 1],
                         interval_1_fake=count_val[:,
                                                   0], interval_2_fake=count_val[:, 1],
                         msg='scatter_count_'+str(iteration), save=True)

            plt.figure()
            plt.semilogy(G_cost_record)
            plt.title('G_cost')
            plt.savefig(result_dir + 'G_cost.png')
            plt.close()

            plt.figure()
            plt.plot(lr_record)
            plt.title('LR')
            plt.savefig(result_dir + 'LR_record.png')
            plt.close()

        if iteration % SAVE_FREQ == 0 and iteration != 0:
            torch.save(netG.state_dict(), result_dir +
                       'netG_{}.pth'.format(iteration))


if EVAL:
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

    print(color + 'Now computing CI' + attr('reset'))
    for i in progressbar.progressbar(range(n_rep_CI)):
        # get count_WGAN_mat
        if NOISE == 'normal':
            noise = torch.randn(n_sample, SEED_DIM)
        if NOISE == 'uniform':
            noise = torch.rand(n_sample, SEED_DIM)
        G_Z = netG(noise)
        if ONLY_MLP:
            intensity_val = G_Z.cpu().data.numpy()
            sign = np.sign(intensity_val)
            intensity_val = intensity_val * sign
            count_val = np.random.poisson(intensity_val)
            count_WGAN_mat = count_val * sign
        else:
            ppf_G_Z = torch.empty_like(G_Z)
            for i in range(p):
                ppf_G_Z[:, i] = GammaPPF_torch(G_Z[:, i], alpha[i], 1/alpha[i])
            intensity_pred = ppf_G_Z * B

            intensity_val = intensity_pred.cpu().data.numpy()
            count_WGAN_mat = np.random.poisson(intensity_val)

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
    plt.savefig(result_dir + 'figures/{}_compare_mean.pdf'.format(DATASET))

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
    plt.savefig(result_dir + 'figures/{}_compare_var.pdf'.format(DATASET))

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
        result_dir + 'figures/{}_compare_past_future_corr.pdf'.format(DATASET))
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
    plt.savefig(result_dir + 'figures/{}_compare_w_dist.pdf'.format(DATASET))
    plt.close('all')

    # arrival count mat for ecdf and histogram
    n_sample = 10000
    # get count_WGAN_mat
    if NOISE == 'normal':
        noise = torch.randn(n_sample, SEED_DIM)
    if NOISE == 'uniform':
        noise = torch.rand(n_sample, SEED_DIM)
    G_Z = netG(noise)
    if ONLY_MLP:
        intensity_val = G_Z.cpu().data.numpy()
        sign = np.sign(intensity_val)
        intensity_val = intensity_val * sign
        count_val = np.random.poisson(intensity_val)
        count_WGAN_mat = count_val * sign
    else:
        ppf_G_Z = torch.empty_like(G_Z)
        for i in range(p):
            ppf_G_Z[:, i] = GammaPPF_torch(G_Z[:, i], alpha[i], 1/alpha[i])
        intensity_pred = ppf_G_Z * B

        intensity_val = intensity_pred.cpu().data.numpy()
        count_WGAN_mat = np.random.poisson(intensity_val)

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

        plt.savefig(result_dir + 'figures/appendix/' + '{}_marginal_count_ecdf'.format(DATASET) +
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
        plt.savefig(result_dir + 'figures/appendix/' + '{}_marginal_count_hist'.format(DATASET) +
                    str(interval) + '.pdf')
        plt.close()
