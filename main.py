import argparse
from ast import Not
from unittest import result
from core.pgnorta import estimate_PGnorta
from core.evaluate import evaluate_joint, evaluate_marginal, compare_plot, scatter_plot, eval_infinite_server_queue, eval_multi_server_queue
from core.dataset import get_real_dataset, get_synthetic_dataset, visualize
from core.utils.arrival_epoch_simulator import arrival_epoch_simulator
from core.arrival_process import BatchCIR, BatchArrivalProcess
from core.utils.des.des import *
from scipy.stats import gamma, norm
from geomloss import SamplesLoss
from colored import attr, fg
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch
import seaborn as sns
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, datetime
import shutil
import os
import glob

from config import args_cir, args_bikeshare, args_bimodal, args_callcenter, args_uniform

# [x] Arrival epochs simulator
# given arrival count list, simulate arrival epochs


# [x] Sample CIR process

# [x] Run-through-queue
# given arrival epochs and service time settings, simulate waiting time

# [ ] speed up multi server queue by multi-threading or C++ reimplement

# [ ] evaluate mode: partially implemented, ecdf plot left unfinalized
# [ ] make the poisson layer into a pytorch layer
# [x] save model


today = date.today().strftime("%y_%m_%d")

# if os.path.isdir(result_dir) is False:
#     os.mkdir(result_dir)

# shutil.copyfile(__file__,
#                 result_dir + 'source.py')
# if os.path.exists(result_dir + 'core'):
#     shutil.rmtree(result_dir + 'core')
# shutil.copytree("core",
#                 result_dir + 'core')


def prepare_data(args, result_dir=''):
    dataset = args.dataset
    new_data = args.resample

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
    elif dataset in ['uniform', 'bimodal']:
        if new_data:
            intensity, train = get_synthetic_dataset(
                P, args.n_train, dataset, new_data, data_dir)
    else:
        # raise error with message 'dataset not supported'
        raise NotImplemented('Dataset not supported')
        # else:
        #     train = np.load(data_dir + 'train_{}.npy'.format(dataset))
        #     intensity = np.load(data_dir + 'intensity_{}.npy'.format(dataset))

    # if new_data:
    #     if dataset in ['bikeshare', 'callcenter']:
    #         train, test = get_real_dataset(dataset, data_dir)
    #     else:
    #         intensity, train = get_synthetic_dataset(
    #             P, N_TRAIN, dataset, new_data, data_dir)
    # else:
    #     train = np.load(data_dir + 'train_{}.npy'.format(dataset))
    #     intensity = np.load(data_dir + 'intensity_{}.npy'.format(dataset))

    # shutil.copyfile(data_dir + 'train_{}.npy'.format(dataset),
    #                 result_dir + 'train_{}.npy'.format(dataset))
    # if REAL_DATA:
    #     shutil.copyfile(data_dir + 'test_{}.npy'.format(dataset),
    #                     result_dir + 'test_{}.npy'.format(dataset))
    # else:
    #     shutil.copyfile(data_dir + 'intensity_{}.npy'.format(dataset),
    #                     result_dir + 'intensity_{}.npy'.format(dataset))

    # plot an overview of the dataset #
    # dir_name = os.path.join(
    #     result_dir, 'figures/{}_train_mean_scatter.pdf'.format(dataset))
    visualize(dataset, os.path.join(result_dir, 'figures'), train)

    return_val['train'] = torch.tensor(train, dtype=torch.float)
    return return_val


def train(exp_data, seed_dim, iters, save_freq, dim, args):
    # Build the generator network.
    training_set = exp_data['train']
    P = training_set.shape[1]
    n_layer = len(dim)

    class Generator(nn.Module):
        def __init__(self):
            super(Generator, self).__init__()
            self.normal = torch.distributions.normal.Normal(
                loc=0, scale=1, validate_args=None)

            nonlinear_layers = [
                nn.Linear(seed_dim, dim[0]), nn.LeakyReLU(0.1, True)]
            for i in range(n_layer-1):
                nonlinear_layers.append(nn.Linear(dim[i], dim[i+1]))
                nonlinear_layers.append(nn.LeakyReLU(0.1, True))
            nonlinear_layers.append(nn.Linear(dim[-1], P))

            self.nonlinear = nn.Sequential(*nonlinear_layers)
            self.linear = nn.Linear(seed_dim, P)

        def forward(self, noise):
            return self.nonlinear(noise)

    netG = Generator()

    # # Estimate PGnorta Gamma marginal distribution
    # norta_model = estimate_PGnorta(training_set.numpy(), zeta=9/16, max_T=1000,
    #                                M=100, img_dir_name=None, rho_mat_dir_name=None)
    norta_model = estimate_PGnorta(training_set.numpy(), zeta=9/16, max_T=20,
                                   M=100, img_dir_name=None, rho_mat_dir_name=None)
    norta_model.save_model(os.path.join(
        result_dir, 'models', 'PGnorta_model'))
    count_norta = norta_model.sample_count(n_sample=100000)
    intensity_norta = norta_model.sample_intensity(n_sample=100000)
    PGnorta_count_mean = np.mean(count_norta, axis=0)
    PGnorta_count_var = np.var(count_norta, axis=0)
    PGnorta_intensity_mean = np.mean(intensity_norta, axis=0)
    PGnorta_intensity_var = np.var(intensity_norta, axis=0)

    # B = torch.mean(training_set, dim=0)
    # var_X = torch.var(training_set, dim=0)
    # alpha = B ** 2 / (var_X - B)

    # if torch.min(alpha) < 0:
    #     print(
    #         'The arrival count of the {}-th time interval does not satisfy variance >= mean'.format(np.where(alpha < 0)[0]))

    # alpha[alpha < 0] = 10000  # alpha 越大，则生成的arrival count的mean和variance越接近

    # P = np.shape(training_set)[1]
    TRAIN_SIZE = np.shape(training_set)[0]

    # # get marginal mean & var of PGnorta
    # n_sample = 100000  # number of samples to visualize
    # lam = B.numpy() if torch.is_tensor(B) else B
    # PGnorta_count_mean = np.zeros_like(lam)
    # PGnorta_count_var = np.zeros_like(lam)
    # PGnorta_intensity_mean = np.zeros_like(lam)
    # PGnorta_intensity_var = np.zeros_like(lam)
    # for interval in progressbar.progressbar(range(P)):
    #     base_intensity_t = lam[interval]
    #     alpha_t = alpha[interval]
    #     intensity_PGnorta, count_PGnorta = sample_PGnorta_marginal(
    #         base_intensity_t, alpha_t, n_sample)

    #     PGnorta_count_mean[interval] = np.mean(count_PGnorta)
    #     PGnorta_count_var[interval] = np.var(count_PGnorta)
    #     PGnorta_intensity_mean[interval] = np.mean(intensity_PGnorta)
    # PGnorta_intensity_var[interval] = np.var(intensity_PGnorta)

    # We'll compute the sinkhorn distance between the real samples and fake samples.
    # Sinkhorn distance is an approximated wassestein distance.
    # For details, see https://www.kernel-operations.io/geomloss/
    # This is an alternative version of GAN.
    # On the current experiment instance, sinkhorn distance runs much more faster than the full GAN version and
    # thus is used for this demo.
    sinkorn_loss = SamplesLoss("sinkhorn", p=1, blur=0.05, scaling=0.5)
    # B = torch.tensor(B, dtype=torch.float)

    # may consider smaller learning rate
    gamma_G = (args.lr_final/args.lr_initial)**(1/iters)
    optimizerG = optim.Adam(
        netG.parameters(), lr=args.lr_initial, betas=(0.5, 0.9))
    optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(
        optimizerG, gamma=gamma_G, last_epoch=-1)

    G_cost_record = []
    lr_record = []

    for iteration in progressbar.progressbar(range(iters), redirect_stdout=True):
        noise = torch.randn(TRAIN_SIZE, seed_dim)
        intensity_pred = netG(noise)

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

        if iteration % args.distribution_eval_freq == 0:
            # Visualize the real and fake intensity every 100 training iterations.
            intensity_pred_np = intensity_pred.detach().numpy()
            print(
                'Generator loss in {}-th iteration: {}'.format(iteration, G_cost.item()))

            # if not REAL_DATA:
            #     compare_plot(real=intensity, fake=intensity_pred_np, msg='intensity'+str(iteration),
            #                  PGnorta_mean=PGnorta_intensity_mean, PGnorta_var=PGnorta_intensity_var, save=True)
            compare_plot(real=training_set, fake=torch.tensor(count_val), msg='count'+str(iteration),
                         PGnorta_mean=PGnorta_count_mean, PGnorta_var=PGnorta_count_var, save=True, result_dir=result_dir)

            # if not REAL_DATA:
            #     scatter_plot(interval_1_real=intensity[:, 0], interval_2_real=intensity[:, 1],
            #                  interval_1_fake=intensity_pred_np[:,
            #                                                    0], interval_2_fake=intensity_pred_np[:, 1],
            #                  msg='scatter_intensity_'+str(iteration), save=True)
            scatter_plot(interval_1_real=training_set[:, 0], interval_2_real=training_set[:, 1],
                         interval_1_fake=count_val[:,
                                                   0], interval_2_fake=count_val[:, 1],
                         msg='scatter_count_'+str(iteration), save=True, result_dir=result_dir)

            plt.figure()
            plt.semilogy(G_cost_record)
            plt.title('G_cost')
            plt.savefig(os.path.join(result_dir, 'G_cost.png'))
            plt.close()

            plt.figure()
            plt.plot(lr_record)
            plt.title('LR')
            plt.savefig(os.path.join(result_dir, 'LR_record.png'))
            plt.close()

            if args.eval_marginal:
                count_WGAN = count_val
                count_train = training_set.numpy()
                # plot marginal ecdf and compute wassestein distance
                evaluate_marginal(count_WGAN, count_norta,
                                  count_train, result_dir, iteration)
                evaluate_joint(count_WGAN, count_norta,
                               count_train, result_dir, iteration)

        if (args.eval_infinite_server_queue) and (iteration % args.eval_infinite_freq == 0):
            print('Evaluate infinite server queue performance.')
            eval_infinite_server_queue(count_WGAN, exp_data, os.path.join(
                result_dir, 'figures', 'mean_var_occupied_{}.png'.format(iteration)))

        if (args.eval_multi_server_queue) and (iteration % args.eval_multi_freq == 0):
            print('Evaluate multi server queue performance.')
            eval_multi_server_queue(count_WGAN, exp_data, os.path.join(
                result_dir, 'figures', 'mean_var_occupied_{}.png'.format(iteration)))

        if iteration % save_freq == 0 and iteration != 0:
            # if iteration == save_freq:
            # os.mkdir(os.path.join(result_dir, 'models'))
            torch.save(netG.state_dict(), os.path.join(
                result_dir, 'models', 'netG_{}.pth'.format(iteration)))
    return netG


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset', default='cir')

    args, rest_args = parser.parse_known_args()
    dataset = args.dataset
    if dataset == 'cir':
        args = args_cir.get_args(rest_args)
    elif dataset == 'uniform':
        args = args_uniform.get_args(rest_args)
    elif dataset == 'bimodal':
        args = args_bimodal.get_args(rest_args)
    elif dataset == 'bikeshare':
        args = args_bikeshare.get_args(rest_args)
    elif dataset == 'callcenter':
        args = args_callcenter.get_args(rest_args)

    # prepare result directory
    if not os.path.exists(os.path.join(args.result_dir, args.dataset)):
        os.mkdir(os.path.join(args.result_dir, args.dataset))
    strftime = date.today().strftime("%y_%m_%d_")+datetime.now().strftime("%H_%M_%S")
    result_dir = os.path.join(args.result_dir, args.dataset,
                              strftime)

    os.mkdir(result_dir)
    os.mkdir(os.path.join(result_dir, 'figures'))
    os.mkdir(os.path.join(result_dir, 'models'))

    exp_data = prepare_data(args, result_dir=result_dir)

    train(exp_data, args.seed_dim, args.iters,
          args.save_freq, args.network_dim, args)

    # if args.eval_model:
    #     evaluate(args)

    # args.dataset =
    # NEW_DATA = False

    # TRAIN = True
    # P = 16  # works only when TRAIN == True and DATASE == 'bikeshare' or 'callcenter'

    # lr_final = 1e-4
    # lr_initial = 0.001

    # N_TRAIN = 700
    # # RELEASE = True  # whether modify the CI to match with previous result

    # REAL_DATA = args.dataset in ['bikeshare', 'callcenter']

    # result_dir = '/Users/hector/Offline Documents/Doubly_Stochastic_WGAN/tmp_results/' + \
    #     today + '_' + \
    #     '{}__P_{}__seed_dim_{}/'.format(args.dataset, P, args.seed_dim)
    # if args.dataset == 'bikeshare':
    #     data_dir = 'Dataset/bikeshare_dswgan_real/'
    # elif args.dataset == 'callcenter':
    #     data_dir = 'Dataset/oakland_callcenter_dswgan/'
    # else:
    #     data_dir = 'Dataset/{}__P_{}/'.format(args.dataset, P)

    # DEBUG = True
    # if DEBUG:
    #     MAX_T = 10
    # else:
    #     MAX_T = 4000
