from core.pgnorta import estimate_PGnorta
from core.evaluate import evaluate_joint, evaluate_marginal, compare_plot, scatter_plot, eval_infinite_server_queue, eval_multi_server_queue
from core.des.des_py.des import *
from core.utils.to_device import to_numpy
import torch.optim as optim
from geomloss import SamplesLoss
import os
from models.poisson_simulator import DSSimulator
import platform


def train(exp_data, seed_dim, iters, save_freq, dim, result_dir, args):

    device = get_device(args)
    training_set = exp_data['train'].to(device)
    P = training_set.shape[1]
    simulator = DSSimulator(seed_dim=seed_dim, dim=dim,
                            n_interval=P).to(device)

    # # Estimate PGnorta Gamma marginal distribution
    # norta_model = estimate_PGnorta(training_set.numpy(), zeta=9/16, max_T=1000,
    #                                M=100, img_dir_name=None, rho_mat_dir_name=None)
    if args.verbose:
        print('Estimating PGnorta model...')
    norta_model = estimate_PGnorta(training_set.cpu().numpy(), zeta=9/16, max_T=20,
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
        simulator.parameters(), lr=args.lr_initial, betas=(0.5, 0.9))
    optimizerG_lrdecay = torch.optim.lr_scheduler.ExponentialLR(
        optimizerG, gamma=gamma_G, last_epoch=-1)

    G_cost_record = []
    lr_record = []

    for iteration in progressbar.progressbar(range(iters), redirect_stdout=True):
        noise = torch.randn(TRAIN_SIZE, seed_dim, device=device)
        count_WGAN, pred_intensity = simulator(noise, return_intensity=True)
        G_cost = sinkorn_loss(count_WGAN, training_set)
        simulator.zero_grad()
        G_cost.backward()
        G_cost_record.append(G_cost.detach().cpu().numpy())
        optimizerG.step()
        optimizerG_lrdecay.step()

        lr_record.append(optimizerG_lrdecay.get_last_lr())

        if iteration % args.distribution_eval_freq == 0:
            pred_intensity_np = pred_intensity.detach().cpu().numpy()
            print(
                'DS-Simulator loss in {}-th iteration: {}'.format(iteration, G_cost.item()))

            # if not REAL_DATA:
            #     compare_plot(real=intensity, fake=pred_intensity_np, msg='intensity'+str(iteration),
            #                  PGnorta_mean=PGnorta_intensity_mean, PGnorta_var=PGnorta_intensity_var, save=True)
            # if args.verbose:
            #     print('Making comparison plot...')
            compare_plot(real=training_set, fake=count_WGAN, msg='count'+str(iteration),
                         PGnorta_mean=PGnorta_count_mean, PGnorta_var=PGnorta_count_var, save=True, result_dir=result_dir)

            # if not REAL_DATA:
            #     scatter_plot(interval_1_real=intensity[:, 0], interval_2_real=intensity[:, 1],
            #                  interval_1_fake=pred_intensity_np[:,
            #                                                    0], interval_2_fake=pred_intensity_np[:, 1],
            #                  msg='scatter_intensity_'+str(iteration), save=True)
            scatter_plot(interval_1_real=to_numpy(training_set[:, 0]), interval_2_real=to_numpy(training_set[:, 1]),
                         interval_1_fake=to_numpy(count_WGAN[:,
                                                             0]), interval_2_fake=to_numpy(count_WGAN[:, 1]),
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
                count_train = to_numpy(training_set)
                # plot marginal ecdf and compute wassestein distance
                evaluate_marginal(to_numpy(count_WGAN), count_norta,
                                  count_train, result_dir, iteration)
                evaluate_joint(to_numpy(count_WGAN), count_norta,
                               count_train, result_dir, iteration)

        if (args.eval_infinite_server_queue) and (iteration % args.eval_infinite_freq == 0):
            print('Evaluate infinite server queue performance.')
            eval_infinite_server_queue(to_numpy(count_WGAN), exp_data, os.path.join(
                result_dir, 'figures', 'mean_var_occupied_{}.png'.format(iteration)))

        if (args.eval_multi_server_queue) and (iteration % args.eval_multi_freq == 0):
            print('Evaluate multi server queue performance.')
            eval_multi_server_queue(to_numpy(count_WGAN), exp_data, os.path.join(
                result_dir, 'figures'), iteration, backend=args.des_backend)

        if iteration % save_freq == 0 and iteration != 0:
            # if iteration == save_freq:
            # os.mkdir(os.path.join(result_dir, 'models'))
            torch.save(simulator.state_dict(), os.path.join(
                result_dir, 'models', 'simulator_{}.pth'.format(iteration)))

        # if args.verbose:
        #     print('Keep training...')
    return simulator


def get_device(args):
    OS = platform.system()
    if not args.enable_gpu:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            print('No GPU available. Use CPU instead.')
            device = torch.device('cpu')
        # if OS == 'Windows' or OS == 'Linux':
        #     if torch.cuda.is_available():
        #         device = torch.device('cuda')
        #     else:
        #         print('No GPU available. Use CPU instead.')
        #         device = torch.device('cpu')
        # if OS == 'Darwin':
        #     if not torch.backends.mps.is_available():
        #         if not torch.backends.mps.is_built():
        #             print("MPS not available because the current PyTorch install was not "
        #                 "built with MPS enabled. Use CPU instead.")
        #         else:
        #             print("MPS not available because the current MacOS version is not 12.3+ "
        #                 "and/or you do not have an MPS-enabled device on this machine. Use CPU instead.")
        #     else:
        #         device = torch.device("mps")

    print('Using device: {}'.format(device))
    return device
