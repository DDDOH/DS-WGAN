import argparse
from pickle import FALSE


def get_args(rest_args):
    parser = argparse.ArgumentParser()

    # dataset and result directory settings
    # 'uniform' or 'bimodal' or 'bikeshare' or 'callcenter' or 'cir'
    parser.add_argument('--dataset', type=str, default='cir')
    # if --dataset is 'self-defined', then --data_dir should be specified
    # parser.add_argument('--data_dir', type=str, default='')
    # if --dataset is 'uniform' or 'bimodal',
    # --resample set True means resample the data,
    # --resample set False means use the previously stored data
    parser.add_argument('--resample', type=bool, default=False)
    # if --dataset is 'uniform' or 'bimodal' or 'cir', how many data in training set
    parser.add_argument('--n_train', type=int, default=700)
    # if --dataset is cir, how many cir is sampled to get the true performance measure
    parser.add_argument('--n_test', type=int, default=500)
    parser.add_argument('--result_dir', type=str,
                        default='/Users/shuffleofficial/Offline_Documents/Doubly_Stochastic_WGAN/tmp_results')

    # network settings
    # use list in parser: https://stackoverflow.com/a/15753721
    parser.add_argument('--network_dim', nargs='+',
                        help='<Required> Set flag', default=[512, 512, 512, 512, 256])

    # training hyperparameters
    parser.add_argument('--seed_dim', type=int, default=4)
    parser.add_argument('--iters', type=int, default=30000)
    parser.add_argument('--save_freq', type=int, default=1000)
    parser.add_argument('--lr_initial', type=float, default=0.001)
    parser.add_argument('--lr_final', type=float, default=1e-4)

    # evaluate settings
    # evaluate means comparing marginal distribution (ecdf and wasserstein distance)
    parser.add_argument('--eval_marginal', type=bool, default=False)
    # parser.add_argument('--eval_run_through_queue', type=bool, default=True)
    parser.add_argument('--eval_multi_server_queue', type=bool, default=True)
    parser.add_argument('--eval_infinite_server_queue',
                        type=bool, default=True)
    parser.add_argument('--eval_multi_freq', type=bool, default=1000)
    parser.add_argument('--eval_infinite_freq', type=bool, default=500)
    parser.add_argument('--distribution_eval_freq', type=int, default=500)
    # 'cpp' or 'python' or 'both'
    parser.add_argument('--des_backend', type=str, default='cpp')
    parser.add_argument('--verbose', type=bool, default=True)

    # others
    parser.add_argument('--enable_gpu', type=bool, default=True)

    return parser.parse_args(rest_args)
