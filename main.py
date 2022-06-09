import argparse
from datetime import date, datetime
import shutil
import os
import glob
from train import train
from colored import fg, attr
from core.dataset import prepare_data

from config import args_cir, args_bikeshare, args_bimodal, args_callcenter, args_uniform, args_pgnorta

today = date.today().strftime("%y_%m_%d")

# if os.path.isdir(result_dir) is False:
#     os.mkdir(result_dir)

# shutil.copyfile(__file__,
#                 result_dir + 'source.py')
# if os.path.exists(result_dir + 'core'):
#     shutil.rmtree(result_dir + 'core')
# shutil.copytree("core",
#                 result_dir + 'core')


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
    elif dataset == 'pgnorta':
        args = args_pgnorta.get_args(rest_args)
    else:
        raise NotImplementedError('Dataset not supported')

    # prepare result directory
    args.result_dir = '/home/lizo/DS-WGAN_tmp/'
    if not os.path.exists(os.path.join(args.result_dir, args.dataset)):
        os.mkdir(os.path.join(args.result_dir, args.dataset))
    strftime = date.today().strftime("%y_%m_%d_")+datetime.now().strftime("%H_%M_%S")
    result_dir = os.path.join(args.result_dir, args.dataset,
                              strftime)

    os.mkdir(result_dir)
    print(fg('blue') + 'Result directory: {}'.format(result_dir) + attr('reset'))
    os.mkdir(os.path.join(result_dir, 'figures'))
    os.mkdir(os.path.join(result_dir, 'models'))

    exp_data = prepare_data(args, result_dir=result_dir)

    train(exp_data, args.seed_dim, args.iters,
          args.save_freq, args.network_dim, result_dir, args)
