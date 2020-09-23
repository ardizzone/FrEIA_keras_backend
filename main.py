import argparse
import configparser
import sys
import os
import warnings

usage = "main.py [--no-hpdlf] <train|test|both> <target directory>"

argparser = argparse.ArgumentParser(usage=usage)
argparser.add_argument('--no-hpdlf', action='store_true', dest='no_hpdlf',
                       help='Only run locally, do not use tarantella')
argparser.add_argument('mode', type=str,
                       help='can be "train", "test", or "both" (currently only training implemented')
argparser.add_argument('directory', type=str, metavar='target directory',
                       help='Directory with a conf.ini file. Outputs will be saved there.')

cli_args = argparser.parse_args()
mode = cli_args.mode
output_dir = cli_args.directory
use_tarantella = (not cli_args.no_hpdlf)

conf_file = os.path.join(output_dir, 'conf.ini')

assert mode in ['train', 'test', 'both'], 'Usage: ' + usage
assert os.path.isdir(output_dir), 'Usage: ' + usage + f'\nNo such directory: "{output_dir}"'

# initialize the options with the defaults, overwrite the ones specified.
args = configparser.ConfigParser()
args.read(os.path.join(os.getcwd(), 'default.ini'))
#args.read('/home/diz/code/freia_keras/default.ini')

if os.path.isfile(conf_file):
    args.read(conf_file)
else:
    warnings.warn(f'No config file found under "{conf_file}", using default')

args['checkpoints']['output_dir'] = output_dir
args['training']['use_tarantella'] = str(use_tarantella)

try:
    args['data']['data_root_folder'] = os.environ['DATASET_DIR']
except KeyError:
    raise ValueError("Please set the DATASET_DIR environment variable")

# to ensure reproducibility in case the defaults changed,
# save the entire set of current options too
conf_full = os.path.join(output_dir, 'config_full.ini')
with open(conf_full, 'w') as f:
    args.write(f)
args.write(sys.stdout)

if mode in ['test', 'both']:
    raise NotImplementedError("TODO: evaluation")

if mode in ['train', 'both']:
    import train
    train.train(args)
