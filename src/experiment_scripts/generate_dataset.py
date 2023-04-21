import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import configargparse

from utils import cond_mkdir
from datasets import train_val_datasets
from main import init_config
from starfile_utils import create_starfile
from torch.utils.data import DataLoader


def main():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True,
                        help='Path to config file.')
    parser.add_argument('--sim_starfile_dir', required=True, help='Root path for saving the dataset.')
    init_config(parser)  # use the default arguments from main.init_config to stay synchronized
    config = parser.parse_args()
    config.map_shape = [config.side_len] * 3

    if config.experiment_type == 'exp_relion_reconstruct':
        return -1, 'Error: experiment_type exp_relion_reconstruct not compatible with generate dataset'

    # Create root directory where models, logs and config files will be written
    config.root_dir = os.path.join(config.log_dir, config.experiment_name)
    if not cond_mkdir(config.root_dir):
        return -1, "Error: cannot create root path."

    print(f"Generating dataset for {config.experiment_type}")
    train_dataset, _ = train_val_datasets(config)
    dataloader = DataLoader(train_dataset, shuffle=False,
                            batch_size=config.train_chunk_sz,
                            pin_memory=False, num_workers=0)

    root_dir = os.path.join(config.sim_starfile_dir, config.experiment_name)
    if not cond_mkdir(root_dir):
        return -1, "Error: cannot create root path."

    relative_mrcs_path_prefix = 'Particles/'
    mrcs_dir = os.path.join(root_dir, relative_mrcs_path_prefix)
    if not cond_mkdir(mrcs_dir):
        return -1, "Error: cannot create mrcs path."

    print("Creating starfile ...")
    create_starfile(dataloader,
                    config,
                    root_dir,
                    relative_mrcs_path_prefix,
                    config.experiment_name)

    return 0, 'Generate dataset successful.'


if __name__ == '__main__':
    try:
        retval, status_message = main()
    except Exception as e:
        print(traceback.format_exc(), file=sys.stderr)
        retval = 1
        status_message = 'Error: Generate dataset failed.'

    print(status_message)
    exit(retval)
