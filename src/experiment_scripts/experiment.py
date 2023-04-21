import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torch.utils.data import DataLoader

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from cryonet import CryoNet
from training_chunks import train
from loss_functions import complex_proj_l2_loss, complex_proj_cc_loss
from summary_functions import write_summary


class DistDataParallelWrapper(DistributedDataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


def experiment(config, train_dataset, val_dataset):
    sampler = DistributedSampler(train_dataset, shuffle=True) if dist.is_initialized() else None
    train_dataloader = DataLoader(train_dataset, drop_last=True,
                                  shuffle=(sampler is None),
                                  sampler=sampler, batch_size=config.train_batch_sz,
                                  pin_memory=True, num_workers=config.train_num_workers)

    if val_dataset:
        val_dataloader = DataLoader(val_dataset,
                                    shuffle=False, batch_size=config.val_chunk_sz,
                                    pin_memory=True, num_workers=config.train_num_workers)
    else:
        val_dataloader = None

    cryonet = CryoNet(config)
    torch.cuda.set_device(config.local_rank)
    cryonet.to(torch.cuda.current_device())
    if config.world_size > 1:
        cryonet = DistDataParallelWrapper(cryonet, find_unused_parameters=True,
                                          device_ids=[config.local_rank],
                                          output_device=[config.local_rank])
    loss_fn = complex_proj_l2_loss if config.data_loss == "L2" else complex_proj_cc_loss

    # Launch the training
    loss_schedule = {'data_term': lambda e, t: 1,
                     'kld_term': lambda e, t: float(config.encoder == "VAE") / config.encoder_conv_layers[-1],
                     'nma_reg_term': lambda e, t: config.nma_reg_weight}

    train(model=cryonet, phases=config.train_phases,
          train_dataloader=train_dataloader, val_dataloader=val_dataloader,
          sampler=sampler, epochs=config.train_epochs, lr=config.train_learning_rate,
          steps_til_summary=config.train_steps_til_summary,
          steps_til_val_summary=config.val_steps_til_summary,
          epochs_til_checkpoint=config.train_epochs_til_checkpoint,
          root_dir=config.root_dir,
          model_dir=config.model_dir,
          loss_fn=loss_fn,
          loss_schedules=loss_schedule,
          summary_fn=write_summary,
          max_chunk_sz=config.train_chunk_sz)
