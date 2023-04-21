import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
from utils import cond_mkdir

def dict2cuda(a_dict):
   tmp = {}
   for key, value in a_dict.items():
       if isinstance(value,torch.Tensor):
           tmp.update({key: value.cuda()})
       else:
           tmp.update({key: value})
   return tmp

def dict2cpu(a_dict):
    tmp = {}
    for key, value in a_dict.items():
        if isinstance(value,torch.Tensor):
            tmp.update({key: value.cpu()})
        elif isinstance(value, dict):
            tmp.update({key: dict2cpu(value)})
        else:
            tmp.update({key: value})
    return tmp

def split_dict_with_tensors(model_input, gt, max_chunk_sz):
    # Chunking = we split tensors organizes in dictionnaries in smaller tensors
    # in dictionnaries of the same format.
    model_input_chunked = []
    for key in model_input:
        chunks = torch.split(model_input[key], max_chunk_sz, dim=0) # split along the batch dimension
        model_input_chunked.append(chunks)

    list_chunked_model_input = [{k:v for k,v in zip(model_input.keys(), curr_chunks)} \
                                    for curr_chunks in zip(*model_input_chunked)]

    gt_chunked = []
    for key in gt:
        chunks = torch.split(gt[key], max_chunk_sz, dim=0)  # split along the batch dimension
        gt_chunked.append(chunks)

    list_chunked_gt = [{k: v for k, v in zip(gt.keys(), curr_chunks)} \
                                for curr_chunks in zip(*gt_chunked)]

    return list_chunked_model_input, list_chunked_gt

def init_torch_dict(d):
    return {k: [v] for k, v in d.items()}

def update_torch_dicts(d1 , d2):
    if d1 is None:
        return init_torch_dict(d2)
    else:
        for k, v in d2.items():
            d1[k].append(v)
        return d1

def merge_torch_dict(d):
    return {k: torch.cat(d[k]) if d[k][0] is not None else None for k, _ in d.items()}

def evaluate_model(model, dataloader, pbar=None, cpu_output=False):
    model.eval()
    val_gt = val_input = val_output = None
    with torch.no_grad():
        for val_step, (chunk_val_input, chunk_val_gt) in enumerate(dataloader):
            # Evaluate the trained model
            chunk_val_output = model(dict2cuda(chunk_val_input))
            if cpu_output:
                chunk_val_output = dict2cpu(chunk_val_output)
            val_gt = update_torch_dicts(val_gt, chunk_val_gt)
            val_input = update_torch_dicts(val_input, chunk_val_input)
            val_output = update_torch_dicts(val_output, chunk_val_output)
            if pbar:
                pbar.update(1)
    model.train()
    return merge_torch_dict(val_gt), merge_torch_dict(val_input), merge_torch_dict(val_output)

def change_phase(model, phase_name, lr):
    for param_name, param in model.named_parameters():
        param.requires_grad = False
        if phase_name in param_name or phase_name == "all":
            param.requires_grad = True
    optim = torch.optim.Adam(lr=lr, params=model.parameters(), amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.9)
    return optim, scheduler

def train(model, phases,
          train_dataloader, val_dataloader,
          sampler, epochs, lr,
          steps_til_summary,
          steps_til_val_summary,
          epochs_til_checkpoint,
          root_dir,
          model_dir,
          loss_fn, summary_fn,
          prefix_model_dir='',
          double_precision=False,
          clip_grad=True,
          loss_schedules=None,
          chunk_lists_from_batch_fn=split_dict_with_tensors,
          max_chunk_sz = 32):

    if os.path.exists(model_dir):
        pass
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        #    shutil.rmtree(model_dir)
        # shutil.rmtree(model_dir)
    else:
        os.makedirs(model_dir)

    model_dir_postfixed = os.path.join(model_dir, prefix_model_dir)

    summaries_dir = os.path.join(model_dir_postfixed, 'summaries')
    cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir_postfixed, 'checkpoints')
    cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    phase_index = 0
    optim, scheduler = change_phase(model, phases[phase_index]['name'], lr * phases[phase_index]['lr_mult'])

    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

                if (total_steps+1) % phases[phase_index]['end'] == 0 and\
                        phase_index < len(phases)-1 and phases[phase_index]['end'] > 0:
                    phase_index += 1
                    optim, scheduler = change_phase(model, phases[phase_index]['name'],
                                                   lr * phases[phase_index]['lr_mult'])

                ''' Restart optimizer '''
                optim.zero_grad()

                ''' Proceed to chunking '''
                list_chunked_model_input, list_chunked_gt = \
                    chunk_lists_from_batch_fn(model_input, gt, max_chunk_sz)

                ''' Accumulate gradient over all the chunks '''
                num_chunks = len(list_chunked_gt)
                batch_avgd_losses = {}
                batch_avgd_tot_loss = 0.
                for chunk_idx, (chunked_model_input, chunked_gt) \
                    in enumerate(zip(list_chunked_model_input, list_chunked_gt)):

                    chunked_model_input = dict2cuda(chunked_model_input)
                    chunked_gt = dict2cuda(chunked_gt)

                    chunked_model_output = model(chunked_model_input)
                    losses = loss_fn(chunked_model_output, chunked_gt)

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            single_loss *= loss_schedules[loss_name](epoch,total_steps)

                        train_loss += single_loss / num_chunks
                        batch_avgd_tot_loss += float(single_loss / num_chunks)
                        if loss_name in batch_avgd_losses:
                            batch_avgd_losses[loss_name] += single_loss / num_chunks
                        else:
                            batch_avgd_losses.update({loss_name: single_loss / num_chunks})

                    train_loss.backward()

                ''' Write losses avgd in tensorboard '''
                for loss_name, loss in batch_avgd_losses.items():
                    writer.add_scalar(loss_name, loss, total_steps)
                    if loss_schedules is not None and loss_name in loss_schedules:
                        writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](epoch,total_steps), total_steps)
                train_losses.append(batch_avgd_tot_loss)
                writer.add_scalar("total_train_loss", batch_avgd_tot_loss, total_steps)

                ''' Clip gradients '''
                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

                ''' Optimize '''
                optim.step()

                ''' Perform summary'''
                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))
                    summary_fn(model, chunked_gt, chunked_model_output,
                               writer, total_steps, root_dir, write_mrc=True)
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (
                        epoch, train_loss, time.time() - start_time))

                ''' Perform validation summary'''
                if (not total_steps % steps_til_val_summary) and total_steps > 0 and val_dataloader:
                    val_start_time = time.time()
                    val_gt, _, val_output = evaluate_model(model, val_dataloader, None, True)
                    val_gt = dict2cuda(val_gt)
                    val_output = dict2cuda(val_output)
                    val_losses = loss_fn(val_output, val_gt)
                    val_loss = 0.
                    for loss_name, loss in val_losses.items():
                        single_loss = loss.mean()
                        if loss_schedules is not None and loss_name in loss_schedules:
                            loss_weight = loss_schedules[loss_name](epoch, total_steps)
                            single_loss *= loss_weight
                            writer.add_scalar("val: " + loss_name + "_weight", loss_weight, total_steps)
                        val_loss += single_loss
                        writer.add_scalar("val: " + loss_name, single_loss, total_steps)
                    writer.add_scalar("val: total_loss", val_loss, total_steps)
                    summary_fn(model, val_gt, val_output, writer, total_steps, root_dir, summary_prefix='val')
                    tqdm.write("Epoch %d, validation time %0.6f" % (epoch, time.time() - val_start_time))

                pbar.update(1)
                total_steps += 1

        scheduler.step()
        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)
