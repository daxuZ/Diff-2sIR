import argparse
import os
import warnings
import torch
import torch.multiprocessing as mp

from core.logger import VisualWriter, InfoLogger
import core.praser as Praser
import core.util as Util
from data import define_dataloader
from models import create_model, define_network, define_loss, define_metric


def main_worker(gpu, ngpus_per_node, opt):
    """  threads running on each GPU """
    if 'local_rank' not in opt:
        opt['local_rank'] = opt['global_rank'] = gpu
    if opt['distributed']:
        torch.cuda.set_device(int(opt['local_rank']))
        print('using GPU {} for training'.format(int(opt['local_rank'])))
        torch.distributed.init_process_group(backend='nccl',
            init_method=opt['init_method'],
            world_size=opt['world_size'],
            rank=opt['global_rank'],
            group_name='mtorch'
        )
    '''set seed and cuDNN environment '''
    torch.backends.cudnn.enabled = True
    warnings.warn('You have chosen to use cudnn for accleration. torch.backends.cudnn.enabled=True')
    Util.set_seed(opt['seed'])

    ''' set logger '''
    phase_logger = InfoLogger(opt)
    phase_writer = VisualWriter(opt, phase_logger)  
    phase_logger.info('Create the log file in directory {}.\n'.format(opt['path']['experiments_root']))

    '''set networks and dataset'''
    phase_loader, val_loader, debug_loader = define_dataloader(phase_logger, opt)  # val_loader is None if phase is test.

    if opt["use_v"]:
        networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['v_networks']]
        if opt["phase"] == "test":
            networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
            networks_v = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['v_networks']]
        else:
            networks_v = None
    else:
        networks = [define_network(phase_logger, opt, item_opt) for item_opt in opt['model']['which_networks']]
        networks_v = None

    ''' set metrics, loss, optimizer and  schedulers '''
    metrics = [define_metric(phase_logger, item_opt) for item_opt in opt['model']['which_metrics']]
    losses = [define_loss(phase_logger, item_opt) for item_opt in opt['model']['which_losses']]

    model = create_model(
        opt=opt,
        networks=networks,
        v_networks=networks_v,
        phase_loader=phase_loader,
        val_loader=val_loader,
        debug_loader=debug_loader,
        losses=losses,
        metrics=metrics,
        logger=phase_logger,
        writer=phase_writer
    )

    phase_logger.info('Begin model {}.'.format(opt['phase']))
    model.print_network(networks[0])
    try:
        if opt['phase'] == 'train':
            model.train()
        else:
            model.test()
    finally:
        phase_writer.close()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str,
                        default=r'E:\Pycharm\Palette_Windows\config\inpainting_celebahq.json',
                        help='JSON file for configuration')  # config/inpainting_placesval.json  config/inpainting_celebahq.json
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'],
                        help='Run train or test', default='test')
    parser.add_argument('-b', '--batch', type=int, default=None, help='Batch size in every gpu')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('-P', '--port', default='21012', type=str)
    parser.add_argument('-m', '--mask_path', type=str,
                        default=r'E:\Pycharm\Palette_Windows\datasets\celebahq_masks_256_small',
                        help='Mask path for evaluation')

    ''' parser configs '''
    args = parser.parse_args()
    opt = Praser.parse(args)
    
    ''' cuda devices '''
    gpu_str = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print('export CUDA_VISIBLE_DEVICES={}'.format(gpu_str))

    ''' use DistributedDataParallel(DDP) and multiprocessing for multi-gpu training'''
    # [Todo]: multi GPU on multi machine
    if opt['distributed']:
        ngpus_per_node = len(opt['gpu_ids']) # or torch.cuda.device_count()
        opt['world_size'] = ngpus_per_node
        opt['init_method'] = 'tcp://127.0.0.1:'+ args.port 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))
    else:
        opt['world_size'] = 1 
        main_worker(0, 1, opt)
