import os
import socket
import time
from tqdm import tqdm

import torch
from torch.utils.tensorboard.writer import SummaryWriter
# import lightning.pytorch as pl
import numpy as np

from muvo.config import get_parser, get_cfg
from muvo.data.dataset import DataModule
from muvo.trainer import WorldModelTrainer
from lightning.pytorch.callbacks import ModelSummary

from clearml import Task, Dataset, Model


def main():
    args = get_parser().parse_args()
    cfg = get_cfg(args)

    task = Task.init(project_name=cfg.CML_PROJECT, task_name=cfg.CML_TASK, task_type=cfg.CML_TYPE, tags=cfg.TAG)
    task.connect(cfg)
    cml_logger = task.get_logger()
    #
    # dataset_root = Dataset.get(dataset_project=cfg.CML_PROJECT,
    #                            dataset_name=cfg.CML_DATASET,
    #                            ).get_local_copy()

    # data = DataModule(cfg, dataset_root=dataset_root)
    data = DataModule(cfg)
    data.setup()

    input_model = Model(model_id='').get_local_copy() if cfg.PRETRAINED.CML_MODEL else None
    model = WorldModelTrainer(cfg.convert_to_dict(), pretrained_path=input_model)
    # model.get_cml_logger(cml_logger)

    save_dir = os.path.join(
        cfg.LOG_DIR, time.strftime('%d%B%Yat%H:%M:%S%Z') + '_' + socket.gethostname() + '_' + cfg.TAG
    )
    # writer = SummaryWriter(log_dir=save_dir)

    dataloader = data.test_dataloader()[2]

    pbar = tqdm(total=len(dataloader),  desc='Prediction')
    model.cuda()

    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()

    # n_prediction_samples = model.cfg.PREDICTION.N_SAMPLES
    upload_data = {
        'rgb_label': list(), 
        'throttle_brake': list(), 
        'steering': list(), 
        'pcd_label': list(), 
        'voxel_label': list(), 
        'rgb_re': list(), 
        'pcd_re': list(), 
        'voxel_re': list(), 
        'rgb_im': list(), 
        'pcd_im': list(), 
        'voxel_im': list(),
        }
    
    for i, batch in enumerate(dataloader):
        batch = {key: value.cuda() for key, value in batch.items()}
        with torch.no_grad():
            batch = model.preprocess(batch)
            output, output_imagine = model.model.sim_forward(batch, is_dreaming=False)
            
            voxel_label = torch.where(batch['voxel_label_1'].squeeze()[0].cpu() != 0)
            voxel_label = torch.stack(voxel_label).transpose(0, 1).numpy()
            
            voxel_re = torch.where(torch.argmax(output['voxel_1'][0][0].detach(), dim=-4).cpu() != 0)
            voxel_re = torch.stack(voxel_re).transpose(0, 1).numpy()
            
            voxel_im = torch.where(torch.argmax(output_imagine['voxel_1'][0][(0, 3, 9), ...].detach(), dim=-4).cpu() != 0)
            voxel_im = torch.stack(voxel_im).transpose(0, 1).numpy()
            
            upload_data['rgb_label'].append((batch['rgb_label_1'][0][0].cpu().numpy() * 255).astype(np.uint8))
            upload_data['throttle_brake'].append(batch['throttle_brake'][0][0].cpu().numpy())
            upload_data['steering'].append(batch['steering'][0][0].cpu().numpy())
            upload_data['pcd_label'].append(batch['range_view_label_1'][0][0].cpu().numpy())
            upload_data['voxel_label'].append(voxel_label)
            upload_data['rgb_re'].append((output['rgb_1'][0][0].detach().cpu().numpy() * 255).astype(np.uint8))
            upload_data['pcd_re'].append(output['lidar_reconstruction_1'][0][0].detach().cpu().numpy())
            upload_data['voxel_re'].append(voxel_re)
            upload_data['rgb_im'].append((output_imagine['rgb_1'][0][(0, 3, 9), ...].detach().cpu().numpy() * 255).astype(np.uint8))
            upload_data['pcd_im'].append(output_imagine['lidar_reconstruction_1'][0][(0, 3, 9), ...].detach().cpu().numpy())
            upload_data['voxel_im'].append(voxel_im)

            if i % 500 == 0 and i != 0:
                print(f'Uploading data {i}')
                task.upload_artifact(f'data_{i}', np.array(upload_data))
                upload_data = {
                    'rgb_label': list(),
                    'throttle_brake': list(),
                    'steering': list(),
                    'pcd_label': list(),
                    'voxel_label': list(),
                    'rgb_re': list(),
                    'pcd_re': list(),
                    'voxel_re': list(),
                    'rgb_im': list(),
                    'pcd_im': list(),
                    'voxel_im': list(),
                }

        pbar.update(1)

    if i % 500 != 0:
        task.upload_artifact(f'data_{i}', np.array(upload_data))


if __name__ == '__main__':
    main()
