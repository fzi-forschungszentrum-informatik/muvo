import numpy as np
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import shutil
import scipy.sparse as sp
import re
from tqdm import tqdm
from clearml import Task
import logging
from multiprocessing import Pool, RLock, Pipe
from threading import Thread

from data_preprocessing import *

log = logging.getLogger(__name__)


def progress_bar_total(parent, total_len, desc):
    desc = desc if desc else "Main"
    pbar_main = tqdm(total=total_len, desc=desc, position=0)
    nums = 0
    while True:
        msg = parent.recv()[0]
        if msg is not None:
            pbar_main.update()
            nums += 1
        if nums == total_len:
            break
    pbar_main.close()


def voxelize_dir(data_path, cfg, task_idx, all_task, pipe):
    log.info(f'Converting Depth Image to Voxels in {data_path}.')
    save_path = data_path.parent
    # if not save_path.exists():
    #     save_path.mkdir()
    file_list = sorted([str(f) for f in data_path.glob('*.png')])
    data_dict = {}
    voxels_dict = {}
    for file in tqdm(file_list, desc=f'{task_idx + 1:04} / {all_task:04}', position=task_idx % cfg.n_process + 1):
        depth, semantic, _ = read_img(file)
        points_list, sem_list = get_all_points(
            depth, semantic, fov=cfg.fov, size=cfg.size, offset=cfg.offset, mask_ego=cfg.mask_ego)
        voxel_points, semantics = voxel_filter(points_list, sem_list, cfg.voxel_size, cfg.center)
        data = np.concatenate([voxel_points, semantics[:, None]], axis=1)
        voxels = np.zeros(shape=(2 * np.asarray(cfg.center) / cfg.voxel_size).astype(int), dtype=np.uint8)
        voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
        name = re.match(r'.*/.*_(\d{9})\.png', file).group(1)
        # np.savez_compressed(f"{save_path}/{name}.npz", data=data)
        coo_voxels = sp.coo_matrix(voxels.reshape(voxels.shape[0], -1))
        # np.savez_compressed(f"{save_path}/v{name}.npz", data=coo_voxels)
        voxels_dict[name] = coo_voxels
        data_dict[name] = data

        if pipe is not None:
            pipe.send(['x'])

    np.savez_compressed(f"{save_path}/voxels.npz", coo_voxels=voxels_dict, voxel_points=data_dict)
    log.info(f"Saved Voxels Data in {save_path}/voxels.npz.")


def voxelize_one(depth_file, lidar_file, cfg, save_name, pipe=None):
    pcd, sem = merge_pcd(depth_file, lidar_file, cfg.camera_position, cfg.lidar_position, cfg.fov)
    offset_x = cfg.bev_offset_forward * cfg.bev_resolution
    offset_z = cfg.offset_z * cfg.voxel_resolution
    voxel_points, semantics = voxel_filter(pcd, sem, cfg.voxel_resolution, cfg.voxel_size, [offset_x, 0, offset_z])
    data = np.concatenate([voxel_points, semantics[:, None]], axis=1)
    # voxels = np.zeros(shape=cfg.voxel_size, dtype=np.uint8)
    # voxels[voxel_points[:, 0], voxel_points[:, 1], voxel_points[:, 2]] = semantics
    # csr_voxels = sp.csr_matrix(voxels.reshape(voxels.shape[0], -1))
    np.save(f'{save_name}', data)
    # np.save(f'{save_path}/voxel_coo/voxel_coo_{name}.npy', csr_voxels)

    if pipe is not None:
        pipe.send(['x'])


@hydra.main(config_path='./', config_name='data_preprocess')
def main_(cfg: DictConfig):
    task = Task.init(project_name=cfg.cml_project, task_name=cfg.cml_task_name, task_type=cfg.cml_task_type,
                     tags=cfg.cml_tags)
    task.connect(cfg)
    cml_logger = task.get_logger()

    root_path = Path(cfg.root)
    data_paths = sorted([p for p in root_path.glob('**/depth_semantic') if p.is_dir()])
    n_files = len([f for f in root_path.glob('**/depth_semantic/*.png')])

    if not root_path.exists() or n_files == 0:
        print('Root Path does not EXSIT or there are NO LEGAL files!!!')
        return

    log.info(f'{n_files} will be proceed.')

    parent, child = Pipe()
    main_thread = Thread(target=progress_bar_total, args=(parent, n_files, "Total"))
    main_thread.start()
    p = Pool(cfg.n_process, initializer=tqdm.set_lock, initargs=(RLock(),))
    for i, path in enumerate(data_paths):
        p.apply_async(func=voxelize_dir, args=(path, cfg, i, len(data_paths), child))
    p.close()
    p.join()
    main_thread.join()

    log.info("Finished Voxelization!")


@hydra.main(config_path='./', config_name='data_preprocess')
def main(cfg: DictConfig):
    # task = Task.init(project_name=cfg.cml_project, task_name=cfg.cml_task_name, task_type=cfg.cml_task_type,
    #                  tags=cfg.cml_tags)
    # task.connect(cfg)
    # cml_logger = task.get_logger()
    root_path = Path(cfg.root)
    data_paths = sorted([p for p in root_path.glob('**/Town*/*/') if p.is_dir()])

    if not root_path.exists() or len(data_paths) == 0:
        print('Root Path does not EXIST or there are NO LEGAL files!!!')
        return

    log.info(f'{len(data_paths)} runs will be voxelized.')
    log.info(f'{data_paths}')

    for i, path in enumerate(data_paths):
        pd_file = f'{path}/pd_dataframe.pkl'
        pd_dataframe = pd.read_pickle(pd_file)
        data_len = len(pd_dataframe)

        # parent, child = Pipe()
        # main_thread = Thread(target=progress_bar_total, args=(parent, data_len, f'{i+1}/{len(data_paths)}'))
        # main_thread.start()
        p = Pool(cfg.n_process)

        log.info(f'start voxelizing in dir {path}.')

        save_path = path.joinpath('voxel')
        if save_path.exists():
            shutil.rmtree(f'{save_path}')
        save_path.mkdir()

        voxel_paths = []
        pbar = tqdm(total=data_len, desc=f'{i+1}/{len(data_paths)}')
        for j in range(data_len):
            # voxelize_one(depth_file, lidar_file, cfg, save_path)
            data_row = pd_dataframe.iloc[j]
            depth_file = str(path.joinpath(data_row['depth_semantic_path']))
            lidar_file = str(path.joinpath(data_row['points_semantic_path']))
            name = re.match(r'.*/.*_(\d{9})\.png', depth_file).group(1)
            name_ = re.match(r'.*/.*_(\d{9})\.npy', lidar_file).group(1)
            assert name == name_, 'file sequence is false.'
            file_name = f'{save_path.name}/voxel_{name}.npy'
            save_name = f'{path}/{file_name}'
            p.apply_async(func=voxelize_one, args=(depth_file, lidar_file, cfg, save_name),
                          callback=lambda _: pbar.update())
            voxel_paths.append(file_name)
        p.close()
        p.join()
        # main_thread.join()
        pbar.close()
        pd_dataframe['voxel_path'] = voxel_paths
        pd_dataframe.to_pickle(pd_file)
        log.info(f'finished, save in {save_path}')


if __name__ == '__main__':
    main()
