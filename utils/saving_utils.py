"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""

import os
import numpy as np
import pandas as pd
import logging
import cv2
from PIL import Image
from tqdm import tqdm
import tempfile
import shutil

from data.dataset_utils import preprocess_birdview_and_routemap, binary_to_integer
from constants import CARLA_FPS

log = logging.getLogger(__name__)


def report_dataset_size(dataset_dir):
    list_runs = list(dataset_dir.glob('*'))

    n_frames = 0
    for run in list_runs:
        n_frames += len(os.listdir(os.path.join(run, 'image')))

    log.warning(f'{dataset_dir}: {len(list_runs)} episodes, '
                f'{n_frames} saved frames={n_frames / (CARLA_FPS * 3600):.2f} hours')


class DataWriter:
    def __init__(self, dir_path, ev_id, im_stack_idx=[-1], run_info=None, save_birdview_label=False,
                 render_image=False):
        self._dir_path = dir_path
        self._ev_id = ev_id
        self._im_stack_idx = np.array(im_stack_idx)
        self.run_info = run_info
        self.weather_keys = [
            'cloudiness', 'fog_density', 'fog_distance', 'fog_falloff', 'precipitation', 'precipitation_deposits',
            'sun_altitude_angle', 'sun_azimuth_angle', 'wetness', 'wind_intensity',
        ]

        assert self._im_stack_idx[0] == -1, 'Not handled'
        self.save_birdview_label = save_birdview_label
        self.render_image = render_image

        os.makedirs(self._dir_path, exist_ok=True)
        self._tmp_dir = tempfile.mkdtemp(dir=self._dir_path)
        print(f'tempdir: {self._tmp_dir}')

        self._data_list = []

    def write(self, timestamp, obs, supervision, reward, control_diff=None, weather=None):
        assert self._ev_id in obs and self._ev_id in supervision
        obs = obs[self._ev_id]
        render_rgb = None

        data_dict = {
            'step': timestamp['step'],
            'obs': {
                'central_rgb': None,
                'left_rgb': None,
                'right_rgb': None,
                'depth_semantic': None,
                # 'all_rgb': None,
                'gnss': None,
                'speed': None,
                'route_plan': None,
                'birdview': None,
                'point_cloud': None,
                'point_cloud_multi': None,
                'point_cloud_semantic': None,
            },
            'supervision': None,
            'control_diff': None,
            'weather': None,
            'reward': None,
            'critical': True,
        }

        # central_rgb
        data_dict['obs']['central_rgb'] = obs['central_rgb']
        # gnss speed
        data_dict['obs']['gnss'] = obs['gnss']
        data_dict['obs']['speed'] = obs['speed']

        # Route plan and birdview
        data_dict['obs']['route_plan'] = obs['route_plan']

        if self.save_birdview_label:
            data_dict['obs']['birdview'] = obs['birdview_label']
        else:
            data_dict['obs']['birdview'] = obs['birdview']

        # left_rgb & right_rgb
        if 'left_rgb' in obs and 'right_rgb' in obs:
            data_dict['obs']['left_rgb'] = obs['left_rgb']
            data_dict['obs']['right_rgb'] = obs['right_rgb']

            if self.render_image:
                render_rgb = np.concatenate([obs['central_rgb']['data'],
                                             obs['left_rgb']['data'],
                                             obs['right_rgb']['data']], axis=0)
        elif self.render_image:
            render_rgb = obs['central_rgb']['data']

        # depth_semantic
        if 'depth_semantic' in obs:
            data_dict['obs']['depth_semantic'] = obs['depth_semantic']

        # point cloud
        if 'lidar_points' in obs:
            data_dict['obs']['point_cloud'] = obs['lidar_points']

        if 'lidar_points_semantic' in obs:
            data_dict['obs']['point_cloud_semantic'] = obs['lidar_points_semantic']

        if 'lidar_points_multi' in obs:
            data_dict['obs']['point_cloud_multi'] = obs['lidar_points_multi']

        # supervision
        data_dict['supervision'] = supervision[self._ev_id]
        # Add reward in supervision
        data_dict['supervision']['reward'] = reward[self._ev_id]

        # reward
        data_dict['reward'] = reward[self._ev_id]

        # control_diff
        if control_diff is not None:
            data_dict['control_diff'] = control_diff[self._ev_id]

        if weather is not None:
            data_dict['weather'] = self.convert_weather_to_dict(weather)

        tmp = tempfile.NamedTemporaryFile(dir=self._tmp_dir, delete=False)
        np.save(tmp, data_dict)
        tmp.close()
        self._data_list.append(tmp.name)
        # self._data_list.append(data_dict)

        if self.render_image:
            # put text
            action_str = np.array2string(supervision[self._ev_id]['action'],
                                         precision=2, separator=',', suppress_small=True)
            speed = supervision[self._ev_id]['speed']
            txt_1 = f'{action_str} spd:{speed[0]:5.2f}'
            render_rgb = cv2.putText(render_rgb, txt_1, (0, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        return render_rgb

    def convert_weather_to_dict(self, weather):
        weather_dict = {}

        for key in self.weather_keys:
            weather_dict[key] = getattr(weather, key)

        return weather_dict

    @staticmethod
    def _write_dict_to_group(group, key, my_dict):
        group_key = group.create_group(key)
        for k, v in my_dict.items():
            if type(v) == np.ndarray and v.size > 2000:
                group_key.create_dataset(k, data=v, compression="gzip", compression_opts=4)
            else:
                group_key.create_dataset(k, data=v)

    def close(self, terminal_debug, remove_final_steps, last_value=None):
        # clean up data
        log.info(f'Episode finished, len={len(self._data_list)}')

        # behaviour cloning dataset
        valid = True
        if remove_final_steps:
            if terminal_debug['traffic_rule_violated']:
                step_to_delete = min(300, len(self._data_list))
                del self._data_list[-step_to_delete:]
                if len(self._data_list) < 300:
                    valid = False
                log.warning(f'traffic_rule_violated, valid={valid}, len={len(self._data_list)}')

            if terminal_debug['blocked']:
                step_to_delete = min(600, len(self._data_list))
                del self._data_list[-step_to_delete:]
                if len(self._data_list) < 300:
                    valid = False
                log.warning(f'blocked, valid={valid}, len={len(self._data_list)}')

        if terminal_debug['route_deviation']:
            valid = False
            log.warning(f'route deviation, valid={valid}')

        if valid:
            self.save_files()

        self._data_list.clear()
        shutil.rmtree(self._tmp_dir)
        return valid

    def save_files(self):
        os.makedirs(os.path.join(self._dir_path, 'image'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'image_left'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'image_right'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'image_all'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'depth_semantic'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'birdview'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'routemap'), exist_ok=True)
        # os.makedirs(os.path.join(self._dir_path, 'points'), exist_ok=True)
        os.makedirs(os.path.join(self._dir_path, 'points_semantic'), exist_ok=True)

        dict_dataframe = {
            'action_mu': [],
            'action_sigma': [],
            'action': [],
            'speed': [],
            'reward': [],
            'value': [],
            'features': [],
            'gnss': [],
            'target_gps': [],
            'imu': [],
            'command': [],
            'target_gps_next': [],
            'command_next': [],
            'image_path': [],
            'depth_semantic_path': [],
            # 'depth_semantic_trans': [],
            'birdview_path': [],
            'routemap_path': [],
            # 'points_path': [],
            # 'point_cloud_multi_path': [],
            'points_semantic_path': [],
            'n_classes': [],  # Number of classes in the bev
        }

        for k in self.run_info.keys():
            dict_dataframe[k] = []
        for k in self.weather_keys:
            dict_dataframe[k] = []

        points_list = {}
        # points_list_multi = {}
        points_list_semantic = {}

        log.info(f'Saving {self._dir_path}, data_len={len(self._data_list)}')

        for i, data_name in enumerate(tqdm(self._data_list, desc='Saving data')):
            data = np.load(data_name, allow_pickle=True).item()
            os.remove(data_name)

            obs = data['obs']
            supervision = data['supervision']

            for k, v in supervision.items():
                dict_dataframe[k].append(v)

            if 'action_mu' not in supervision.keys():
                # Using autopilot, fill with dummy values
                for k in ['action_mu', 'action_sigma', 'value', 'features']:
                    dict_dataframe[k].append(np.zeros(1))

            for k, v in obs['gnss'].items():
                dict_dataframe[k].append(v)

            # Add weather information
            for k, v in data['weather'].items():
                dict_dataframe[k].append(v)

            # Add run information
            for k, v in self.run_info.items():
                dict_dataframe[k].append(v)

            image = obs['central_rgb']['data']

            if obs['left_rgb'] is not None and obs['right_rgb'] is not None:
                image_left = obs['left_rgb']['data']
                image_right = obs['right_rgb']['data']
                image_all = np.concatenate([obs['left_rgb']['data'],
                                            obs['central_rgb']['data'],
                                            obs['right_rgb']['data']], axis=1)
            else:
                image_all, image_left, image_right = None, None, None

            if obs['depth_semantic'] is not None:
                depth_semantic = obs['depth_semantic']['data']
            else:
                depth_semantic = None

            if obs['point_cloud'] is not None:
                points = obs['point_cloud']['data']
            else:
                points = None

            if obs['point_cloud_semantic'] is not None:
                points_semantic = obs['point_cloud_semantic']['data']
            else:
                points_semantic = None

            # Process birdview and save as png
            birdview, route_map = preprocess_birdview_and_routemap(obs['birdview']['masks'])
            birdview, route_map = birdview.numpy(), route_map.numpy()
            n_bits, h, w = birdview.shape
            birdview = birdview.reshape(n_bits, -1)
            birdview = birdview.transpose((1, 0))
            # Convert bits to integer for storage
            birdview = binary_to_integer(birdview, n_bits).reshape(h, w)

            image_path = os.path.join(f'image', f'image_{i:09d}.png')
            birdview_path = os.path.join(f'birdview', f'birdview_{i:09d}.png')
            routemap_path = os.path.join(f'routemap', f'routemap_{i:09d}.png')
            dict_dataframe['image_path'].append(image_path)
            dict_dataframe['birdview_path'].append(birdview_path)
            dict_dataframe['routemap_path'].append(routemap_path)
            dict_dataframe['n_classes'].append(n_bits)
            # Save RGB images
            Image.fromarray(image).save(os.path.join(self._dir_path, image_path))
            Image.fromarray(birdview, mode='I').save(os.path.join(self._dir_path, birdview_path))
            Image.fromarray(route_map, mode='L').save(os.path.join(self._dir_path, routemap_path))
            if image_all is not None:
                image_left_path = os.path.join(f'image_left', f'image_left_{i:09d}.png')
                image_right_path = os.path.join(f'image_right', f'image_right_{i:09d}.png')
                image_all_path = os.path.join(f'image_all', f'image_all_{i:09d}.png')
                Image.fromarray(image_left).save(os.path.join(self._dir_path, image_left_path))
                Image.fromarray(image_right).save(os.path.join(self._dir_path, image_right_path))
                Image.fromarray(image_all).save(os.path.join(self._dir_path, image_all_path))
            if depth_semantic is not None:
                depth_semantic_path = os.path.join(f'depth_semantic', f'depth_semantic_{i:09d}.png')
                Image.fromarray(depth_semantic).save(os.path.join(self._dir_path, depth_semantic_path))
                # dict_dataframe['depth_semantic_trans'].append(obs['depth_semantic']['trans'])
                dict_dataframe['depth_semantic_path'].append(depth_semantic_path)

            # store point cloud
            if points is not None:
                points_path = os.path.join(f'points', f'points_{i:09d}.npy')
                np.save(os.path.join(self._dir_path, points_path), points)
                dict_dataframe['points_path'].append(points_path)
            if points_semantic is not None:
                points_semantic_path = os.path.join(f'points_semantic', f'points_semantic_{i:09d}.npy')
                np.save(os.path.join(self._dir_path, points_semantic_path), points_semantic)
                dict_dataframe['points_semantic_path'].append(points_semantic_path)

        pd_dataframe = pd.DataFrame(dict_dataframe)
        pd_dataframe.to_pickle(os.path.join(self._dir_path, 'pd_dataframe.pkl'))

