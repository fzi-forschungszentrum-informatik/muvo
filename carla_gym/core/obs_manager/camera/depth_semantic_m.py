# import time

import numpy as np
import weakref
import copy
import carla
from queue import Queue, Empty
from gym import spaces
# from matplotlib import cm
# import open3d as o3d

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase


class ObsManager(ObsManagerBase):
    def __init__(self, obs_configs):
        self._sensor_types = ('camera.depth', 'camera.semantic_segmentation')
        self._height = obs_configs['height']
        self._width = obs_configs['width']
        self._fov = obs_configs['fov']
        self._channels = 4

        self._camera_transform_list = []
        # self._depth_queue_list = []
        # self._semantic_queue_list = []
        self._data_queue_list = []
        self._sensor_list = []
        self._rotation = carla.Rotation(roll=0, pitch=-90, yaw=0)  # roll, pitch, yaw
        # self._scale = ((2, -1, 1), (2, 0, 1), (2, 1, 1),
        #                (1, -1, 1), (1, 0, 1), (1, 1, 1),
        #                (0, -1, 1), (0, 0, 1), (0, 1, 1),
        #                (-1, -1, 1), (-1, 0, 1), (-1, 1, 1),
        #                (-2, -1, 1), (-2, 0, 1), (-2, 1, 1))
        self._scale = []
        self._hw = obs_configs['sensor_num']
        for i in range(2 * self._hw[0] + 1):
            for j in range(2 * self._hw[1] + 1):
                self._scale.append((-i + self._hw[0], j - self._hw[1], 1))
        # self._scale = ((1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1), (0, 0, 1))
        # self._scale = ((1, 0, 1), (-1, 0, 1), (0, 0, 1),
        #                (0.5, 1, 1), (-0.5, 1, 1), (-0.5, -1, 1), (0.5, -1, 1))
        self._box_size = (float(obs_configs['box_size'][0]),
                          float(obs_configs['box_size'][1]),
                          float(obs_configs['box_size'][2]))
        x, y, z = self._box_size
        for x_scale, y_scale, z_scale in self._scale:
            location = carla.Location(
                x=x * x_scale,
                y=y * y_scale,
                z=z * z_scale
            )
            self._camera_transform_list.append((carla.Transform(location, self._rotation)))

        self._queue_timeout = 10.0

        super(ObsManager, self).__init__()

    def _define_obs_space(self):

        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Box(
                low=0, high=255,
                shape=((2 * self._hw[0] + 1) * self._height, (2 * self._hw[1] + 1) * self._width, self._channels),
                dtype=np.uint8)
        })

    def create_sensor(self, world, bp, transform, vehicle, i):
        self._data_queue_list.append(Queue())
        sensor_type = bp.tags[0]
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        weak_self = weakref.ref(self)
        sensor.listen(lambda data: self._parse_points_m(weak_self, data, sensor_type, i))
        self._sensor_list.append(sensor)

    def attach_ego_vehicle(self, parent_actor):
        self._world = parent_actor.vehicle.get_world()
        bps = [self._world.get_blueprint_library().find("sensor." + sensor) for sensor in self._sensor_types]
        for bp in bps:
            bp.set_attribute('image_size_x', str(self._width))
            bp.set_attribute('image_size_y', str(self._height))
            bp.set_attribute('fov', str(self._fov))

            for i, camera_transform in enumerate(self._camera_transform_list):
                self.create_sensor(self._world, bp, camera_transform, parent_actor.vehicle, i)

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        data_all = []
        for transf, data_queue in zip(self._camera_transform_list, self._data_queue_list):
            assert data_queue.qsize() <= 2
            datas = {}

            try:
                for _ in range(len(self._sensor_types)):
                    frame, sensor_type, data = data_queue.get(True, self._queue_timeout)
                    assert snap_shot.frame == frame
                    datas[sensor_type] = data
            except Empty:
                raise Exception(f'{sensor_type} sensor took too long!')

            data_all.append(np.concatenate([datas['depth'], datas['semantic_segmentation']], axis=2))
        h_ = 2 * self._hw[0] + 1
        w_ = 2 * self._hw[1] + 1
        data = np.concatenate([np.concatenate(
            [data_all[j] for j in range(w_*i, w_*i+w_)], axis=1) for i in range(h_)], axis=0)

        obs = {'frame': snap_shot.frame,
               'data': data,
               'trans': self._box_size}

        return obs

    def clean(self):
        for sensor in self._sensor_list:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensor_list = {}
        self._world = None

        self._data_queue_list = {}

    @staticmethod
    def _parse_points_m(weak_self, data, sensor_type, i):
        self = weak_self()

        np_img = np.frombuffer(data.raw_data, dtype=np.dtype('uint8'))
        np_img = np.reshape(copy.deepcopy(np_img), (data.height, data.width, 4))
        assert (sensor_type == 'depth' or sensor_type == 'semantic_segmentation'), 'sensor_type error'
        if sensor_type == 'depth':
            np_img = np_img[..., :3]
        elif sensor_type == 'semantic_segmentation':
            np_img = np_img[..., 2][..., None]

        self._data_queue_list[i].put((data.frame, sensor_type, np_img))
