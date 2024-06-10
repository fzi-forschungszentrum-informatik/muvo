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

        # Coordinates are forward-right-up https://carla.readthedocs.io/en/latest/ref_sensors/
        location = carla.Location(
            x=float(obs_configs['location'][0]),
            y=float(obs_configs['location'][1]),
            z=float(obs_configs['location'][2]))
        rotation = carla.Rotation(
            roll=float(obs_configs['rotation'][0]),
            pitch=float(obs_configs['rotation'][1]),
            yaw=float(obs_configs['rotation'][2]))

        self._camera_transform = carla.Transform(location, rotation)

        self._sensor_list = []
        self._data_queue = None
        self._queue_timeout = 10.0

        super(ObsManager, self).__init__()

    def _define_obs_space(self):

        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Box(
                low=0, high=255,
                shape=(self._height, self._width, self._channels),
                dtype=np.uint8)
        })

    def create_sensor(self, world, bp, transform, vehicle):
        sensor_type = bp.tags[0]
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        weak_self = weakref.ref(self)
        sensor.listen(lambda data: self._parse_points(weak_self, data, sensor_type))
        self._sensor_list.append(sensor)

    def attach_ego_vehicle(self, parent_actor):
        self._data_queue = Queue()
        self._world = parent_actor.vehicle.get_world()
        bps = [self._world.get_blueprint_library().find("sensor." + sensor) for sensor in self._sensor_types]
        for bp in bps:
            bp.set_attribute('image_size_x', str(self._width))
            bp.set_attribute('image_size_y', str(self._height))
            bp.set_attribute('fov', str(self._fov))

            self.create_sensor(self._world, bp, self._camera_transform, parent_actor.vehicle)

    def get_observation(self):
        snap_shot = self._world.get_snapshot()

        assert self._data_queue.qsize() <= len(self._sensor_types)
        datas = {}

        try:
            for _ in range(len(self._sensor_types)):
                frame, sensor_type, data = self._data_queue.get(True, self._queue_timeout)
                assert snap_shot.frame == frame
                datas[sensor_type] = data
        except Empty:
            raise Exception(f'{sensor_type} sensor took too long!')

        data = np.concatenate([datas['depth'], datas['semantic_segmentation']], axis=2)

        obs = {'frame': snap_shot.frame,
               'data': data}

        return obs

    def clean(self):
        for sensor in self._sensor_list:
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensor_list = {}
        self._world = None

        self._data_queue = None

    @staticmethod
    def _parse_points(weak_self, data, sensor_type):
        self = weak_self()

        np_img = np.frombuffer(data.raw_data, dtype=np.dtype('uint8'))
        np_img = np.reshape(copy.deepcopy(np_img), (data.height, data.width, 4))
        assert (sensor_type == 'depth' or sensor_type == 'semantic_segmentation'), 'sensor_type error'
        if sensor_type == 'depth':
            np_img = np_img[..., :3]
        elif sensor_type == 'semantic_segmentation':
            np_img = np_img[..., 2][..., None]

        self._data_queue.put((data.frame, sensor_type, np_img))
