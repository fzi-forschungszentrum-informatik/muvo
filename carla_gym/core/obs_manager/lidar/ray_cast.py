import time

import numpy as np
import weakref
import carla
from queue import Queue, Empty
from gym import spaces
from matplotlib import cm
import open3d as o3d

from carla_gym.core.obs_manager.obs_manager import ObsManagerBase
from constants import CARLA_FPS

VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


class ObsManager(ObsManagerBase):
    """
    Template configs:
    obs_configs = {
        "module": "lidar.ray_cast",
        "location": [-5.5, 0, 2.8],
        "rotation": [0, 0, 0],
        "frame_stack": 1,
        "render_o3d": False,
        "show_axis": False,
        "no_noise": False,
        "lidar_options": {
            "width": 1920,
            "height": 1080,
            # https://github.com/carla-simulator/leaderboard/blob/master/leaderboard/autoagents/agent_wrapper.py
            "channels": 64,
            "range": 100,
            "rotation_frequency": 20
            "points_per_second": 100000
            "upper_fov": 15.0,
            "lower_fov": 25.0, # -30.0
            "atmosphere_attenuation_rate": 0.004,
            # if no_noise
            "dropoff_general_rate": 0.45,
            "dropoff_intensity_limit": 0.8,
            "dropoff_zero_intensity": 0.4,
        },
    }
    frame_stack: [Image(t-2), Image(t-1), Image(t)]
    """

    def __init__(self, obs_configs):

        self._sensor_type = 'lidar.ray_cast'

        self._lidar_options = obs_configs['lidar_options']
        self._no_noise = obs_configs['no_noise']
        self._render_o3d = obs_configs["render_o3d"]
        self._show_axis = obs_configs["show_axis"]

        # rewrite the 'rotation_frequency' to the same as carla_fps
        self._lidar_options['rotation_frequency'] = CARLA_FPS

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

        self._sensor = None
        self._queue_timeout = 10.0
        self._points_queue = None
        if self._render_o3d:
            self._point_list = o3d.geometry.PointCloud()
            self._point_list.points = o3d.utility.Vector3dVector(10 * np.random.randn(1000, 3))

            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                window_name='Carla Lidar',
                width=960,
                height=540,
                left=480,
                top=270)
            self.vis.get_render_option().background_color = [0.05, 0.05, 0.05]
            self.vis.get_render_option().point_size = 1
            self.vis.get_render_option().show_coordinate_frame = True
            if self._show_axis:
                add_open3d_axis(self.vis)
            self.vis.add_geometry(self._point_list)

        super(ObsManager, self).__init__()

    def _define_obs_space(self):

        self.obs_space = spaces.Dict({
            'frame': spaces.Discrete(2**32-1),
            'data': spaces.Dict({
                'x': spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                'y': spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                'z': spaces.Box(low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32),
                'i': spaces.Box(low=0, high=1, shape=(1, ), dtype=np.float32),
            })
        })

    def attach_ego_vehicle(self, parent_actor):
        self._points_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor."+self._sensor_type)
        for key, value in self._lidar_options.items():
            bp.set_attribute(key, str(value))
        if self._no_noise:
            bp.set_attribute('dropoff_general_rate', '0.0')
            bp.set_attribute('dropoff_intensity_limit', '1.0')
            bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            bp.set_attribute('noise_stddev', '0.2')

        self._sensor = self._world.spawn_actor(bp, self._camera_transform, attach_to=parent_actor.vehicle)
        weak_self = weakref.ref(self)
        self._sensor.listen(lambda data: self._parse_points(weak_self, data))

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        assert self._points_queue.qsize() <= 1

        try:
            frame, data = self._points_queue.get(True, self._queue_timeout)
            assert snap_shot.frame == frame
        except Empty:
            raise Exception('RGB sensor took too long!')

        if self._render_o3d:
            self.vis.update_geometry(self._point_list)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.005)

        obs = {'frame': frame,
               'data': data}

        return obs

    def clean(self):
        if self._sensor and self._sensor.is_alive:
            self._sensor.stop()
            self._sensor.destroy()
        self._sensor = None
        self._world = None

        self._points_queue = None

    @staticmethod
    def _parse_points(weak_self, data):
        self = weak_self()

        # get 4D points data
        point_cloud = np.copy(np.frombuffer(data.raw_data, dtype=np.dtype('f4')))
        point_cloud = np.reshape(point_cloud, (int(point_cloud.shape[0] / 4), 4))

        # Isolate the intensity
        intensity = point_cloud[:, -1]

        # Isolate the 3D points data
        points = point_cloud[:, :-1]
        # points[:, :1] = -points[:, :1]

        if self._render_o3d:
            intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
            int_color = np.c_[
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
                np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

            self._point_list.points = o3d.utility.Vector3dVector(points)
            self._point_list.colors = o3d.utility.Vector3dVector(int_color)

        self._points_queue.put((data.frame, {"points_xyz": points, "intensity": intensity}))
