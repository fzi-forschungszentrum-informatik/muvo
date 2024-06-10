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

LABEL_COLORS = np.array([
    (0, 0, 0),  # unlabeled
    # cityscape
    (128, 64, 128),     # road = 1
    (244, 35, 232),     # sidewalk = 2
    (70, 70, 70),       # building = 3
    (102, 102, 156),    # wall = 4
    (190, 153, 153),    # fence = 5
    (153, 153, 153),    # pole = 6
    (250, 170, 30),     # traffic light = 7
    (220, 220, 0),      # traffic sign = 8
    (107, 142, 35),     # vegetation = 9
    (152, 251, 152),    # terrain = 10
    (70, 130, 180),     # sky = 11
    (220, 20, 60),      # pedestrian = 12
    (255, 0, 0),        # rider = 13
    (0, 0, 142),        # Car = 14
    (0, 0, 70),         # truck = 15
    (0, 60, 100),       # bs = 16
    (0, 80, 100),       # train = 17
    (0, 0, 230),        # motorcycle = 18
    (119, 11, 32),      # bicycle = 19
    # custom
    (110, 190, 160),    # static = 20
    (170, 120, 50),     # dynamic = 21
    (55, 90, 80),       # other = 22
    (45, 60, 150),      # water = 23
    (157, 234, 50),     # road line = 24
    (81, 0, 81),        # grond = 25
    (150, 100, 100),    # bridge = 26
    (230, 150, 140),    # rail track = 27
    (180, 165, 180)     # gard rail = 28
]) / 255.0  # normalize each channel [0-1] since is what Open3D uses


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
        },
    }
    frame_stack: [Image(t-2), Image(t-1), Image(t)]
    """

    def __init__(self, obs_configs):

        self._sensor_type = 'lidar.ray_cast_semantic'

        self._lidar_options = obs_configs['lidar_options']
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

        self._world = None
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
                'CosAngle': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
                'ObjIdx': spaces.Box(low=0, high=28, shape=(1,), dtype=np.uint32),
                'ObjTag': spaces.Box(low=0, high=28, shape=(1,), dtype=np.uint32),
            })
        })

    def attach_ego_vehicle(self, parent_actor):
        self._points_queue = Queue()

        self._world = parent_actor.vehicle.get_world()

        bp = self._world.get_blueprint_library().find("sensor."+self._sensor_type)
        for key, value in self._lidar_options.items():
            bp.set_attribute(key, str(value))

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
        point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

        # Isolate the 3D points data
        points = np.array([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T

        if self._render_o3d:
            labels = np.array(point_cloud['ObjTag'])
            int_color = LABEL_COLORS[labels]

            self._point_list.points = o3d.utility.Vector3dVector(points)
            self._point_list.colors = o3d.utility.Vector3dVector(int_color)

        self._points_queue.put((data.frame, {"points_xyz": points,
                                             # "CosAngel": np.array(point_cloud['CosAngle']),
                                             # "ObjIdx": np.array(point_cloud['ObjIdx']),
                                             "ObjTag": np.array(point_cloud['ObjTag'], dtype=np.uint8)}))
