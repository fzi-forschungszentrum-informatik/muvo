import time

import numpy as np
import weakref
import carla
from queue import Queue, Empty
from gym import spaces
from matplotlib import cm
import open3d as o3d

from carla_gym.core.obs_manager.lidar.ray_cast_semantic import ObsManager as OM
from carla_gym.core.obs_manager.lidar.ray_cast_semantic import LABEL_COLORS


class ObsManager(OM):
    def __init__(self, obs_configs):
        super(ObsManager, self).__init__(obs_configs)
        self._camera_transform_list = []
        self._points_queue_list = {}
        self._sensor_list = {}
        self._rotations = ((90, 0, 0), (90, 0, 0), (90, 0, 0), (0, 90, 0), (0, 90, 0), (0, 90, 0))
        # self._scale = ((1, 0, 1), (0, 1, 1), (-1, 0, 1), (0, -1, 1), (0, 0, 1),
        #                (0.7, 0.7, 0.7), (-0.7, 0.7, 0.7), (-0.7, -0.7, 0.7), (0.7, -0.7, 0.7))
        self._scale = ((1, 0, 0.8), (-1, 0, 0.8), (0, 0, 1), (0, 1, 0.8), (0, -1, 0.8), (0, 0, 1))
        # self._scale = ((1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, 1), (0, 0, 1))
        # self._scale = ((1, 0, 1), (-1, 0, 1), (0, 0, 1),
        #                (0.5, 1, 1), (-0.5, 1, 1), (-0.5, -1, 1), (0.5, -1, 1))
        self._box_size = (float(obs_configs['box_size'][0]),
                          float(obs_configs['box_size'][1]),
                          float(obs_configs['box_size'][2]))
        x, y, z = self._box_size
        for (x_scale, y_scale, z_scale), (roll, pitch, yaw) in zip(self._scale, self._rotations):
            location = carla.Location(
                x=x * x_scale,
                y=y * y_scale,
                z=z * z_scale
            )
            rotation = carla.Rotation(
                roll=roll,
                pitch=pitch,
                yaw=yaw
            )
            self._camera_transform_list.append((carla.Transform(location, rotation)))

    def create_sensor(self, world, bp, transform, vehicle, i):
        self._points_queue_list[i] = Queue()
        sensor = world.spawn_actor(bp, transform, attach_to=vehicle)
        weak_self = weakref.ref(self)
        sensor.listen(lambda data: self._parse_points_m(weak_self, data, i))
        self._sensor_list[i] = sensor

    def attach_ego_vehicle(self, parent_actor):
        self._world = parent_actor.vehicle.get_world()
        bp = self._world.get_blueprint_library().find("sensor." + self._sensor_type)
        for key, value in self._lidar_options.items():
            bp.set_attribute(key, str(value))

        for i, camera_transform in enumerate(self._camera_transform_list):
            self.create_sensor(self._world, bp, camera_transform, parent_actor.vehicle, i)

    def get_observation(self):
        snap_shot = self._world.get_snapshot()
        points = []
        for transf, points_queue_key in zip(self._camera_transform_list, self._points_queue_list):
            points_queue = self._points_queue_list[points_queue_key]
            assert points_queue.qsize() <= 1
            # transf_matrix = transf.get_matrix()

            try:
                frame, data = points_queue.get(True, self._queue_timeout)
                assert snap_shot.frame == frame
                point_cloud = data['points_xyz']
                # point_cloud = np.append(point_cloud, np.ones((point_cloud.shape[0], 1)), axis=1)
                # point_cloud = np.dot(transf_matrix, point_cloud.T).T
                # point_cloud = point_cloud[:, :-1]
                ObjTag = data['ObjTag']
                CosAngel = data['CosAngel']
                ObjIdx = data['ObjIdx']
                points.append(
                    np.concatenate([point_cloud, CosAngel[:, None], ObjIdx[:, None], ObjTag[:, None]], axis=1))
                # obs.append({'frame': frame,
                #             'data': data,
                #             'transformation': transf_matrix})
                assert snap_shot.frame == frame
            except Empty:
                raise Exception('RGB sensor took too long!')
        points = np.concatenate(points, axis=0)
        data_all = {
            'points_xyz': points[:, :3],
            'CosAngel': points[:, 3],
            'ObjIdx': points[:, 4].astype(np.uint32),
            'ObjTag': points[:, 5].astype(np.uint32),
        }
        obs = {'frame': snap_shot.frame,
               'data': data_all}

        if self._render_o3d:
            self._point_list.points = o3d.utility.Vector3dVector(points[:, :3])
            self._point_list.colors = o3d.utility.Vector3dVector(points[5])
            self.vis.update_geometry(self._point_list)
            self.vis.poll_events()
            self.vis.update_renderer()
            time.sleep(0.005)

        return obs

    def clean(self):
        for key in self._sensor_list:
            sensor = self._sensor_list[key]
            if sensor and sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        self._sensor_list = {}
        self._world = None

        self._points_queue_list = {}

    @staticmethod
    def _parse_points_m(weak_self, data, i):
        self = weak_self()

        # get 4D points data
        point_cloud = np.frombuffer(data.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

        # Isolate the 3D points data
        points = np.array([point_cloud['x'], point_cloud['y'], point_cloud['z']]).T
        transf_matrix = self._camera_transform_list[i].get_matrix()

        points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
        points = np.dot(transf_matrix, points.T).T
        points = points[:, :-1]
        idx = (-50 <= points[:, 0]) & (points[:, 0] <= 50) \
            & (-40 <= points[:, 1]) & (points[:, 1] <= 40) \
            & (-20 <= points[:, 2]) & (points[:, 2] <= 20)

        self._points_queue_list[i].put((data.frame, {"points_xyz": points[idx],
                                                     "CosAngel": np.array(point_cloud['CosAngle'])[idx],
                                                     "ObjIdx": np.array(point_cloud['ObjIdx'])[idx],
                                                     "ObjTag": np.array(point_cloud['ObjTag'])[idx]}))
