import numpy as np
import carla
import math

EARTH_RADIUS_EQUA = 6378137.0


def vec_global_to_ref(target_vec_in_global, ref_rot_in_global):
    """
    :param target_vec_in_global: carla.Vector3D in global coordinate (world, actor)
    :param ref_rot_in_global: carla.Rotation in global coordinate (world, actor)
    :return: carla.Vector3D in ref coordinate
    """
    R = carla_rot_to_mat(ref_rot_in_global)
    np_vec_in_global = np.array([[target_vec_in_global.x],
                                 [target_vec_in_global.y],
                                 [target_vec_in_global.z]])
    np_vec_in_ref = R.T.dot(np_vec_in_global)
    target_vec_in_ref = carla.Vector3D(x=np_vec_in_ref[0, 0], y=np_vec_in_ref[1, 0], z=np_vec_in_ref[2, 0])
    return target_vec_in_ref


def carla_rot_to_mat(carla_rotation):
    """
    Transform rpy in carla.Rotation to rotation matrix in np.array

    :param carla_rotation: carla.Rotation
    :return: np.array rotation matrix
    """
    roll = np.deg2rad(carla_rotation.roll)
    pitch = np.deg2rad(carla_rotation.pitch)
    yaw = np.deg2rad(carla_rotation.yaw)

    yaw_matrix = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    pitch_matrix = np.array([
        [np.cos(pitch), 0, -np.sin(pitch)],
        [0, 1, 0],
        [np.sin(pitch), 0, np.cos(pitch)]
    ])
    roll_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(roll), np.sin(roll)],
        [0, -np.sin(roll), np.cos(roll)]
    ])

    rotation_matrix = yaw_matrix.dot(pitch_matrix).dot(roll_matrix)
    return rotation_matrix


def gps_to_location(gps):
    lat, lon, z = gps
    lat = float(lat)
    lon = float(lon)
    z = float(z)

    location = carla.Location(z=z)

    location.x = lon / 180.0 * (math.pi * EARTH_RADIUS_EQUA)

    location.y = -1.0 * math.log(math.tan((lat + 90.0) * math.pi / 360.0)) * EARTH_RADIUS_EQUA

    return location
