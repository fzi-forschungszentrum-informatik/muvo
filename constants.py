import numpy as np

CARLA_FPS = 10
DISPLAY_SEGMENTATION = True
DISTORT_IMAGES = False
WHEEL_BASE = 2.8711279296875
# Ego-vehicle is 4.902m long and 2.128m wide. See `self._parent_actor.vehicle.bounding_box` in chaffeurnet_label
EGO_VEHICLE_DIMENSION = [4.902, 2.128, 1.511]

# https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/agents/navigation/local_planner.py
# However when processed, see "process_obs" function, unknown becomes lane_follow and the rest has a value between
# [0, 5] by substracting 1.
ROUTE_COMMANDS = {0: 'UNKNOWN',
                  1: 'LEFT',
                  2: 'RIGHT',
                  3: 'STRAIGHT',
                  4: 'LANEFOLLOW',
                  5: 'CHANGELANELEFT',
                  6: 'CHANGELANERIGHT',
                  }

BIRDVIEW_COLOURS = np.array([[255, 255, 255],          # Background
                             [225, 225, 225],       # Road
                             [160, 160, 160],      # Lane marking
                             [0, 83, 138],        # Vehicle
                             [127, 255, 212],      # Pedestrian
                             [50, 205, 50],        # Green light
                             [255, 215, 0],      # Yellow light
                             [220, 20, 60],        # Red light and stop sign
                             ], dtype=np.uint8)

# Obtained with sqrt of inverse frequency
SEMANTIC_SEG_WEIGHTS = np.array([1.0, 1.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0])

# VOXEL_SEG_WEIGHTS = np.ones(23, dtype=float)
# VOXEL_SEG_WEIGHTS[4] = 3.0
# VOXEL_SEG_WEIGHTS[10] = 2.0

VOXEL_SEG_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.5, 2.0, 3.0, 1.0, 1.0, 1.0])

VOXEL_LABEL_CARLA = {
    0:   'Background',  # None
    1:   'Building',  # Building
    2:   'Fences',  # Fences
    3:   'Other',  # Other
    4:   'Pedestrian',  # Pedestrian
    5:   'Pole',  # Pole
    6:   'RoadLines',  # RoadLines
    7:   'Road',  # Road
    8:   'Sidewalk',  # Sidewalk
    9:   'Vegetation',  # Vegetation
    10:  'Vehicle',  # Vehicle
    11:  'Wall',  # Wall
    12:  'TrafficSign',  # TrafficSign
    13:  'Sky',  # Sky
    14:  'Ground',  # Ground
    15:  'Bridge',  # Bridge
    16:  'RailTrack',  # RailTrack
    17:  'GuardRail',  # GuardRail
    18:  'TrafficLight',  # TrafficLight
    19:  'Static',  # Static
    20:  'Dynamic',  # Dynamic
    21:  'Water',  # Water
    22:  'Terrain',  # Terrain
}

# VOXEL_LABEL = {
#     0:  'Background',
#     1:  'Road',
#     2:  'RoadLines',
#     3:  'Sidewalk',
#     4:  'Vehicle',
#     5:  'Pedestrian',
#     6:  'TrafficSign',
#     7:  'TrafficLight',
#     8:  'Others'
# }
VOXEL_LABEL = {
    0:  'Background',
    1:  'Occupancy',
}
# VOXEL_LABEL = VOXEL_LABEL_CARLA

# VOXEL_COLOURS = np.array([[255, 255, 255],  # Background
#                           [150, 150, 150],  # Road
#                           [200, 200, 20],  # Road Lines
#                           [200, 200, 200],  # Sidewalk
#                           [0, 83, 138],  # Vehicle
#                           [127, 255, 212],  # Pedestrian
#                           [220, 20, 60],  # Traffic Sign
#                           [100, 150, 35],  # Traffic light
#                           [0, 0, 0],  # Others
#                           ], dtype=np.uint8)
VOXEL_COLOURS = np.array([[255, 255, 255],  # Background
                          [115, 115, 115],  # Others
                          ], dtype=np.uint8)
# VOXEL_COLOURS = np.array([[255, 255, 255],  # None
#                           [70, 70, 70],     # Building
#                           [100, 40, 40],    # Fences
#                           [55, 90, 80],     # Other
#                           [220, 20, 60],    # Pedestrian
#                           [153, 153, 153],  # Pole
#                           [157, 234, 50],   # RoadLines
#                           [128, 64, 128],   # Road
#                           [244, 35, 232],   # Sidewalk
#                           [107, 142, 35],   # Vegetation
#                           [0, 0, 142],      # Vehicle
#                           [102, 102, 156],  # Wall
#                           [220, 220, 0],    # TrafficSign
#                           [70, 130, 180],   # Sky
#                           [81, 0, 81],      # Ground
#                           [150, 100, 100],  # Bridge
#                           [230, 150, 140],  # RailTrack
#                           [180, 165, 180],  # GuardRail
#                           [250, 170, 30],   # TrafficLight
#                           [110, 190, 160],  # Static
#                           [170, 120, 50],   # Dynamic
#                           [45, 60, 150],    # Water
#                           [145, 170, 100],  # Terrain
#                           ], dtype=np.uint8)

# VOXEL_COLOURS = np.array([[0, 0, 0],  # unlabeled
#                          # cityscape
#                           [128, 64, 128],     # road = 1
#                           [244, 35, 232],     # sidewalk = 2
#                           [70, 70, 70],       # building = 3
#                           [102, 102, 156],    # wall = 4
#                           [190, 153, 153],    # fence = 5
#                           [153, 153, 153],    # pole = 6
#                           [250, 170, 30],     # traffic light = 7
#                           [220, 220, 0],      # traffic sign = 8
#                           [107, 142, 35],     # vegetation = 9
#                           [152, 251, 152],    # terrain = 10
#                           [70, 130, 180],     # sky = 11
#                           [220, 20, 60],      # pedestrian = 12
#                           [255, 0, 0],        # rider = 13
#                           [0, 0, 142],        # Car = 14
#                           [0, 0, 70],         # truck = 15
#                           [0, 60, 100],       # bs = 16
#                           [0, 80, 100],       # train = 17
#                           [0, 0, 230],        # motorcycle = 18
#                           [119, 11, 32],      # bicycle = 19
#                           # custom
#                           [110, 190, 160],    # static = 20
#                           [170, 120, 50],     # dynamic = 21
#                           [55, 90, 80],       # other = 22
#                           [45, 60, 150],      # water = 23
#                           [157, 234, 50],     # road line = 24
#                           [81, 0, 81],        # grond = 25
#                           [150, 100, 100],    # bridge = 26
#                           [230, 150, 140],    # rail track = 27
#                           [180, 165, 180],    # gard rail = 28
#                           ], dtype=np.uint8)

# LABEL_MAP = {
#     0:  0,  # None
#     1:  8,  # Building
#     2:  8,  # Fences
#     3:  8,  # Other
#     4:  5,  # Pedestrian
#     5:  8,  # Pole
#     6:  2,  # RoadLines
#     7:  1,  # Road
#     8:  3,  # Sidewalk
#     9:  8,  # Vegetation
#     10: 4,  # Vehicle
#     11: 8,  # Wall
#     12: 6,  # TrafficSign
#     13: 0,  # Sky
#     14: 8,  # Ground
#     15: 8,  # Bridge
#     16: 8,  # RailTrack
#     17: 8,  # GuardRail
#     18: 7,  # TrafficLight
#     19: 8,  # Static
#     20: 8,  # Dynamic
#     21: 8,  # Water
#     22: 8,  # Terrain
# }
LABEL_MAP = {
    0:  0,  # None
    1:  1,  # Building
    2:  1,  # Fences
    3:  1,  # Other
    4:  1,  # Pedestrian
    5:  1,  # Pole
    6:  1,  # RoadLines
    7:  1,  # Road
    8:  1,  # Sidewalk
    9:  1,  # Vegetation
    10: 1,  # Vehicle
    11: 1,  # Wall
    12: 1,  # TrafficSign
    13: 0,  # Sky
    14: 1,  # Ground
    15: 1,  # Bridge
    16: 1,  # RailTrack
    17: 1,  # GuardRail
    18: 1,  # TrafficLight
    19: 1,  # Static
    20: 1,  # Dynamic
    21: 1,  # Water
    22: 1,  # Terrain
}
# LABEL_MAP = {
#     0:  0,  # None
#     1:  1,  # Building
#     2:  2,  # Fences
#     3:  3,  # Other
#     4:  4,  # Pedestrian
#     5:  5,  # Pole
#     6:  6,  # RoadLines
#     7:  7,  # Road
#     8:  8,  # Sidewalk
#     9:  9,  # Vegetation
#     10: 10,  # Vehicle
#     11: 11,  # Wall
#     12: 12,  # TrafficSign
#     13: 13,  # Sky
#     14: 14,  # Ground
#     15: 15,  # Bridge
#     16: 16,  # RailTrack
#     17: 17,  # GuardRail
#     18: 18,  # TrafficLight
#     19: 19,  # Static
#     20: 20,  # Dynamic
#     21: 21,  # Water
#     22: 22,  # Terrain
# }
