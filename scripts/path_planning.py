import math
import numpy as np
from scipy.spatial.distance import euclidean
from itertools import permutations

class DisinfectionPlanner:
    def __init__(self, robot_width=0.6, safety_distance=0.2):
        self.robot_radius = robot_width / 2
        self.safety_distance = safety_distance
        self.d = self.robot_radius + self.safety_distance

    def calculate_stop_pose(self, obj_x, obj_y, obj_type, approach_angle=0):

        if obj_type in ['chair', 'shelf']: 
            # small stuff：沿approach_angle反方向停靠，面向物體
            stop_x = obj_x - self.d * math.cos(approach_angle)
            stop_y = obj_y - self.d * math.sin(approach_angle)
            orientation = approach_angle
        elif obj_type in ['bed', 'counter']:
            # big stuff：朝向與approach_angle垂直
            # side_angle = approach_angle - math.pi/2
            # stop_x = obj_x + self.d * math.cos(side_angle)
            # stop_y = obj_y + self.d * math.sin(side_angle)
            # orientation = approach_angle + math.pi/2 
            stop_x = obj_x - self.d * math.cos(approach_angle)
            stop_y = obj_y - self.d * math.sin(approach_angle)
            orientation = approach_angle + math.pi/2
        else:
            stop_x, stop_y, orientation = obj_x, obj_y, 0
        return (stop_x, stop_y, orientation)

    def tsp_path_schedule(self, objects, start_pose):
        stop_poses = []
        for obj in objects:
            dx = obj[0] - start_pose[0] 
            dy = obj[1] - start_pose[1] 
            approach_angle = math.atan2(dy, dx)
            pose = self.calculate_stop_pose(obj[0], obj[1], obj[2], approach_angle) 
            stop_poses.append(pose)

        min_distance = float('inf')
        best_order = []
        for order in permutations(range(len(stop_poses))):
            distance = euclidean(start_pose[:2], stop_poses[order[0]][:2])
            for i in range(len(order)-1):
                distance += euclidean(stop_poses[order[i]][:2], stop_poses[order[i+1]][:2])
            if distance < min_distance:
                min_distance = distance
                best_order = order

        return [stop_poses[i] for i in best_order]

objects = [(-2, 2, 'bed'), (1, 2, 'counter'), (1, 0.6, 'counter'), (-0.6, 1, 'shelf')] 
start_pose = (0.0, 2, 0)

planner = DisinfectionPlanner()
path = planner.tsp_path_schedule(objects, start_pose)

print("最優路徑順序:")
for point in path:
    print(f"位置: ({point[0]:.4f}, {point[1]:.4f}), 朝向: {math.degrees(point[2]):.2f}°")
