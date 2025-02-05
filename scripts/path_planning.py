#!/usr/bin/env python
import math
import numpy as np
import rospy
from scipy.spatial.distance import euclidean
from itertools import permutations
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion

class DisinfectionPlanner:
    def __init__(self, robot_width=0.6, safety_distance=0.2):
        self.robot_radius = robot_width / 2
        self.safety_distance = safety_distance
        self.d = self.robot_radius + self.safety_distance

    def calculate_stop_pose(self, obj_x, obj_y, obj_type, approach_angle=0):
        if obj_type in ['chair', 'shelf']: 
            # small stuff：沿 approach_angle 的反方向停靠，面向物體
            stop_x = obj_x - self.d * math.cos(approach_angle)
            stop_y = obj_y - self.d * math.sin(approach_angle)
            orientation = approach_angle
        elif obj_type in ['bed', 'counter']:
            # big stuff：朝向與 approach_angle 垂直
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
    
    def publish_goal(self, stop_pose, pub):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = stop_pose[0]
        goal_msg.pose.position.y = stop_pose[1]
        goal_msg.pose.position.z = 0.0
        
        quat = quaternion_from_euler(0, 0, stop_pose[2])
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]
        
        pub.publish(goal_msg)
        rospy.loginfo("Published goal: ({:.4f}, {:.4f}, {:.2f} rad)".format(stop_pose[0], stop_pose[1], stop_pose[2]))

current_start_pose = None

def amcl_pose_callback(msg):
    global current_start_pose

    pos = msg.pose.pose.position
    # 將四元數轉換成 yaw 角度
    quat = msg.pose.pose.orientation
    (_, _, yaw) = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    current_start_pose = (pos.x, pos.y, yaw)
    rospy.loginfo("AMCL Pose received: (%.4f, %.4f, %.2f rad)", pos.x, pos.y, yaw)

if __name__ == "__main__":
    rospy.init_node("disinfection_planner")

    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)
    pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

    rospy.loginfo("Waiting for AMCL")
    while current_start_pose is None and not rospy.is_shutdown():
        rospy.sleep(0.1)

    # 可根據需要，從其他來源取得消毒物件的資訊
    objects = [(-2, 2, 'bed'), (1, 2, 'counter'), (1, 0.6, 'counter'), (-0.6, 1, 'shelf')] 
    
    planner = DisinfectionPlanner()

    path = planner.tsp_path_schedule(objects, current_start_pose)
    
    rospy.loginfo("Start publishing goals")
    for point in path:
        planner.publish_goal(point, pub)
        rospy.sleep(2)  # 等待機器人執行
