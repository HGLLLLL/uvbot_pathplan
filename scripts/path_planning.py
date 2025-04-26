#!/usr/bin/env python
import math
import numpy as np
import rospy
from scipy.spatial.distance import euclidean
from itertools import permutations
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Point
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import defaultdict
from uvbot_pathplan.msg import ObjectWallPosition 
from uvbot_pathplan.msg import FilteredSegment
from visualization_msgs.msg import Marker

class DisinfectionPlanner:
    def __init__(self, robot_width=0.6, safety_distance=0.2, cluster_threshold=0.3):
        self.robot_radius = robot_width / 2.0
        self.safety_distance = safety_distance
        self.d = self.robot_radius + self.safety_distance
        self.cluster_threshold = cluster_threshold 
        
        # 儲存所有收到的檢測數據，格式：
        # { type: [(x_obj, y_obj, wall1, wall2), ...] }
        self.object_dict = defaultdict(list)
        
        # 儲存牆壁座標，由 /filtered_segments 訊息提供，格式：
        # list of (x_start, y_start, x_end, y_end)
        self.wall_segments = []
        
        # 訂閱 /filtered_position 訊息（包含物體座標及牆壁座標）
        rospy.Subscriber("/filtered_position", ObjectWallPosition, self.object_wall_callback)
        # 訂閱 /filtered_segments 訊息（包含房間中所有牆壁的座標）
        rospy.Subscriber("/filtered_segments", FilteredSegment, self.wall_segment_callback)
        
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)
        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.marker_id = 0

    def object_wall_callback(self, msg):
        """
        處理 /object_wall_position 的訂閱訊息，
        儲存物體座標以及原始牆壁座標 (wall1, wall2)
        """
        obj_type = msg.type.strip()
        x_obj = msg.x_obj
        y_obj = msg.y_obj
        wall1 = (msg.x_wall1, msg.y_wall1)
        wall2 = (msg.x_wall2, msg.y_wall2)
        self.object_dict[obj_type].append((x_obj, y_obj, wall1, wall2))
        rospy.loginfo("Received detection: type=%s, obj=(%.2f, %.2f), wall1=(%.2f, %.2f), wall2=(%.2f, %.2f)",
                      obj_type, x_obj, y_obj, msg.x_wall1, msg.y_wall1, msg.x_wall2, msg.y_wall2)

    def wall_segment_callback(self, msg):
        """
        處理 /filtered_segments 的訊息，存取房間中所有牆壁的起始與結束座標
        訊息格式：
            float64 x_start
            float64 y_start
            float64 x_end
            float64 y_end
        """
        wall = (msg.x_start, msg.y_start, msg.x_end, msg.y_end)
        self.wall_segments.append(wall)

        # Too noisy, comment out
        # rospy.loginfo("Received wall segment: start=(%.2f, %.2f), end=(%.2f, %.2f)", msg.x_start, msg.y_start, msg.x_end, msg.y_end)

    def calculate_wall_direction(self, wall_p1, wall_p2):
        """根據牆面兩端點計算牆面方向角及其垂直方向角"""
        wall_vec = np.array([wall_p1[0] - wall_p2[0], wall_p1[1] - wall_p2[1]])
        wall_angle = math.atan2(wall_vec[1], wall_vec[0])
        perp_angle = wall_angle + math.pi/2
        return wall_angle, perp_angle

    def calculate_stop_pose(self, obj_x, obj_y, obj_type, wall_p1, wall_p2):
        """
        根據物體與牆面資訊計算機器人的停靠點。
        計算方式保持不變，僅用傳入的牆面座標 (wall_p1, wall_p2)
        """
        wall_angle, perp_angle = self.calculate_wall_direction(wall_p1, wall_p2)

        if obj_type.lower() in ['chair']:
            stop_x = obj_x + self.d * math.cos(perp_angle)
            stop_y = obj_y + self.d * math.sin(perp_angle)
            reverse_perp_angle = perp_angle + math.pi
            orientation = reverse_perp_angle

        elif obj_type.lower() in ["switch"]:
            reverse_perp_angle = perp_angle + math.pi
            stop_x = obj_x + self.d * math.cos(reverse_perp_angle)
            stop_y = obj_y + self.d * math.sin(reverse_perp_angle)
            orientation = perp_angle

        elif obj_type.lower() in ['side rail']:
            stop_x = obj_x + self.d * math.cos(perp_angle)
            stop_y = obj_y + self.d * math.sin(perp_angle)
            orientation = wall_angle
        else:
            stop_x, stop_y, orientation = obj_x, obj_y, 0
        return (stop_x, stop_y, orientation)

    def cluster_detections(self, det_list, cluster_threshold=None):
        """
        對同一類型內所有檢測進行聚類（僅根據物體位置聚類）
        每個元素展平成一個列表：[x_obj, y_obj, wall1_x, wall1_y, wall2_x, wall2_y]
        返回聚類後的列表，每個元素為 (avg_obj_x, avg_obj_y, avg_wall1, avg_wall2)
        """
        if cluster_threshold is None:
            cluster_threshold = self.cluster_threshold

        clusters = []
        used = [False] * len(det_list)
        for i in range(len(det_list)):
            if used[i]:
                continue
            cluster = [det_list[i]]
            used[i] = True
            for j in range(i+1, len(det_list)):
                if used[j]:
                    continue
                if euclidean((det_list[i][0], det_list[i][1]), (det_list[j][0], det_list[j][1])) < cluster_threshold:
                    cluster.append(det_list[j])
                    used[j] = True
            avg_obj_x = np.mean([pt[0] for pt in cluster])
            avg_obj_y = np.mean([pt[1] for pt in cluster])
            avg_wall1_x = np.mean([pt[2][0] for pt in cluster])
            avg_wall1_y = np.mean([pt[2][1] for pt in cluster])
            avg_wall2_x = np.mean([pt[3][0] for pt in cluster])
            avg_wall2_y = np.mean([pt[3][1] for pt in cluster])
            clusters.append((avg_obj_x, avg_obj_y, (avg_wall1_x, avg_wall1_y), (avg_wall2_x, avg_wall2_y)))
        return clusters

    def find_closest_wall(self, obj_x, obj_y):
        """
        將物體座標與所有牆壁進行比對，
        找出與物體垂直距離最近的牆壁，
        返回該牆壁的 (start, end) 座標，分別為 (x_start, y_start) 與 (x_end, y_end)
        """
        if not self.wall_segments:
            rospy.logwarn("No wall segments available.")
            return None

        min_distance = float('inf')
        closest_wall = None
        P = np.array([obj_x, obj_y])
        for wall in self.wall_segments:
            A = np.array([wall[0], wall[1]])
            B = np.array([wall[2], wall[3]])
            AB = B - A
            if np.linalg.norm(AB) == 0:
                continue
            distance = np.abs(np.cross(AB, A - P)) / np.linalg.norm(AB)
            if distance < min_distance:
                min_distance = distance
                closest_wall = wall
        if closest_wall is None:
            rospy.logwarn("No valid wall segment found for object at (%.2f, %.2f)", obj_x, obj_y)
            return None
        return ((closest_wall[0], closest_wall[1]), (closest_wall[2], closest_wall[3]))

    def generate_stop_poses(self):
        """
        根據 object_dict 中儲存的所有檢測數據（物體座標及原始牆壁座標），
        對每一類型進行聚類，然後依據每個聚類找到與該物體垂直距離最近的牆壁，
        與原始牆壁座標比較：若方向相反則對調 filtered_segments 的 start 與 end，
        最後計算機器人的停靠點，返回 (stop_x, stop_y, orientation)
        """
        stop_poses = []
        for obj_type, det_list in self.object_dict.items():
            if obj_type.lower() == 'chair':
                threshold = 1
            elif obj_type.lower() == 'side rail':
                threshold = 0.2
            else:
                threshold = self.cluster_threshold 

            clusters = self.cluster_detections(det_list, threshold)
            for cluster in clusters:
                x_obj, y_obj, orig_wall1, orig_wall2 = cluster
                filtered_wall = self.find_closest_wall(x_obj, y_obj)
                if filtered_wall is None:
                    rospy.logwarn("Skipping object at (%.2f, %.2f) due to no wall found", x_obj, y_obj)
                    continue
                wall_p1, wall_p2 = filtered_wall
                # 與原始 wall 比較，確保方向一致：計算兩個向量的內積
                v_orig = np.array([orig_wall2[0] - orig_wall1[0], orig_wall2[1] - orig_wall1[1]])
                v_filt = np.array([wall_p2[0] - wall_p1[0], wall_p2[1] - wall_p1[1]])
                if np.dot(v_orig, v_filt) < 0:
                    wall_p1, wall_p2 = wall_p2, wall_p1
                    rospy.loginfo("Swapped wall segment for object at (%.2f, %.2f)", x_obj, y_obj)
                pose = self.calculate_stop_pose(x_obj, y_obj, obj_type, wall_p1, wall_p2)
                stop_poses.append(pose)
        return stop_poses

    def merge_stop_poses(self, stop_poses, merge_threshold=0.3):
        """
        將計算得到的停靠點進行合併：
        若多個停靠點彼此間距離小於 merge_threshold，
        則將它們取平均合併成一個停靠點。
        返回合併後的停靠點列表，每個元素為 (stop_x, stop_y, orientation)
        """
        merged = []
        used = [False] * len(stop_poses)
        for i in range(len(stop_poses)):
            if used[i]:
                continue
            cluster = [stop_poses[i]]
            used[i] = True
            for j in range(i+1, len(stop_poses)):
                if used[j]:
                    continue
                if euclidean(stop_poses[i][:2], stop_poses[j][:2]) < merge_threshold:
                    cluster.append(stop_poses[j])
                    used[j] = True
            avg_x = np.mean([p[0] for p in cluster])
            avg_y = np.mean([p[1] for p in cluster])
            angles = [p[2] for p in cluster]
            avg_angle = math.atan2(np.mean(np.sin(angles)), np.mean(np.cos(angles)))
            merged.append((avg_x, avg_y, avg_angle))
        return merged

    def tsp_path_schedule(self, start_pose, stop_poses):
        """根據 TSP 計算從起始點到所有停靠點的最佳路徑順序（以最短距離為目標）。"""
        if not stop_poses:
            return []
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

    def publish_goal(self, stop_pose):
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
        self.pub.publish(goal_msg)
        rospy.loginfo("Published goal: ({:.2f}, {:.2f}, {:.2f} rad)".format(*stop_pose))

        marker = Marker()
        marker.header.stamp = rospy.Time.now()
        marker.header.frame_id = "map"
        marker.ns = "stop_poses"
        marker.id = self.marker_id
        self.marker_id += 1
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose.position.x = stop_pose[0]
        marker.pose.position.y = stop_pose[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        marker.scale.x = 0.8
        marker.scale.y = 0.05
        marker.scale.z = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)

    def process_and_plan(self, current_start_pose):
        """
        掃描 scan_time 秒後，直接呼叫 generate_stop_poses() 取得所有停靠點，
        並進行停靠點合併、TSP 路徑規劃與目標點發佈。
        """
        rospy.loginfo("Route planning started...")

        stop_poses = self.generate_stop_poses()
       
        self.object_dict.clear()

        if not stop_poses:
            rospy.loginfo("No detections to process")
            return

        rospy.loginfo("Generated %d stop poses", len(stop_poses))
        merged_stop_poses = self.merge_stop_poses(stop_poses, merge_threshold=0.4)
        rospy.loginfo("Merged to %d stop poses after filtering", len(merged_stop_poses))
        
        ordered_poses = self.tsp_path_schedule(current_start_pose, merged_stop_poses)
        if not ordered_poses:
            rospy.loginfo("No stop poses to plan")
            return
        
        rospy.loginfo("Publishing planned goals...")
        for pose in ordered_poses:
            self.publish_goal(pose)
            rospy.sleep(0.5)


current_start_pose = None

def amcl_pose_callback(msg):
    global current_start_pose
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    (_, _, yaw) = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    current_start_pose = (pos.x, pos.y, yaw)
    rospy.loginfo("AMCL Pose received: (%.4f, %.4f, %.2f rad)", pos.x, pos.y, yaw)


if __name__ == "__main__":
    rospy.init_node("disinfection_planner")
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)
    
    planner = DisinfectionPlanner()
    rospy.loginfo("Waiting for AMCL pose...")
    while current_start_pose is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    scan_time = rospy.get_param("~scan_time", 70.0)
    rospy.loginfo("Scanning for detections for %.1f seconds...", scan_time)
    rospy.sleep(scan_time)
    
    planner.process_and_plan(current_start_pose)
    
    rospy.spin()
