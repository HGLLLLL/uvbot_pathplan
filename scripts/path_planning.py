#!/usr/bin/env python
import math
import numpy as np
import rospy
from scipy.spatial.distance import euclidean
from itertools import permutations
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from collections import defaultdict
from uvbot_pathplan.msg import ObjectWallPosition  # 引入自定義訊息
from visualization_msgs.msg import Marker

class DisinfectionPlanner:
    def __init__(self, robot_width=0.6, safety_distance=0.2, cluster_threshold=0.5):
        self.robot_radius = robot_width / 2.0
        self.safety_distance = safety_distance
        # d: 停靠點沿牆面垂直方向的偏移距離
        self.d = self.robot_radius + self.safety_distance
        self.cluster_threshold = cluster_threshold 
        
        # 儲存所有收到的檢測數據，格式：
        # { type: [(x_obj, y_obj, wall_p1, wall_p2), ...] }
        self.object_dict = defaultdict(list)
        
        # 訂閱 /object_wall_position 訊息（型態為 ObjectWallPosition）
        rospy.Subscriber("/object_wall_position", ObjectWallPosition, self.object_wall_callback)
        self.pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=10)

        self.marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
        self.marker_id = 0


    def object_wall_callback(self, msg):
        """
        處理 /object_wall_position 的訂閱訊息
        訊息格式：
            string type
            float64 x_obj
            float64 y_obj
            float64 x_wall1
            float64 y_wall1
            float64 x_wall2
            float64 y_wall2
        此處僅將接收到的訊息原始存入 object_dict 中。
        """
        obj_type = msg.type.strip()
        x_obj = msg.x_obj
        y_obj = msg.y_obj
        wall_p1 = (msg.x_wall1, msg.y_wall1)
        wall_p2 = (msg.x_wall2, msg.y_wall2)
        
        self.object_dict[obj_type].append((x_obj, y_obj, wall_p1, wall_p2))
        rospy.loginfo("Received detection: type=%s, obj=(%.2f, %.2f)", obj_type, x_obj, y_obj)

    def calculate_wall_direction(self, wall_p1, wall_p2):
        """根據牆面兩端點計算牆面方向角及其垂直方向角"""
        wall_vec = np.array([wall_p1[0] - wall_p2[0], wall_p1[1] - wall_p2[1]])
        wall_angle = math.atan2(wall_vec[1], wall_vec[0])
        perp_angle = wall_angle + math.pi/2
        return wall_angle, perp_angle

    def calculate_stop_pose(self, obj_x, obj_y, obj_type, wall_p1, wall_p2):
        """
        根據物體與牆面資訊計算機器人的停靠點。
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

    def cluster_detections(self, det_list, cluster_threshold=None, outlier_factor=1.0):
        """
        對同一類型內所有檢測進行聚類（僅聚類位置接近的數據），
        並在每個聚類組內剔除離群值（與群組中心差距過大的數據）。
        每個元素展平成一個列表：[x_obj, y_obj, wall1_x, wall1_y, wall2_x, wall2_y]
        返回聚類後的列表，每個元素為 (x_obj, y_obj, wall_p1, wall_p2)。
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
            # 將該聚類中的每個檢測展平成一個列表，方便計算
            flat_cluster = []
            for elem in cluster:
                flat_cluster.append([elem[0], elem[1], elem[2][0], elem[2][1], elem[3][0], elem[3][1]])
            flat_cluster = np.array(flat_cluster)  # shape: (n,6)

            # 計算初始聚類中心（僅針對物體位置 x, y）
            center_x = np.mean(flat_cluster[:, 0])
            center_y = np.mean(flat_cluster[:, 1])
            # 計算每個點與中心的歐式距離
            dists = np.sqrt((flat_cluster[:, 0] - center_x)**2 + (flat_cluster[:, 1] - center_y)**2)
            # 定義離群值剔除閾值（例如：cluster_threshold 的一半）
            outlier_threshold = cluster_threshold * outlier_factor
            # 篩選出不超過閾值的點
            valid_indices = np.where(dists <= outlier_threshold)[0]
            if valid_indices.size > 0:
                flat_cluster = flat_cluster[valid_indices]
            # 以過濾後的數據重新計算平均值
            avg_obj_x = np.mean(flat_cluster[:, 0])
            avg_obj_y = np.mean(flat_cluster[:, 1])
            avg_wall1 = (np.mean(flat_cluster[:, 2]), np.mean(flat_cluster[:, 3]))
            avg_wall2 = (np.mean(flat_cluster[:, 4]), np.mean(flat_cluster[:, 5]))
            clusters.append((avg_obj_x, avg_obj_y, avg_wall1, avg_wall2))
        return clusters



    def generate_stop_poses(self):
        """
        根據 object_dict 中儲存的所有檢測數據，對每一類型進行一次聚類，
        然後依據每個聚類計算機器人的停靠點，
        返回一個列表，每個項目為 (stop_x, stop_y, orientation)。
        """
        stop_poses = []
        for obj_type, det_list in self.object_dict.items():
            # 根據檢測類型設置不同的聚類閾值
            if obj_type.lower() == 'chair':
                threshold = 1.3  # 椅子類型使用較大的聚類閾值
            elif obj_type.lower() == 'side rail':
                threshold = 0.4  # 側軌類型使用較小的聚類閾值
            else:
                threshold = self.cluster_threshold  # 其他類型使用預設值

            clusters = self.cluster_detections(det_list, threshold)
            for cluster in clusters:
                obj_x, obj_y, wall_p1, wall_p2 = cluster
                pose = self.calculate_stop_pose(obj_x, obj_y, obj_type, wall_p1, wall_p2)
                stop_poses.append(pose)
        return stop_poses
    
    def merge_stop_poses(self, stop_poses, merge_threshold=0.75):
        """
        將計算得到的停靠點進行合併：
        若多個停靠點彼此間距離小於 merge_threshold，
        則將它們取平均合併成一個停靠點。
        返回合併後的停靠點列表，每個元素為 (stop_x, stop_y, orientation)。
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
            # 計算位置平均
            avg_x = np.mean([p[0] for p in cluster])
            avg_y = np.mean([p[1] for p in cluster])
            # 平均角度採用向量平均法計算
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
        marker.scale.x = 0.8  # 箭桿長度
        marker.scale.y = 0.05 # 箭桿寬度
        marker.scale.z = 0.1 # 箭頭大小
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        # lifetime 設為 0 表示永久顯示
        marker.lifetime = rospy.Duration(0)
        self.marker_pub.publish(marker)

    def process_and_plan(self, current_start_pose):
        """
        掃描 scan_time 秒後，對收集到的所有 /object_wall_position 訊息進行一次聚類，
        並在每個聚類組內剔除離群值，生成最終的檢測點，再根據這些點計算機器人停靠點，
        若停靠點彼此過於接近則進行合併，最後依據 TSP 規劃路徑並依序發佈目標點。
        """
        rospy.loginfo("Route planning started...")
        aggregated = []
        for obj_type, det_list in self.object_dict.items():
            # 根據檢測類型設定不同的聚類閾值
            if obj_type.lower() == 'chair':
                threshold = 1.3
            elif obj_type.lower() == 'side rail':
                threshold = 0.4
            else:
                threshold = self.cluster_threshold

            clusters = self.cluster_detections(det_list, threshold)
            for cluster in clusters:
                obj_x, obj_y, wall_p1, wall_p2 = cluster
                aggregated.append((obj_type, obj_x, obj_y, wall_p1[0], wall_p1[1],
                                   wall_p2[0], wall_p2[1]))
        # 清空檢測數據，為下一次掃描做準備
        self.object_dict.clear()

        if not aggregated:
            rospy.loginfo("No detections to process")
            return

        stop_poses = []
        for item in aggregated:
            obj_type, x_obj, y_obj, wall1_x, wall1_y, wall2_x, wall2_y = item
            pose = self.calculate_stop_pose(x_obj, y_obj, obj_type, (wall1_x, wall1_y), (wall2_x, wall2_y))
            stop_poses.append(pose)
        rospy.loginfo("Generated %d stop poses", len(stop_poses))
        
        # 將彼此過於接近的停靠點進行合併
        merged_stop_poses = self.merge_stop_poses(stop_poses, merge_threshold=0.3)
        rospy.loginfo("Merged to %d stop poses after filtering", len(merged_stop_poses))
        
        ordered_poses = self.tsp_path_schedule(current_start_pose, merged_stop_poses)
        if not ordered_poses:
            rospy.loginfo("No stop poses to plan")
            return
        
        rospy.loginfo("Publishing planned goals...")
        for pose in ordered_poses:
            self.publish_goal(pose)
            rospy.sleep(0.5)


# -----------------------------------------------------------------------------
# Global current start pose (from AMCL)
current_start_pose = None

def amcl_pose_callback(msg):
    global current_start_pose
    pos = msg.pose.pose.position
    quat = msg.pose.pose.orientation
    (_, _, yaw) = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    current_start_pose = (pos.x, pos.y, yaw)
    rospy.loginfo("AMCL Pose received: (%.4f, %.4f, %.2f rad)", pos.x, pos.y, yaw)

# -----------------------------------------------------------------------------
# Main
if __name__ == "__main__":
    rospy.init_node("disinfection_planner")
    rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, amcl_pose_callback)
    
    planner = DisinfectionPlanner()
    rospy.loginfo("Waiting for AMCL pose...")
    while current_start_pose is None and not rospy.is_shutdown():
        rospy.sleep(0.1)
    
    # 設定掃描時間 (scan_time) 秒內收集 /object_wall_position 訊息
    scan_time = 80
    rospy.loginfo("Scanning for detections for %.1f seconds...", scan_time)
    rospy.sleep(scan_time)
    
    # 掃描結束後，處理並規劃路徑
    planner.process_and_plan(current_start_pose)
    
    rospy.spin()
