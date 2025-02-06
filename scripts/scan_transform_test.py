#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf2_ros
import tf2_geometry_msgs
import math
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import LaserScan, PointCloud2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Header

tfBuffer = None

pub_cloud = None

def scan_callback(scan_msg):
    """
    當接收到 /scan 主題的 LaserScan 訊息時，依序將每個測量點從 base_link 轉換到 tm_tool0_controller 座標系，
    並以 PointCloud2 訊息發布到 /scan_trans 主題。
    """
    transformed_points = []  # store transformed points (x, y, z)
    angle = scan_msg.angle_min

    # if scan_msg.header.frame_id != "base_link":
    #     rospy.logwarn("LaserScan frame_id is not 'base_link', it is: %s", scan_msg.header.frame_id)
    #     return

    for r in scan_msg.ranges:

        if math.isinf(r) or math.isnan(r):
            angle += scan_msg.angle_increment
            continue

        # 在 base_link 座標系中建立一個點 (注意：LaserScan 的 frame_id 應為 "base_link")
        point_in_base = PointStamped()
        point_in_base.header.stamp = scan_msg.header.stamp
        point_in_base.header.frame_id = scan_msg.header.frame_id
        # point_in_base.header.frame_id = "base_link"
        point_in_base.point.x = r * math.cos(angle)
        point_in_base.point.y = r * math.sin(angle)
        point_in_base.point.z = 0.0

        try:
            # 查詢從 base_link 到 tm_tool0_controller 的轉換
            transform = tfBuffer.lookup_transform("tm_tool0_controller",
                                                  "base_link",  # 來源座標系
                                                  rospy.Time(0),  # 使用最新可用轉換
                                                  rospy.Duration(1.0))
            # 將點從 base_link 轉換到 tm_tool0_controller 座標系
            point_in_tool = tf2_geometry_msgs.do_transform_point(point_in_base, transform)
            # 將轉換後的點加入列表 (提取 x, y, z)
            transformed_points.append([point_in_tool.point.x,
                                       point_in_tool.point.y,
                                       point_in_tool.point.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn("Transform error: %s", ex)
        
        angle += scan_msg.angle_increment

    # rospy.loginfo("Transformed %d points", len(transformed_points))
    # rospy.loginfo(f"Transformed {len(transformed_points)} points")
    rospy.loginfo("Transformed %d points" % len(transformed_points))

    
    # 建立 PointCloud2 訊息
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "tm_tool0_controller"  # 新點雲以此 frame 表示

    cloud_msg = pc2.create_cloud_xyz32(header, transformed_points)
    
    # 發布轉換後的點雲訊息到 /scan_trans 主題
    pub_cloud.publish(cloud_msg)

if __name__ == "__main__":
    rospy.init_node("scan_transformer", anonymous=True)
    
    # 建立 tf2 緩衝區與監聽器
    tfBuffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tfBuffer)
    
    # 訂閱 /scan 主題，訊息型態為 LaserScan
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    
    # 建立發布器，發布轉換後的點雲訊息到 /scan_trans 主題
    pub_cloud = rospy.Publisher("/scan_trans", PointCloud2, queue_size=10)
    
    rospy.loginfo("scan_transformer 節點啟動，等待 /scan 訊息...")
    rospy.spin()


