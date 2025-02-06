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
    When a LaserScan message is received from the /scan topic, each point is transformed from the
    original frame (merged_laser_frame) to the target frame (tm_tool0_controller), 
    and the transformed points are published as PointCloud2 messages to the /scan_trans topic.
    """
    transformed_points = []  
    angle = scan_msg.angle_min

    # Check if the frame_id of LaserScan is 'merged_laser_frame'
    if scan_msg.header.frame_id != "merged_laser_frame":
        rospy.logwarn("LaserScan frame_id is not 'merged_laser_frame', it is: %s", scan_msg.header.frame_id)
        return

    for r in scan_msg.ranges:
        if math.isinf(r) or math.isnan(r):
            angle += scan_msg.angle_increment
            continue

        point_in_laser = PointStamped()
        point_in_laser.header.stamp = scan_msg.header.stamp
        point_in_laser.header.frame_id = scan_msg.header.frame_id  # 'merged_laser_frame'
        point_in_laser.point.x = r * math.cos(angle)
        point_in_laser.point.y = r * math.sin(angle)
        point_in_laser.point.z = 0.0

        try:
            # Lookup the transform from merged_laser_frame to tm_tool0_controller
            # transform = tfBuffer.lookup_transform("tm_tool0_controller",
            #                                       "merged_laser_frame",  # Source frame
            #                                       rospy.Time(0),  # Use the latest available transform
            #                                       rospy.Duration(1.0))
            transform = tfBuffer.lookup_transform("disinfect_cam",
                                                  "merged_laser_frame",  # Source frame
                                                  rospy.Time(0),  # Use the latest available transform
                                                  rospy.Duration(1.0))            
            # Transform the point from merged_laser_frame to tm_tool0_controller
            point_in_tool = tf2_geometry_msgs.do_transform_point(point_in_laser, transform)
            # print(transform)
 
            transformed_points.append([point_in_tool.point.x,
                                       point_in_tool.point.y,
                                       point_in_tool.point.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as ex:
            rospy.logwarn("Transform error: %s", ex)

        angle += scan_msg.angle_increment

    # rospy.loginfo("Transformed %d points", len(transformed_points))

    # Create PointCloud2 message
    header = Header()
    header.stamp = rospy.Time.now()
    # header.frame_id = "tm_tool0_controller"  # The new point cloud will be in this frame
    header.frame_id = "disinfect_cam"  # The new point cloud will be in this frame

    cloud_msg = pc2.create_cloud_xyz32(header, transformed_points)
    
    # Publish the transformed point cloud to the /scan_trans topic
    pub_cloud.publish(cloud_msg)

if __name__ == "__main__":
    rospy.init_node("scan_transformer", anonymous=True)
    
    # Initialize tf2 buffer and listener
    tfBuffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tfBuffer)
    
    # Subscribe to the /scan topic with LaserScan message type
    rospy.Subscriber("/scan", LaserScan, scan_callback)
    
    # Create a publisher to publish the transformed point cloud to /scan_trans topic
    pub_cloud = rospy.Publisher("/scan_trans", PointCloud2, queue_size=10)
    
    rospy.loginfo("scan_transformer node started, waiting for /scan messages...")
    rospy.spin()
