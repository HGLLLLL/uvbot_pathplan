#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import tf
import numpy as np
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped
from laser_line_extraction.msg import LineSegmentList

# Import the custom message for a filtered (merged) segment.
# Replace 'your_package' with your actual package name.
from uvbot_pathplan.msg import FilteredSegment

from sklearn.cluster import DBSCAN

# Global variable to store transformed line segments (in map frame)
# Each segment is stored as [start_x, start_y, end_x, end_y]
line_segments = []
scanning = True
tf_listener = None

def transform_point(listener, point, from_frame, to_frame):
    """
    Transform a 2D point (with z=0) from the 'from_frame' to 'to_frame'
    """
    pt_stamped = PointStamped()
    pt_stamped.header.frame_id = from_frame
    pt_stamped.header.stamp = rospy.Time(0)  # Latest available transform
    pt_stamped.point.x = point[0]
    pt_stamped.point.y = point[1]
    pt_stamped.point.z = 0.0
    try:
        transformed = listener.transformPoint(to_frame, pt_stamped)
        return [transformed.point.x, transformed.point.y]
    except Exception as e:
        rospy.logerr("TF transformation failed: %s", e)
        return None

def line_segment_callback(msg):
    global line_segments, scanning, tf_listener
    if not scanning:
        return
    # For each segment in the message (field: line_segments)
    for segment in msg.line_segments:
        start = segment.start  # [x, y]
        end   = segment.end    # [x, y]
        # Immediately transform from "merged_laser_frame" to "map"
        t_start = transform_point(tf_listener, [start[0], start[1]], "merged_laser_frame", "map")
        t_end   = transform_point(tf_listener, [end[0], end[1]], "merged_laser_frame", "map")
        if t_start is not None and t_end is not None:
            line_segments.append([t_start[0], t_start[1], t_end[0], t_end[1]])

def merge_segments(segments):
    """
    Merge segments that belong to the same complete line.
    
    For each segment, we compute a feature vector [2*angle, r]:
      - The segment's angle is computed (in [0, pi)) so that reversed segments are equivalent.
      - A normal vector n = [-sin(angle), cos(angle)] is defined and r is computed as dot(midpoint, n).
      - The angle dimension is scaled (multiplied by 2) for better balancing.
      
    DBSCAN clusters segments with similar features. Then, for each cluster, all endpoints are collected,
    and PCA is performed to find the principal (line) direction. The endpoints are projected onto this direction.
    
    The projected points are then grouped into contiguous clusters if the gap between consecutive points
    exceeds a threshold (~max_gap, in meters, defined by a ROS parameter). Each group is merged separately,
    so that segments far apart along the line are not merged together.
    
    Returns:
      A list of merged segments, each as a tuple: (start, end) with start and end being [x, y].
    """
    if len(segments) == 0:
        rospy.loginfo("No segments to merge!")
        return []
    
    # Build feature vectors for each segment.
    features = []
    for seg in segments:
        start = np.array(seg[0:2])
        end   = np.array(seg[2:4])
        midpoint = (start + end) / 2.0
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = np.arctan2(dy, dx)
        if angle < 0:
            angle += np.pi
        # Normal vector and r value.
        n = np.array([-np.sin(angle), np.cos(angle)])
        r = np.dot(midpoint, n)
        features.append([2 * angle, r])
    features = np.array(features)
    
    # Cluster segments based on line orientation and offset.
    clustering = DBSCAN(eps=0.15, min_samples=1).fit(features) # eps=0.28, min_samples=1
    labels = clustering.labels_
    
    merged_segments = []
    # Get the maximum gap allowed between points (in meters) along the line.
    max_gap = rospy.get_param("~max_gap", 1.0)
    
    # Process each cluster from DBSCAN.
    for label in set(labels):
        indices = np.where(labels == label)[0]
        # Collect all endpoints from segments in the cluster.
        cluster_points = []
        for i in indices:
            seg = segments[i]
            cluster_points.append(seg[0:2])
            cluster_points.append(seg[2:4])
        cluster_points = np.array(cluster_points)
        # Compute the mean and use SVD to obtain the principal direction.
        mean = np.mean(cluster_points, axis=0)
        centered = cluster_points - mean
        u, s, vh = np.linalg.svd(centered)
        direction = vh[0]  # Principal direction
        
        # Project endpoints onto the principal direction.
        projections = np.dot(centered, direction)
        # Sort projections and corresponding points.
        sorted_indices = np.argsort(projections)
        sorted_projs = projections[sorted_indices]
        sorted_points = cluster_points[sorted_indices]
        
        # Group adjacent projections that are within max_gap.
        groups = []
        current_group_projs = [sorted_projs[0]]
        for i in range(1, len(sorted_projs)):
            if sorted_projs[i] - sorted_projs[i-1] > max_gap:
                groups.append(current_group_projs)
                current_group_projs = [sorted_projs[i]]
            else:
                current_group_projs.append(sorted_projs[i])
        groups.append(current_group_projs)
        
        # For each group, determine extreme endpoints.
        for group in groups:
            group = np.array(group)
            min_proj = np.min(group)
            max_proj = np.max(group)
            p_min = mean + min_proj * direction
            p_max = mean + max_proj * direction
            merged_segments.append((p_min.tolist(), p_max.tolist()))
    
    return merged_segments

def publish_marker(marker_pub, segments):
    """
    Publish a Marker.LINE_LIST message showing each merged segment in yellow on RViz.
    """
    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.Time.now()
    marker.ns = "merged_line_segments"
    marker.id = 0
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = 0.05  # Line width
    marker.color.r = 1.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    marker.color.a = 1.0
    for seg in segments:
        start, end = seg
        marker.points.append(Point(x=start[0], y=start[1], z=0.0))
        marker.points.append(Point(x=end[0], y=end[1], z=0.0))
    marker.lifetime = rospy.Duration(0)
    marker_pub.publish(marker)

def main():
    global scanning, tf_listener, line_segments
    rospy.init_node('line_segment_merge_node')
    
    # Get scan_time (seconds) from the parameter server; default is 10 seconds.
    scan_time = rospy.get_param("~scan_time", 30.0)
    
    # Initialize TF listener and wait briefly for TF data.
    tf_listener = tf.TransformListener()
    rospy.sleep(1.0)
    
    # Subscribe to /line_segments (message type: LineSegmentList)
    rospy.Subscriber("/line_segments", LineSegmentList, line_segment_callback)
    
    marker_pub = rospy.Publisher("visualization_marker", Marker, queue_size=10)
    filtered_seg_pub = rospy.Publisher("filtered_segments", FilteredSegment, queue_size=10)
    
    rospy.loginfo("Collecting line_segment data for %.2f seconds...", scan_time)
    rospy.sleep(scan_time)
    scanning = False
    
    # Merge segments into complete lines.
    merged_segments = merge_segments(line_segments)
    
    # Log the number of final merged segments.
    rospy.loginfo("Final merged segments count: %d", len(merged_segments))
    
    if not merged_segments:
        rospy.loginfo("No merged segments obtained!")
        return
    
    rospy.loginfo("Publishing markers and merged segment messages...")
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        publish_marker(marker_pub, merged_segments)
        # Publish each merged segment as a FilteredSegment message.
        for seg in merged_segments:
            filtered_msg = FilteredSegment()
            filtered_msg.x_start = seg[0][0]
            filtered_msg.y_start = seg[0][1]
            filtered_msg.x_end   = seg[1][0]
            filtered_msg.y_end   = seg[1][1]
            filtered_seg_pub.publish(filtered_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass
