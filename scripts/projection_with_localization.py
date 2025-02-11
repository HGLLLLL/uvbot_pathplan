#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import sys
from ultralytics import YOLO
import os
import rospkg
import threading
import queue
import tf

from uvbot_pathplan.msg import ObjectWallPosition

pub_obj_wall_pos = rospy.Publisher('/object_wall_position', ObjectWallPosition, queue_size=10)
# --------------------
# Camera intrinsic parameters
K = np.array([[611.72529, 0, 323.12238],
              [0, 612.57867, 248.12445],
              [0, 0, 1]], dtype=np.float32)
D = np.zeros(5, dtype=np.float32)

# --------------------
# Load YOLO model
rospack = rospkg.RosPack()
package_path = rospack.get_path('uvbot_pathplan')
MODEL_PATH = os.path.join(package_path, 'scripts', 'hospital.pt')
model = YOLO(MODEL_PATH)

bridge = CvBridge()

# --------------------
# Global variables
latest_display_image = None   
window_name = "Realtime Scan Projection"
processing_queue = queue.Queue()

tf_listener = None

def sync_callback(image_msg, scan_msg):
    """
    Put the incoming image and point cloud pair into the queue.
    If the queue already has data, discard old messages to avoid delay.
    """
    while not processing_queue.empty():
        try:
            processing_queue.get_nowait()
        except queue.Empty:
            break
    processing_queue.put((image_msg, scan_msg))

    #test
    # rospy.loginfo("Enqueued new message pair (old frames discarded).")

def processing_thread():
    """
    Processing thread:
      1. Convert the ROS image to an OpenCV image.
      2. Run YOLO v8 detection.
      3. Read the point cloud and project it onto the image plane.
         Also, extract the full 3D coordinates (x,y,z) from the point cloud.
      4. For each detected bounding box, choose the points i want to use.
      5. Filter out strange points and compute the average of the remaining points.
      6. Transform the calculated result from the camera coordinate to the map coordinate 
         and display result next to the bounding box.
    """
    global latest_display_image, tf_listener

    allowed_classes = ["chair", "television", "handrail", "monitor", "side rail"]
    outlier_threshold = 0.5  # in meters
    vertical_offset = 150    # vertical extension in pixels for the region below the box
    margin = 5              # margin to the image border

    # optional: wait for tf_listener to initialize properly
    # rospy.sleep(0.08) # (seconds)

    while not rospy.is_shutdown():
        try:
            image_msg, scan_msg = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        rospy.loginfo("Processing thread: received new message pair.")

        # 1. Convert image to OpenCV image.
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            img_h, img_w = cv_image.shape[:2]
            rospy.loginfo("Image size: {} x {}".format(cv_image.shape[1], cv_image.shape[0]))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            continue

        # 2. YOLO object detection.
        results = model(cv_image)

        # 3. Process point cloud.
        points = np.array([p for p in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=np.float32)
        if points.size == 0:
            rospy.loginfo("No valid scan points received.")
            pixel_points = None
            world_points = None
        else:
            valid_idx = (points[:, 2] > 0.0) & (points[:, 2] < 6.0)
            points = points[valid_idx]
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.zeros((3, 1), dtype=np.float32)
            pixel_points, _ = cv.projectPoints(points, rvec, tvec, K, D)
            pixel_points = pixel_points.squeeze()
            if len(pixel_points.shape) == 1:
                pixel_points = pixel_points.reshape(1, 2)
            world_points = points[:, :3]

        # 4. For each detected object, compute its 3D position (in camera coordinate).
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item() if len(box.conf) > 0 else 0
                if conf < 0.85:
                    continue
                label = result.names[int(box.cls[0])] if len(box.cls) > 0 else "obj"
                if label.lower() not in allowed_classes:
                    continue
                # Check if the bounding box is complete.
                if x1 < margin or y1 < margin or x2 > (img_w - margin) or y2 > (img_h - margin):
                    # Skip bounding boxes that are too close to the image border.
                    continue

                cv.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # start to calculate object position (offers two ways to choose points)

                # take points in my chosen area
                # obj_position_cam = None
                # if (pixel_points is not None) and (world_points is not None):
                #     # points inside the bounding box
                #     indices_in_box = np.where((pixel_points[:, 0] >= x1) & (pixel_points[:, 0] <= x2) &
                #                                 (pixel_points[:, 1] >= y1) & (pixel_points[:, 1] <= y2))[0]
                #     # points below the bounding box(look down)
                #     search_y1 = y2
                #     search_y2 = min(y2 + vertical_offset, cv_image.shape[0] - 1)
                #     indices_below = np.where((pixel_points[:, 0] >= x1) & (pixel_points[:, 0] <= x2) &
                #                               (pixel_points[:, 1] >= search_y1) & (pixel_points[:, 1] <= search_y2))[0]
                #     # merge two sets of points
                #     selected_indices = np.union1d(indices_in_box, indices_below)
                #     if selected_indices.size > 0:
                #         selected_points = world_points[selected_indices]
                #         # filter strange points
                #         mean_point = np.mean(selected_points, axis=0)
                #         distances = np.linalg.norm(selected_points - mean_point, axis=1)
                #         filtered_points = selected_points[distances < outlier_threshold]
                #         if filtered_points.shape[0] > 0:
                #             obj_position_cam = np.mean(filtered_points, axis=0)
                #         else:
                #             obj_position_cam = mean_point

                
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                obj_position_cam = None
                if (pixel_points is not None) and (world_points is not None):
                    selected_indices = np.where(
                        (pixel_points[:, 0] >= center_x - 20) & (pixel_points[:, 0] <= center_x + 20) &
                        (pixel_points[:, 1] >= center_y) & (pixel_points[:, 1] <= y2 + 100)
                    )[0]
                    if selected_indices.size > 0:
                        selected_points = world_points[selected_indices]
                        mean_point = np.mean(selected_points, axis=0)
                        distances = np.linalg.norm(selected_points - mean_point, axis=1)
                        filtered_points = selected_points[distances < outlier_threshold]
                        if filtered_points.shape[0] > 0:
                            obj_position_cam = np.mean(filtered_points, axis=0)
                        else:
                            obj_position_cam = mean_point

                if obj_position_cam is not None:
                # Transform point from camera frame to map frame.
                    point_cam = PointStamped()
                    point_cam.header.stamp = rospy.Time(0)  
                    point_cam.header.frame_id = "disinfect_cam" 
                    point_cam.point.x = float(obj_position_cam[0])
                    point_cam.point.y = float(obj_position_cam[1])
                    point_cam.point.z = float(obj_position_cam[2])
                    if tf_listener.canTransform("map", point_cam.header.frame_id, rospy.Time(0)):
                        point_map = tf_listener.transformPoint("map", point_cam)
                        map_x = point_map.point.x
                        map_y = point_map.point.y
                        cv.putText(cv_image, f"Map: ({map_x:.2f}, {map_y:.2f})",
                                    (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        rospy.logwarn("Can't get scan points of this bounding box")
                        continue

                    # calcualte wall position
                    if tf_listener.canTransform("map", "disinfect_cam", rospy.Time(0)):
                        # define the region of interest (ROI) for wall detection
                        # left boundary ROI: x range [x1-10, x1+10], y range [y1, y2+vertical_offset]
                        left_x_min = x1 - 10
                        left_x_max = x1 + 10
                        roi_left_indices = np.where(
                            (pixel_points[:, 0] >= left_x_min) & (pixel_points[:, 0] <= left_x_max) &
                            (pixel_points[:, 1] >= y1) & (pixel_points[:, 1] <= y2 + vertical_offset)
                        )[0]
                        wall_left_cam = None
                        if roi_left_indices.size > 0:
                            left_points = world_points[roi_left_indices]
                            mean_left = np.mean(left_points, axis=0)
                            distances_left = np.linalg.norm(left_points - mean_left, axis=1)
                            filtered_left = left_points[distances_left < outlier_threshold]
                            if filtered_left.shape[0] > 0:
                                wall_left_cam = np.mean(filtered_left, axis=0)
                            else:
                                wall_left_cam = mean_left

                        # right boundary ROI: x range [x2-10, x2+10], y range [y1, y2+vertical_offset]
                        right_x_min = x2 - 10
                        right_x_max = x2 + 10
                        roi_right_indices = np.where(
                            (pixel_points[:, 0] >= right_x_min) & (pixel_points[:, 0] <= right_x_max) &
                            (pixel_points[:, 1] >= y1) & (pixel_points[:, 1] <= y2 + vertical_offset)
                        )[0]
                        wall_right_cam = None
                        if roi_right_indices.size > 0:
                            right_points = world_points[roi_right_indices]
                            mean_right = np.mean(right_points, axis=0)
                            distances_right = np.linalg.norm(right_points - mean_right, axis=1)
                            filtered_right = right_points[distances_right < outlier_threshold]
                            if filtered_right.shape[0] > 0:
                                wall_right_cam = np.mean(filtered_right, axis=0)
                            else:
                                wall_right_cam = mean_right

                        # Transform left boundary point
                        if wall_left_cam is not None:
                            point_left = PointStamped()
                            point_left.header.stamp = rospy.Time(0)
                            point_left.header.frame_id = "disinfect_cam" 
                            point_left.point.x = float(wall_left_cam[0])
                            point_left.point.y = float(wall_left_cam[1])
                            point_left.point.z = float(wall_left_cam[2])
                            if tf_listener.canTransform("map", point_left.header.frame_id, rospy.Time(0)):
                                point_left_map = tf_listener.transformPoint("map", point_left)
                                map_left_x = point_left_map.point.x
                                map_left_y = point_left_map.point.y
                            else:
                                rospy.logwarn("Can't get left wall scan points。")
                                map_left_x = map_left_y = None

                        # Transform right boundary point
                        if wall_right_cam is not None:
                            point_right = PointStamped()
                            point_right.header.stamp = rospy.Time(0)
                            point_right.header.frame_id = "disinfect_cam"
                            point_right.point.x = float(wall_right_cam[0])
                            point_right.point.y = float(wall_right_cam[1])
                            point_right.point.z = float(wall_right_cam[2])
                            if tf_listener.canTransform("map", point_right.header.frame_id, rospy.Time(0)):
                                point_right_map = tf_listener.transformPoint("map", point_right)
                                map_right_x = point_right_map.point.x
                                map_right_y = point_right_map.point.y
                            else:
                                rospy.logwarn("Can' get right wall scan points.")
                                map_right_x = map_right_y = None

                        # Show wall positions on the image
                        if map_left_x is not None and map_right_x is not None:
                            cv.putText(cv_image, f"Wall: ({map_left_x:.2f}, {map_left_y:.2f}), ({map_right_x:.2f}, {map_right_y:.2f})",
                                    (x1, y2 - 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            
                        # Publish object and wall positions
                        msg = ObjectWallPosition()
                        msg.type = label
                        msg.x_obj = map_x
                        msg.y_obj = map_y
                        msg.x_wall1 = map_left_x if 'map_left_x' in locals() else 0
                        msg.y_wall1 = map_left_y if 'map_left_y' in locals() else 0
                        msg.x_wall2 = map_right_x if 'map_right_x' in locals() else 0
                        msg.y_wall2 = map_right_y if 'map_right_y' in locals() else 0

                        pub_obj_wall_pos.publish(msg)
                                                    
                    else:
                        rospy.logwarn("Map transform for wall positions not available, skipping wall computation.")

        if pixel_points is not None:
            for pt in pixel_points:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
                    cv.circle(cv_image, (x, y), 3, (255, 0, 0), -1)

        latest_display_image = cv_image

def display_thread():
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    while not rospy.is_shutdown():
        if latest_display_image is not None:
            cv.imshow(window_name, latest_display_image)
        if cv.waitKey(1) & 0xFF == 27:
            rospy.signal_shutdown("User pressed ESC")
            break

def main():
    global tf_listener
    rospy.init_node("realtime_scan_projection", anonymous=True)
    
    # Create the tf listener after node initialization.
    tf_listener = tf.TransformListener() 

    proc_thread = threading.Thread(target=processing_thread)
    proc_thread.daemon = True
    proc_thread.start()

    ui_thread = threading.Thread(target=display_thread)
    ui_thread.daemon = True
    ui_thread.start()

    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    scan_sub = message_filters.Subscriber("/scan_trans", PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, scan_sub],
                                                      queue_size=3,
                                                      slop=0.05)
    ts.registerCallback(sync_callback)
    
    rospy.loginfo("Realtime scan projection node started.")
    
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    finally:
        rospy.loginfo("Destroying OpenCV window...")
        cv.destroyAllWindows()
        sys.exit(0)

if __name__ == '__main__':
    main()
