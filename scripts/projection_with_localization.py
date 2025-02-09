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
MODEL_PATH = os.path.join(package_path, 'scripts', 'no_background.pt')
model = YOLO(MODEL_PATH)

bridge = CvBridge()

# --------------------
# Global variables
latest_display_image = None   
window_name = "Realtime Scan Projection"
processing_queue = queue.Queue()

# We now declare the tf_listener as None.
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
    # rospy.loginfo("Enqueued new message pair (old frames discarded).")

def processing_thread():
    """
    Processing thread:
      1. Convert the ROS image to an OpenCV image.
      2. Run YOLO object detection.
      3. Read the point cloud and project it onto the image plane.
         Also, extract the full 3D coordinates (x,y,z) from the point cloud.
      4. For each detected bounding box, regardless of whether points exist inside,
         select points that fall within a region defined relative to the box’s center:
             - Horizontally: from (center_x - 10) to (center_x + 10)
             - Vertically: from center_y to (center_y + 100)
      5. Filter out outlier points and compute the average of the remaining points.
      6. Transform the resulting 3D position from the camera coordinate system to the map coordinate system.
      7. Finally, display the transformed coordinates next to the bounding box.
    """
    global latest_display_image, tf_listener

    allowed_classes = ["bed", "chair", "television", "handrail", "monitor", "side rail"]
    outlier_threshold = 0.5  # in meters
    vertical_offset = 100    # vertical extension in pixels for the region below the box

    # Give tf_listener some time to get transforms.
    rospy.sleep(1.0)

    while not rospy.is_shutdown():
        try:
            image_msg, scan_msg = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        rospy.loginfo("Processing thread: received new message pair.")

        # 1. Convert image to OpenCV image.
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
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

        # 4. For each detected object, compute its 3D position.
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item() if len(box.conf) > 0 else 0
                if conf < 0.1:
                    continue
                label = result.names[int(box.cls[0])] if len(box.cls) > 0 else "obj"
                if label.lower() not in allowed_classes:
                    continue

                cv.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # start to calculate object position
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
                    # 6. Transform point from camera frame to map frame.
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
                        rospy.logwarn("Transform not available, skipping map coordinate display.")

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
    # rospy.sleep(1.0)  # Allow tf_listener to initialize properly 

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
