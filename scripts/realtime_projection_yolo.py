#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2 as cv
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
import sys
from ultralytics import YOLO
import os
import rospkg
import threading
import queue

# --------------------
# camera parameters from old code
K = np.array([[615.186, 0, 330.623],
              [0, 615.263, 239.772],
              [0, 0, 1]], dtype=np.float32)
D = np.zeros(5, dtype=np.float32)

# --------------------
# load yolo model
rospack = rospkg.RosPack()
package_path = rospack.get_path('uvbot_pathplan')
MODEL_PATH = os.path.join(package_path, 'scripts', 'no_background.pt')
model = YOLO(MODEL_PATH)

bridge = CvBridge()

# --------------------
# global variables
latest_display_image = None   
window_name = "Realtime Scan Projection"
processing_queue = queue.Queue()

def sync_callback(image_msg, scan_msg):
    """
    將收到的影像與光達訊息對放入 Queue，
    若 Queue 中已有資料，先捨棄以避免延遲。
    """
    while not processing_queue.empty():
        try:
            processing_queue.get_nowait()
        except queue.Empty:
            break
    processing_queue.put((image_msg, scan_msg))
    rospy.loginfo("Enqueued new message pair (old frames discarded).")

def processing_thread():
    """
    重運算線程：
    1. 取得 ROS 訊息後進行影像轉換、YOLO 物件偵測，
    2. 讀取點雲並將其投影至影像平面，並取得對應的 2D 世界座標（從光達數據中取出）。
    3. 對於每個偵測框，不論內部是否有光達點，皆取出該框內與框下方一定區域內的光達點，
       然後過濾離群值，再計算平均作為該物體在 2D 平面上的位置，並標示在影像上。
    """
    global latest_display_image

    allowed_classes = ["bed", "chair", "television", "handrail", "monitor", "side rail"]
    # filter strange points
    outlier_threshold = 0.5
    # look down
    vertical_offset = 100

    while not rospy.is_shutdown():
        try:
            image_msg, scan_msg = processing_queue.get(timeout=1)
        except queue.Empty:
            continue

        rospy.loginfo("Processing thread: received new message pair.")

        # image message to OpenCV image
        try:
            cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
            rospy.loginfo("Image size: {} x {}".format(cv_image.shape[1], cv_image.shape[0]))
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)
            continue

        # yolo object detection
        results = model(cv_image)

        # scan point cloud projection
        points = np.array([p for p in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=np.float32)
        if points.size == 0:
            rospy.loginfo("No valid scan points received.")
            pixel_points = None
            world_points = None
        else:
            # filter points with z values outside the range [0, 6]
            valid_idx = (points[:, 2] > 0.0) & (points[:, 2] < 6.0)
            points = points[valid_idx]
            rvec = np.zeros((3, 1), dtype=np.float32)
            tvec = np.zeros((3, 1), dtype=np.float32)
            pixel_points, _ = cv.projectPoints(points, rvec, tvec, K, D)
            pixel_points = pixel_points.squeeze()
            if len(pixel_points.shape) == 1:
                pixel_points = pixel_points.reshape(1, 2)
            # take only x and y coordinates
            world_points = points[:, :2]

        # calculate object position of each detected object
        for result in results:
            for box in result.boxes:
                # get box coordinates and confidence
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item() if len(box.conf) > 0 else 0
                if conf < 0.7:
                    continue
                label = result.names[int(box.cls[0])] if len(box.cls) > 0 else "obj"
                if label.lower() not in allowed_classes:
                    continue

                # generate bounding box and label
                cv.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # take points in my chosen area
                obj_position = None
                if (pixel_points is not None) and (world_points is not None):
                    # points inside the bounding box
                    indices_in_box = np.where((pixel_points[:, 0] >= x1) & (pixel_points[:, 0] <= x2) &
                                                (pixel_points[:, 1] >= y1) & (pixel_points[:, 1] <= y2))[0]
                    # points below the bounding box(look down)
                    search_y1 = y2
                    search_y2 = min(y2 + vertical_offset, cv_image.shape[0] - 1)
                    indices_below = np.where((pixel_points[:, 0] >= x1) & (pixel_points[:, 0] <= x2) &
                                              (pixel_points[:, 1] >= search_y1) & (pixel_points[:, 1] <= search_y2))[0]
                    # merge two sets of points
                    selected_indices = np.union1d(indices_in_box, indices_below)
                    if selected_indices.size > 0:
                        selected_points = world_points[selected_indices]
                        # filter strange points
                        mean_point = np.mean(selected_points, axis=0)
                        distances = np.linalg.norm(selected_points - mean_point, axis=1)
                        filtered_points = selected_points[distances < outlier_threshold]
                        if filtered_points.shape[0] > 0:
                            obj_position = np.mean(filtered_points, axis=0)
                        else:
                            obj_position = mean_point

                if obj_position is not None:
                    # get the position of the object(need to be changed)
                    cv.putText(cv_image, f"Pos: ({obj_position[0]:.2f}, {obj_position[1]:.2f})",
                               (x1, y2 + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # scan projection
        if pixel_points is not None:
            for pt in pixel_points:
                x, y = int(pt[0]), int(pt[1])
                if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
                    cv.circle(cv_image, (x, y), 3, (255, 0, 0), -1)

        # update the latest display image
        latest_display_image = cv_image

def display_thread():
    """
    UI thread: display latest_display_image in OpenCV window.
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    while not rospy.is_shutdown():
        if latest_display_image is not None:
            cv.imshow(window_name, latest_display_image)
        if cv.waitKey(1) & 0xFF == 27:
            rospy.signal_shutdown("User pressed ESC")
            break

def main():
    rospy.init_node("realtime_scan_projection", anonymous=True)

    # start processing thread
    proc_thread = threading.Thread(target=processing_thread)
    proc_thread.daemon = True
    proc_thread.start()

    # start display thread
    ui_thread = threading.Thread(target=display_thread)
    ui_thread.daemon = True
    ui_thread.start()

    # synchronize image and scan messages
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
