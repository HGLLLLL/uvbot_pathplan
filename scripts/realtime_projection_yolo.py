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

# camera parameters from old code
K = np.array([[615.186, 0, 330.623],
              [0, 615.263, 239.772],
              [0, 0, 1]], dtype=np.float32)
D = np.zeros(5, dtype=np.float32)

# load YOLO model
rospack = rospkg.RosPack()
package_path = rospack.get_path('uvbot_pathplan')
MODEL_PATH = os.path.join(package_path, 'scripts', 'no_background.pt')
# MODEL_PATH = "/scripts/no_background.pt"  
model = YOLO(MODEL_PATH)

bridge = CvBridge()
latest_display_image = None   
window_name = "Realtime Scan Projection"

def sync_callback(image_msg, scan_msg):
    global latest_display_image
    rospy.loginfo("Callback triggered")
    
    # transform image message to OpenCV image
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        rospy.loginfo("Image size: {} x {}".format(cv_image.shape[1], cv_image.shape[0]))
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)
        return

    # object detection
    results = model(cv_image)
    for result in results:

        for box in result.boxes:
            # get box coordinates and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item() if len(box.conf) > 0 else 0
            label = result.names[int(box.cls[0])] if len(box.cls) > 0 else "obj"
            # show bounding box and label
            cv.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10), 
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # read scan points (transformed numpy array)
    points = []
    for p in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    if len(points) == 0:
        rospy.loginfo("No valid scan points received")
        latest_display_image = cv_image
        return
    points = np.array(points, dtype=np.float32)

    # pick points within this range (meters)
    z_min, z_max = 0.0, 6.0
    filtered_points = [p for p in points if z_min < p[2] < z_max]
    points = np.array(filtered_points, dtype=np.float32)

    # 3D points projection to 2D image plane
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    pixel_points, _ = cv.projectPoints(points, rvec, tvec, K, D)
    pixel_points = pixel_points.squeeze()  # 轉為 (N, 2) 陣列

    # projected points on image
    for pt in pixel_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
            cv.circle(cv_image, (x, y), 3, (255, 0, 0), -1)

    # update display
    latest_display_image = cv_image

def main():
    global latest_display_image

    rospy.init_node("realtime_scan_projection", anonymous=True)

    # sync image and scan messages
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    scan_sub = message_filters.Subscriber("/scan_trans", PointCloud2)
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, scan_sub],
                                                      queue_size=10,
                                                      slop=0.5)
    ts.registerCallback(sync_callback)
    
    rospy.loginfo("Realtime scan projection node started.")
    
    # open OpenCV window
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    rate = rospy.Rate(30)  # 30 Hz rate
    try:
        while not rospy.is_shutdown():
            if latest_display_image is not None:
                cv.imshow(window_name, latest_display_image)
            if cv.waitKey(1) & 0xFF == 27:  # esc to exit
                rospy.loginfo("Closing OpenCV window.")
                break
            rate.sleep()
    
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS node interrupted.")
    
    except Exception as e:
        rospy.logerr("Unexpected error: %s", str(e))
    
    finally:
        rospy.loginfo("Destroying OpenCV window...")
        cv.destroyAllWindows()
        sys.exit(0)

if __name__ == '__main__':
    main()
