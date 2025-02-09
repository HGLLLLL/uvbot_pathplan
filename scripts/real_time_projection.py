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

# parameters from old code
K = np.array([[611.72529, 0, 323.12238],
              [0, 612.57867, 248.12445],
              [0, 0, 1]], dtype=np.float32)
D = np.zeros(5, dtype=np.float32)

bridge = CvBridge()
latest_display_image = None 
window_name = "Realtime Scan Projection"

def sync_callback(image_msg, scan_msg):

    global latest_display_image
    rospy.loginfo("Callback triggered") #test
    
    # convert image message to OpenCV image
    try:
        cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
        rospy.loginfo("Image size: {} x {}".format(cv_image.shape[1], cv_image.shape[0]))
    except CvBridgeError as e:
        rospy.logerr("CvBridge Error: %s", e)
        return

    # read scan points
    points = []
    for p in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True):
        points.append([p[0], p[1], p[2]])
    
    if len(points) == 0:
        rospy.loginfo("No valid scan points received")
        latest_display_image = cv_image
        return

    points = np.array(points, dtype=np.float32)

    z_min, z_max = 0.0, 6 # only take points within this range (meters)
    filtered_points = [p for p in points if z_min < p[2] < z_max]
    points = np.array(filtered_points, dtype=np.float32)

    # project 3D points to 2D image plane
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    pixel_points, _ = cv.projectPoints(points, rvec, tvec, K, D)
    pixel_points = pixel_points.squeeze() 

    # generate projected points on image
    for pt in pixel_points:
        x, y = int(pt[0]), int(pt[1])
        if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
            cv.circle(cv_image, (x, y), 3, (0, 255, 0), -1)  

    latest_display_image = cv_image  

def main():
    global latest_display_image

    rospy.init_node("realtime_scan_projection", anonymous=True)

    # sychronize image and scan messages
    image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
    scan_sub = message_filters.Subscriber("/scan_trans", PointCloud2)
    
    ts = message_filters.ApproximateTimeSynchronizer([image_sub, scan_sub], queue_size=10, slop=0.5)
    ts.registerCallback(sync_callback)
    
    rospy.loginfo("Realtime scan projection node started.")
    
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
