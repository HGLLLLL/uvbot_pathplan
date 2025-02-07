from ultralytics import YOLO
import os
import rospkg
import cv2

rospack = rospkg.RosPack()
package_path = rospack.get_path('uvbot_pathplan')
MODEL_PATH = os.path.join(package_path, 'scripts', 'no_background.pt')
# MODEL_PATH = "/scripts/no_background.pt"  
model = YOLO(MODEL_PATH) 


image_path =  os.path.join(package_path, 'bags', '6.jpg')


results = model(image_path)


for result in results:
    # Generate an image with annotations (bounding boxes, labels, etc.)
    annotated_img = result.plot()

    # Display the image in a window
    # cv2.imshow("Detection Results", annotated_img)

    cv2.namedWindow("Detection Results", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Detection Results", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    cv2.imshow("Detection Results", annotated_img)


    # Wait for a key press to close the window
    cv2.waitKey(0)

# Clean up and close the window
cv2.destroyAllWindows()


# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# import rospy
# import cv2 as cv
# import numpy as np
# import sensor_msgs.point_cloud2 as pc2
# from sensor_msgs.msg import Image, PointCloud2
# from cv_bridge import CvBridge, CvBridgeError
# import message_filters
# import sys
# from ultralytics import YOLO
# import os
# import rospkg
# import threading
# import queue

# # camera parameters from old code
# K = np.array([[615.186, 0, 330.623],
#               [0, 615.263, 239.772],
#               [0, 0, 1]], dtype=np.float32)
# D = np.zeros(5, dtype=np.float32)

# # load YOLO model
# rospack = rospkg.RosPack()
# package_path = rospack.get_path('uvbot_pathplan')
# MODEL_PATH = os.path.join(package_path, 'scripts', 'no_background.pt')
# model = YOLO(MODEL_PATH)

# bridge = CvBridge()

# # global variables for image display
# latest_display_image = None   
# window_name = "Realtime Scan Projection"

# # establish a queue for synchronizing image and scan messages
# processing_queue = queue.Queue()

# def sync_callback(image_msg, scan_msg):
#     # If the queue already has an item, remove it before adding the new one.
#     while not processing_queue.empty():
#         try:
#             processing_queue.get_nowait()
#         except queue.Empty:
#             break
#     processing_queue.put((image_msg, scan_msg))
#     rospy.loginfo("Enqueued new message pair (old frames discarded).")

# def processing_thread():
#     """
#     重運算線程：從 Queue 中取得 ROS 訊息,進行影像轉換、YOLO 推理與點雲投影，
#     並將處理後的影像存入全域變數 latest_display_image,供 UI 執行緒顯示。
#     """
#     global latest_display_image

#     # List of allowed object classes (in lower case)
#     allowed_classes = ["bed", "chair", "television", "handrail", "monitor","side rail"]

#     while not rospy.is_shutdown():
#         try:
#             # wait for new message pair
#             image_msg, scan_msg = processing_queue.get(timeout=1)
#         except queue.Empty:
#             continue  # wait for new message pair

#         rospy.loginfo("Processing thread: received new message pair.")

#         # image message to OpenCV image
#         try:
#             cv_image = bridge.imgmsg_to_cv2(image_msg, "bgr8")
#             rospy.loginfo("Image size: {} x {}".format(cv_image.shape[1], cv_image.shape[0]))
#         except CvBridgeError as e:
#             rospy.logerr("CvBridge Error: %s", e)
#             continue

#         # object detection with YOLO
#         results = model(cv_image)
#         for result in results:
#             for box in result.boxes:
#                 # get box coordinates and confidence
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = box.conf[0].item() if len(box.conf) > 0 else 0
#                 if conf < 0.7:
#                     continue
#                 label = result.names[int(box.cls[0])] if len(box.cls) > 0 else "obj"
#                 if label.lower() not in allowed_classes:
#                     continue
#                 # generate bounding box and label
#                 cv.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
#                 cv.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10), 
#                            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # point cloud projection
#         # transform PointCloud2 message to numpy array
#         points = np.array([p for p in pc2.read_points(scan_msg, field_names=("x", "y", "z"), skip_nans=True)], dtype=np.float32)
#         if points.size == 0:
#             rospy.loginfo("No valid scan points received.")
#         else:
#             # filter out points outside the valid range
#             valid_idx = (points[:, 2] > 0.0) & (points[:, 2] < 6.0)
#             points = points[valid_idx]
#             # 3D points to 2D pixel points
#             rvec = np.zeros((3, 1), dtype=np.float32)
#             tvec = np.zeros((3, 1), dtype=np.float32)
#             pixel_points, _ = cv.projectPoints(points, rvec, tvec, K, D)
#             pixel_points = pixel_points.squeeze()
#             # generate pixel points on the image
#             for pt in pixel_points:
#                 x, y = int(pt[0]), int(pt[1])
#                 if 0 <= x < cv_image.shape[1] and 0 <= y < cv_image.shape[0]:
#                     cv.circle(cv_image, (x, y), 3, (255, 0, 0), -1)

#         # update the latest display image
#         latest_display_image = cv_image

# def display_thread():
#     """
#     UI 執行緒：負責在 OpenCV 視窗中顯示 latest_display_image。
#     """
#     cv.namedWindow(window_name, cv.WINDOW_NORMAL)
#     while not rospy.is_shutdown():
#         if latest_display_image is not None:
#             cv.imshow(window_name, latest_display_image)

#         if cv.waitKey(1) & 0xFF == 27:
#             rospy.signal_shutdown("User pressed ESC")
#             break

# def main():
#     rospy.init_node("realtime_scan_projection", anonymous=True)

#     # start processing thread
#     proc_thread = threading.Thread(target=processing_thread)
#     proc_thread.daemon = True
#     proc_thread.start()

#     # start display thread
#     ui_thread = threading.Thread(target=display_thread)
#     ui_thread.daemon = True
#     ui_thread.start()

#     # synchronize image and scan messages
#     image_sub = message_filters.Subscriber("/camera/color/image_raw", Image)
#     scan_sub = message_filters.Subscriber("/scan_trans", PointCloud2)
#     ts = message_filters.ApproximateTimeSynchronizer([image_sub, scan_sub],
#                                                       queue_size=3,
#                                                       slop=0.05) #queue_size=5, slop=0.1
#     ts.registerCallback(sync_callback)
    
#     rospy.loginfo("Realtime scan projection node started.")
    
#     try:
#         rospy.spin()
#     except rospy.ROSInterruptException:
#         rospy.loginfo("ROS node interrupted.")
#     finally:
#         rospy.loginfo("Destroying OpenCV window...")
#         cv.destroyAllWindows()
#         sys.exit(0)

# if __name__ == '__main__':
#     main()


