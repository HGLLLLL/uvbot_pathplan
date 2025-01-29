#!/usr/bin/env python3
import rospy
from std_msgs.msg import Header
from ball_detect.msg import BoundingBox, BoundingBoxArray
import cv2
import threading
from ObjectDetection import ObjectDetection
from Camera import Monitor

class ObjectDetectionPublisher:
    def __init__(self, camera_id, yolo_model_path):
        self.camera_id = camera_id
        
        self.monitor = Monitor(camera_id)
        self.object_detector = ObjectDetection(yolo_model_path)
        self.pub = rospy.Publisher('object_detection', BoundingBoxArray, queue_size=10)
        self.monitor.start()

    def run(self):
        while not rospy.is_shutdown():
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.monitor.Exit()
                break

            # Get the frame from the camera thread 
            frame = self.monitor.get_frame()  

            # Check if the frame is valid
            if frame is not None and frame.size > 0:
                #cv2.imshow('frame', frame)

                # Processing the image
                frame_processed, detections = self.object_detector(frame)
                #check1:
                for detection in detections:
                    print("check")
                    print(detection)
                cv2.imshow('Object Detection', frame_processed)

                # Prepare BoundingBoxArray message
                bounding_boxes = BoundingBoxArray()
                bounding_boxes.header = Header()
                bounding_boxes.header.stamp = rospy.Time.now()

                for detection in detections:
                    bbox = BoundingBox()
                    bbox.x = detection['x']
                    bbox.y = detection['y']
                    bbox.w = detection['w']
                    bbox.h = detection['h']
                    bbox.bbox_class = detection['class']
                    bbox.d = detection['distance']
                    bounding_boxes.bounding_boxes.append(bbox)

                # Publish the bounding box array
                self.pub.publish(bounding_boxes)
                #pub.publish(bounding_boxes)



            else:
                rospy.logwarn("Failed to capture frame from camera.")

        # Release the resource
        self.monitor.join()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    rospy.init_node('object_detection_publisher', anonymous=True)
    #camera_id = 5
    camera_id = rospy.get_param('~camera_id', '') 
    yolo_model = rospy.get_param('~yolo_model', '')
    detection_publisher = ObjectDetectionPublisher(camera_id, yolo_model)
    detection_publisher.run()
