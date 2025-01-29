import rospy
import cv2
import threading
from ObjectDetection import ObjectDetection
from Camera import Monitor

if __name__ == '__main__':
    
    #camera_id = 0
    camera_id = rospy.get_param('~camera_id', '')
    yolo_model = rospy.get_param('~yolo_model', '')

    # Init the camera thread
    monitor = Monitor(camera_id)
    monitor.start()

    # Init the object of image processor
    object_detector = ObjectDetection(yolo_model)

    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            monitor.Exit()
            break

        # Get the frame from the camera thread
        frame = monitor.get_frame()

        # Check if the frame is valid
        if frame is not None and frame.size > 0:
            #cv2.imshow('frame', frame)

            # Processing the image
            frame_processed, detections = object_detector(frame)
            cv2.imshow('Object Detection', frame_processed)
            for detection in detections:
                distance_readings.append(detection['distance'])
                print(detection)

            # for detection in detections:
            #     print(detection)
        else:
            print("Failed to capture frame from camera.")
    
    # Release the resource
    monitor.join()
    cv2.destroyAllWindows()
