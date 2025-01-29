import torch
import numpy as np
import cv2
from ultralytics import YOLO
import os 

class ObjectDetection:
    """
    Class implements Yolo model to detect objects in the image.
    """

    def __init__(self, yolo_model_path):

        self.model = self.load_model(yolo_model_path)
        self.classes = self.model.names

        self.device = 'cpu'
        print('\n\nDevice used: ', self.device, '\n')
        self.known_width = 15
        self.focal_length = 800

    def load_model(self, model_path):
        """
        Loads Yolo model from the local weight file.
        :return: Trained Pytorch model./home/huang/catkin_ws/src/ball_detect/scripts/best.pt
        """
        model = YOLO(model_path)
        print("Model loaded")
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo model
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        results = self.model(frame)
        return results

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label 
        """
        return self.classes[int(x)]

    def calculate_distance(self, width):

        distance = (self.known_width * self.focal_length) / width
        return distance

    def plot_boxes(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes on the frame.
        Also stores the position info of the target area 
        :param results: contains labels and coordinates predicted by model on the given frame.
        :return: Frame with bounding boxes and labels plotted on it and a list of detection dictionaries.
        """
        boxes = results[0].boxes.data.cpu().numpy()
        detections = []

        for i, box in enumerate(boxes):

            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            distance = self.calculate_distance(width)

            #detections.append({'x': cx, 'y': cy, 'w': width, 'h': height, 'class': int(cls), 'distance': distance})
            detections.append({
                'x': cx, 
                'y': cy, 
                'w': width, 
                'h': height, 
                'class': self.class_to_label(cls) , 
                'distance': distance
                })

            label = f"{self.class_to_label(cls)}: {distance:.2f}cm"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        return frame, detections

    def __call__(self, frame):
        """
        This function is called when class is executed
        :return: the frame with the objects boxed and the coordinates of the objects
        """
        results = self.score_frame(frame)
        frame, detections = self.plot_boxes(results, frame)
        return frame, detections
