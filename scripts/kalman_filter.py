#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from scipy.spatial.distance import euclidean
from uvbot_pathplan.msg import ObjectWallPosition
from tf.transformations import quaternion_from_euler
from copy import deepcopy

class KalmanFilter2D:
    def __init__(self, x0, y0, q=0.01, r=0.1): # Q(0.01) less means less noise # R(0.1) more means less noise
        # array x is [px, py](狀態向量)
        self.x = np.array([x0, y0], dtype=float)
        # array P is covariance matrix(狀態誤差協方差矩陣)
        self.P = np.eye(2) * 1.0
        # array F is state transition matrix(狀態轉移矩陣)
        self.F = np.eye(2)
        # array H is observation matrix(觀測矩陣)
        self.H = np.eye(2)
        # array Q is process noise covariance(過程噪聲協方差矩陣)
        self.Q = np.eye(2) * q
        #array R is measurement noise covariance(量測噪聲協方差矩陣)
        self.R = np.eye(2) * r

    def predict(self):
        # x = F x
        self.x = self.F.dot(self.x)
        # P = F P F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, z):
        # z: 量測向量 [x_meas, y_meas]
        # calculate Kalman Gain
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        # update state
        y = z - self.H.dot(self.x)
        self.x = self.x + K.dot(y)
        # update covariance
        I = np.eye(2)
        self.P = (I - K.dot(self.H)).dot(self.P)

class ObjectWallKalmanNode:
    def __init__(self):
        rospy.init_node("kalman_filter_node")
        # 重新發佈至 /filtered_position，same type as /object_wall_position
        self.pub = rospy.Publisher("/filtered_position", ObjectWallPosition, queue_size=10)
        # each type of object will have its own Kalman filter
        self.filters = {}
        # over 0.4 m will have a new Kalman filter(new group)
        self.reset_dist = rospy.get_param("~reset_distance", 0.4)

        rospy.Subscriber("/object_wall_position", ObjectWallPosition, self.cb)
        rospy.loginfo("Kalman filter node started.")
        rospy.spin()

    def cb(self, msg):
        key = msg.type.strip()
        meas = np.array([msg.x_obj, msg.y_obj])

        if key not in self.filters:
            self.filters[key] = KalmanFilter2D(meas[0], meas[1])
        else:
            kf = self.filters[key]
            if euclidean(meas, kf.x) > self.reset_dist:
                self.filters[key] = KalmanFilter2D(meas[0], meas[1])
            else:
                kf.predict()
                kf.update(meas)

        kf = self.filters[key]
        fx, fy = kf.x

        out = deepcopy(msg)
        out.x_obj = fx
        out.y_obj = fy
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        ObjectWallKalmanNode()
    except rospy.ROSInterruptException:
        pass
