#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import matplotlib.pyplot as plt
from uvbot_pathplan.msg import ObjectWallPosition

class KalmanVisualizer:
    def __init__(self, duration):
        # 只關心 type == "chair"
        self.raw_pts = []
        self.filt_pts = []
        self.duration = duration

        rospy.Subscriber("/object_wall_position", ObjectWallPosition, self.cb_raw)
        rospy.Subscriber("/filtered_position", ObjectWallPosition, self.cb_filt)
        rospy.loginfo("Collecting data for %.1f seconds...", self.duration)

    def cb_raw(self, msg):
        if msg.type.strip().lower() == "chair":
            self.raw_pts.append((msg.x_obj, msg.y_obj))

    def cb_filt(self, msg):
        if msg.type.strip().lower() == "chair":
            self.filt_pts.append((msg.x_obj, msg.y_obj))

    def plot(self):
        n = min(len(self.raw_pts), len(self.filt_pts))
        if n == 0:
            rospy.logwarn("No 'chair' data collected, skipping plot.")
            return

        raw = np.array(self.raw_pts[:n])
        filt = np.array(self.filt_pts[:n])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # 左圖：未濾波前
        ax1.plot(raw[:,0], raw[:,1], 'o-', label="raw")
        ax1.set_title("Raw Measurements")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.grid(True)

        # 固定 x, y 範圍與刻度
        ax1.set_xlim(0.66, 0.84)
        ax1.set_ylim(1.750, 1.975)
        ax1.set_xticks(np.arange(0.66, 0.82 + 1e-6, 0.02))
        ax1.set_yticks(np.arange(1.775, 1.950 + 1e-6, 0.025))
        ax1.set_aspect('equal', 'box')

        # 右圖：濾波後
        ax2.plot(filt[:,0], filt[:,1], 'o-', label="filtered")
        ax2.set_title("After Kalman Filter")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.grid(True)

        # 同步刻度與範圍
        ax2.set_xlim(0.66, 0.84)
        ax2.set_ylim(1.750, 1.975)
        ax2.set_xticks(np.arange(0.66, 0.82 + 1e-6, 0.02))
        ax2.set_yticks(np.arange(1.775, 1.950 + 1e-6, 0.025))
        ax2.set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    rospy.init_node("kalman_visualized", anonymous=True)
    duration = rospy.get_param("~plot_duration", 30.0)
    viz = KalmanVisualizer(duration)
    rospy.sleep(duration)
    viz.plot()
    rospy.signal_shutdown("Plotting done")
