#!/usr/bin/env python3
import numpy as np
import rospy
from collections import deque
from geometry_msgs.msg import PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w
Quaternion.__iter__ = _it

class Optitracker:
    def __init__(self, asset_name):
        self.trace = deque(maxlen=20)
        self.frequency = 30

        self.pose = PoseStamped()
        rospy.Subscriber(f'/natnet_ros/{asset_name}/pose', PoseStamped, self.cb, queue_size=1)
        self.pub = rospy.Publisher(f'optitrack_state_estimator/{asset_name}/state', Odometry, queue_size=10)

    def cb(self, msg):
        self.trace.append(msg)

        # if len(self.trace) <= 1:
        #     init_state = np.matrix([[msg.pose.position.x], [0], [msg.pose.position.y], [0]])
        #     self.KF = KalmanFilter(
        #         init_state=init_state, 
        #         frequency=self.frequency,
        #         measurement_variance=0.000001,
        #         state_variance=0.10
        #     )
        #     return
        # else:
        #     measurement = np.matrix([[msg.pose.position.x], [msg.pose.position.y]])
        #     self.KF.predict()
        #     self.KF.correct(measurement)
            # print(self.KF.pred_state)

        if len(self.trace) <= 1:
            return 
        filtered_state_msg = Odometry()
        filtered_state_msg.header = msg.header
        filtered_state_msg.pose.pose = self.trace[-1].pose
        filtered_state_msg.twist.twist.linear.x = (self.trace[-1].pose.position.x - self.trace[-2].pose.position.x)*self.frequency 
        filtered_state_msg.twist.twist.linear.y = (self.trace[-1].pose.position.y - self.trace[-2].pose.position.y)*self.frequency 
        filtered_state_msg.twist.twist.angular.z = (self._to_yaw(self.trace[-1].pose.orientation) - self._to_yaw(self.trace[-2].pose.orientation))*self.frequency 

        self.pub.publish(filtered_state_msg)

    def _to_yaw(self, quat):
        _, _, yaw = euler_from_quaternion(list(quat))
        return yaw 


class KalmanFilter:
    def __init__(
        self,
        init_state,
        frequency: float,
        state_variance: float = 0.01,
        measurement_variance: float = 0.01,
        method: str = 'velocity'
    ):
        self.stateVariance = state_variance
        self.measurementVariance = measurement_variance

        self.U = 1 if method == "Acceleration" else 0

        dt = 1 / frequency
        self.A = np.matrix([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

        self.B = np.matrix([[dt**2 / 2], [dt], [dt**2 / 2], [dt]])

        self.H = np.matrix([[1, 0, 0, 0], [0, 0, 1, 0]])

        self.P = np.matrix(self.stateVariance * np.identity(self.A.shape[0]))
        self.R = np.matrix(self.measurementVariance * np.identity(self.H.shape[0]))

        self.Q = np.matrix(
            [
                [dt**4 / 4, dt**3 / 2, 0, 0],
                [dt**3 / 2, dt**2, 0, 0],
                [0, 0, dt**4 / 4, dt**3 / 2],
                [0, 0, dt**3 / 2, dt**2],
            ]
        )

        self.state = init_state
        self.pred_state = init_state

        self.err_cov = self.P
        self.pred_err_cov = self.P

    def predict(self):
        self.pred_state = self.A * self.state + self.B * self.U
        self.pred_err_cov = self.A * self.err_cov * self.A.T + self.Q

    def correct(self, measurement):
        kalman_gain = (
            self.pred_err_cov
            * self.H.T
            * np.linalg.pinv(self.H * self.pred_err_cov * self.H.T + self.R)
        )
        self.state = self.pred_state + kalman_gain * (
            measurement - (self.H * self.pred_state)
        )
        self.err_cov = (
            np.identity(self.P.shape[0]) - kalman_gain * self.H
        ) * self.pred_err_cov


if __name__ == '__main__':
    rospy.init_node(name=f"optitrack_state_estimator")
    Optitracker(asset_name='Heijn')
    Optitracker(asset_name='Crate')
    rospy.spin()

