import numpy as np
import rospy
from collections import deque
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class Optitracker:
    def __init__(self):

        self.trace = deque(maxlen=20)

        self.pose = PoseStamped()
        rospy.Subscriber('/natnet_ros_cpp/pose', PoseStamped, self.cb, queue_size=1)
        self.pub = rospy.Publisher('~state', Odometry, queue_size=10)

    def cb(self, msg):
        self.trace.append(msg)

        state = np.matrix([[msg.position.x], [0], [msg.position.y], [0]])
        if self.first_pose:
            self.KF = KalmanFilter(init_state=state, frequency=100)
        else:
            self.KF.correct(state.reshape(2, 1))

        filtered_state_msg = Odometry()

        filtered_state_msg.pose.position.x = self.KF.pred_state[0, 0]
        filtered_state_msg.pose.position.y = self.KF.pred_state[2, 0]
        filtered_state_msg.twist.linear.x = self.KF.pred_state[1, 0]
        filtered_state_msg.twist.linear.y = self.KF.pred_state[3, 0]

        self.pub.publish(filtered_state_msg)


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
    Optitracker()

