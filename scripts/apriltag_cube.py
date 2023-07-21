#!/usr/bin/env python
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TransformStamped, Point
from apriltag_ros.msg import AprilTagDetectionArray
import tf2_ros

np.set_printoptions(precision=3, suppress=True)


# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z


Point.__iter__ = _it


# Helpers
def _it(self):
    yield self.x
    yield self.y
    yield self.z
    yield self.w


Quaternion.__iter__ = _it
import tf


class CubeDetector:
    def __init__(self):
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.cb)

        self.tf_listener = tf.TransformListener()
        self.red_cube_ids = [13, 14, 15, 16, 17, 18]
        self.red_cube_quats = [
            [0, 0, 0, 1],
            [ -0.5, -0.5, -0.5, 0.5 ],
            [ 0, -0.7071068, 0.7071068, 0 ],
            [ 1, 0, 0, 0 ],
            [ -0.5, 0.5, 0.5, 0.5 ],
            [ 0, 0.7071068, 0.7071068, 0 ]
        ]

        self.blue_cube_ids = [23, 22, 21, 20, 19, 12]
        self.blue_cube_quats = [
            [-0.7071068, 0, 0, 0.7071068],
            [-0.7071068, 0.7071068, 0, 0],
            [0, 0, 0, 1],
            [ 0.7071068, 0, 0.7071068, 0 ],
            [ 0.5, 0.5, -0.5, 0.5 ],
            [ 0.5, 0.5, 0.5, 0.5 ]
        ]

        self.cube_pose = PoseStamped()
        self.cube_pose.pose.position = Point(x=0, y=0, z=-0.02)
        self.cube_pose.pose.orientation = Quaternion(x=0, y=0, z=0.0, w=0.0)

    def cb(self, msg: AprilTagDetectionArray):
        red_cube_positions = []
        blue_cube_positions = []
        for det in msg.detections:
            if det.id[0] in self.red_cube_ids:
                self.cube_pose.header.frame_id = "tag_" + str(det.id[0])
                self.cube_pose.pose.orientation = Quaternion(
                    *self.red_cube_quats[self.red_cube_ids.index(det.id[0])]
                )
                try:
                    pos = self.tf_listener.transformPose(
                        "/camera_color_optical_frame", self.cube_pose
                    )

                    red_cube_positions.append(list(pos.pose.position))
                    red_cube_orientation = pos.pose.orientation

                except Exception as e:
                    rospy.logwarn(e)
            elif det.id[0] in self.blue_cube_ids:
                self.cube_pose.header.frame_id = "tag_" + str(det.id[0])
                self.cube_pose.pose.orientation = Quaternion(
                    *self.blue_cube_quats[self.blue_cube_ids.index(det.id[0])]
                )
                try:
                    pos = self.tf_listener.transformPose(
                        "/camera_color_optical_frame", self.cube_pose
                    )

                    blue_cube_positions.append(list(pos.pose.position))
                    blue_cube_orientation = pos.pose.orientation
                except Exception as e:
                    rospy.logwarn(e)

        if len(blue_cube_positions) > 0:
            blue_cube_trans = TransformStamped()
            blue_cube_trans.header.stamp = rospy.Time.now()
            blue_cube_trans.header.frame_id = "camera_color_optical_frame"
            blue_cube_trans.child_frame_id = f"BlueBlock"

            blue_cube_trans.transform.translation = Vector3(
                *np.mean(np.array(blue_cube_positions), axis=0)
            )
            blue_cube_trans.transform.rotation = blue_cube_orientation
            self.tf_broadcaster.sendTransform(blue_cube_trans)

        if len(red_cube_positions) > 0:
            red_cube_trans = TransformStamped()
            red_cube_trans.header.stamp = rospy.Time.now()
            red_cube_trans.header.frame_id = "camera_color_optical_frame"
            red_cube_trans.child_frame_id = f"RedBlock"

            red_cube_trans.transform.translation = Vector3(
                *np.mean(np.array(red_cube_positions), axis=0)
            )
            red_cube_trans.transform.rotation = red_cube_orientation
            self.tf_broadcaster.sendTransform(red_cube_trans)


if __name__ == "__main__":
    rospy.init_node(name="apriltag_cube_detector")
    CubeDetector()
    rospy.spin()
