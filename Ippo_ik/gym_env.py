# Yongliang Wang
# July 2022
# PyBullet UR5e_robotiq140 Environment
import random
import time
from turtle import width, window_width
import numpy as np
import sys
from gym import spaces
import gym
import scipy
import os
import math
import pybullet
import pybullet_data
from datetime import datetime
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools
from scipy.spatial.transform import Rotation
from Ippo_ik.ur5e_fk import *

# ROBOT_URDF_PATH = "./ur_e_description/urdf/ur5e_robotiq140.urdf"
ROBOT_URDF_PATH = "./Ippo_ik/ur_e_description/urdf/ur5e.urdf"
# PLANE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "plane.urdf")
# TABLE_URDF_PATH = "./Ippo_ik/ur_e_description/urdf/objects/table.urdf"
SPHERE_URDF_PATH = "./Ippo_ik/ur_e_description/urdf/objects/sphere.urdf"  # boxes for target
BLOCK_URDF_PATH = "./Ippo_ik/ur_e_description/urdf/objects/block.urdf"  # boxes for target

class ur5eGymEnv(gym.Env):
    def __init__(self,
                 camera_attached = False,
                 actionRepeat = 100,
                 renders = False,
                 maxSteps = 100,
                 simulatedGripper = False,
                 randObjPos = False,
                 task = 0, # here target number
                 learning_param = 0):

        self.renders = renders
        self.actionRepeat = actionRepeat

        self.goal_roll_line_id = None
        self.goal_pitch_line_id = None
        self.goal_yaw_line_id = None

        self.tool_roll_line_id = None
        self.tool_pitch_line_id = None
        self.tool_yaw_line_id = None

        # setup pybullet sim:
        if self.renders:
            pybullet.connect(pybullet.GUI)
        else:
            pybullet.connect(pybullet.DIRECT)

        pybullet.setTimeStep(1./240.)
        pybullet.setGravity(0,0,-10)

        pybullet.setRealTimeSimulation(False)
        pybullet.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=60, cameraPitch=-30, cameraTargetPosition=[0,0,0])

        # setup robot arm:
        self.end_effector_index = 7
        # self.plane = pybullet.loadURDF(PLANE_URDF_PATH)
        # self.table = pybullet.loadURDF(TABLE_URDF_PATH, [0, 0.75, 0.01], [0, 0, 0, 1])

        flags = pybullet.URDF_USE_SELF_COLLISION
        self.ur5 = pybullet.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags)
        self.num_joints = pybullet.getNumJoints(self.ur5)
        self.control_joints = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])

        self.joints = AttrDict()
        for i in range(self.num_joints):
            info = pybullet.getJointInfo(self.ur5, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = self.joint_type_list[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.control_joints else False
            info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            if info.type == "REVOLUTE":
                pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.POSITION_CONTROL, targetPosition=0, positionGain=0.1, velocityGain=0.1, force=info.maxForce)
                # pybullet.setJointMotorControl2(self.ur5, info.id, pybullet.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
        # explicitly deal with mimic joints
        def controlGripper(robotID, parent, children, mul, **kwargs):
            controlMode = kwargs.pop("controlMode")
            if controlMode==pybullet.POSITION_CONTROL:
                pose = kwargs.pop("targetPosition")
                # move parent joint
                pybullet.setJointMotorControl2(robotID, parent.id, controlMode, targetPosition=pose,
                                        force=parent.maxForce, maxVelocity=parent.maxVelocity)
                # move child joints
                for name in children:
                    child = children[name]
                    childPose = pose * mul[child.name]
                    pybullet.setJointMotorControl2(robotID, child.id, controlMode, targetPosition=childPose,
                                            force=child.maxForce, maxVelocity=child.maxVelocity)
            else:
                raise NotImplementedError("controlGripper does not support \"{}\" control mode".format(controlMode))
            # check if there
            if len(kwargs) is not 0:
                raise KeyError("No keys {} in controlGripper".format(", ".join(kwargs.keys())))

        # object:
        self.initial_obj_pos = [0.5, 0.4, 0.5] # initial object pos
        self.obj = pybullet.loadURDF(BLOCK_URDF_PATH, self.initial_obj_pos)

        # obstacles
        self.initial_obs1_pos = [0.5, 0.3, 0.1] # initial object pos
        self.initial_obs2_pos = [0.5, 0.3, 0.2] # initial object pos
        self.initial_obs3_pos = [0.5, 0.3, 0.3] # initial object pos

        self.obs1 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs1_pos)
        self.obs2 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs2_pos)
        self.obs3 = pybullet.loadURDF(SPHERE_URDF_PATH, self.initial_obs3_pos)

        self.name = 'ur5eGymEnv'
        self.simulatedGripper = simulatedGripper
        self.action_dim = 6
        self.stepCounter = 0
        self.maxSteps = maxSteps
        self.terminated = False
        self.randObjPos = randObjPos
        self.observation = np.array(0)

        self.task = task
        self.learning_param = learning_param

        self._action_bound = 3.14 # delta limits
        action_high = np.array([self._action_bound] * self.action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype='float32')
        self.reset(self.initial_obj_pos, self.initial_obs1_pos, self.initial_obs2_pos, self.initial_obs3_pos)
        high = np.array([10]*self.observation.shape[0])
        self.observation_space = spaces.Box(-high, high, dtype='float32')

    # x,y,z distance
    def goal_distance(self, goal_a, goal_b):
        goal_a = np.array(goal_a)
        goal_b = np.array(goal_b)
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)

    # x,y distance
    def goal_distance2d(self, goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a[0:2] - goal_b[0:2], axis=-1)

    def quaternion_angle_diff(self, quat1, quat2):
        """Calculate the shortest angle (radians) difference between two quaternions"""
        def shortest_angular_difference(theta1, theta2):
            theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
            theta2 = np.arctan2(np.sin(theta2), np.cos(theta2))
            delta_theta = abs(theta2 - theta1)
            if delta_theta > np.pi:
                delta_theta = 2 * np.pi - delta_theta
            return delta_theta

        q1 = np.array(quat1)
        q2 = np.array(quat2)
        dot_product = np.dot(q1, q2)
        # Ensure dot product is in range [-1, 1] due to potential floating point errors
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate the shortest angular difference
        angle_diff = 2 * np.arccos(dot_product)
        return shortest_angular_difference(0, angle_diff)

    def generate_points(self, position, quaternion, distance = 0.001):
        rpy_rad = pybullet.getEulerFromQuaternion(quaternion)
        # Calculate the rotation matrices for roll, pitch, and yaw
        roll_matrix = np.array([[1, 0, 0],
                                [0, math.cos(rpy_rad[0]), -math.sin(rpy_rad[0])],
                                [0, math.sin(rpy_rad[0]), math.cos(rpy_rad[0])]])

        pitch_matrix = np.array([[math.cos(rpy_rad[1]), 0, math.sin(rpy_rad[1])],
                                 [0, 1, 0],
                                 [-math.sin(rpy_rad[1]), 0, math.cos(rpy_rad[1])]])

        yaw_matrix = np.array([[math.cos(rpy_rad[2]), -math.sin(rpy_rad[2]), 0],
                               [math.sin(rpy_rad[2]), math.cos(rpy_rad[2]), 0],
                               [0, 0, 1]])

        # Calculate the points along the roll, pitch, and yaw orientations
        points_roll = position + distance * roll_matrix[:, 0]
        points_pitch = position + distance * pitch_matrix[:, 1]
        points_yaw = position + distance * yaw_matrix[:, 2]
        return points_roll, points_pitch, points_yaw

    def set_joint_angles(self, joint_angles):
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints):
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        pybullet.setJointMotorControlArray(
            self.ur5, indexes,
            pybullet.POSITION_CONTROL,
            targetPositions=joint_angles,
            targetVelocities=[0]*len(poses),
            positionGains=[0.05]*len(poses),
            forces=forces
        )

    def get_joint_angles(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joints = [i[0] for i in j]
        return joints

    def get_joint_velocities(self):
        j = pybullet.getJointStates(self.ur5, [1,2,3,4,5,6])
        joint_velocities = [i[1] for i in j]
        return joint_velocities

    def check_collisions(self):
        collisions = pybullet.getContactPoints()
        # print(len(collisions))
        if len(collisions) > 1:
            # print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False

    def get_current_pose(self):
        linkstate = pybullet.getLinkState(self.ur5, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)

    def line_to_point_distance(self, p, q, r):
        """Calculate the distance between point r and a line formed by points p and q."""
        v = q - p
        w = r - p

        # Calculate the projection of w onto v
        projection = np.dot(w, v) / np.dot(v, v)

        if 0 <= projection <= 1:
            closest_point = p + projection * v
        else:
            # If the projection is outside the line segment, choose the closest endpoint
            dist_p = np.linalg.norm(r - p)
            dist_q = np.linalg.norm(r - q)
            closest_point = p if dist_p < dist_q else q

        distance = np.linalg.norm(r - closest_point)
        return distance

    def cal_inter_point(self, a_position, a_quat, length):

        rotation_matrix = Rotation.from_quat(a_quat).as_matrix()
        direction_vector = np.dot(rotation_matrix, [0, 0, length]) #0.424
        b_position = np.array(a_position) + direction_vector
        return b_position


    def cal_lines2obs(self, links_positions, links_quaternions, obstacle_position):

        l1 = np.array(links_positions[0])
        l2 = np.array(links_positions[1])
        l3 = np.array(links_positions[2])
        l4 = np.array(links_positions[3])
        l5 = np.array(links_positions[4])
        l6 = np.array(links_positions[5])
        l1_5 = self.cal_inter_point(l1, links_quaternions[0], 0.14)
        l2_5 = self.cal_inter_point(l2, links_quaternions[1], 0.14)
        # l7 = self.cal_inter_point(l6, links_quaternions[5], 0.23)
        op = np.array(obstacle_position)
        d1 = self.line_to_point_distance(l1,l1_5,op)
        d2 = self.line_to_point_distance(l1_5,l2_5,op)
        d3 = self.line_to_point_distance(l2_5,l2,op)
        d4 = self.line_to_point_distance(l2,l3,op)
        d5 = self.line_to_point_distance(l3,l4,op)
        d6 = self.line_to_point_distance(l4,l5,op)
        d7 = self.line_to_point_distance(l5,l6,op)
        # d8 = self.line_to_point_distance(l6,l7,op)
        d = np.min([d1, d2, d3, d4, d5, d6, d7])
        return d

    def get_euclidean_dist(self, p_in, p_pout):
        distance = np.linalg.norm(p_in-p_pout)
        return distance

    def update_goal_axes_lines(self, pos, orn):

        # If the lines have been drawn before, remove them.
        if self.goal_roll_line_id is not None:
            pybullet.removeUserDebugItem(self.goal_roll_line_id)
            pybullet.removeUserDebugItem(self.goal_pitch_line_id)
            pybullet.removeUserDebugItem(self.goal_yaw_line_id)

        euler = pybullet.getEulerFromQuaternion(orn)
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape((3, 3))
        roll_axis, pitch_axis, yaw_axis = rot_matrix.T

        line_width = 10  # Set the line width to 10 (bold line)
        line_length = 0.08

        # Draw the lines and store their IDs.
        self.goal_roll_line_id = pybullet.addUserDebugLine(pos, pos + line_length * roll_axis, [1, 0, 0], lineWidth=line_width) # Roll in red.
        self.goal_pitch_line_id = pybullet.addUserDebugLine(pos, pos + line_length * pitch_axis, [0, 1, 0], lineWidth=line_width) # Pitch in green.
        self.goal_yaw_line_id = pybullet.addUserDebugLine(pos, pos + line_length * yaw_axis, [0, 0, 1], lineWidth=line_width) # Yaw in blue.

    def update_tool_axes_lines(self, pos, orn):

        # If the lines have been drawn before, remove them.
        if self.tool_roll_line_id is not None:
            pybullet.removeUserDebugItem(self.tool_roll_line_id)
            pybullet.removeUserDebugItem(self.tool_pitch_line_id)
            pybullet.removeUserDebugItem(self.tool_yaw_line_id)

        euler = pybullet.getEulerFromQuaternion(orn)
        rot_matrix = np.array(pybullet.getMatrixFromQuaternion(orn)).reshape((3, 3))
        roll_axis, pitch_axis, yaw_axis = rot_matrix.T

        line_width = 10  # Set the line width to 3 (bold line)
        line_length = 0.08

        # Draw the lines and store their IDs.
        self.tool_roll_line_id = pybullet.addUserDebugLine(pos, pos + line_length * roll_axis, [1, 0, 0], lineWidth=line_width) # Roll in red.
        self.tool_pitch_line_id = pybullet.addUserDebugLine(pos, pos + line_length * pitch_axis, [0, 1, 0], lineWidth=line_width) # Pitch in green.
        self.tool_yaw_line_id = pybullet.addUserDebugLine(pos, pos + line_length * yaw_axis, [0, 0, 1], lineWidth=line_width) # Yaw in blue.

    def reset(self, obj_pos, obs1_pos, obs2_pos, obs3_pos):
        self.stepCounter = 0
        self.terminated = False
        self.getExtendedObservation(obj_pos, obs1_pos, obs2_pos, obs3_pos)
        return self.observation

    def step(self, action, obj_pos, obs1_pos, obs2_pos, obs3_pos, lp):
        action = np.array(action)
        arm_action = action[0:self.action_dim].astype(float) # j1-j6 - range: [-1,1]

        joint_angles = arm_action
        self.set_joint_angles(joint_angles)
        for i in range(self.actionRepeat):
            pybullet.stepSimulation()
            if self.renders: time.sleep(1./240.)

        self.getExtendedObservation(obj_pos, obs1_pos, obs2_pos, obs3_pos)
        reward = self.compute_reward(self.obj_tool_pos, self.obj_tool_ori, obs1_pos, obs2_pos, obs3_pos, lp, joint_angles, None)
        done = self.my_task_done()
        info = {'is_success': False}
        if self.terminated == self.task:
            info['is_success'] = True
        self.stepCounter += 1
        return self.observation, reward, done, info

    def getExtendedObservation(self, obj_pos, obs1_pos, obs2_pos, obs3_pos):
        # sensor values:
        js = self.get_joint_angles()
        jv = self.get_joint_velocities()

        links_positions, links_quaternions = fwd_kin(js)
        obstacle_dis1 = self.cal_lines2obs(links_positions, links_quaternions, obs1_pos)
        obstacle_dis2 = self.cal_lines2obs(links_positions, links_quaternions, obs2_pos)
        obstacle_dis3 = self.cal_lines2obs(links_positions, links_quaternions, obs3_pos)

        tool_pos = self.get_current_pose()[0] # XYZ
        tool_ori = self.get_current_pose()[1] # Quaternion

        self.obj_pos, self.obj_ori = obj_pos[0:3], pybullet.getQuaternionFromEuler(obj_pos[-3:])

        self.obj_tool_pos = np.array(np.concatenate((self.obj_pos, tool_pos)))
        self.obj_tool_ori = np.array(np.concatenate((self.obj_ori, tool_ori)))

        orient_error = self.quaternion_angle_diff(self.obj_tool_ori[-4:], self.obj_tool_ori[:4])

        tool_p1, tool_p2, tool_p3 = self.generate_points(self.obj_tool_pos[-3:], self.obj_tool_ori[-4:])
        goal_p1, goal_p2, goal_p3 = self.generate_points(self.obj_tool_pos[:3], self.obj_tool_ori[:4])

        p1_dist = self.goal_distance(tool_p1[:3], goal_p1[:3])
        p2_dist = self.goal_distance(tool_p2[:3], goal_p2[:3])
        p3_dist = self.goal_distance(tool_p3[:3], goal_p3[:3])

        obs_dis = np.array([obstacle_dis1, obstacle_dis2, obstacle_dis3])

        error = np.array([p1_dist + p2_dist + p3_dist, orient_error])

        self.observation = np.array(np.concatenate((js, tool_pos, tool_ori, self.obj_pos, self.obj_ori, error, obs_dis)))

    def my_task_done(self):
        c = (self.terminated == True or self.stepCounter > self.maxSteps)
        return c

    def compute_reward(self, obj_tool_pos, obj_tool_ori, obs1_pos, obs2_pos, obs3_pos, lp, joint_angles, info):
        reward = 0
        obstacle_dis1, obstacle_dis2, obstacle_dis3 = self.observation[-3:]
        distance_threshold = 0.08
        fa1 = max(0, 1 - obstacle_dis1/distance_threshold)
        fa2 = max(0, 1 - obstacle_dis2/distance_threshold)
        fa3 = max(0, 1 - obstacle_dis3/distance_threshold)
        fa = fa1+fa2+fa3
        error1 = self.observation[-5]
        error2 = self.observation[-4]
        errors = 0.5*(error1) + 0.25*(error2)
        reward += -0.001*pow(errors, 2) - np.log(pow(errors, 2)+0.0001) - 0.1*fa

        if errors < lp:
            self.terminated = True

        if self.check_collisions():
            reward += -1

        return reward




