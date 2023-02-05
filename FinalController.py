from controller import Camera, InertialUnit, Motor, PositionSensor, Robot, TouchSensor
from enum import Enum
import math
import numpy
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import gestureCaptures
import time

TIME_STEP = 16

# PR2 constants
MAX_WHEEL_SPEED = 3
WHEELS_DISTANCE = 0.4492
SUB_WHEELS_DISTANCE = 0.098
WHEEL_RADIUS = 0.08
TOLERANCE = 0.05

# function to check if a double is almost equal to another
def ALMOST_EQUAL(a, b, t = TOLERANCE):
    # global TOLERANCE
    return ((a < b + t) and (a > b - t))

# helper constants to distinguish the motors
class Wheel(Enum):
    FLL_WHEEL, FLR_WHEEL, FRL_WHEEL, FRR_WHEEL, BLL_WHEEL, BLR_WHEEL, BRL_WHEEL, BRR_WHEEL = range(0,8)

class Rotation(Enum):
    FL_ROTATION, FR_ROTATION, BL_ROTATION, BR_ROTATION = range(0,4)

class Arm(Enum):
    SHOULDER_ROLL, SHOULDER_LIFT, UPPER_ARM_ROLL, ELBOW_LIFT, WRIST_ROLL = range(0,5)

class Hand(Enum):
    LEFT_FINGER, RIGHT_FINGER, LEFT_TIP, RIGHT_TIP = range(0,4)

# robot
robot = Robot()

# PR2 motors and their sensors
wheel_motors = [None] * 8
wheel_sensors = [None] * 8
rotation_motors = [None] * 4
rotation_sensors = [None] * 4
left_arm_motors = [None] * 5
left_arm_sensors = [None] * 5
right_arm_motors = [None] * 5
right_arm_sensors = [None] * 5
right_finger_motors = [None] * 4
right_finger_sensors = [None] * 4
left_finger_motors = [None] * 4
left_finger_sensors = [None] * 4
head_tilt_motor = None
torso_motor = None
torso_sensor = None

# PR2 sensor devices
left_finger_contact_sensors = [None] * 2
right_finger_contact_sensors = [None] * 2
imu_sensor = None
wide_stereo_l_stereo_camera_sensor = None
wide_stereo_r_stereo_camera_sensor = None
high_def_sensor = None
r_forearm_cam_sensor = None
l_forearm_cam_sensor = None
laser_tilt = None
base_laser = None

# static vars
torques = [0.0] * 8
firstCall = True
maxTorque = 0
targetOpenValue = 0.5
targetCloseValue = 0.0

# Simpler step function
def step():
    global TIME_STEP
    global robot
    if(robot.step(TIME_STEP) == -1):
        exit()

# Retrieve all the pointers to the PR2 devices
def initialize_devices():
    global robot
    global wheel_motors
    global wheel_sensors
    global rotation_motors
    global rotation_sensors
    global left_arm_motors
    global left_arm_sensors
    global right_arm_motors
    global right_arm_sensors
    global right_finger_motors
    global right_finger_sensors
    global left_finger_motors
    global left_finger_sensors
    global head_tilt_motor
    global torso_motor
    global torso_sensor
    global left_finger_contact_sensors
    global right_finger_contact_sensors
    global imu_sensor
    global wide_stereo_l_stereo_camera_sensor
    global wide_stereo_r_stereo_camera_sensor
    global high_def_sensor
    global r_forearm_cam_sensor
    global l_forearm_cam_sensor
    global laser_tilt
    global base_laser

    wheel_motors[Wheel.FLL_WHEEL.value] = robot.getDevice("fl_caster_l_wheel_joint")
    wheel_motors[Wheel.FLR_WHEEL.value] = robot.getDevice("fl_caster_r_wheel_joint")
    wheel_motors[Wheel.FRL_WHEEL.value] = robot.getDevice("fr_caster_l_wheel_joint")
    wheel_motors[Wheel.FRR_WHEEL.value] = robot.getDevice("fr_caster_r_wheel_joint")
    wheel_motors[Wheel.BLL_WHEEL.value] = robot.getDevice("bl_caster_l_wheel_joint")
    wheel_motors[Wheel.BLR_WHEEL.value] = robot.getDevice("bl_caster_r_wheel_joint")
    wheel_motors[Wheel.BRL_WHEEL.value] = robot.getDevice("br_caster_l_wheel_joint")
    wheel_motors[Wheel.BRR_WHEEL.value] = robot.getDevice("br_caster_r_wheel_joint")

    wheel_sensors[Wheel.FLL_WHEEL.value] = robot.getDevice("fl_caster_l_wheel_joint_sensor")
    wheel_sensors[Wheel.FLR_WHEEL.value] = robot.getDevice("fl_caster_r_wheel_joint_sensor")
    wheel_sensors[Wheel.FRL_WHEEL.value] = robot.getDevice("fr_caster_l_wheel_joint_sensor")
    wheel_sensors[Wheel.FRR_WHEEL.value] = robot.getDevice("fr_caster_r_wheel_joint_sensor")
    wheel_sensors[Wheel.BLL_WHEEL.value] = robot.getDevice("bl_caster_l_wheel_joint_sensor")
    wheel_sensors[Wheel.BLR_WHEEL.value] = robot.getDevice("bl_caster_r_wheel_joint_sensor")
    wheel_sensors[Wheel.BRL_WHEEL.value] = robot.getDevice("br_caster_l_wheel_joint_sensor")
    wheel_sensors[Wheel.BRR_WHEEL.value] = robot.getDevice("br_caster_r_wheel_joint_sensor")

    rotation_motors[Rotation.FL_ROTATION.value] = robot.getDevice("fl_caster_rotation_joint")
    rotation_motors[Rotation.FR_ROTATION.value] = robot.getDevice("fr_caster_rotation_joint")
    rotation_motors[Rotation.BL_ROTATION.value] = robot.getDevice("bl_caster_rotation_joint")
    rotation_motors[Rotation.BR_ROTATION.value] = robot.getDevice("br_caster_rotation_joint")

    rotation_sensors[Rotation.FL_ROTATION.value] = robot.getDevice("fl_caster_rotation_joint_sensor")
    rotation_sensors[Rotation.FR_ROTATION.value] = robot.getDevice("fr_caster_rotation_joint_sensor")
    rotation_sensors[Rotation.BL_ROTATION.value] = robot.getDevice("bl_caster_rotation_joint_sensor")
    rotation_sensors[Rotation.BR_ROTATION.value] = robot.getDevice("br_caster_rotation_joint_sensor")

    left_arm_motors[Arm.SHOULDER_ROLL.value] = robot.getDevice("l_shoulder_pan_joint")
    left_arm_motors[Arm.SHOULDER_LIFT.value] = robot.getDevice("l_shoulder_lift_joint")
    left_arm_motors[Arm.UPPER_ARM_ROLL.value] = robot.getDevice("l_upper_arm_roll_joint")
    left_arm_motors[Arm.ELBOW_LIFT.value] = robot.getDevice("l_elbow_flex_joint")
    left_arm_motors[Arm.WRIST_ROLL.value] = robot.getDevice("l_wrist_roll_joint")

    left_arm_sensors[Arm.SHOULDER_ROLL.value] = robot.getDevice("l_shoulder_pan_joint_sensor")
    left_arm_sensors[Arm.SHOULDER_LIFT.value] = robot.getDevice("l_shoulder_lift_joint_sensor")
    left_arm_sensors[Arm.UPPER_ARM_ROLL.value] = robot.getDevice("l_upper_arm_roll_joint_sensor")
    left_arm_sensors[Arm.ELBOW_LIFT.value] = robot.getDevice("l_elbow_flex_joint_sensor")
    left_arm_sensors[Arm.WRIST_ROLL.value] = robot.getDevice("l_wrist_roll_joint_sensor")

    right_arm_motors[Arm.SHOULDER_ROLL.value] = robot.getDevice("r_shoulder_pan_joint")
    right_arm_motors[Arm.SHOULDER_LIFT.value] = robot.getDevice("r_shoulder_lift_joint")
    right_arm_motors[Arm.UPPER_ARM_ROLL.value] = robot.getDevice("r_upper_arm_roll_joint")
    right_arm_motors[Arm.ELBOW_LIFT.value] = robot.getDevice("r_elbow_flex_joint")
    right_arm_motors[Arm.WRIST_ROLL.value] = robot.getDevice("r_wrist_roll_joint")

    right_arm_sensors[Arm.SHOULDER_ROLL.value] = robot.getDevice("r_shoulder_pan_joint_sensor")
    right_arm_sensors[Arm.SHOULDER_LIFT.value] = robot.getDevice("r_shoulder_lift_joint_sensor")
    right_arm_sensors[Arm.UPPER_ARM_ROLL.value] = robot.getDevice("r_upper_arm_roll_joint_sensor")
    right_arm_sensors[Arm.ELBOW_LIFT.value] = robot.getDevice("r_elbow_flex_joint_sensor")
    right_arm_sensors[Arm.WRIST_ROLL.value] = robot.getDevice("r_wrist_roll_joint_sensor")

    left_finger_motors[Hand.LEFT_FINGER.value] = robot.getDevice("l_gripper_l_finger_joint")
    left_finger_motors[Hand.RIGHT_FINGER.value] = robot.getDevice("l_gripper_r_finger_joint")
    left_finger_motors[Hand.LEFT_TIP.value] = robot.getDevice("l_gripper_l_finger_tip_joint")
    left_finger_motors[Hand.RIGHT_TIP.value] = robot.getDevice("l_gripper_r_finger_tip_joint")

    left_finger_sensors[Hand.LEFT_FINGER.value] = robot.getDevice("l_gripper_l_finger_joint_sensor")
    left_finger_sensors[Hand.RIGHT_FINGER.value] = robot.getDevice("l_gripper_r_finger_joint_sensor")
    left_finger_sensors[Hand.LEFT_TIP.value] = robot.getDevice("l_gripper_l_finger_tip_joint_sensor")
    left_finger_sensors[Hand.RIGHT_TIP.value] = robot.getDevice("l_gripper_r_finger_tip_joint_sensor")

    right_finger_motors[Hand.LEFT_FINGER.value] = robot.getDevice("r_gripper_l_finger_joint")
    right_finger_motors[Hand.RIGHT_FINGER.value] = robot.getDevice("r_gripper_r_finger_joint")
    right_finger_motors[Hand.LEFT_TIP.value] = robot.getDevice("r_gripper_l_finger_tip_joint")
    right_finger_motors[Hand.RIGHT_TIP.value] = robot.getDevice("r_gripper_r_finger_tip_joint")

    right_finger_sensors[Hand.LEFT_FINGER.value] = robot.getDevice("r_gripper_l_finger_joint_sensor")
    right_finger_sensors[Hand.RIGHT_FINGER.value] = robot.getDevice("r_gripper_r_finger_joint_sensor")
    right_finger_sensors[Hand.LEFT_TIP.value] = robot.getDevice("r_gripper_l_finger_tip_joint_sensor")
    right_finger_sensors[Hand.RIGHT_TIP.value] = robot.getDevice("r_gripper_r_finger_tip_joint_sensor")

    head_tilt_motor = robot.getDevice("head_tilt_joint")
    head_tilt_sensor = robot.getDevice("head_tilt_joint_sensor")
    torso_motor = robot.getDevice("torso_lift_joint")
    torso_sensor = robot.getDevice("torso_lift_joint_sensor")

    left_finger_contact_sensors[Hand.LEFT_FINGER.value] = robot.getDevice("l_gripper_l_finger_tip_contact_sensor")
    left_finger_contact_sensors[Hand.RIGHT_FINGER.value] = robot.getDevice("l_gripper_r_finger_tip_contact_sensor")
    right_finger_contact_sensors[Hand.LEFT_FINGER.value] = robot.getDevice("r_gripper_l_finger_tip_contact_sensor")
    right_finger_contact_sensors[Hand.RIGHT_FINGER.value] = robot.getDevice("r_gripper_r_finger_tip_contact_sensor")

    imu_sensor = robot.getDevice("imu_sensor")

    wide_stereo_l_stereo_camera_sensor = robot.getDevice("wide_stereo_l_stereo_camera_sensor")
    wide_stereo_r_stereo_camera_sensor = robot.getDevice("wide_stereo_r_stereo_camera_sensor")
    high_def_sensor = robot.getDevice("high_def_sensor")
    r_forearm_cam_sensor = robot.getDevice("r_forearm_cam_sensor")
    l_forearm_cam_sensor = robot.getDevice("l_forearm_cam_sensor")
    laser_tilt = robot.getDevice("laser_tilt")
    base_laser = robot.getDevice("base_laser")


# enable the robot devices
def enable_devices():
    global wheel_sensors
    global wheel_motors
    global rotation_sensors
    global left_finger_contact_sensors
    global right_finger_contact_sensors
    global left_finger_sensors
    global right_finger_sensors
    global left_arm_sensors
    global right_arm_sensors
    global torso_sensor
    global TIME_STEP

    for i in range(8):
        wheel_sensors[i].enable(TIME_STEP)
        wheel_motors[i].setPosition(float('inf'))
        wheel_motors[i].setVelocity(0)

    for i in range(4):
        rotation_sensors[i].enable(TIME_STEP)

    for i in range(2):
        left_finger_contact_sensors[i].enable(TIME_STEP)
        right_finger_contact_sensors[i].enable(TIME_STEP)

    for i in range(4):
        left_finger_sensors[i].enable(TIME_STEP)
        right_finger_sensors[i].enable(TIME_STEP)
    
    for i in range(5):
        left_arm_sensors[i].enable(TIME_STEP)
        right_arm_sensors[i].enable(TIME_STEP)
    
    torso_sensor.enable(TIME_STEP)

# set the speeds of the robot wheels
def set_wheels_speeds(fll, flr, frl, frr, bll, blr, brl, brr):
    global wheel_motors
    wheel_motors[Wheel.FLL_WHEEL.value].setVelocity(fll)
    wheel_motors[Wheel.FLR_WHEEL.value].setVelocity(flr)
    wheel_motors[Wheel.FRL_WHEEL.value].setVelocity(frl)
    wheel_motors[Wheel.FRR_WHEEL.value].setVelocity(frr)
    wheel_motors[Wheel.BLL_WHEEL.value].setVelocity(bll)
    wheel_motors[Wheel.BLR_WHEEL.value].setVelocity(blr)
    wheel_motors[Wheel.BRL_WHEEL.value].setVelocity(brl)
    wheel_motors[Wheel.BRR_WHEEL.value].setVelocity(brr)

def set_wheels_speed(speed):
    set_wheels_speeds(speed, speed, speed, speed, speed, speed, speed, speed)

def stop_wheels():
    set_wheels_speeds(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

# enable/disable the torques on the wheels motors
def enable_passive_wheels(enable):
    global wheel_motors
    global torques
    if(enable):
        for i in range(8):
            torques[i] = wheel_motors[i].getAvailableTorque()
            wheel_motors[i].setAvailableTorque(0)
    else:
        for i in range(8):
            wheel_motors[i].setAvailableTorque(torques[i])

# Set the rotation wheels angles.
# If wait_on_feedback is true, the function is left when the rotational motors have reached their target positions.
def set_rotation_wheels_angles(fl, fr, bl, br, wait_on_feedback):
    global rotation_motors
    if(wait_on_feedback):
        stop_wheels()
        enable_passive_wheels(True)

    rotation_motors[Rotation.FL_ROTATION.value].setPosition(fl)
    rotation_motors[Rotation.FR_ROTATION.value].setPosition(fr)
    rotation_motors[Rotation.BL_ROTATION.value].setPosition(bl)
    rotation_motors[Rotation.BR_ROTATION.value].setPosition(br)

    if(wait_on_feedback):
        target = [fl, fr, bl, br]

        while(True):
            all_reached = True
            for i in range(4):
                current_position = rotation_sensors[i].getValue()
                if not(ALMOST_EQUAL(current_position, target[i])):
                    all_reached = False
                    break
            if(all_reached):
                break
            else:
                step()
        enable_passive_wheels(False)


# High level function to rotate the robot around itself of a given angle [rad]
# Note: the angle can be negative
def robot_rotate(angle):
    global MAX_WHEEL_SPEED
    global WHEELS_DISTANCE
    global SUB_WHEELS_DISTANCE
    global WHEEL_RADIUS

    stop_wheels()
    set_rotation_wheels_angles(3.0 * (math.pi / 4), (math.pi / 4), -3.0 * (math.pi / 4), -1 * (math.pi / 4), True)
    max_wheel_speed = MAX_WHEEL_SPEED if angle > 0 else MAX_WHEEL_SPEED * -1
    set_wheels_speed(max_wheel_speed)
    initial_wheel0_position = wheel_sensors[Wheel.FLL_WHEEL.value].getValue()

    # expected travel distance done by the wheel
    expected_travel_distance = math.fabs(angle * 0.5 * (WHEELS_DISTANCE + SUB_WHEELS_DISTANCE))

    while(True):
        wheel0_position = wheel_sensors[Wheel.FLL_WHEEL.value].getValue()
        # travel distance done by the wheel
        wheel0_travel_distance = math.fabs(WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))
        if(wheel0_travel_distance > expected_travel_distance):
            break

        # reduce the speed before reaching the target
        if (expected_travel_distance - wheel0_travel_distance < 0.025):
            set_wheels_speed(0.1 * max_wheel_speed)
        step()
    
    # reset wheels
    set_rotation_wheels_angles(0.0, 0.0, 0.0, 0.0, True)
    stop_wheels()

# High level function to go forward for a given distance [m]
# Note: the distance can be negative
def robot_go_forward(distance):
    global MAX_WHEEL_SPEED
    global WHEEL_RADIUS

    max_wheel_speed = MAX_WHEEL_SPEED if distance > 0 else MAX_WHEEL_SPEED * -1
    set_wheels_speed(max_wheel_speed)

    initial_wheel0_position = wheel_sensors[Wheel.FLL_WHEEL.value].getValue()

    while(True):
        wheel0_position = wheel_sensors[Wheel.FLL_WHEEL.value].getValue()
        # travel distance done by the wheel
        wheel0_travel_distance = math.fabs(WHEEL_RADIUS * (wheel0_position - initial_wheel0_position))
        if (wheel0_travel_distance > math.fabs(distance)):
            break

        # reduce the speed before reaching the target
        if (math.fabs(distance) - wheel0_travel_distance < 0.025):
            set_wheels_speed(0.1 * max_wheel_speed)

        step()

    stop_wheels()

# Open or close the gripper.
# If wait_on_feedback is true, the gripper is stopped either when the target is reached,
# or either when something has been gripped
def set_gripper(left, open, torqueWhenGripping, wait_on_feedback):
    global left_finger_motors
    global right_finger_motors
    global left_finger_contact_sensors
    global right_finger_contact_sensors
    global firstCall
    global maxTorque
    global targetOpenValue
    global targetCloseValue

    motors = [None] * 4
    motors[Hand.LEFT_FINGER.value] = left_finger_motors[Hand.LEFT_FINGER.value] if left else right_finger_motors[Hand.LEFT_FINGER.value]
    motors[Hand.RIGHT_FINGER.value] = left_finger_motors[Hand.RIGHT_FINGER.value] if left else right_finger_motors[Hand.RIGHT_FINGER.value]
    motors[Hand.LEFT_TIP.value] = left_finger_motors[Hand.LEFT_TIP.value] if left else right_finger_motors[Hand.LEFT_TIP.value]
    motors[Hand.RIGHT_TIP.value] = left_finger_motors[Hand.RIGHT_TIP.value] if left else right_finger_motors[Hand.RIGHT_TIP.value]
    
    sensors = [None] * 4
    sensors[Hand.LEFT_FINGER.value] = left_finger_sensors[Hand.LEFT_FINGER.value] if left else right_finger_sensors[Hand.LEFT_FINGER.value]
    sensors[Hand.RIGHT_FINGER.value] = left_finger_sensors[Hand.RIGHT_FINGER.value] if left else right_finger_sensors[Hand.RIGHT_FINGER.value]
    sensors[Hand.LEFT_TIP.value] = left_finger_sensors[Hand.LEFT_TIP.value] if left else right_finger_sensors[Hand.LEFT_TIP.value]
    sensors[Hand.RIGHT_TIP.value] = left_finger_sensors[Hand.RIGHT_TIP.value] if left else right_finger_sensors[Hand.RIGHT_TIP.value]

    contacts = [None] * 2
    contacts[Hand.LEFT_FINGER.value] = left_finger_contact_sensors[Hand.LEFT_FINGER.value] if left else right_finger_contact_sensors[Hand.LEFT_FINGER.value]
    contacts[Hand.RIGHT_FINGER.value] = left_finger_contact_sensors[Hand.RIGHT_FINGER.value] if left else right_finger_contact_sensors[Hand.RIGHT_FINGER.value]

    if(firstCall):
        maxTorque = motors[Hand.LEFT_FINGER.value].getAvailableTorque()
        firstCall = False

    for i in range(4):
        motors[i].setAvailableTorque(maxTorque)
    
    if(open):
        for i in range(4):
            motors[i].setPosition(targetOpenValue)
        
        if(wait_on_feedback):
            while(not(ALMOST_EQUAL(sensors[Hand.LEFT_FINGER.value].getValue(), targetOpenValue))):
                step()
    else:
        for i in range(4):
            motors[i].setPosition(targetCloseValue)
        
        if(wait_on_feedback):
            # wait until the 2 touch sensors are fired or the target value is reached
            while((contacts[Hand.LEFT_FINGER.value].getValue() == 0 or contacts[Hand.RIGHT_FINGER.value].getValue() == 0) and (not(ALMOST_EQUAL(sensors[Hand.LEFT_FINGER.value].getValue(), targetCloseValue)))):
                step()

        current_position = sensors[Hand.LEFT_FINGER.value].getValue()
        for i in range(4):
            motors[i].setAvailableTorque(torqueWhenGripping)
            motors[i].setPosition(numpy.fmax(0.0, 0.95 * current_position))

# Set the right arm position (forward kinematics)
# If wait_on_feedback is enabled, the function is left when the target is reached.
def set_right_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift, wrist_roll, wait_on_feedback):
    global right_arm_motors
    global right_arm_sensors

    right_arm_motors[Arm.SHOULDER_ROLL.value].setPosition(shoulder_roll)
    right_arm_motors[Arm.SHOULDER_LIFT.value].setPosition(shoulder_lift)
    right_arm_motors[Arm.UPPER_ARM_ROLL.value].setPosition(upper_arm_roll)
    right_arm_motors[Arm.ELBOW_LIFT.value].setPosition(elbow_lift)
    right_arm_motors[Arm.WRIST_ROLL.value].setPosition(wrist_roll)

    if(wait_on_feedback):
        while((not(ALMOST_EQUAL(right_arm_sensors[Arm.SHOULDER_ROLL.value].getValue(), shoulder_roll))) or (not(ALMOST_EQUAL(right_arm_sensors[Arm.SHOULDER_LIFT.value].getValue(), shoulder_lift))) or (not(ALMOST_EQUAL(right_arm_sensors[Arm.UPPER_ARM_ROLL.value].getValue(), upper_arm_roll))) or (not(ALMOST_EQUAL(right_arm_sensors[Arm.ELBOW_LIFT.value].getValue(), elbow_lift))) or (not(ALMOST_EQUAL(right_arm_sensors[Arm.WRIST_ROLL.value].getValue(), wrist_roll)))):
            step()

def set_left_arm_position(shoulder_roll, shoulder_lift, upper_arm_roll, elbow_lift, wrist_roll, wait_on_feedback):
    global left_arm_motors
    global left_arm_sensors

    left_arm_motors[Arm.SHOULDER_ROLL.value].setPosition(shoulder_roll)
    left_arm_motors[Arm.SHOULDER_LIFT.value].setPosition(shoulder_lift)
    left_arm_motors[Arm.UPPER_ARM_ROLL.value].setPosition(upper_arm_roll)
    left_arm_motors[Arm.ELBOW_LIFT.value].setPosition(elbow_lift)
    left_arm_motors[Arm.WRIST_ROLL.value].setPosition(wrist_roll)

    if(wait_on_feedback):
        while((not(ALMOST_EQUAL(left_arm_sensors[Arm.SHOULDER_ROLL.value].getValue(), shoulder_roll))) or (not(ALMOST_EQUAL(left_arm_sensors[Arm.SHOULDER_LIFT.value].getValue(), shoulder_lift))) or (not(ALMOST_EQUAL(left_arm_sensors[Arm.UPPER_ARM_ROLL.value].getValue(), upper_arm_roll))) or (not(ALMOST_EQUAL(left_arm_sensors[Arm.ELBOW_LIFT.value].getValue(), elbow_lift))) or (not(ALMOST_EQUAL(left_arm_sensors[Arm.WRIST_ROLL.value].getValue(), wrist_roll)))):
            step()

# Set the torso height
# If wait_on_feedback is enabled, the function is left when the target is reached.
def set_torso_height(height, wait_on_feedback):
    global torso_motor
    global torso_sensor
    torso_motor.setPosition(height)
    if(wait_on_feedback):
        while(not(ALMOST_EQUAL(torso_sensor.getValue(), height))):
            step()

# Convenient initial position
def set_initial_position():
    set_left_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, False)
    set_right_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, True)

    set_gripper(False, True, 0.0, False)
    set_gripper(True, True, 0.0, False)

    set_torso_height(0.2, True)
    
    
    # use from initial position
def use_phone():
    set_initial_position()
    robot_rotate(-math.pi/2)
    set_right_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    set_left_arm_position(2.0, 1.00, 0.0, -2.2, 0.0, False) 
    robot_go_forward(.35)    
    set_gripper(False, False, 50.0, True)
    # shRoll, shLift, upArmRoll, elbLift, wrRoll, wait
    head_tilt_motor.setPosition(1.0)
    set_right_arm_position(-.5, 0.0, -1.55, -2.0, 2*math.pi/3, True)
    head_tilt_motor.setPosition(0.0)
    #time.sleep(2.0)
    set_right_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
 
    time.sleep(.4)
    set_gripper(False, True, 0.0, True)
    set_initial_position()
    robot_go_forward(-.35)
    robot_rotate(math.pi/2)
    print("done using phone")
    
    
def feed_cat():

    set_initial_position()
    set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    
    robot_go_forward(0.35)
    #grab the cookies
    set_gripper(True, False, 200.0, True)
    set_left_arm_position(0.05, 0.47, 0.0, -0.47, 0.0, True)
    set_left_arm_position(-0.05, 0.5, 0.0, -0.5, 0.0, True)
    set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    set_gripper(True, True, 0.0, True)
    set_gripper(True, False, 50.0, True)
    
    set_left_arm_position(0.0, 0.5, 0.0, -1.0, 0.0, True)
    robot_go_forward(-0.35)
    #go to cat
    robot_rotate(math.pi)
    set_torso_height(.33, True)  
    head_tilt_motor.setPosition(.4)  
    robot_go_forward(.6)
    
    # pour the cookies in the bowl
    # position arm
    # shRoll, shLift, upArmRoll, elbLift, wrRoll, wait
    set_left_arm_position(0.38, 0.65, 0.0, -1.0, 0.0, True)
    set_left_arm_position(0.33, 0.65, 0.0, -0.8, 5*math.pi/7, True)
    time.sleep(1.0)
    # unpour the cookies and put the thing back
    set_left_arm_position(0.38, 0.65, 0.0, -1.0, 0.0, True)
    head_tilt_motor.setPosition(0.0)
    set_left_arm_position(0.0, 0.5, 0.0, -1.0, 0.0, True)
    robot_go_forward(-0.6)
    set_torso_height(0.2, True)
    robot_rotate(math.pi)
    robot_go_forward(.35)
    
    set_left_arm_position(0.0, 0.499, 0.0, -0.47, 0.0, True)
    set_left_arm_position(0.0, 0.501, 0.0, -0.5, 0.0, True)
    set_left_arm_position(0.0, 0.5, 0.0, -0.5, 0.0, True)
    set_gripper(True, True, 0.0, True)
    set_gripper(True, False, 20.0, True)
    time.sleep(1.0)
    set_gripper(True, True, 0.0, True)
    robot_go_forward(-.35)
    # set initial position, but with feedback
    set_left_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, True)
    set_right_arm_position(0.0, 1.35, 0.0, -2.2, 0.0, True)

    set_gripper(False, True, 0.0, True)
    set_gripper(True, True, 0.0, True)

    set_torso_height(0.2, True)
    print("done feeding the cat!")
    
    
def live_long():
         # shRoll, shLift, upArmRoll, elbLift, wrRoll, wait
    set_gripper(False, False, 10.0, True)
    set_torso_height(0.23, True)
    set_gripper(True, False, 10.0, True)
    set_right_arm_position(-math.pi/2, 0.0, 0.0, -1.5, math.pi/2, True)
    print("live long and prosper")
    set_gripper(False, True, 0.0, True)
    time.sleep(1.5)
    set_initial_position()



def run_robot():

    initialize_devices()
    enable_devices()
    set_initial_position()
    

    while (True):
        output = gestureCaptures.run()
        print("output:", output)
        if (output == "use phone"):
            use_phone()        
        elif(output == "feed cat"):
            feed_cat()
        elif(output == "thumbs up"):
            print("yay!")
            for i in range(5):
                set_torso_height(0.10, True)
                set_torso_height(.33, True)
        elif(output == "live long"):
            live_long()
            
if __name__ == "__main__":
    run_robot()
    #initialize_devices()
    #enable_devices()
    #set_initial_position()
    #use_phone()
    #feed_cat()
    #run.run()
    quit()