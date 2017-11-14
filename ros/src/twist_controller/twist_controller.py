from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import math

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
TIMEOUT = 2.0

LPF_TAU = 0.2
LPF_TS = 1.0

class Controller(object):
    def __init__(self, data):
        self.mass = data['vehicle_mass'] + data['fuel_capacity'] * GAS_DENSITY
        self.brake_deadband = data['brake_deadband']
        self.wheel_radius = data['wheel_radius']
        self.steer_ratio = data['steer_ratio']
        self.max_steer = abs(data['max_steer_angle'])
        self.last_t = rospy.rostime.get_time()
        self.max_decel = abs(data['decel_limit'])
        self.max_accel = data['accel_limit']

        # Define Filters
        Kp = rospy.get_param('~throttle_Kp', 0.1)
        Ki = rospy.get_param('~throttle_Ki', 0.1)
        Kd = rospy.get_param('~throttle_Kd', 0.005)
        self.throttle_PID = PID(Kp, Ki, Kd, -data['decel_limit'], data['accel_limit'])
        
        self.steer_control = YawController(data['wheel_base'], data['steer_ratio'], ONE_MPH, \
                                            data['max_lat_accel'], self.max_steer)
                                    
        self.LPF = LowPassFilter(LPF_TAU, LPF_TS)

    def control(self, target_v, target_yaw, current_v, enabled):
        if self.last_t is None:
            return 0.0, 0.0, 0.0

        # Update timeset
        delta_t = rospy.rostime.get_time() - self.last_t
        self.last_t = rospy.rostime.get_time()

        # If not dbw enabled, reset filter and break
        if not enabled:
            self.throttle_PID.reset()
            return 0.0, 0.0, 0.0

        throttle, brake, steer = 0, 0, 0

        # Define throttle and break
        diff_v = target_v - current_v
        accel = self.throttle_PID.step(diff_v, delta_t)

        if diff_v < 0:
            brake = self.get_torque(abs(accel), self.mass, self.wheel_radius)
        else:
            throttle = accel

        # Force hard braking when the car must stop
        if target_v < 0.3:
            brake = self.get_torque(self.max_decel, self.mass, self.wheel_radius)

        # Reset PID when we get to the objective velocity to avoid
        # having the error pile up
        if abs(diff_v) < 0.2:
            self.throttle_PID.reset()

        # Define steering
        steer = self.steer_control.get_steering(target_v * ONE_MPH, target_yaw, current_v * ONE_MPH)
        steer = self.LPF.filt(steer)

        # rospy.logwarn("{} {} <{} {} {}>".format(target_v, current_v, throttle, brake, steer))
        return throttle, brake, steer

    # Return brake torque [N*m]
    def get_torque(self, accel, weight, radius):
        return accel * weight * radius
