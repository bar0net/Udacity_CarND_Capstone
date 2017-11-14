from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import math

import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704
TIMEOUT = 2.0

class Controller(object):
    def __init__(self, data):
        self.mass = data['vehicle_mass'] + data['fuel_capacity'] * GAS_DENSITY
        self.brake_deadband = data['brake_deadband']
        self.wheel_radius = data['wheel_radius']
        self.steer_ratio = data['steer_ratio']
        self.max_steer = abs(data['max_steer_angle'])
        self.last_t = rospy.rostime.get_time()

        # Define Filters
        self.velocity_PID = PID(0.35, 0.0, 0.0, -abs(data['decel_limit']), abs(data['accel_limit']))
        
        self.steer_control = YawController(data['wheel_base'], data['steer_ratio'], ONE_MPH, \
                                            data['max_lat_accel'], self.max_steer)
                                    
        self.LPF = LowPassFilter(0.2, 1.0)

    def control(self, target_v, target_yaw, current_v, enabled):
        if self.last_t is None:
            return 0.0, 0.0, 0.0

        delta_t = rospy.rostime.get_time() - self.last_t
        self.last_t = rospy.rostime.get_time()

        if not enabled:
            self.velocity_PID.reset()

        accel = self.velocity_PID.step(target_v - current_v, delta_t)

        throttle = max(0.0, accel)
        brake = max(0.0, -accel)

        steer = self.steer_control.get_steering(target_v, target_yaw, current_v)
        steer = self.LPF.filt(steer)

        return throttle, brake, steer
