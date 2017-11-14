#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math
import time
import tf
import numpy as np

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''
# Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 30

# Distance in [m] before a "Stopping Signal" to start car deceleration
STOP_DISTANCE = 3.0

class VehicleData():
    def __init__(self):
        self.position = None
        self.yaw = None
        self.velocity = None

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # TODO: Add a subscriber for /obstacle_waypoint below (pending clarification)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below
        self.car = VehicleData()
        self.waypoints = []
        self.next_wp_id = None

        self.target_speed = 10.0 # rospy.get_param('~/waypoint_loader/velocity', 64.0)
        self.brake_limit = rospy.get_param('~/twist_controller/decel_limit', -5)
        self.accel_limit = rospy.get_param('~/twist_controller/accel_limit', 1)
        
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.loop()
            rate.sleep()

    def loop(self):
        # Sanity check
        if self.lack_information():
            return

        # Get the index of the next waypoint in front of the car
        next_wp_id = self.get_next_waypoint_index()

        # generate the list of next waypoints
        final_waypoints = self.generate_waypoints(next_wp_id)

        # publish the list of next waypoints
        self.publish(final_waypoints)   

    def pose_cb(self, msg):
        # Update Car position and Orientation
        self.car.position = msg.pose.position
        self.car.yaw = self.get_yaw(msg.pose.orientation)

    def waypoints_cb(self, msg):
        self.waypoints = msg.waypoints

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        # rospy.logwarn('[Traffic_cb::msg] {}'.format(msg.data))
        if msg.data == -1:
            self.target_speed = 10.0
        else:
            self.target_speed = 0.44

    def velocity_cb(self, msg):
        #self.car.velocity = self.get_velocity(msg.twist.linear.x, msg.twist.linear.y)
        self.car.velocity = msg.twist.linear.x

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def module(self, pos1, pos2):
        return math.sqrt( (pos1.x - pos2.x)*(pos1.x - pos2.x) + (pos1.y - pos2.y)*(pos1.y - pos2.y))

    def get_yaw(self,orientation):
        angles = tf.transformations.euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return angles[2]

    def get_velocity(self, x, y):
        return math.sqrt(x*x + y*y)

    # Check if the expected information is recieved
    def lack_information(self):
        output = False
        if self.car.position == None:
            rospy.loginfo("[waypoint_updater::lack_information] No position recieved")
            output = True
        
        if len(self.waypoints) == 0:
            rospy.loginfo("[waypoint_updater::lack_information] No waypoints recieved")
            output = True
        
        if self.car.velocity == None:
            rospy.loginfo("[waypoint_updater::lack_information] No velocity recieved")
            output = True

        # Todo: Register Warning when no traffic_waypoints recieved

        return output

    # Get the next waypoint in front of the car
    def get_next_waypoint_index(self):
        
        # get closest waypoint to the car
        min_dst = float("inf")
        index = -1

        for i, wp in enumerate(self.waypoints):
            dst = self.module(self.car.position,wp.pose.pose.position)

            if dst < min_dst:
                min_dst = dst
                index = i

        # If the waypoint is behind the car, get the next one
        wp = self.waypoints[index]
        angle = self.car.yaw - math.atan2(wp.pose.pose.position.y - self.car.position.y, wp.pose.pose.position.x - self.car.position.x)

        if abs(angle) > math.pi / 4:
            index = (index + 1) % len(self.waypoints)
            wp = self.waypoints[index]

        return index

    def generate_waypoints(self, start_id):
        final_waypoints = []
        for i in range(LOOKAHEAD_WPS):
            curr_id = (start_id + i) % len(self.waypoints)
            
            dst = self.distance(self.waypoints, start_id, curr_id)
            vel = math.sqrt(self.car.velocity * self.car.velocity + 2 * self.accel_limit * dst)
            if vel > self.target_speed:
                vel = self.target_speed

            self.waypoints[curr_id].twist.twist.linear.x = self.target_speed
            final_waypoints.append(self.waypoints[curr_id])
        return final_waypoints

    def publish(self, final_waypoints):
        out_msg = Lane()
        out_msg.header.frame_id = '/World'
        out_msg.header.stamp = rospy.Time(0)
        out_msg.waypoints = list(final_waypoints)
        self.final_waypoints_pub.publish(out_msg)

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
