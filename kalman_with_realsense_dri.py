#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
import numpy as np
import math

KF_data = Float32MultiArray()

def camera_data_callback(datas):
    global camera_data
    camera_data = datas.data
    rospy.loginfo(" All sensor data recieved successfully")

##### Pixhawk IMU data subscribe
def callback(data):
    global  acc_imu_data
    global ang_vel_imu
    global imu_orientation 
    global  cov_acc_imu
    global cov_ang_vel_imu
    global cov_imu_orientation 
    acc_imu_data = data.linear_acceleration
    ang_vel_imu = data.angular_velocity
    imu_orientation = data.orientation
    cov_acc_imu = data.linear_acceleration_covariance
    cov_ang_vel_imu = data.angular_velocity_covariance
    cov_imu_orientation  = data.orientation_covariance

def euler_from_quaternion(x, y, z, w):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return roll_x, pitch_y, yaw_z # in radians

def listner_node():
    rospy.init_node('kalman_filtering',anonymous=True)
    rospy.Subscriber('/mavros/imu/data', Imu,  callback)
    rospy.Subscriber('/visual_odom/rt', Float32MultiArray,camera_data_callback)

    pub = rospy.Publisher('/Kalman_filter',Float32MultiArray, queue_size=10)

    rate = rospy.Rate(10)
    rate.sleep()
    X_t = np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    P_t = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    while not rospy.is_shutdown():

        imu_roll, imu_pitch, imu_yaw = euler_from_quaternion(imu_orientation.x,imu_orientation.y,imu_orientation.z,imu_orientation.w)

        gx = ang_vel_imu.x
        gy = ang_vel_imu.y
        gz = ang_vel_imu.z
        ax = acc_imu_data.x
        ay = acc_imu_data.y
        az = acc_imu_data.z
        phi = imu_roll
        theta = imu_pitch
        psi = imu_yaw
        c_phi = camera_data[0]
        c_theta = camera_data[1]
        c_psi = camera_data[2]
        px = camera_data[3]
        py = camera_data[4]
        pz = camera_data[5]
        dt = camera_data[6]


        conv_mat = np.array([[1 , 0, -math.sin(theta)],[0, math.cos(phi),math.sin(phi)*math.cos(theta)],[0, -math.sin(phi), math.cos(phi)*math.cos(theta)]])
        inv_mat = np.linalg.inv(conv_mat)
        Ang_rate = inv_mat.dot(np.array([[gx],[gy],[gz]]))
        
        ### Motion Model
        ## X_t1 = A @ X_t + B @ T_t + w
        A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],[0, 1, 0, 0, dt, 0, 0, 0, 0],[0, 0, 1, 0, 0, dt, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 0, 0, 1]])

        B = np.array([[0.5*dt**2, 0, 0, 0, 0, 0],[0, 0.5*dt**2, 0, 0, 0, 0],[0, 0, 0.5*dt**2, 0, 0, 0], [dt, 0, 0, 0, 0, 0],[0, dt, 0, 0, 0, 0],[0, 0, dt, 0, 0, 0],[0,0, 0, dt, 0, 0],[0, 0, 0, 0, dt, 0],[0, 0, 0, 0, 0, dt]])
        u = np.array([[ax],[ay],[az],Ang_rate[0],Ang_rate[1],Ang_rate[2]])
        Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
        H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],[0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 0, 0, 1]])
        R = np.diag([0.01,0.01,0.01,0.01,0.01,0.01])
        ######
        x_t1 = A @ X_t + B @ u
        z_tp = H @ x_t1 + np.array([[0.01],[0.01],[0.01],[0.01],[0.01],[0.01]])     
        p_t1 = A @ P_t @ A.T + Q
        
        KG = p_t1 @ H.T @ np.linalg.inv(H @ p_t1 @ H.T + R)
        m_z = np.array([[px],[py],[pz],[c_phi],[c_theta],[c_psi]])

        X = x_t1 + KG @ (m_z - z_tp)
        P = p_t1 - KG @ H @ p_t1

        print(X[0][0],)
        #print(KG)
        X_t = X
        P_t = P

        KF_data.data = [X[0][0],X[1][0],X[2][0],X[6][0],X[7][0],X[8][0]]
        pub.publish(KF_data)

    rospy.spin()
         
if __name__=='__main__':
    listner_node()
