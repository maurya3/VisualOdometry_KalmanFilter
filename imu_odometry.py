#from realsense_depth.realsense_depth import DepthCamera
import cv2
import numpy as np
import rospy
from numpy.core.fromnumeric import shape
from realsense_depth import DepthCamera
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as RT
from rospy.core import is_shutdown
from std_msgs.msg import Float32MultiArray 
import time
import math
### Realsense Depth camera calling

rtdata = Float32MultiArray()

def callback(data):
    global imu_data
    imu_data = data.data

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]
	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2

def kalman_filter_estimate(ax,ay,az,phi,theta,psi,gx,gy,gz,XXp,PPp,dt=0.01):

    conv_mat = np.array([[1 , 0, -math.sin(theta)],[0, math.cos(phi),math.sin(phi)*math.cos(theta)],[0, -math.sin(phi), math.cos(phi)*math.cos(theta)]])
    inv_mat = np.linalg.inv(conv_mat)
    p,q,r = inv_mat.dot(np.array([[gx],[gy],[gz]]))
    
    ### Motion Model
    ## X_t1 = A @ X_t + B @ T_t + w
    A = np.array([[1, 0, 0, dt, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, dt, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, dt, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                             [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                            [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                    [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    B = np.array([[0.5*(dt**2), 0, 0, 0, 0, 0],
                    [0, 0.5*(dt**2), 0, 0, 0, 0],
                         [0, 0, 0.5*(dt**2), 0, 0, 0],
                            [dt, 0, 0, 0, 0, 0],
                                [0, dt, 0, 0, 0, 0],
                                    [0, 0, dt, 0, 0, 0],
                                        [0,0, 0, dt, 0, 0],
                                            [0, 0, 0, 0, dt, 0],
                                                [0, 0, 0, 0, 0, dt]])


    u = np.array([[ax],[ay],[az],
                            p,q,r])

    Q = np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
    
    ######
    X = A @ XXp + B @ u    
    P = A @ PPp @ A.T + Q
    
    return X, P

def kalman_filter_update(X,P, mx,my,mz, c_phi, c_theta, c_psi):

    H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                            [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    R = np.diag([0.5,0.51,0.51,0.1,0.1,0.1])

    Z = H @ X       #+ np.array([[0.01],[0.01],[0.01],[0.01],[0.01],[0.01]])

    KG = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)
    MZ = np.array([[mx],[my],[mz],[c_phi],[c_theta],[c_psi]])

    X = X + KG @ (MZ - Z)
    P = P - KG @ H @ P  

    return X, P


dc = DepthCamera()
ret, ref_depth_frame, ref_color_frame = dc.get_frame()

## Detector Creat
detector = cv2.ORB_create(edgeThreshold=15,patchSize=20,nlevels=10,fastThreshold=20,
                                            scaleFactor=1.2,WTA_K=2,scoreType=cv2.ORB_FAST_SCORE,firstLevel=0,nfeatures=500)

## Lucas Kanade Parameters
lk_params = dict(winSize  = (21, 21), maxLevel = 8,criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
# convert the image to trhe gray scale
grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
## detect the feature points in the first frame  || draw the key points
px_ref = detector.detect(grey_r)
img_kp_ref = cv2.drawKeypoints(grey_r,px_ref,None,(0,220,0),cv2.DrawMatchesFlags_DEFAULT)
#cv2.imshow('First_Refrence_frame',img_kp_ref)

# convert the feature points to the float32 numbers
px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)

# camera parameters
cx, cy = 322.399, 243.355
fx ,fy = 603.917, 603.917
pp = (cx,cy)
focal = (fx, fy)
#Initial rotation and translation vectors
curp_R = np.ones(3,dtype=None)
curp_t = np.array([[0],[0],[0]])
depth_arr = []
pax = 0
pay = 0
paz = 0
pre_time = time.time()
#print(pre_time)
Xp=np.array([[0],[0],[0],[0],[0],[0],[0],[0],[0]])
Pp=np.diag([0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])

while True:
    rospy.init_node('Imu_odom',anonymous=True)
    pub = rospy.Publisher('/my_odom', Float32MultiArray, queue_size=10 )
    rospy.Subscriber('/custom_imu',Float32MultiArray, callback)
    rate = rospy.Rate(30)
    rate.sleep()

    cur_time = time.time()
    #dt = cur_time - pre_time
    dt = 1/100

    ax  =   imu_data[0]
    ay  =   imu_data[1]
    az  =   imu_data[2]
    gx  =   imu_data[3]
    gy  =   imu_data[4]
    gz  =   imu_data[5]
    roll =  imu_data[6]
    pitch = imu_data[7]
    yaw =   imu_data[8]

    ##### frame treansfornation from imu to camera

    rxi = RT.from_euler('zyx', [yaw, pitch, roll], degrees=True)
    r1 = rxi.as_matrix()
    rx = RT.from_euler('zyx', [0, 0, 90], degrees=True)
    r = rx.as_matrix()
    RR = r @ r1 
    ri = RT.from_matrix(RR)
    roll, pitch, yaw = ri.as_euler('zyx',degrees = True)

    ### scale recovery

    s_x = (ax-pax)*dt*dt
    s_y = (ay-pay)*dt*dt
    s_z = (az-paz)*dt*dt
    
    scale = np.array([np.sqrt(s_x**2)+np.sqrt(s_y**2)+np.sqrt(s_z**2)])
    #print(scale)
    ## capture the current frame

    ret, cur_depth_frame,cur_color_frame = dc.get_frame()

    #convert the frames in the grey scale
    grey_r = cv2.cvtColor(ref_color_frame,cv2.COLOR_BGR2GRAY)
    grey_c = cv2.cvtColor(cur_color_frame,cv2.COLOR_BGR2GRAY)
    
    px_ref = detector.detect(grey_r)
    img_kp_ref = cv2.drawKeypoints(grey_r,px_ref,None,(0,220,0),cv2.DrawMatchesFlags_DEFAULT)
    cv2.putText(img_kp_ref,'Reference Image',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,0),1)
    #cv2.imshow('Previous_reference_frame',img_kp_ref)
    px_ref = np.array([x.pt for x in px_ref], dtype=np.float32)
    
    px_cur = detector.detect(grey_c)
    img_kp_cur = cv2.drawKeypoints(cur_color_frame,px_cur,None,(255,0,0),cv2.DrawMatchesFlags_DEFAULT)
    cv2.putText(img_kp_cur,'Current Image',(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1)
    Windowd_img = np.concatenate((img_kp_ref,img_kp_cur),axis=1)
    cv2.imshow('Images',Windowd_img)
    px_cur = np.array([x.pt for x in px_cur], dtype=np.float32)  
    kp2, st, err = cv2.calcOpticalFlowPyrLK(grey_r,grey_c, px_ref, None, **lk_params)
    st = st.reshape(st.shape[0])
   
    camMatrix = np.array([[fx,0, cx],[0, fy, cy],[0, 0, 1]])
    E, mask = cv2.findEssentialMat(kp2, px_ref,cameraMatrix=camMatrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
     
    s, R, t, mask = cv2.recoverPose(E, kp2, px_ref, cameraMatrix=camMatrix, mask=mask)
    
    cur_t = curp_t + 1000*scale*curp_R.dot(t)
    cur_R = -R.dot(curp_R)
     
    rotation = RT.from_rotvec(cur_R)
    pre_rotation = RT.from_rotvec(curp_R)
    Roll, Pitch, Yaw = rotation.as_euler('xyz',degrees=True)
    pRoll, pPitch, pYaw = pre_rotation.as_euler('xyz',degrees=True)

    #rot = rotation.as_euler('xyz',degrees=True) - pre_rotation.as_euler('xyz',degrees=True)
    XXp = Xp
    PPp = Pp

    X, P = kalman_filter_estimate(ax,ay,az,roll,pitch,yaw,gx,gy,gz,XXp,PPp,(1/30))
    
    X,P = kalman_filter_update(X,P,cur_t[0],cur_t[1],cur_t[2],Roll,Pitch,Yaw)
    print(np.array(X))
    Xp = X
    Pp = P

    ref_color_frame  = cur_color_frame
    ref_depth_frame = cur_depth_frame

    curp_R = cur_R
    curp_t = cur_t
 
    r11 = float(cur_R[0])
    r12 = float(cur_R[1])
    r13 = float(cur_R[2])
    t11 = float(cur_t[0])
    t12 = float(cur_t[1])
    t13 = float(cur_t[2])

    rtdata.data = [r11, r12, r13, t11, t12, t13]
    
    pub.publish(rtdata)
    #rospy.spin()     
    pre_time  = cur_time
    pax = ax
    pay = ay
    paz = az
    key = cv2.waitKey(100)
    if key==27:
        break

dc.release()
cv2.destroyAllWindows()
