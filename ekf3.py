
import math
from math import pi
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d

ax = plt.axes(projection='3d')
# define covariance matrix
fig = plt.figure()
Q =np.diag([0.1, 0.1, np.deg2rad(1.0)]) **2
R = np.diag([1.0, 1.0]) **2

# simulation parameter
Input_noise = np.diag([1.0, np.deg2rad(30.0)]) **2
Meas_noise = np.diag([0.5, 0.5]) **2

dt = 0.1
Sim_Time = 100.0
show_animation = True


def input_vector(xGoal,xEst,u):
    theta = math.atan((xGoal[1,0]-xEst[1,0])/(xGoal[0,0]-xEst[0,0]))
    if ((xGoal[1,0]-xEst[1,0])<0):
        theta = theta - pi
    elif ((xGoal[0,0]-xEst[0,0])<0):
        theta = theta - pi
    
    u[1,0]=theta -xEst[2,0]
    print('xEst = ', xEst)
    print('theta =', theta)
    print('u = ', u[1])
    return u

def motion_model(x, u):
    A = np.array([[1.0, 0, 0],[0, 1.0, 0],[0, 0, 1.0]])

    B = np.array([[dt * math.cos(x[2, 0]), 0],
                  [dt * math.sin(x[2, 0]), 0],
                  [0.0, dt]])

    x= A @ x + B @ u

    return x


def observation_model(x):
    H = np.array([[1, 0, 0],[0, 1, 0]])

    z = H @ x
    return z

def observation(xTrue, xd, u):
    xTrue = motion_model(xTrue, u)

    #add noise to measurement x-y
    z = observation_model(xTrue) + Meas_noise @ np.random.rand(2, 1)

    d=np.shape(z)
    print('ddddddddd =',d)

    # add noise to input 
    ud = u + Input_noise @ np.random.randn(2, 1)

    xd = motion_model(xd, ud)

    return xTrue, z, xd, ud


def jacob_f(x,u):
    alpha = x[2, 0]
    v = u[0, 0]
    jA = np.array([
        [1.0, 0.0, -dt * v * math.sin(alpha)],
        [0.0, 1.0, dt * v * math.cos(alpha)],
        [0.0, 0.0, 1.0]])

    return jA


def jacob_h():
    # jacobion for observation model
    jH  =np.array([[1, 0, 0],[0, 1, 0]])

    return jH

def ekf_estimation(xEst, PEst, z, u):

    #prediction
    xPred = motion_model(xEst, u)
    jA = jacob_f(xEst, u)

    PPred = jA @ PEst @ jA.T + Q

    # updation correction
    jH = jacob_h()
    zPred = observation_model(xPred)

    y = z - zPred

    # process covariance matrix
    S = jH @ PPred @ jH.T + R

    # Kalman gain 
    K = PPred @ jH.T @ np.linalg.inv(S)

    xEst = xPred + K @ y
    #xEst[2]= normalizeangle(xEst[2])

    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred

    return xEst, PEst


def main():
    time = 0.0

    # state vector [x y alpha]

    xEst = np.zeros((3, 1))
    xTrue = np.zeros((3, 1))
    PEst = np.eye(3)

    xDR = np.zeros((3, 1))

    # History 

    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    # input Vectors for initialization
    v = 0.5            # in m/s
    w = 0.10            # in rad/s

    u = np.array([[v], [w]])

    # goal state
    xGoal = np.array([[3],[3]])

    while Sim_Time >= time: # add goal condition
        time += dt
        u = input_vector(xGoal,xEst,u)

        xTrue, z, xDR, ud = observation(xTrue, xDR, u) #add noise

        xEst, PEst = ekf_estimation(xEst, PEst, z, ud) # estimation

        # store data history
        hxEst= np.hstack((hxEst,xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue,xTrue))
        print(hxEst)
        hz = np.hstack((hz, z))

        if ((abs(xGoal[0, 0] - xEst[0, 0]))<0.1):

            if ((abs(xGoal[1,0]-xEst[1,0]))<0.1):
                print('Reached goal')
                break

        elif show_animation:
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0,:].flatten(),hxTrue[1, :].flatten(), "-b")

            plt.plot(hxEst[0, :].flatten(), hxEst[1, :].flatten(),"-r")

            ax.scatter3D(hxTrue[0,:].flatten(),hxTrue[1, :].flatten(),hxEst[0,:].flatten(),"-c")

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__=='__main__':
    main()
    plt.pause(100)
