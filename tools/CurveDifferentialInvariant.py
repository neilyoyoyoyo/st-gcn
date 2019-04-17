import os 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import splprep, splev 
import scipy.signal as signal 
from ntu_read_skeleton import read_xyz 

def SimpleFeatures(px, py, pz): 

    t = np.arange(len(px)) / 30.0 
    SF = np.zeros((6, len(t))) 

    SF[0, :] = px ** 2 
    SF[1, :] = py ** 2 
    SF[2, :] = pz ** 2 
    SF[3, :] = px * py 
    SF[4, :] = py * pz 
    SF[5, :] = px * pz 

    return SF 

def CurveDifferential(px, py, pz, eps=1e-15, kernel_size=5, smooth_factor=0.1, tanh_factor=0.2, visualization=False):

    px = signal.medfilt(px, kernel_size) 
    py = signal.medfilt(py, kernel_size) 
    pz = signal.medfilt(pz, kernel_size) 

    t = np.arange(len(px)) / 30.0  
    CD = np.zeros((9, len(t))) 

    tck, _ = splprep([px, py, pz], u=t, k=5, s=smooth_factor) 

    f1 = splev(t, tck, der=1) 
    CD[0:3, :] = f1  
    f2 = splev(t, tck, der=2) 
    CD[3:6, :] = f2 
    f3 = splev(t, tck, der=3) 
    CD[6:9, :] = f3 

    CD = np.tanh(CD * tanh_factor)  
    
    if visualization: 
        for i in range(9): 
            plt.subplot(3, 3, i+1) 
            plt.plot(CD[i, :], linewidth=1) 
        plt.show() 

    return CD

def CurveDifferentialInvariant(px, py, pz, eps=1e-15, kernel_size=5, smooth_factor=0.1, tanh_factor=0.2, visualization=False):

    px = signal.medfilt(px, kernel_size) 
    py = signal.medfilt(py, kernel_size) 
    pz = signal.medfilt(pz, kernel_size) 

    t = np.arange(len(px)) / 30.0  
    CDI = np.zeros((8, len(t))) 

    tck, _ = splprep([px, py, pz], u=t, k=5, s=smooth_factor) 

    f0 = splev(t, tck, der=0) 
    fx0, fy0, fz0 = f0[0], f0[1], f0[2] 
    f1 = splev(t, tck, der=1) 
    fx1, fy1, fz1 = f1[0], f1[1], f1[2] 
    f2 = splev(t, tck, der=2) 
    fx2, fy2, fz2 = f2[0], f2[1], f2[2] 
    f3 = splev(t, tck, der=3) 
    fx3, fy3, fz3 = f3[0], f3[1], f3[2] 
    f4 = splev(t, tck, der=4) 
    fx4, fy4, fz4 = f4[0], f4[1], f4[2] 

    fx0 = fx0 - np.mean(fx0) 
    fy0 = fy0 - np.mean(fy0) 
    fz0 = fz0 - np.mean(fz0) 

    for i in range(len(t)): 

        M012 = np.array([[fx0[i], fx1[i], fx2[i]], [fy0[i], fy1[i], fy2[i]], [fz0[i], fz1[i], fz2[i]]]) 
        M013 = np.array([[fx0[i], fx1[i], fx3[i]], [fy0[i], fy1[i], fy3[i]], [fz0[i], fz1[i], fz3[i]]]) 
        M023 = np.array([[fx0[i], fx2[i], fx3[i]], [fy0[i], fy2[i], fy3[i]], [fz0[i], fz2[i], fz3[i]]]) 
        M014 = np.array([[fx0[i], fx1[i], fx4[i]], [fy0[i], fy1[i], fy4[i]], [fz0[i], fz1[i], fz4[i]]])
        M123 = np.array([[fx1[i], fx2[i], fx3[i]], [fy1[i], fy2[i], fy3[i]], [fz1[i], fz2[i], fz3[i]]]) 
        M024 = np.array([[fx0[i], fx2[i], fx4[i]], [fy0[i], fy2[i], fy4[i]], [fz0[i], fz2[i], fz4[i]]]) 
        M124 = np.array([[fx1[i], fx2[i], fx4[i]], [fy1[i], fy2[i], fy4[i]], [fz1[i], fz2[i], fz4[i]]]) 
        M034 = np.array([[fx0[i], fx3[i], fx4[i]], [fy0[i], fy3[i], fy4[i]], [fz0[i], fz3[i], fz4[i]]]) 
        M134 = np.array([[fx1[i], fx3[i], fx4[i]], [fy1[i], fy3[i], fy4[i]], [fz1[i], fz3[i], fz4[i]]]) 
        M234 = np.array([[fx2[i], fx3[i], fx4[i]], [fy2[i], fy3[i], fy4[i]], [fz2[i], fz3[i], fz4[i]]]) 

        u012 = np.linalg.det(M012) 
        u013 = np.linalg.det(M013) 
        u023 = np.linalg.det(M023) 
        u014 = np.linalg.det(M014) 
        u123 = np.linalg.det(M123) 
        u024 = np.linalg.det(M024) 
        u124 = np.linalg.det(M124) 
        u034 = np.linalg.det(M034) 
        u134 = np.linalg.det(M134) 
        u234 = np.linalg.det(M234) 

        CDI[0, i] = u023 / (u014+eps) # I1 = (023) / (014) 
        CDI[1, i] = u024 / (u123+eps) # I2 = (024) / (123) 
        CDI[2, i] = u034 / (u124+eps) # I3 = (034) / (124) 
        CDI[3, i] = (u012*u023) / (u013**2+eps**2) # I4 = (012)*(023) / (013)**2
        CDI[4, i] = (u013*u123) / (u014**2+eps**2) # I5 = (013)*(123) / (014)**2  
        CDI[5, i] = (u023*u124) / (u123**2+eps**2) # I6 = (023)*(124) / (123)**2 
        CDI[6, i] = (u123*u134) / (u124**2+eps**2) # I7 = (123)*(134) / (124)**2
        CDI[7, i] = (u124*u234) / (u134**2+eps**2) # I8 = (124)*(234) / (134)**2 

    CDI = np.tanh(CDI * tanh_factor)  
    
    if visualization: 
        for i in range(8): 
            plt.subplot(3, 3, i+1) 
            plt.plot(CDI[i, :], linewidth=1) 
        plt.show() 

    return CDI

if __name__ == '__main__': 

    data_path = '/exp/datasets/NTURGB-D/nturgb+d_skeletons' 
    test_skeleton = 'S014C002P037R002A050.skeleton' 
    test_joint = 8 
    test_human = 0 

    data = read_xyz(os.path.join(data_path, test_skeleton))
    px, py, pz = data[0, :, test_joint, test_human], data[1, :, test_joint, test_human], data[2, :, test_joint, test_human]
    CD = CurveDifferential(px, py, pz, visualization=True)  
    print(CD) 