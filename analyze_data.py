import numpy as np
import matplotlib.pyplot as plot
import json
import os
from scipy import integrate
import constants as c
import fns


def read_data():
    try:
        with open(c.data_path, 'r') as infile:
                data_array = np.array(json.load(infile))
                print("File found")
                return data_array
    except FileNotFoundError:
         print(f"File not found at {c.data_path}")
         return None 


def summary_stats(data_array):
    arr_x = data_array[:,:,0]
    arr_y = data_array[:,:,1]
    arr_vx = data_array[:,:,2]
    arr_vy = data_array[:,:,3]
    arr_omega_integrate = data_array[:,0,4]


    energy_array = fns.energy(arr_x, arr_y, arr_vx, arr_vy)
    magmoment_array = fns.mag_moment(arr_x, arr_y, arr_vx, arr_vy)
    print("Simulated ", c.num_part, " particles")
    print("Initial energies are ", energy_array[0,:])
    print("Initial magnetic moments are ",magmoment_array[0,:])

    energy_diff_array = energy_array - np.tensordot(np.ones(c.num_record),energy_array[0,:], axes=0)
    print("Max energy deviation from t=0 is ", np.max(np.abs(energy_diff_array)))

    magmoment_diff_array = magmoment_array - np.tensordot(np.ones(c.num_record),magmoment_array[0,:], axes=0)
    print("Max magnetic moment deviation from t=0 is ", np.max(np.abs(magmoment_diff_array)))

    fig, axes = plot.subplots(nrows=1, ncols=3, figsize=(4, 3))
    t = np.linspace(0,c.total_t, c.num_record)
    axes[0].scatter(t, energy_array[:,0])
    axes[0].set_title("energy of 0th particle")

    axes[1].scatter(t, magmoment_array[:,0])
    axes[1].set_title("magnetic moment of 0th particle")

    axes[2].scatter(t,arr_omega_integrate)
    axes[2].set_title("loop phase shift")


     
    plot.show()


def loop_visualization(data_array):
    arr_x = data_array[:,:,0]
    arr_y = data_array[:,:,1]
    arr_vx = data_array[:,:,2]
    arr_vy = data_array[:,:,3]
    arr_omega_integrate = data_array[:,0,4]

    
    (arr_rescaledx,arr_rescaledy) = mean_removed_POST(arr_x,arr_y)

    fig, axes = plot.subplots(nrows=2, ncols=2, figsize=(8, 6))

    time1_frac = 0 ## What fraction of the total simulation time do you want to plot, in each of the four subplots
    time2_frac = .7
    time3_frac = .8
    time4_frac = .9

    time1 = int(c.num_record * time1_frac) * c.dt * c.recordinterval_iter
    time2 = int(c.num_record * time2_frac) * c.dt * c.recordinterval_iter
    time3 = int(c.num_record * time3_frac) * c.dt * c.recordinterval_iter
    time4  =int(c.num_record * time4_frac) * c.dt * c.recordinterval_iter

    axes[0,0].scatter(arr_rescaledx[int(c.num_record * time1_frac)],arr_rescaledy[int(c.num_record * time1_frac)])
    axes[0,0].set_title(f"time = {time1}")

    axes[1,0].scatter(arr_rescaledx[int(c.num_record * time2_frac)],arr_rescaledy[int(c.num_record * time2_frac)])
    axes[1,0].set_title(f"time = {time2}")

    axes[0,1].scatter(arr_rescaledx[int(c.num_record * time3_frac)],arr_rescaledy[int(c.num_record * time3_frac)])
    axes[0,1].set_title(f"time = {time3}")

    axes[1,1].scatter(arr_rescaledx[int(c.num_record * time4_frac)],arr_rescaledy[int(c.num_record * time4_frac)])
    axes[1,1].set_title(f"time = {time4}")

    plot.show()


def slowmanifold_visualization(data_array):
    (harmonic0_x,harmonic0_y,harmonic1p_x,harmonic1p_y,harmonic1m_x, harmonic1m_y,harmonic0_vx,harmonic0_vy,
            harmonic1p_vx,harmonic1p_vy,harmonic1m_vx, harmonic1m_vy) = harmonic_arrays(data_array)
    fig, axes = plot.subplots(nrows=3, ncols=2, figsize=(6, 4))
    t = np.linspace(0,c.total_t, c.num_record)

    axes[0,0].scatter(t, harmonic0_vx[:,0])
    axes[0,0].set_title("0th harmonic vx  ")

    axes[0,1].scatter(t, harmonic0_vy[:,0])
    axes[0,1].set_title("0th harmonic vy ")

    axes[1,0].scatter(t, harmonic1p_vx[:]-harmonic1m_vy[:])
    axes[1,0].set_title("adiab velocity -- plus")

    axes[1,1].scatter(t, harmonic1p_vy[:]+harmonic1m_vx[:])
    axes[1,1].set_title("adiab velocity -- minus")

    axes[2,0].scatter(t, harmonic1p_vx[:]+harmonic1m_vy[:])
    axes[2,0].set_title("nonadiab velocity -- plus")

    axes[2,1].scatter(t, harmonic1p_vy[:]-harmonic1m_vx[:])
    axes[2,1].set_title("nonadiab velocity -- minus")
     
    plot.show()

def harmonic_visualization(data_array):
    (harmonic0_x,harmonic0_y,harmonic1p_x,harmonic1p_y,harmonic1m_x, harmonic1m_y,harmonic0_vx,harmonic0_vy,
            harmonic1p_vx,harmonic1p_vy,harmonic1m_vx, harmonic1m_vy) = harmonic_arrays(data_array)
    
    fig, axes = plot.subplots(nrows=3, ncols=2, figsize=(6, 4))
    t = np.linspace(0,c.total_t, c.num_record)

    axes[0,0].scatter(t, harmonic0_x[:,0])
    axes[0,0].set_title("0th harmonic x -- mean removed ")

    axes[0,1].scatter(t, harmonic0_y[:,0])
    axes[0,1].set_title("0th harmonic y -- mean removed")

    axes[1,0].scatter(t, harmonic1p_x[:])
    axes[1,0].set_title("1st harmonic x -- plus")

    axes[1,1].scatter(t, harmonic1p_y[:])
    axes[1,1].set_title("1st harmonic y -- plus")

    axes[2,0].scatter(t, harmonic1m_x[:])
    axes[2,0].set_title("1st harmonic x -- minus")

    axes[2,1].scatter(t, harmonic1m_y[:])
    axes[2,1].set_title("1st harmonic y -- minus")
     
    plot.show()

def harmonic_arrays(data_array): ## At present, returns the 0th and 1st harmonics for position and velocity
    arr_x = data_array[:,:,0]
    arr_y = data_array[:,:,1]
    arr_vx = data_array[:,:,2]
    arr_vy = data_array[:,:,3]
    arr_omega_integrate = data_array[:,0,4]
    (arr_rescaledx,arr_rescaledy) = mean_removed_POST(arr_x,arr_y)

    harmonic0_x = nth_harmonic_array(arr_rescaledx, arr_omega_integrate, 0)
    harmonic0_y = nth_harmonic_array(arr_rescaledy, arr_omega_integrate, 0)

    harmonic0_vx = nth_harmonic_array(arr_vx, arr_omega_integrate, 0)
    harmonic0_vy = nth_harmonic_array(arr_vy, arr_omega_integrate, 0)

    temp_harmonic1x = nth_harmonic_array(arr_rescaledx, arr_omega_integrate, 1)
    temp_harmonic1y = nth_harmonic_array(arr_rescaledx, arr_omega_integrate, 1)
    harmonic1p_x = temp_harmonic1x[:,0]
    harmonic1p_y = temp_harmonic1y[:,0]
    harmonic1m_x = temp_harmonic1x[:,1]
    harmonic1m_y = temp_harmonic1y[:,1]

    temp_harmonic1vx = nth_harmonic_array(arr_vx, arr_omega_integrate, 1)
    temp_harmonic1vy = nth_harmonic_array(arr_vy, arr_omega_integrate, 1)
    harmonic1p_vx = temp_harmonic1vx[:,0]
    harmonic1p_vy = temp_harmonic1vy[:,0]
    harmonic1m_vx = temp_harmonic1vx[:,1]
    harmonic1m_vy = temp_harmonic1vy[:,1]

    return (harmonic0_x,harmonic0_y,harmonic1p_x,harmonic1p_y,harmonic1m_x, harmonic1m_y,harmonic0_vx,harmonic0_vy,
            harmonic1p_vx,harmonic1p_vy,harmonic1m_vx, harmonic1m_vy)

def center_POST(arr_x,arr_y): ## the input arrays are num_record x num_part, ## POST indicates that this computes on the array post simulation
    return np.mean(arr_x, axis=1), np.mean(arr_y,axis=1)
    

def mean_removed_POST(arr_x,arr_y): ## POST indicates that this computes on the array post simulation
    arr_center_x, arr_center_y = center_POST(arr_x,arr_y)
    return (1/c.eps*(arr_x - np.tensordot(arr_center_x, np.ones(c.num_part), axes=0) ), 
            1/c.eps*(arr_y - np.tensordot(arr_center_y, np.ones(c.num_part), axes=0)))


def nth_harmonic_array(arr_input, arr_omega_integrate, n): ## here arr_input has shape (num_record,num_part)
    if(n != 0):
         arr_return = np.zeros((c.num_record,2)) ## will return the sin, cos harmonics at each recorded time
         for i in range(0, c.num_record):
            arr_return[i,:] = nth_harmonic_fixedT(arr_input[i,:], arr_omega_integrate[i], n)
         return arr_return
    
    if(n == 0):
        arr_return = np.zeros((c.num_record,1)) ## will return the 0th harmonic (the mean)
        for i in range(0, c.num_record):
            arr_return[i,:] = nth_harmonic_fixedT(arr_input[i,:], arr_omega_integrate[i], n)
        return arr_return 

def nth_harmonic_fixedT(arr_input, omega_integrate, n ): ## this is integration at fixed time, n = which harmonic (non-negative integer)
    arr_theta = np.arange(0,c.num_part) * c.theta_sep

    if(n != 0):
        arr_input_modified_sin = np.multiply(arr_input, np.sin(n * (arr_theta + omega_integrate)))
        arr_input_modified_cos = np.multiply(arr_input, np.cos(n * (arr_theta + omega_integrate)))
    
        arr_return = [0,0]
        arr_return[0] = 1/(2*np.pi)*integrate.simpson(arr_input_modified_sin, arr_theta)
        arr_return[1] = 1/(2*np.pi)*integrate.simpson(arr_input_modified_cos, arr_theta)

        return arr_return
    
    if(n == 0):
        arr_return = [0]
        arr_return[0] = 1/(2*np.pi)*integrate.simpson(arr_input, arr_theta)

        return arr_return
        
