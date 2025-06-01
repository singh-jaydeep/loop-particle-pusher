import constants as c
from constants import bfield_mag
import numpy as np
from numba import njit, prange
import json


def main_iter_loop(): #[Will implement options to choose algorithm, override things like b field, epsilon, dt., etc.]
    (arr_x,arr_y,arr_vx,arr_vy,arr_global) = init_array()

    omega_curr = np.abs(c.bfield_mag( np.average(arr_x), np.average(arr_y)))
    omega_next = 0
    omega_integrate = 0

    for iters in range(0,c.total_iter):

        ############################ Main time-step
        if(c.PARALLEL):
            (arr_x,arr_y,arr_vx,arr_vy) = time_step_parallel(arr_x,arr_y,arr_vx,arr_vy, c.eps, c.dt)
        else:
            (arr_x,arr_y,arr_vx,arr_vy) = time_step(arr_x,arr_y,arr_vx,arr_vy, c.eps, c.dt)

        ############################ Processing which needs to be done during the simulation
        omega_next = np.abs(c.bfield_mag( np.average(arr_x), np.average(arr_y)))
        omega_integrate += .5 * c.dt * 1/c.eps * (omega_curr + omega_next)
        omega_integrate = np.mod(omega_integrate, 2*np.pi)
        omega_curr = omega_next


        ########################### Recording data
        if iters % c.recordinterval_iter == 0 and iters > 0 and c.RECORDING:
            curr_recording = int(iters / c.recordinterval_iter)
            arr_global[curr_recording, :, 0] = arr_x
            arr_global[curr_recording, :, 1] = arr_y
            arr_global[curr_recording, :, 2] = arr_vx
            arr_global[curr_recording, :, 3] = arr_vy
            arr_global[curr_recording, 0, 4] = omega_integrate

            if curr_recording == c.num_record-1:
                c.RECORDING = False ## You've finished your snapshots. Should probably just finish the simulation here.

            print("On iteration ", iters, " out of ", c.total_iter)
            

    ######################## Things to do after simulation
    with open(c.data_path, 'w') as outfile:
        json.dump(arr_global.tolist(), outfile)
    print("Simulated a total of ", c.num_part, " particles")


def init_array():
    arr0 = np.arange(0,c.num_part)
    arr_theta = arr0 * c.theta_sep

    arr_global = np.zeros((c.num_record, c.num_part, 5)) ## for each recording time and particle, store the four values of position (x,y) 
                                                         ## and velocity (vx, vy), and integrated omega
    arr_x = c.init_x *np.ones(c.num_part) + c.eps * np.cos(arr_theta) ## particles uniformly spaced in a circle of radius eps, centered at (init_x,init_y)
    arr_y = c.init_y *np.ones(c.num_part) + c.eps * np.sin(arr_theta)
    arr_vx = np.sin(arr_theta) 
    arr_vy = -np.cos(arr_theta)

    arr_global[0,:,0]=arr_x
    arr_global[0,:,1]=arr_y
    arr_global[0,:,2]=arr_vx
    arr_global[0,:,3]=arr_vy

    return (arr_x,arr_y,arr_vx,arr_vy,arr_global)


@njit ## just in time compiled using numba. Meaningful speedup 
def time_step(arr_x,arr_y,arr_vx,arr_vy, eps, dt): ## Basic Lorentz force equations, using either RK4 or Boris. Specified in 
                                                   ## constants.py

    
    if c.alg == "RK4":

        inv_eps = 1.0 / eps
        half_dt = 0.5 * dt

        bfield_k1 = bfield_mag(arr_x,arr_y)

        arr_x_k1 = arr_vx
        arr_y_k1 = arr_vy
        arr_vx_k1 = inv_eps * np.multiply(arr_vy, bfield_k1)
        arr_vy_k1 = - inv_eps * np.multiply(arr_vx, bfield_k1)

        bfield_k2 = bfield_mag(arr_x + half_dt*arr_x_k1,arr_y +half_dt*arr_y_k1)

        arr_x_k2 = arr_vx+ half_dt*arr_vx_k1
        arr_y_k2 = arr_vy+ half_dt*arr_vy_k1
        arr_vx_k2 = inv_eps * np.multiply( arr_vy + half_dt*arr_vy_k1, bfield_k2)
        arr_vy_k2 = - inv_eps * np.multiply( arr_vx + half_dt*arr_vx_k1, bfield_k2)

        bfield_k3 = bfield_mag(arr_x + half_dt*arr_x_k2,arr_y + half_dt*arr_y_k2)

        arr_x_k3 = arr_vx+ half_dt*arr_vx_k2
        arr_y_k3 = arr_vy+ half_dt*arr_vy_k2
        arr_vx_k3 = inv_eps * np.multiply( arr_vy + half_dt*arr_vy_k2, bfield_k3)
        arr_vy_k3 = - inv_eps * np.multiply( arr_vx + half_dt*arr_vx_k2, bfield_k3)

        bfield_k4 = bfield_mag(arr_x +  dt*arr_x_k3,arr_y +  dt*arr_y_k3)

        arr_x_k4 = arr_vx+ dt*arr_vx_k3
        arr_y_k4 = arr_vy+ dt*arr_vy_k3
        arr_vx_k4 = inv_eps * np.multiply( arr_vy + dt*arr_vy_k3, bfield_k4)
        arr_vy_k4 = - inv_eps * np.multiply( arr_vx + dt*arr_vx_k3, bfield_k4)

        arr_x_next = arr_x + 1/6 * dt * ( arr_x_k1 + 2 * arr_x_k2 + 2 *arr_x_k3 + arr_x_k4)
        arr_y_next = arr_y + 1/6 * dt * ( arr_y_k1 + 2 * arr_y_k2 + 2 *arr_y_k3 + arr_y_k4)
        arr_vx_next = arr_vx + 1/6 * dt * ( arr_vx_k1 + 2 * arr_vx_k2 + 2 *arr_vx_k3 + arr_vx_k4)
        arr_vy_next = arr_vy + 1/6 * dt * ( arr_vy_k1 + 2 * arr_vy_k2 + 2 *arr_vy_k3 + arr_vy_k4)

        return (arr_x_next, arr_y_next, arr_vx_next, arr_vy_next)
    
    if c.alg == "BORIS":
        temp = 0.5 * 1.0/eps * dt
        t_vec = temp * bfield_mag(arr_x,arr_y)
        s_vec = 2.0 / (1 + np.power(t_vec,2))*t_vec

        arr_vx_inter = arr_vx + np.multiply(arr_vy, t_vec)
        arr_vy_inter = arr_vy - np.multiply(arr_vx, t_vec)

        arr_vx_next = arr_vx + np.multiply(arr_vy_inter, s_vec)
        arr_vy_next = arr_vy - np.multiply(arr_vx_inter, s_vec)

        arr_x_next = arr_x + dt*arr_vx_next
        arr_y_next = arr_y + dt*arr_vy_next

        return (arr_x_next, arr_y_next, arr_vx_next, arr_vy_next)    

@njit(parallel=True) ## Code with parallel loops. Haven't tested, but should be faster for larger particle numbers
def time_step_parallel(arr_x,arr_y,arr_vx,arr_vy, eps, dt): 
    N = arr_x.shape[0] 
    inv_eps = 1.0 / eps
    half_dt = 0.5 * dt

    arr_x_next = np.empty(N)
    arr_y_next = np.empty(N)
    arr_vx_next = np.empty(N)
    arr_vy_next = np.empty(N)

    if c.alg == "RK4":
        for i in prange(N):
            x = arr_x[i]
            y = arr_y[i]
            vx = arr_vx[i]
            vy = arr_vy[i]

            bfield_k1=bfield_mag(x,y)

            x_k1 = vx
            y_k1 = vy
            vx_k1 = inv_eps * vy * bfield_k1
            vy_k1 = - inv_eps * vx * bfield_k1

            bfield_k2 = bfield_mag(x + half_dt*x_k1, y + half_dt*y_k1)

            x_k2 = vx + half_dt * vx_k1
            y_k2 = vy+ half_dt*vy_k1
            vx_k2 = inv_eps * np.multiply( vy + half_dt*vy_k1, bfield_k2)
            vy_k2 = - inv_eps * np.multiply( vx + half_dt*vx_k1, bfield_k2)

            bfield_k3 = bfield_mag(x + half_dt*x_k2,y + half_dt*y_k2)

            x_k3 = vx+ half_dt*vx_k2
            y_k3 = vy+ half_dt*vy_k2
            vx_k3 = inv_eps * np.multiply( vy + half_dt*vy_k2, bfield_k3)
            vy_k3 = - inv_eps * np.multiply( vx + half_dt*vx_k2, bfield_k3)

            bfield_k4 = bfield_mag(x +  dt*x_k3,arr_y +  dt*y_k3)

            x_k4 = vx+ dt*vx_k3
            y_k4 = vy+ dt*vy_k3
            vx_k4 = inv_eps * np.multiply( vy + dt*vy_k3, bfield_k4)
            vy_k4 = - inv_eps * np.multiply( vx + dt*vx_k3, bfield_k4)

            arr_x_next[i] = x + 1/6 * dt * ( x_k1 + 2 * x_k2 + 2 *x_k3 + x_k4)
            arr_y_next[i] = y + 1/6 * dt * ( y_k1 + 2 * y_k2 + 2 *y_k3 + y_k4)
            arr_vx_next[i] = vx + 1/6 * dt * ( vx_k1 + 2 * vx_k2 + 2 *vx_k3 + vx_k4)
            arr_vy_next[i] = vy + 1/6 * dt * ( vy_k1 + 2 * vy_k2 + 2 *vy_k3 + vy_k4)


        return (arr_x_next, arr_y_next, arr_vx_next, arr_vy_next)
    
    
    if c.alg == "BORIS":
        temp = 0.5 * 1.0/eps * dt
        for i in prange(N):
            x = arr_x[i]
            y = arr_y[i]
            vx = arr_vx[i]
            vy = arr_vy[i]
            bfield = bfield_mag(x,y)
            tval = temp * bfield

            vx_inter = vx + vy *tval
            vy_inter = vy - vx*tval
            arr_vx_next[i] = vx + vy_inter * 2/(1 + np.power(tval,2))*tval
            arr_vy_next[i] = vy - vx_inter * 2/(1 + np.power(tval,2))*tval
            arr_x_next[i] = x + dt*arr_vx_next[i]
            arr_y_next[i] = y + dt*arr_vy_next[i]
        return (arr_x_next, arr_y_next, arr_vx_next, arr_vy_next)
    
    return (arr_x_next, arr_y_next, arr_vx_next, arr_vy_next)

def energy(arr_x,arr_y,arr_vx, arr_vy):
    return .5*( np.power(arr_vx,2) + np.power(arr_vy,2))

def mag_moment(arr_x,arr_y,arr_vx,arr_vy):
    return np.divide(.5*( np.power(arr_vx,2) + np.power(arr_vy,2)), bfield_mag(arr_x,arr_y))





