import os
import csv
import matplotlib
#import tkinter as tk

matplotlib.use('TkAgg')  # o 'Qt5Agg' si prefieres
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.insert(0, "/eos/user/s/scrivens/SWAN_projects/lib/")  # for generalFunctions and semgrid
#import generalFunctions as gf
#import semgrid as sg
import scipy.stats
import scipy.special
from scipy.optimize import curve_fit

import json

from scipy import special
fits = {"mu":-1.5, "sig": 7.5, "Am": -0.125, "A0":0.25, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=80A"}    
mu=2
sig=4
import matplotlib.pyplot as plt
x = np.linspace(-30, 30,200)
plt.plot(x,scipy.special.erf((x-mu)/sig))
plt.xlabel('$x$')
plt.ylabel('$erf(x)$')
plt.show()


def find_dict_with_key_value(items, key_name, value):
    for item in items:
        if isinstance(item, dict) and item.get(key_name) == value:
            return item
    return None  # Return None if no matching dictionary is found

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def plot_individual(traces,x_axis):
            
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            #ax.plot(range(500), traces.T[:,1]) 
            mediciones = traces[12]  # Esto tiene forma (3, 500)
            #mediciones2 = traces[44]  # Esto tiene forma (3, 500)
            # Graficar cada medición
            for i in range(3):
                #plt.figure()
                plt.plot(mediciones[i])
                #plt.plot(mediciones2[i])

                plt.title(f"step {23}")
                plt.xlabel("Índice de dato")
                plt.ylabel("Valor")
                plt.grid(True)
            
            
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            traces_av = np.average(traces, axis=1)  # Shape: (45, 500)
            traces_std = np.std(traces, axis=1)     # Shape: (45, 500), std across 3 measurements

            axs[0].plot(range(500), traces_av.T)  # Plot each averaged trace

            # --- Now for the second plot (average over time window) ---
            # Averages over time windows:
            traces_av_av = np.average(traces_av[:, 180:200], axis=1)   # (45,)
            traces_av_av2 = np.average(traces_av[:, 340:360], axis=1)  # (45,)

            # Error bars: std of time window 200:250 for each trace
            traces_std = np.std(traces_av[:, 180:200], axis=1)        # (45,)
            traces_std2 = np.std(traces_av[:, 340:360], axis=1)        # (45,)

            # Plot lines
            axs[1].plot(x_axis, traces_av_av, label='Avg 180–200')
            axs[1].plot(x_axis, traces_av_av2, label='Avg 340–360')

            # Plot scatter points with error bars
            axs[1].errorbar(
                x_axis,
                traces_av_av2,
                yerr=traces_std2,
                fmt='o',
                capsize=3,
                
            )
            axs[1].errorbar(
                x_axis,
                traces_av_av,
                yerr=traces_std,
                fmt='o',
                capsize=3
            )

            axs[1].set_title('Average over time windows with error bars')
            axs[1].legend()
            axs[1].grid(True)

            plt.tight_layout()
            plt.show()
            
def read_file(filename):
    json_data = read_json_file(filename)
    data_names = json_data["datanames"]

    scan_details = find_dict_with_key_value(json_data["header"], "type", "scan")   # Look for the first type: scan item
    x0 = scan_details["start_value"]
    dx = scan_details["step_value"]
    nx = scan_details["steps"]
    x_axis = [ x0 + dx*i for i in range(nx) ]

    for dd in json_data["data"]:
        if dd["type"]=="record" and dd["class"]=="oasis":
            traces = np.array(dd["data"])                      # data (for oasis) is in form of scan_step, meas_per_step, pts_traces
            print(traces.shape,filename)
            #plot_individual(traces,x_axis)  # Plot individual traces
            
            traces_av = np.average(traces, axis=1)
            #traces_av_av = np.average(traces_av[:,200:350], axis=1) 
            traces_av_av = np.average(traces_av[:,200:220], axis=1) 

    return(x_axis, traces_av_av)

def erf2(x, mu, sig):
    val = scipy.special.erf((x-mu)/sig)
    return(val)

def erf21(x, mu, sig):
    val = scipy.special.erf((x-mu)/sig)+1
    return(val)

def plot_meas(scan_data):
    sd = scan_data
    fig, ax = plt.subplots()
    ax.plot(sd["LEFT_X"], sd["LEFT_V"])
    ax.plot(sd["RIGHT_X_2"], sd["RIGHT_V_2"])
    if sd.get("mu") is not None:
        xxx = np.linspace(-20.0, 20.0, 100)
        yyy = sd["Am"]*erf21(xxx, sd["mu"], sd["sig"])+sd["A0"]
        ax.plot(xxx, yyy)
    if sd.get("title") is not None:
        ax.set_title(sd["title"])
    ax.set_ylabel("FCup Current (mA)")
    ax.set_xlabel("ITL.SLH01 - Position (mm)")
    fig.show()


filenames = [
    "scanrecord_data_2025-04-28-11-10-19_ITL_SLH01_LEFT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-18-17_ITL_SLH01_RIGHT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-46-08_ITL_SLH01_LEFT_SOL_75.json",
    "scanrecord_data_2025-04-28-11-25-08_ITL_SLH01_RIGHT_SOL_75.json",
]
BASE = ""
d_list = [ {"LEFT_F": BASE+filenames[ii*2], "RIGHT_F": BASE+filenames[ii*2+1]} for ii in range(int(len(filenames)*0.5)) ]

for dd in d_list:
    xxl, VL = read_file(dd["LEFT_F"])
    dd["LEFT_X"] = xxl
    dd["LEFT_V"] = VL
    xx, V = read_file(dd["RIGHT_F"])
    dd["RIGHT_X"] = xx
    dd["RIGHT_V"] = V
    dd["MAX_V"] = 0.5*(dd["LEFT_V"][0]+dd["RIGHT_V"][-1])  # Average the "full out values" to get the max intensit

    print("max",dd["MAX_V"])
    
    V2 = [ dd["MAX_V"]-VV for VV in dd["RIGHT_V"] ]        # 

    V2R = [ VV/dd["MAX_V"] for VV in dd["RIGHT_V"] ]        
    V2L = [ -VV/dd["MAX_V"] for VV in dd["LEFT_V"] ]        

    # 
    dd["RIGHT_V_2"] = V2
    dd["RIGHT_X_2"] = [ x2 for x2 in dd["RIGHT_X"] ]
    dd["RIGHT_X_2"] = [ x2 for x2 in dd["RIGHT_X"] ]
    fig, ax = plt.subplots()
    ax.scatter(dd["RIGHT_X_2"],V2R)
    ax.scatter(dd["LEFT_X"],V2L)
    ax.set_title("right x2 v2")
    
    popt, pcov = curve_fit(erf2, xx, V2R, p0=[-1.0, 4.0])  # p0 son valores iniciales para mu y sig
    # Parámetros ajustados
    mu_fit, sig_fit = popt
    print("Parámetros ajustados:")
    print("mu =", mu_fit)
    print("sig =", sig_fit)
    x_fit = np.linspace(min(xx), max(xx), 500)
    y_fit = erf2(x_fit, mu_fit, sig_fit)
    ax.plot(x_fit, y_fit, '-', label=f'Ajuste erf2: mu={mu_fit:.2f}, sig={sig_fit:.2f}')
    x_check = np.linspace(-20, 20, 500)
    ax.plot(x_check, erf2(x_check, mu_fit, sig_fit), 'o', label='Datos originales')
    #ax.plot(xx, erf2(xx, mu_fit,sig_fit))
    #, 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    #ax.plot(xx, erf2(xx, *popt), color='red', label='Ajuste')
fits = {"mu":-1.5, "sig": 16.0, "Am": -0.16, "A0":0.30, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=70A"}    
d_list[0].update(fits)
fits = {"mu":-4.5, "sig": 16.0, "Am": -0.16, "A0":0.30, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=75A"}    
d_list[1].update(fits)

for dd in d_list:
    plot_meas(dd) 
plt.show()  # Show all plots at once





