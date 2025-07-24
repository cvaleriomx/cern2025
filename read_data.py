import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import sys
#sys.path.insert(0, "/eos/user/s/scrivens/SWAN_projects/lib/")  # for generalFunctions and semgrid
#import generalFunctions as gf
#import semgrid as sg
import scipy.stats
import scipy.special
import json

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

def plot_individual(traces):
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            #ax.plot(range(500), traces.T[:,1]) 
            mediciones = traces[23]  # Esto tiene forma (3, 500)
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
            axs[1].plot(range(len(traces_av_av2)), traces_av_av, label='Avg 180–200')
            axs[1].plot(range(len(traces_av_av2)), traces_av_av2, label='Avg 340–360')

            # Plot scatter points with error bars
            axs[1].errorbar(
                range(len(traces_av_av2)),
                traces_av_av2,
                yerr=traces_std2,
                fmt='o',
                capsize=3,
                
            )
            axs[1].errorbar(
                range(len(traces_av_av)),
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
            plot_individual(traces)  # Plot individual traces
            
            traces_av = np.average(traces, axis=1)
            #traces_av_av = np.average(traces_av[:,200:350], axis=1) 
            traces_av_av = np.average(traces_av[:,200:350], axis=1) 

    return(x_axis, traces_av_av)

def erf2(x, mu, sig):
    val = scipy.special.erf((x-mu)/sig)+1.0
    return(val)

def plot_meas(scan_data):
    sd = scan_data
    fig, ax = plt.subplots()
    ax.plot(sd["LEFT_X"], sd["LEFT_V"])
    ax.plot(sd["RIGHT_X_2"], sd["RIGHT_V_2"])
    if sd.get("mu") is not None:
        xxx = np.linspace(-20.0, 20.0, 100)
        yyy = sd["Am"]*erf2(xxx, sd["mu"], sd["sig"])+sd["A0"]
        ax.plot(xxx, yyy)
    if sd.get("title") is not None:
        ax.set_title(sd["title"])
    ax.set_ylabel("FCup Current (mA)")
    ax.set_xlabel("ITL.SLH01 - Position (mm)")
    fig.show()

# First dataset
filenames = [
    "scanrecord_data_2025-04-22-09-23-41_ITL_SLH01_LEFT_SOL_70A.json",
    "scanrecord_data_2025-04-22-10-16-09_ITL_SLH01_RIGHT_SOL_70A.json",
    "scanrecord_data_2025-04-22-10-51-33_ITL_SLH01_LEFT_SOL_75A.json",
    "scanrecord_data_2025-04-22-10-44-05_ITL_SLH01_RIGHT_SOL_75A.json",
    "scanrecord_data_2025-04-22-09-32-05_ITL_SLH01_LEFT_SOL_80A.json",
    "scanrecord_data_2025-04-22-10-25-08_ITL_SLH01_RIGHT_SOL_80A.json",
    "scanrecord_data_2025-04-22-09-39-17_ITL_SLH01_LEFT_SOL_90A.json",
    "scanrecord_data_2025-04-22-10-37-08_ITL_SLH01_RIGHT_SOL_90A.json", 
    "scanrecord_data_2025-04-22-21-37-32_ITL_SLH01_LEFT_SOL_70A.json",
    "scanrecord_data_2025-04-22-21-41-29_ITL_SLH01_RIGHT_SOL_70A.json", 
]
BASE = "/Users/cvalerio/work1/cern_2025/richard_data/"
d_list = [ {"LEFT_F": BASE+filenames[ii*2], "RIGHT_F": BASE+filenames[ii*2+1]} for ii in range(int(len(filenames)*0.5)) ]

for dd in d_list:
    xx, V = read_file(dd["LEFT_F"])
    dd["LEFT_X"] = xx
    dd["LEFT_V"] = V
    xx, V = read_file(dd["RIGHT_F"])
    dd["RIGHT_X"] = xx
    dd["RIGHT_V"] = V
    dd["MAX_V"] = 0.5*(dd["LEFT_V"][0]+dd["RIGHT_V"][-1])  # Average the "full out values" to get the max intensity
    V2 = [ dd["MAX_V"]-VV for VV in dd["RIGHT_V"] ]        # 
    dd["RIGHT_V_2"] = V2
    dd["RIGHT_X_2"] = [ x2-5.5 for x2 in dd["RIGHT_X"] ]

# Plot the results
fits = {"mu":-8.5, "sig": 8.0, "Am": -0.16, "A0":0.32, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=70A"}    
d_list[0].update(fits)
fits = {"mu":-4.5, "sig": 6.0, "Am": -0.15, "A0":0.3, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=75A"}    
d_list[1].update(fits)
fits = {"mu":-1.5, "sig": 7.5, "Am": -0.125, "A0":0.25, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=80A"}    
d_list[2].update(fits)
fits = {"mu":-2.5, "sig": 8.0, "Am": -0.05, "A0":0.1, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=90A"}    
d_list[3].update(fits)
fits = {"mu":-7.5, "sig": 6.0, "Am": -0., "A0":0.0, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=70.9A"}    
d_list[4].update(fits)
for dd in d_list:
    plot_meas(dd) 

# Second dataset
filenames = [
    "scanrecord_data_2025-04-28-11-10-19_ITL_SLH01_LEFT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-18-17_ITL_SLH01_RIGHT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-46-08_ITL_SLH01_LEFT_SOL_75.json",
    "scanrecord_data_2025-04-28-11-25-08_ITL_SLH01_RIGHT_SOL_75.json",
]
BASE = "/Users/cvalerio/work1/cern_2025/richard_data/"
d_list = [ {"LEFT_F": BASE+filenames[ii*2], "RIGHT_F": BASE+filenames[ii*2+1]} for ii in range(int(len(filenames)*0.5)) ]

for dd in d_list:
    xx, V = read_file(dd["LEFT_F"])
    dd["LEFT_X"] = xx
    dd["LEFT_V"] = V
    xx, V = read_file(dd["RIGHT_F"])
    dd["RIGHT_X"] = xx
    dd["RIGHT_V"] = V
    dd["MAX_V"] = 0.5*(dd["LEFT_V"][0]+dd["RIGHT_V"][-1])  # Average the "full out values" to get the max intensity
    V2 = [ dd["MAX_V"]-VV for VV in dd["RIGHT_V"] ]        # 
    dd["RIGHT_V_2"] = V2
    dd["RIGHT_X_2"] = [ x2-6.0 for x2 in dd["RIGHT_X"] ]

# Plot the results




fits = {"mu":-8.5, "sig": 8.0, "Am": -0.0, "A0":0.0, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=70A"}    
d_list[0].update(fits)
fits = {"mu":-4.5, "sig": 6.0, "Am": -0.0, "A0":0.0, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=75A"}    
d_list[1].update(fits)

for dd in d_list:
    plot_meas(dd) 
plt.show()  # Show all plots at once
