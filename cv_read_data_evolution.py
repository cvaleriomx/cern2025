import os
import csv
import matplotlib
#import tkinter as tk

#matplotlib.use('TkAgg')  # o 'Qt5Agg' si prefieres
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
#sys.path.insert(0, "/eos/user/s/scrivens/SWAN_projects/lib/")  # for generalFunctions and semgrid
#import generalFunctions as gf
#import semgrid as sg
import scipy.stats
import scipy.special
from scipy.optimize import curve_fit

import json

from scipy import special
colores = [
    'red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan',
    'magenta', 'teal', 'navy', 'gold', 'darkred', 'darkgreen', 'indigo', 'lime', 'coral', 'darkblue',
    'orchid', 'salmon', 'chocolate', 'maroon', 'turquoise'
]
fits = {"mu":-1.5, "sig": 7.5, "Am": -0.125, "A0":0.25, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=80A"}    
mu=2
sig=4
sigmafitsL=[]
sigmafitsR=[]
sigmafits=[]
mufitL=[]
mufitR=[]
mufitF=[]
#x = np.linspace(-30, 30,200)
#plt.plot(x,scipy.special.erf((x-mu)/sig))
#plt.xlabel('$x$')
#plt.ylabel('$erf(x)$')
#plt.show()


#filenames = [
#    "scanrecord_data_2025-07-24-10-30-20_LEFT_o2_sol_132.1.json",
#    "scanrecord_data_2025-07-24-10-36-48_RIGHT_o2_sol_132.1.json",
#]

filenames = [
    "scanrecord_data_2025-04-28-11-10-19_ITL_SLH01_LEFT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-18-17_ITL_SLH01_RIGHT_SOL_70A.json",
    "scanrecord_data_2025-04-28-11-46-08_ITL_SLH01_LEFT_SOL_75.json",
    "scanrecord_data_2025-04-28-11-25-08_ITL_SLH01_RIGHT_SOL_75.json",
    "scanrecord_data_2025-07-24-10-30-20_LEFT_o2_sol_132.1.json",
    "scanrecord_data_2025-07-24-10-36-48_RIGHT_o2_sol_132.1.json",
]
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
    # "scanrecord_data_2025-07-24-10-30-20_LEFT_o2_sol_132.1.json",
   # "scanrecord_data_2025-07-24-10-36-48_RIGHT_o2_sol_132.1.json"
]
filenames = [
    "scanrecord_data_2025-07-25-14-39-16_LEFT_1.json",
    "scanrecord_data_2025-07-25-14-45-43_RIGHT_1.json",
    "scanrecord_data_2025-07-25-14-53-42_LEFT_2.json",
    "scanrecord_data_2025-07-25-14-58-47_RIGHT_2.json",
    "scanrecord_data_2025-07-25-15-07-53_LEFT_SLIT1_20mm.json",
    "scanrecord_data_2025-07-25-15-13-06_RIGHT_SLIT1_20mm.json",
    "scanrecord_data_2025-07-25-15-22-40_LEFT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-29-03_RIGHT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-36-02_LEFT_SLIT1_10mm_SLIT2_18mm.json",
    "scanrecord_data_2025-07-25-15-42-59_RIGTH_SLIT1_10mm_SLIT2_18mm.json"
            ]
filenames = [
    "scanrecord_data_2025-07-25-15-22-40_LEFT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-29-03_RIGHT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-36-02_LEFT_SLIT1_10mm_SLIT2_18mm.json",
    "scanrecord_data_2025-07-25-15-42-59_RIGTH_SLIT1_10mm_SLIT2_18mm.json"
            ]

filenames22= [
"scanrecord_data_2025-07-29-11-14-40_LEFT_Vertical_0.5_-3_slit2_18_sol_115.json",
"scanrecord_data_2025-07-29-11-08-49_RIGHT__Vertical_0.5_-3_slit2_18_sol_115.json",
"scanrecord_data_2025-07-29-10-51-12_LEFT_Vertical_0.5_-3_slit2_18_sol_124.json",
"scanrecord_data_2025-07-29-10-44-20_RIGHT_Vertical_0.5_-3_slit2_18_sol_124.json"
]
filenames= [
"scanrecord_data_2025-08-06-10-41-38_LEFT_VS_1.json",
"scanrecord_data_2025-08-06-10-53-21_RIGHT_VS_1.json",
"scanrecord_data_2025-08-06-11-04-50_LEFT_VS_-3.json",
"scanrecord_data_2025-08-06-11-13-29_RIGHT_VS_-3.json",
]
filenames11= [
"scanrecord_data_2025-08-06-10-41-38_LEFT_VS_1.json",
"scanrecord_data_2025-08-06-10-53-21_RIGHT_VS_1.json",
"scanrecord_data_2025-08-06-11-04-50_LEFT_VS_-3.json",
"scanrecord_data_2025-08-06-11-13-29_RIGHT_VS_-3.json",
"scanrecord_data_2025-08-06-11-32-56_LEFT_sol_120.json",
"scanrecord_data_2025-08-06-11-39-38_RIGHT_sol_120.json",
"scanrecord_data_2025-08-06-11-49-29_LEFT_sol_124.json",
"scanrecord_data_2025-08-06-11-54-27_RIGHT_sol_124.json",
"scanrecord_data_2025-08-06-12-15-49_LEFT_sol_126.json",
"scanrecord_data_2025-08-06-12-20-36_RIGHT_sol_126.json",
"scanrecord_data_2025-08-06-12-03-34_LEFT_sol_130.json",
"scanrecord_data_2025-08-06-12-08-29_RIGHT_sol_130.json",
]
filenames= [
"scanrecord_data_2025-08-06-10-41-38_LEFT_VS_1.json",
"scanrecord_data_2025-08-06-10-53-21_RIGHT_VS_1.json"
]




### Set to True to plot individual traces, RECOMMENDED for debugging the measurements and chose time window
Plot_individual_traces = False  # Set to True to plot individual traces

# Function to find a dictionary in a list of dictionaries by key-value pair
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

def plot_individual(traces,x_axis,Srange):
            
            fig = plt.figure() 
            ax = fig.add_subplot(111)
            mediciones = traces[6]  # Esto tiene forma (3, 500)            
            n_pasos = traces.shape[0]
            n_values= traces.shape[1]  # Número de mediciones por paso
            n_puntos = traces.shape[2]
 
            # Inicializar arreglo vacío para guardar los nuevos promedios
            if n_values > 3:
                traces_filtrados = np.empty((n_pasos, n_values-2, n_puntos))  # guardará los valores sin min/max
                traces_av = np.empty((n_pasos, n_puntos))
                for i in range(n_pasos):         # por cada paso
                    for j in range(n_puntos):     # por cada punto
                        valores = np.sort(traces[i, :, j])[1:-1]  # quitar min y max → 5 valores
                        traces_filtrados[i, :, j] = valores

                # Para cada paso
                for i in range(n_pasos):
                    # Para cada punto
                    for j in range(n_puntos):
                        valores = traces[i, :, j]         # 7 mediciones en ese punto
                        valores_filtrados = np.sort(valores)[1:-1]  # quitar mínimo y máximo (queda 5)
                        traces_av[i, j] = np.mean(valores_filtrados)
            else:
                traces_av = np.average(traces, axis=1)
            # Inspection of traces in two steps
            pasos_a_ver = [3, 15]

            for paso in pasos_a_ver:
                #plt.figure(figsize=(10, 4))
                
                # Trazas originales (las 7 mediciones)
                for k in range(traces_filtrados.shape[1]):
                    #plt.plot(traces[paso, k, :], alpha=0.4, label=f'Medición {k+1}' if k == 0 else "")
                    plt.plot(traces_filtrados[paso, k, :], alpha=0.4, label=f'Measurements {k+1}' if k == 0 else "")
                
                # Promedio sin extremos
                plt.plot(traces_av[paso], color='black', linewidth=2, label='Average withou xtremes')
                plt.xlabel("Number of saples (arbitrary units)")
                plt.ylabel(f'signal {paso}')
                plt.title(f"Step {paso} - Average without extremes")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

            fig, axs = plt.subplots(2, 1, figsize=(10, 6))
            traces_std = np.std(traces_filtrados, axis=1)     # Shape: (45, 500), std across 3 measurements
        
            # --- Now for the second plot (average over time window) ---
            # Averages over time windows:
            traces_av_av = np.average(traces_av[:, Srange[0]:Srange[1]], axis=1)   # (45,)
            
            # Error bars: std of time window 200:250 for each trace
            #traces_std = np.std(traces_av[:, Srange[0]:Srange[1]], axis=1)        # (45,)          
            traces_std= np.std(traces_filtrados[:, :, Srange[0]:Srange[1]], axis=(1, 2), ddof=1) 
            axs[0].plot(range(500), traces_filtrados[11].T,label=f'Samples Avg {Srange[0]}–{Srange[1]}') 
            # Plot lines
            axs[1].plot(x_axis, traces_av_av,label=f'Samples {Srange[0]}–{Srange[1]}')
            axs[1].set_xlabel("Position (mm)")
           
            # Plot scatter points with error bars
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
            
def read_file(filename,Srange):
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
            if Plot_individual_traces:
                plot_individual(traces,x_axis,Srange)
            
            
            traces_av = np.average(traces, axis=1)
            traces_av_av = np.average(traces_av[:,Srange[0]:Srange[1]], axis=1) 

    return(x_axis, traces_av_av)

def erf2(x, mu, sig):
    val = scipy.special.erf((x-mu)/sig)
    return(val)

def erf21(x, mu, sig):
    val = scipy.special.erf((x-mu)/sig)+1
    return(val)

def plot_meas(scan_data):
    sd = scan_data
    Name1=extraer_info(sd["LEFT_F"])
    fig, ax = plt.subplots()
    ax.plot(sd["LEFT_X"], sd["LEFT_V_2"],label=Name1)
    ax.plot(sd["RIGHT_X_2"], sd["RIGHT_V_2"])
    if sd.get("mu") is not None:
        xxx = np.linspace(-30.0, 30.0, 100)
        yyy = sd["Am"]*erf21(xxx, sd["mu"], sd["sig"])+sd["A0"]
        ax.plot(xxx, yyy)
    if sd.get("title") is not None:
        ax.set_title(sd["title"])
   
    ax.legend()
    ax.set_xlabel("ITL.SLH01 L - Position (mm)")
    fig.show()

def extraer_info(nombre_archivo):
    nombre_archivo = nombre_archivo.upper()  # por si acaso
    if 'LEFT' in nombre_archivo:
        parte = nombre_archivo.split('LEFT', 1)[-1]
    elif 'RIGHT' in nombre_archivo:
        parte = nombre_archivo.split('RIGHT', 1)[-1]
    else:
        parte = ''
    return parte.replace('.JSON', '').strip('_')


def fusionar_y_promediar(x1, v1, x2, v2, tol=1e-8):
    # Combinar todos los puntos
    x_all = np.concatenate([x1, x2])
    v_all = np.concatenate([v1, v2])

    # Agrupar valores cercanos según tolerancia
    grupos = defaultdict(list)

    for xi, vi in zip(x_all, v_all):
        # Buscar si ya hay un grupo cercano
        encontrado = False
        for xg in grupos:
            if abs(xi - xg) < tol:
                grupos[xg].append(vi)
                encontrado = True
                break
        if not encontrado:
            grupos[xi].append(vi)

    # Promediar cada grupo
    x_fusion = []
    v_fusion = []

    for xg, vs in grupos.items():
        x_fusion.append(xg)
        v_fusion.append(np.mean(vs))

    # Ordenar por x
    x_fusion = np.array(x_fusion)
    v_fusion = np.array(v_fusion)
    orden = np.argsort(x_fusion)

    return x_fusion[orden], v_fusion[orden]

## HEre we start the main code difining the way we build the data list
BASE = ""
d_list = [ {"LEFT_F": BASE+filenames[ii*2], "RIGHT_F": BASE+filenames[ii*2+1]} for ii in range(int(len(filenames)*0.5)) ]

ii = 0 #counter for colors 
fig, ax = plt.subplots()
Srange = [300, 350]  # Define the range for averaging

#PLOT ORIGINAL DATA without postprocessing for the FIT 
for dd in d_list:
    xxl, VL = read_file(dd["LEFT_F"],Srange)
    xxR, VR = read_file(dd["RIGHT_F"],Srange)
    #NOW we start postp proccessing the data
    xxl2 = [ x for x in xxl ]  # Define the offset to the left blade
    dd["MAX_V"] = 1*max( max(VL), max(VR)  )  # Average the "full out values" to get the max intensit
    Name1=extraer_info(dd["LEFT_F"])
    Name2=extraer_info(dd["RIGHT_F"])
    ax.plot(xxl2, VL, label=Name1,color=colores[ii])
    ax.plot(xxR, VR,color=colores[ii])
    ax.legend()
    ax.set_title(dd["LEFT_F"])
    ax.set_xlabel("ITL.SLH01 - Position (mm)")
    ax.legend()
    ii= ii + 1
plt.show()


# Now we do the postprocessing of the data
for dd in d_list:
    xxl, VL = read_file(dd["LEFT_F"],Srange)
    xxR, VR = read_file(dd["RIGHT_F"],Srange)
    xxl2 = xxl
    [ -x -6 for x in xxl ]  # Apply offset to the left blade
    xlof=[x +2.0 for x in xxl]
    #convert signal to shape of the fit 
    dd["MAX_V1"] = max( max(VL), max(VR)  )  
    dd["LEFT_V"] = VL/dd["MAX_V1"]  # Normalized to the max value
    dd["RIGHT_V"] = VR/dd["MAX_V1"]  # Normalized to the max value
    
    dd["LEFT_X"] = xlof
    dd["RIGHT_X"] = xxR
    
    dd["MAX_V"] = 0.5*max( max(dd["LEFT_V"]), max(dd["RIGHT_V"])  )  # Average the "full out values" to get the 
    
    V2 = [ (dd["MAX_V"]-VV)/dd["MAX_V"] for VV in dd["RIGHT_V"] ]        # 
    V2R = [ (VV-dd["MAX_V"])/dd["MAX_V"] for VV in dd["RIGHT_V"] ]        
    V2L = [ (-VV + dd["MAX_V"])/dd["MAX_V"] for VV in dd["LEFT_V"] ]  

    dd["RIGHT_V_2"] = V2R  # Normalized to the max value
    dd["LEFT_V_2"] = V2L  # Normalized to the max value
    dd["RIGHT_X_2"] = [ x2 for x2 in dd["RIGHT_X"] ]
    #fig, ax = plt.subplots()
    fig, axr = plt.subplots(2, 1, figsize=(10, 6))
    axr[0].scatter(dd["RIGHT_X_2"],V2R, label="RIGHT_F")
    axr[0].scatter(dd["LEFT_X"],V2L, label="LEFT_F")
    axr[0].set_title(dd["LEFT_F"])
    axr[0].legend()
    #NOw we feed the fit function for the right side 
    popt, pcov = curve_fit(erf2, xxR, V2R, p0=[-1.0, 4.0])  # p0 son valores iniciales para mu y sig
    # results of the fit 
    mu_fitR, sig_fitR = popt
    sigmafitsR.append(sig_fitR)
    mufitR.append(mu_fitR)

    print("Parámetros ajustados:",dd["LEFT_F"])
   # plot fit function
    x_fit = np.linspace(min(xxR), max(xxR), 500)
    y_fit = erf2(x_fit, mu_fitR, sig_fitR)
   # axr[0].plot(x_fit, y_fit, '-', label=f'Ajuste erf2: mu={mu_fitR:.2f}, sig={sig_fitR:.2f}')
    x_check = np.linspace(-20, 20, 500)
    # Fit for the left side
    popt, pcov = curve_fit(erf2, xlof, V2L, p0=[1.0, 4.0])  # Fit the left Blade 
    mu_fitl, sig_fitl = popt
    mufitL.append(mu_fitl)
    
    #print("Parámetros ajustados:",dd["LEFT_F"])
    print("mu =L R ", mu_fitl,mu_fitR)
    print("siig L R =", sig_fitl, sig_fitR)
    sigmafitsL.append(sig_fitl)
    
    #Put together the data for the full fit
    xN1, yN = fusionar_y_promediar(dd["LEFT_X"], V2L, dd["RIGHT_X_2"],V2R, tol=1e-8)
    popt, pcov = curve_fit(erf2, xN1, yN, p0=[-1.0, 4.0])  # p0 son valores iniciales para mu y sig
    mu_fit, sig_fit = popt
    sigmafits.append(sig_fit)
    mufitF.append(mu_fit)
    print("mu = Full", mu_fit)
    print("siig  Full=", sig_fit)
    axr[1].plot(xN1, yN, 'o', label='Average Data')
    axr[1].plot(x_check, erf2(x_check, mu_fit, sig_fit), 'o', label='Datos ajuste')

    axr[1].set_xlabel("ITL.SLH01 - Position (mm)")
    axr[1].set_ylabel("FCup Current (mA)")
    axr[1].legend()
    axr[1].grid(True)
    print("sigmafits","Left","RIGHT","Full")

plt.show()  # Show the individual plots for each measurement

for dd in d_list:
    plot_meas(dd) 
print("sigmafits","Left","RIGHT","Full","mu Left","mu Right","mu Full")
for var in range(len(sigmafitsL)):
    print(var,sigmafitsL[var],sigmafitsR[var],sigmafits[var],mufitL[var],mufitR[var],mufitF[var])
plt.show()  # Show all plots at once


# ...existing code...

scan_start = 200
scan_end = 380
window_size = 10

# Listas para guardar los resultados del scan
scan_centers = []
sigmas_left = []
sigmas_right = []
sigmas_full = []
mus_left = []
mus_right = []
mus_full = []

for start in range(scan_start, scan_end - window_size + 1, window_size):
    Srange = [start, start + window_size]
    # Limpia los resultados temporales para cada ventana
    sigmafitsL_tmp = []
    sigmafitsR_tmp = []
    sigmafits_tmp = []
    mufitL_tmp = []
    mufitR_tmp = []
    mufitF_tmp = []
    for dd in d_list:
        xxl, VL = read_file(dd["LEFT_F"], Srange)
        xxR, VR = read_file(dd["RIGHT_F"], Srange)
        xlof = [x + 2.0 for x in xxl]
        dd["MAX_V1"] = max(max(VL), max(VR))
        dd["LEFT_V"] = VL / dd["MAX_V1"]
        dd["RIGHT_V"] = VR / dd["MAX_V1"]
        dd["LEFT_X"] = xlof
        dd["RIGHT_X"] = xxR
        dd["MAX_V"] = 0.5 * max(max(dd["LEFT_V"]), max(dd["RIGHT_V"]))
        V2R = [(VV - dd["MAX_V"]) / dd["MAX_V"] for VV in dd["RIGHT_V"]]
        V2L = [(-VV + dd["MAX_V"]) / dd["MAX_V"] for VV in dd["LEFT_V"]]
        # Ajuste lado derecho
        popt, _ = curve_fit(erf2, xxR, V2R, p0=[-1.0, 4.0])
        mu_fitR, sig_fitR = popt
        # Ajuste lado izquierdo
        popt, _ = curve_fit(erf2, xlof, V2L, p0=[1.0, 4.0])
        mu_fitl, sig_fitl = popt
        # Ajuste combinado
        xN1, yN = fusionar_y_promediar(dd["LEFT_X"], V2L, dd["RIGHT_X"], V2R, tol=1e-8)
        popt, _ = curve_fit(erf2, xN1, yN, p0=[-1.0, 4.0])
        mu_fit, sig_fit = popt
        # Guarda resultados temporales
        sigmafitsL_tmp.append(sig_fitl)
        sigmafitsR_tmp.append(sig_fitR)
        sigmafits_tmp.append(sig_fit)
        mufitL_tmp.append(mu_fitl)
        mufitR_tmp.append(mu_fitR)
        mufitF_tmp.append(mu_fit)
    # Guarda el promedio de cada ventana
    scan_centers.append(start + window_size // 2)
    sigmas_left.append(np.mean(sigmafitsL_tmp))
    sigmas_right.append(np.mean(sigmafitsR_tmp))
    sigmas_full.append(np.mean(sigmafits_tmp))
    mus_left.append(np.mean(mufitL_tmp))
    mus_right.append(np.mean(mufitR_tmp))
    mus_full.append(np.mean(mufitF_tmp))

# Graficar los resultados del scan
plt.figure(figsize=(10, 6))
plt.plot(scan_centers, sigmas_left, label="Sigma Left")
#plt.plot(scan_centers, sigmas_right, label="Sigma Right")
plt.plot(scan_centers, sigmas_full, label="Sigma Full")
plt.xlabel("Centro de la ventana Srange")
plt.ylabel("Sigma ajustado")
plt.title("Evolución de sigma en ventanas de 10 muestras")
plt.legend()
plt.grid(True)
plt.show()