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
            mediciones = traces[6]  # Esto tiene forma (3, 500)
            #mediciones2 = traces[44]  # Esto tiene forma (3, 500)
            # Graficar cada medición
            #for i in range(6):
            
            #    plt.plot(mediciones[i])
                #plt.plot(mediciones2[i])

            #    plt.title(f"step {23}")
            #    plt.xlabel("Índice de dato")
            #    plt.ylabel("Valor")
            #    plt.grid(True)
            
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

            pasos_a_ver = [7, 8]

            for paso in pasos_a_ver:
                plt.figure(figsize=(10, 4))
                
                # Trazas originales (las 7 mediciones)
                for k in range(traces_filtrados.shape[1]):
                    #plt.plot(traces[paso, k, :], alpha=0.4, label=f'Medición {k+1}' if k == 0 else "")
                    plt.plot(traces_filtrados[paso, k, :], alpha=0.4, label=f'Measurements {k+1}' if k == 0 else "")
                
                # Promedio sin extremos
                #timeaux= np.arange(0, 0.002.500)
                plt.plot(traces_av[paso], color='black', linewidth=2, label='Average withou xtremes')
                #time= np.arange(0, 500)  # Assuming 500 time points
                #plt.title(f'Trace stepdel paso {paso + 1}')
                plt.xlabel("Time (arbitrary units)")
                plt.ylabel("Signal (V)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            fig, axs = plt.subplots(2, 1, figsize=(10, 6))

            #traces_av = np.average(traces, axis=1)  # Shape: (45, 500)
            traces_std = np.std(traces_filtrados, axis=1)     # Shape: (45, 500), std across 3 measurements
            
            #bloques_por_paso = [traces[i] for i in range(43)]  # lista de 43 arrays (cada uno 7x500)

            #promedios = np.mean(traces, axis=(1, 2))  # Promedia sobre ejes 1 y 2 → resultado (43,)
            desviaciones = np.std(traces[:, :, 300:400], axis=(1, 2), ddof=1)  # Desviación estándar de cada bloque

            #print("largo",len(traces))
            axs[0].plot(range(500), traces_filtrados[11].T)  # Plot each averaged trace

            # --- Now for the second plot (average over time window) ---
            # Averages over time windows:
            traces_av_av = np.average(traces_av[:, 180:200], axis=1)   # (45,)
            traces_av_av2 = np.average(traces_av[:, 340:360], axis=1)  # (45,)

            # Error bars: std of time window 200:250 for each trace
            traces_std = np.std(traces_av[:, 240:300], axis=1)        # (45,)
            traces_std = np.std(traces_filtrados[:, :, 240:300] , axis=(1, 2), ddof=1)  # shape: (43,)

            traces_std2 = np.std(traces_av[:, 340:360], axis=1)        # (45,)

            # Plot lines
            axs[1].plot(x_axis, traces_av_av, label='Avg 240–300')
            axs[1].set_xlabel("Position (mm)")
            #axs[1].plot(x_axis, traces_av_av2, label='Avg 340–360')

            # Plot scatter points with error bars
            axs[1].errorbar(
                x_axis,
                traces_av_av,
                yerr=traces_std,
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
           # plot_individual(traces,x_axis)  # Plot individual traces
            
            traces_av = np.average(traces, axis=1)
            #traces_av_av = np.average(traces_av[:,200:350], axis=1) 
            traces_av_av = np.average(traces_av[:,240:300], axis=1) 

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
    ax.plot(sd["LEFT_X"], sd["LEFT_V_2"])
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
filenames22 = [
    "scanrecord_data_2025-07-25-15-22-40_LEFT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-29-03_RIGHT_SLIT1_20mm_SLIT2_18.json",
    "scanrecord_data_2025-07-25-15-36-02_LEFT_SLIT1_10mm_SLIT2_18mm.json",
    "scanrecord_data_2025-07-25-15-42-59_RIGTH_SLIT1_10mm_SLIT2_18mm.json"
            ]

#filenames = [
#    "scanrecord_data_2025-07-24-10-30-20_LEFT_o2_sol_132.1.json",
#    "scanrecord_data_2025-07-24-10-36-48_RIGHT_o2_sol_132.1.json",
#]



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


BASE = ""
d_list = [ {"LEFT_F": BASE+filenames[ii*2], "RIGHT_F": BASE+filenames[ii*2+1]} for ii in range(int(len(filenames)*0.5)) ]

for dd in d_list:
    xxl, VL = read_file(dd["LEFT_F"])
    xxR, VR = read_file(dd["RIGHT_F"])
    xxl2 = [ -x -6 for x in xxl ]  # Normalize to the first point
    xlof= [x +1 for x in xxl]
    dd["MAX_V"] = 1*max( max(VL), max(VR)  )  # Average the "full out values" to get the max intensit
    VLM=2*(-VL/dd["MAX_V"] +0.5)
    VRM=2*(VR/dd["MAX_V"]-0.5)
    fig, ax = plt.subplots()

    ax.plot(xxl2, VL, label="LEFT_F")
    ax.plot(xxR, VR, label="RIGHT_F")
    ax.set_title(dd["LEFT_F"])
    ax.legend()
    
    
    
    #ax.plot(xlof, VLM, label=dd["LEFT_F"])
    #ax.plot(xxR, VRM, label=dd["RIGHT_F"])
    #plt.show()
    dd["LEFT_V"] = VL/dd["MAX_V"]  # Normalized to the max value
    dd["RIGHT_V"] = VR/dd["MAX_V"]  # Normalized to the max value
    print("LEFT_V",dd["LEFT_V"])
    #input("Press Enter to continue...")  # Pause for user input
  

    dd["LEFT_X"] = xlof
    dd["RIGHT_X"] = xxR
    #print("LEFT_X",dd["LEFT_V"])
    #print("RIGHT_X",dd["RIGHT_V"])
    dd["MAX_V"] = 0.5*max( max(dd["LEFT_V"]), max(dd["RIGHT_V"])  )  # Average the "full out values" to get the max intensit
    #dd["MAX_V"] = 0.5*(dd["LEFT_V"][0]+dd["RIGHT_V"][0])  # Average the "full out values" to get the max intensit

    print("max",dd["MAX_V"])
    
    V2 = [ (dd["MAX_V"]-VV)/dd["MAX_V"] for VV in dd["RIGHT_V"] ]        # 
    V2R = [ (VV-dd["MAX_V"])/dd["MAX_V"] for VV in dd["RIGHT_V"] ]        
    V2L = [ (-VV + dd["MAX_V"])/dd["MAX_V"] for VV in dd["LEFT_V"] ]  
    #V2L= VL 
    #V2R = V 
    # 
    dd["RIGHT_V_2"] = V2R  # Normalized to the max value
    dd["LEFT_V_2"] = V2L  # Normalized to the max value
    dd["RIGHT_X_2"] = [ x2 for x2 in dd["RIGHT_X"] ]
    #fig, ax = plt.subplots()
    fig, axr = plt.subplots(2, 1, figsize=(10, 6))
    axr[0].scatter(dd["RIGHT_X_2"],V2R, label="RIGHT_F")
    axr[0].scatter(dd["LEFT_X"],V2L, label="LEFT_F")
    axr[0].set_title(dd["LEFT_F"])
    axr[0].legend()
    
    popt, pcov = curve_fit(erf2, xxR, V2R, p0=[-1.0, 4.0])  # p0 son valores iniciales para mu y sig
    # Parámetros ajustados
    mu_fitR, sig_fitR = popt
    sigmafitsR.append(sig_fitR)
    mufitR.append(mu_fitR)
    print("Parámetros ajustados:",dd["LEFT_F"])
    #print("mu =", mu_fit)
    #print("sig =", sig_fit)
    x_fit = np.linspace(min(xxR), max(xxR), 500)
    y_fit = erf2(x_fit, mu_fitR, sig_fitR)
   # axr[0].plot(x_fit, y_fit, '-', label=f'Ajuste erf2: mu={mu_fitR:.2f}, sig={sig_fitR:.2f}')
    x_check = np.linspace(-20, 20, 500)
    # Fit for the left side
    popt, pcov = curve_fit(erf2, xlof, V2L, p0=[1.0, 4.0])  # p0 son valores iniciales para mu y sig
    mu_fitl, sig_fitl = popt
    mufitL.append(mu_fitl)
    
    #print("Parámetros ajustados:",dd["LEFT_F"])
    print("mu =L R ", mu_fitl,mu_fitR)
    print("siig L R =", sig_fitl, sig_fitR)
    sigmafitsL.append(sig_fitl)
    #axr[0].plot(x_check, erf2(x_check, mu_fitl, sig_fitl), 'o', label='Datos ajuste')
    
    xN1, yN = fusionar_y_promediar(dd["LEFT_X"], V2L, dd["RIGHT_X_2"],V2R, tol=1e-8)
    popt, pcov = curve_fit(erf2, xN1, yN, p0=[-1.0, 4.0])  # p0 son valores iniciales para mu y sig
    mu_fit, sig_fit = popt
    sigmafits.append(sig_fit)
    mufitF.append(mu_fit)
    print("mu = Full", mu_fit)
    print("siig  Full=", sig_fit)
    axr[1].plot(xN1, yN, 'o', label='Average Data')
    #axr[1].plot(x_check, erf2(x_check, mu_fit, sig_fit), 'o', label='Datos ajuste')

    axr[1].set_xlabel("ITL.SLH01 - Position (mm)")
    axr[1].set_ylabel("FCup Current (mA)")
    axr[1].legend()
    axr[1].grid(True)
    print("sigmafits","Left","RIGHT","Full")


    #plt.show()


    #ax.plot(xx, erf2(xx, mu_fit,sig_fit))
    #, 'g--', label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))
    #ax.plot(xx, erf2(xx, *popt), color='red', label='Ajuste')
fits = {"mu":-1.5, "sig": 16.0, "Am": -0.16, "A0":0.30, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=70A"}    
d_list[0].update(fits)
#fits = {"mu":-4.5, "sig": 16.0, "Am": -0.16, "A0":0.30, "title":"LEFT-RIGHT ITL.SLH01 - ITL.SOL01=75A"}    
#d_list[1].update(fits)

for dd in d_list:
    plot_meas(dd) 
print("sigmafits","Left","RIGHT","Full","mu Left","mu Right","mu Full")
for var in range(len(sigmafitsL)):
    print(var,sigmafitsL[var],sigmafitsR[var],sigmafits[var],mufitL[var],mufitR[var],mufitF[var])
plt.show()  # Show all plots at once



