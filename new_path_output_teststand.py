import numpy as np
from scipy.constants import c
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys, os, gzip
import matplotlib.style
import matplotlib as mpl
#mpl.style.use('classic')
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
#plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20


#filename = 'transmission/FC2_BEAM_FULL.TXT'
#filename = 'transmission/TESTVC.TXT'
filename = 'output_RFQ2.44.txt'

def extraer_datos_travel_beam(file_path):
    with open(file_path, 'r') as file:
        lines = [line.strip() for line in file.readlines()[:10]]  # leer primeras líneas necesarias

    datos = {
        "hora": lines[1].split()[1],
        "momento_referencia": float(lines[2].split()[0]),
        "fase_referencia": float(lines[3].split()[0]),
        "frecuencia": float(lines[4].split()[0]),
        "masa_referencia": float(lines[5].split()[0]),
        "estado_de_carga": float(lines[6].split()[0]),
        "numero_de_particulas": int(lines[7].split()[0]),
    }

    return datos

datos_base = extraer_datos_travel_beam(filename)
p_ref = datos_base["momento_referencia"]      # GeV/c
phi_ref=datos_base["fase_referencia"]           #REFERENCE PHASE [rad]
q_ref = datos_base["estado_de_carga"]         # carga
freq_ref= datos_base["frecuencia"]         # freq Hz
masa_ref = datos_base["masa_referencia"]         # !REFERENCE MASS [GeV/c2]




# Convert wake from IW2D to PyHT convention
#DEFINE THE ENERGY AND MAIN PARAMETERS
pi=np.pi      		           # pi

Frf=freq_ref# RF frequency [Hz]   
clight=299792458
lambdabeam=299792458/2998e6 
Omega=Frf*(2*pi)
coveromega=clight/(2*pi*Frf)# CoverOmega
lambdad=1/coveromega
mo=masa_ref
qe=1.69e-19
mc=mo/clight
print ("MC= ",mc)
E_total = np.sqrt(p_ref**2 + masa_ref**2)   # GeV
Ekin= (E_total - masa_ref)*1e9               # GeV
print("cinetica",Ekin)
#Ekin=2.40e6
gammar=1+(Ekin)/mo              # Relativistic gamma
gamar=130
betar=np.sqrt(1-1.0/(gammar**2))   # Relativistic beta
bg=betar*gammar
BDipole1=2.0
deg=180/3.1416
me=9.10953e-31
Rbend = -me*clight*gamar*betar/(qe*BDipole1) 
Lbend = 0.25/np.sin(20/deg) 
print("bends ",Rbend,Lbend)
print("momentum ",mc)
print ("BG= %.4f M0 = %03s MeV Betar = %03s KEnergy= %03s Mev" % (bg,mo/1e6,betar,Ekin/1e6))


#B = np.loadtxt(filename, max_rows=1)
#print(B)
#print("B0 Ar= ",B[0])
#print("Bq Ar= ",B[1])
#print("PZ0 = ",B[5]/mo)
#pz0=B[5]/mo
#px0=B[3]/mo

B=[0,0,0,0,0,0,p_ref]

A = np.loadtxt(filename, skiprows=10)

#A = np.loadtxt('r0.1mm_1ps2.txt', skiprows=1)
#A = np.loadtxt('gun.0012.001_nosc.txt', skiprows=1)
dz=clight*4e-12
dz=1
X=np.array(A[:,1])+B[1]
PX=np.array(A[:,2])+B[2]
Y=np.array(A[:,3])+B[3]
PY=np.array(A[:,4])+B[4]
Z=180*(np.array(A[:,5])+B[5])/3.1416
##DP is around 0.5% for the beam 
DP=np.array(A[:,6])
#DP2=np.random.permutation(DP)
#PY=np.array(A[:,4])/mo+B[4]/mo
E=DP*B[6]/mo+B[6]/mo

status=np.array(A[:,7])
carga=(1.0)*np.array(A[:,8])
masa=(np.array(A[:,9]))
CM=np.array(A[:,8:9])
CM = A[:, [8, 9]]
print('forma ',CM.shape,CM)
#print( np.split(A, A[:,8]<3))
MQ=masa*carga
#*1e9
ME=np.mean(E)
ZE=np.mean(Z)
print("masas ",np.unique(CM))
#ZE=0.18
Mgammaa=np.sqrt(1+ME**2)
Energy=(np.sqrt(1+E**2)-1)*mo


Gamma=np.sqrt(1+E**2)
Gammaexact=np.sqrt(1+PY**2+PX**2+E**2)

betar1=np.sqrt(1-1.0/(Gamma**2))
v=clight*betar1
Time=Z/(clight*betar)

Owtime=Omega*Time
OwE=Omega*ZE/(clight*betar)
#OwE=np.mean(Owtime)
Degressz=180*(Owtime-OwE)/(pi)
MD=np.mean(Degressz)
MD=np.max(Degressz)
phasenew=-Degressz+MD


varx=0
varpx=0
varxpx=0
LS=len(X)
VXP=(PX/Gammaexact)*clight
VYP=(PY/Gammaexact)*clight
VZP=(E/Gammaexact)*clight

PX2=VXP/VZP
PY2=VYP/VZP
PZ2=VZP/VZP


#VXP=(PX/np.sqrt(1+PX*PX))*clight
#ratio123=VXP4/VXP
#VYP=(PY/np.sqrt(1+PY*PY))*clight
#VZP=(E/np.sqrt(1+E*E))*clight
Gammaexact2=1/np.sqrt(1-(VZP**2+VYP**2+VXP**2)/(clight**2))
print('gammas=' ,Gammaexact[4],Gammaexact2[4])
#PX2=VXP/VZP
#PY2=VYP/VZP
#PX2=PX/E
#PY2=PY/E
MX=np.mean(X)
MPX=np.mean(PX2)
chargebunch=1.0e-11
charge = np.ones(LS)*chargebunch/LS
print("LA carga total")
print(charge)
header = "X(m) xp(rad) y yp phase(degrees) E(mev) \n"
#header += "This is a second line"
filename2 = 'out_'+filename

#f = open(filename2, 'wb')

#np.savetxt(f, [], header=header)
#for i in range(LS):
#    if status[i]> 1:
#     data = np.column_stack((X[i],PX2[i], Y[i],PY2[i],Degressz[i],Energy[i],charge[i]))
#     np.savetxt(f, data)

#f.close()


print("media = ",MX)
rmd=np.sqrt(np.mean(X**2))
print("media = ",1e3*rmd)
print("media energia = ",ME,"z ",ZE,"gamma ",Mgammaa,"KE= ",(Mgammaa-1)*mo/1e6)
for i in range(0,len(X)):
    varx   = varx   + (X[i]-MX) *(X[i]-MX)/LS
    varpx  = varpx  + (PX2[i]-MPX)*(PX2[i]-MPX)/LS
    varxpx = varxpx + (X[i]-MX)*(PX2[i]-MPX)/LS

print(varx)
e_rms = bg*np.sqrt(varx*varpx-varxpx*varxpx)
print ("RMS Size X = %.4f mm Emittance =  %03s mm.mrad" % (np.sqrt(varx)*1e3,np.sqrt(1-1.0/(Mgammaa**2))*Mgammaa*e_rms*1000000))

      


#plt.scatter(1000*XDNEW, 1000*YDNEW, s=0.5,alpha=1.0)#,cmap="jet")

#plt.show()
fig1=plt.figure()
ax=fig1.add_subplot(111,title="X vs X\' full", rasterized=True)
nbinsx = 200
limits=[[-50, 50], [-50, 50]]
Hx, xxedges, yxedges = np.histogram2d(1000*X,1000*PX,bins=nbinsx)
#Hx, xxedges, yxedges = np.histogram2d(X,PX2,bins=nbinsx)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax.set_xlabel('X [mm]',fontsize=20)
ax.set_ylabel('X\' [mrad] ',fontsize=20)
ax.grid(True)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)   

fig2=plt.figure()
ax=fig2.add_subplot(111,title="Y vs Y\' full", rasterized=True)
nbinsx = 200

Hx, xxedges, yxedges = np.histogram2d(1000*Y,1000*PY,bins=nbinsx,range=limits)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax.set_xlabel('Y [mm]',fontsize=20)
ax.set_ylabel('Y\' [mrad] ',fontsize=20)
ax.grid(True)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)

fig2=plt.figure()
ax=fig2.add_subplot(111,title="X vs Y full", rasterized=True)
nbinsx = 200
limits=[[-10, 10], [-10, 10]]
Hx, xxedges, yxedges = np.histogram2d(1000*X,1000*Y,bins=nbinsx)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax.set_xlabel('X [mm]',fontsize=20)
ax.set_ylabel('Y [mm] ',fontsize=20)
ax.grid(True)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20)
        

fig2x=plt.figure()
ax=fig2x.add_subplot(111,title="Longitudinal full", rasterized=True)
nbinsx = 200
limits=[[0.05, 0.11], [-0.02, 0.02]]
Hx, xxedges, yxedges = np.histogram2d(Z,DP,bins=nbinsx)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)
cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax.set_xlabel('Phase [degrees]',fontsize=20)
ax.set_ylabel('E [MeV] ',fontsize=20)
ax.grid(True)

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(20) 



fig212=plt.figure()
ax212=fig212.add_subplot(111,title="Z vs X full", rasterized=True)
nbinsx = 1000
limits=[[0.1, 0.122], [-0.02, 0.02]]
Hx, xxedges, yxedges = np.histogram2d(Z,X,bins=nbinsx)
Hx = np.rot90(Hx)
Hx = np.flipud(Hx)
Hmaskedx = np.ma.masked_where(Hx==0,Hx)      
plt.pcolormesh(xxedges,yxedges,Hmaskedx)

cbar = plt.colorbar()
cbar.ax.set_ylabel('Counts')
ax212.set_xlabel('Z [ mm]',fontsize=20)
ax212.set_ylabel('X [mm] ',fontsize=20)
ax212.grid(True)

for item in ([ax212.title, ax212.xaxis.label, ax212.yaxis.label] +
             ax212.get_xticklabels() + ax212.get_yticklabels()):
        item.set_fontsize(30) 


fig3=plt.figure()

num_bins=500
n, bins, patches = plt.hist(X, num_bins, facecolor='blue', alpha=0.5,label="cc")
plt.xlabel('Position X', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title('Histogram of X full', fontsize=16)

fig3p=plt.figure()

num_bins=500
n, bins, patches = plt.hist(PX, num_bins, facecolor='blue', alpha=0.5,label="cc")
plt.xlabel('Position pX', fontsize=14)
plt.title('Histogram of PX full', fontsize=16)

fig4=plt.figure()
num_bins=5000
n, bins, patches = plt.hist(Energy, num_bins, facecolor='blue', alpha=0.5,label="cc")
print(Energy)
#for var in Energy:
#    print(var)
plt.xlabel('Energy (MeV)', fontsize=14)
plt.title('Histogram of Energy full', fontsize=16)
plt.show()

 



if (False):
    A = np.loadtxt('PATHRMSQ29.XLS',skiprows=1)
    Z=1*np.array(A[:,2])
    XRMS=1*np.array(A[:,5])
    YRMS=1*np.array(A[:,7])
    PXRMS=1*np.array(A[:,6])
    CTY=1*np.array(A[:,1])

    fige=plt.figure()
    ax=fige.add_subplot(111,title="Plot R.M.S vs Z", rasterized=True)
    ax.set_xlabel('Z [m]',fontsize=20)
    ax.set_ylabel('1 R.M.S [m] ',fontsize=20)
    ax.plot(Z, XRMS,label='X')
    ax.plot(Z, YRMS,label='Y')
    plt.legend()
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax2 = ax.twinx()
    ax2 .scatter(Z, CTY,color='red')
    ax2.set_ylim([0, 7])



plt.show()
