from math import cos,sin, radians, sqrt
import numpy as np
import matplotlib.pyplot as plt
import os,sys
import subprocess
import scipy.optimize as optimize
import time
plt.rcParams['mathtext.rm'] = "Times New Roman"
plt.rcParams["font.weight"] = "bold"
plt.rcParams['font.family'] = 'Times New Roman'
#h=[x for x in range(1, 10) if x % 2]
#print(h)

#defining cavity data
cavity_Length=1.992

#linea1="cmd  /c prueba2.bat "  
#os.system(linea1)
#subprocess.call(linea1, shell=True)

#time.sleep(10)
x00=0
x002=2
cavityle2=2.6
rmin=0.205
RCAVITY=3.85
L_small=0.5
first_apert=0.99
aper_cavity=first_apert+rmin
radio_topg=0.3175
#radio_top=radio_topg
#raditop2=radio_topg
#numbers = [1.0,1.5,2,2.5,3.0]
drift1=range(5, 20,5)
drift1= [x * 0.01 for x in drift1]

quad1=range(4, 20,4)

quad2=range(-20, -4,4)

dipo=range(0,50,10)
dipo2=range(0,50,10)

drift1 = [0.45]
#quad1 = [18]
#quad2 = [-30]
#dipo= [4]

solfac= 1.19

filename="  "
linea21="del output.txt"
os.system(linea21)
fige=plt.figure()
## 35.90662298 17.61160158  0.74573867 drift 0.2
#0.0331 87.9440 0.0039 24.0267 54.0736 0.2000 1.1900 52.8529
def f(params):
    # print(params)  # <-- you'll see that params is a NumPy array
        solfac = params # <-- for readability you may wish to assign names to the component variables
        var3=0.3      
        #solfac=1.19
#aqui se define la linea

        #print(x[5]-2*raditop2+0.5*cavitylen2)
        filename="beamline_onedipole.in" 
        with open(filename, 'w') as a_writer:
            a_writer.write("cREM Travel python example\n" )
            a_writer.write("	34 1 'PATHRMSq.XLS';	\n" )
            a_writer.write("	33 1 'PATHAVGq.XLS';	\n" )
            a_writer.write("	13 1 1; 	\n" )
            a_writer.write('	c==RFQ;	\n' )
            a_writer.write('	98 1 "fold over the bucket";	\n' )
            a_writer.write("	3 0.0001; 	\n" )
            
            a_writer.write('	24 35 0 0 20 40 5 0 0 0 "current";	\n' )
            a_writer.write('	23 6000  "steps";	\n' )
            #a_writer.write("    55 %03s 'fieldmap_comsol_highfield.ef3'"+ '0 1.03 "RFQ";	\n'% (solfac) )
            a_writer.write(f"    55 {solfac:03} 'fieldmap_comsol_highfield.ef3' 0 1.03 \"RFQ\";\n")

            a_writer.write("	23 4;	\n")

            a_writer.write('	3 0.0001 "slitmeas";	\n' )
            #a_writer.write('    82  1 -0.03 %03s "FIlterH";\n' % (solfac) )
            a_writer.write("	39 'output_RFQ%03s.txt';	\n" % (solfac))
            a_writer.write("	38 'output_RFQ%03s.dst';	\n" % (solfac))

            a_writer.write("	13 1 0;	\n")
            a_writer.write("	23 4;	\n")
            a_writer.write('	3 0.100 "FC2";	\n')
            a_writer.write("	c39 'SLH01loop.txt';	\n" )
            a_writer.write("	34 0 'PATHRMSq.XLS';	\n" )
            a_writer.write("	33 0 'PATHAVGq.XLS';	\n" )
            a_writer.write("	SENTINEL	\n" )
            a_writer.close()


           #aqui se define el ejecutable y el archivo de entrada
        
        #linea="C:\Use\Bin\Travel.exe RFQ_in.dat " + filename
        
        #linea="C:\Program Files (x86)\Path Manager\Travel\Bin\Travel.exe RFQ_in.dat " + filename
        linea="C:\Travel\Bin\Travel.exe RFQ_in.dat " + filename
        print("T ",linea)
        #linea="C:\work\Bin\Travel.exe PB29ONLY_AFTERSOL124.dat " + filename 
        #linea="C:\work\Bin\Travel.exe PB29ONLY_AFTERSOL124.dat " + filename +" > C:\work\ppasks.txt"
        #linea="C:\work\Bin\Travel.exe Pb29_sol.dat " + filename  
        #linea="C:\work\Bin\Travel.exe Pb29_2.dat " + filename
        
        #linea="Travel.exe Pb29_1.dat " + filename
        #este corre el programa

        os.system(linea)
        #subprocess.call(linea, shell=True)
        linea21="del " + filename
        #os.system(linea21)
        #subprocess.call(linea21, shell=True)
        #print(linea)


        valid_lines = []

        with open("PATHRMSQ.XLS", "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i == 0:  # saltar encabezado si lo hay
                    continue
                if '*' in line:
                    continue  # saltar líneas con asteriscos
                valid_lines.append(line)

        # Guardar temporalmente para cargar con np.loadtxt
        with open("temp_cleaned.txt", "w") as out:
            out.writelines(valid_lines)

        # Ahora cargar como float
        A = np.loadtxt("temp_cleaned.txt",skiprows=3)
        
        #A = np.loadtxt('PATHRMSQ29.XLS',skiprows=3)
        #print(A)
        Z=1*np.array(A[:,2])
        XRMS=1*np.array(A[:,5])
        YRMS=1*np.array(A[:,7])
        XPRMS=1*np.array(A[:,6])
        YPRMS=1*np.array(A[:,8])
        Phase_RMS=1*np.array(A[:,9])
        emitxRMS=(np.array(A[:,19]))
        emityRMS=(np.array(A[:,22]))
        gammaxRMS=XPRMS*XPRMS/emitxRMS
        gammayRMS=YPRMS*YPRMS/emityRMS
        AXRMS=1*np.array(A[:,26])
        AYRMS=1*np.array(A[:,28])
        BXRMS=1*np.array(A[:,27])
        BYRMS=1*np.array(A[:,29])

        B = np.loadtxt('PATHAVGq.XLS',skiprows=1)
        ZA=1*np.array(B[:,2])
        AX=1*np.array(B[:,4])
        AY=1*np.array(B[:,6])
        Trans=1*np.array(B[:,3])
        Aphase=1*np.array(B[:,8])
        Akin=1*np.array(B[:,10])
        lenAV=(len(ZA))
        Trans_f=Trans[lenAV-1]
        Akin_f=Akin[-1]
        Aphase_f=Aphase[-1]
        
        alphaxg=0.1
                      
        RXR=np.sqrt(XRMS**2+YRMS**2)
        lenRR=(len(RXR))
        R_final=RXR[lenRR-1]
        AXR_final=AXRMS[lenRR-1]
        AYR_final=AYRMS[lenRR-1]
        BX_final=BXRMS[lenRR-1]
        BY_final=BYRMS[lenRR-1]
        gammax=gammaxRMS[lenRR-1]
        gammay=gammayRMS[lenRR-1]
         
        gamaideal=(1+alphaxg**2)/0.8
        deltagamma=np.abs(gammax-gamaideal)+np.abs(gammay-gamaideal)
        deltaALPHA=np.abs(AXR_final-alphaxg)+np.abs(AYR_final-alphaxg)
        deltaBETA=np.abs(BX_final-0.001)+np.abs(BY_final-0.0001)
        DELTAX=np.abs(AXR_final-alphaxg)*np.abs(AXR_final-alphaxg)-(BX_final-0.001)*(gammax-gamaideal)
        DELTAY=np.abs(AYR_final-alphaxg)*np.abs(AYR_final-alphaxg)-(BY_final-0.001)*(gammay-gamaideal)
        DELTAT=DELTAY+DELTAX
        
        MM=np.sqrt(1+(DELTAT+np.sqrt(DELTAT*(DELTAT+4))/2))
        totalM=R_final
        lenAVE=(len(AX))
        AX_FINAL=AX[lenAVE-1]
        prueba=R_final
        nombre_archivo="parametros.txt"
        datos1=[solfac,R_final,XRMS[-1],YRMS[-1],emitxRMS[-1],emitxRMS[-1], Trans_f, Akin_f,Aphase_f,Phase_RMS[-1]]
        with open(nombre_archivo, 'a') as f:  # 'a' = append mode
                linea = '\t'.join(f"{x:.6f}" for x in datos1)  # Formatea con 6 decimales
                f.write(linea + '\n')
        f.close()
        
        #print(" %.4f %.4f %.4f %.4f %.4f"% (R_final,MM,AX_FINAL,solfac,AXR_final))
        print(" %.4f %.4f %.4f "% (solfac,R_final,Trans_f))
        return R_final, Trans_f
        
        


##initial_guess = [1.9]
#result = optimize.minimize(f, initial_guess,method='Nelder-Mead',options={'xtol': 1e-3, 'disp': True,'ftol':1.0,})
#result = optimize.minimize(f, initial_guess,method='Nelder-Mead',bounds=((-45.0, 40.0), (-40.0, 40.0), (0.8, 1.3)),options={'xtol': 1e-3, 'disp': True,'ftol':1.0,})
#result = optimize.minimize(f, initial_guess,method='L-BFGS-B',options={'xtol': 1e-3, 'disp': True,'ftol':1.0,})

##if result.success:
##    fitted_params = result.x
##    print(fitted_params)
##else:
##    raise ValueError(result.message)

fig2w=plt.figure()
ax=fig2w.add_subplot(111,title="Solenoid Strenght vs RMS", rasterized=True)
ap1=range(-10,20,1)
ap1=range(-20,10,1)
ap1=[3.0596,2.44,3.67]
ap1 = np.linspace(2.8, 3.7, 50)  # 50 puntos incluyendo 3.05 y 3.67


ap2=[4,5,6,10]
XNEW = np.array([]);
YNEW = np.array([]);
transNEW = np.array([]);
for xsol in ap1:
        radiuss, transtotal=f(xsol)
        #print(initial_guess,radiuss) 
        XNEW=np.append(XNEW,xsol)
        YNEW=np.append(YNEW,radiuss)
        transNEW=np.append(transNEW,transtotal)
         
plt.plot(XNEW, transNEW)
#plt.plot(XNEW, YNEW,label=mome)
ax.set_xlabel('Sol Factor')
ax.set_ylabel('1 R.M.S [m] KERMS0=926.7002436')
ax.legend()
plt.legend()
ax2 = ax.twinx()
ax2 .scatter(XNEW, YNEW,color='red')
#ax2.set_ylim([0, 7])
plt.show()

print(transNEW)



A = np.loadtxt('PATHRMSQ29.XLS',skiprows=1)
Z=1*np.array(A[:,2])
XRMS=1*np.array(A[:,5])
YRMS=1*np.array(A[:,7])
PXRMS=1*np.array(A[:,6])
CTY=1*np.array(A[:,1])

fige=plt.figure()
ax=fige.add_subplot(111,title="fff", rasterized=True)
ax.set_xlabel('Z [m]',fontsize=20)
ax.set_ylabel('1 R.M.S [m] ',fontsize=20)
ax.plot(Z, XRMS,label='X')
ax.plot(Z, YRMS,label='Y')
plt.legend()
ax2 = ax.twinx()
ax2 .scatter(Z, CTY,color='red')
ax2.set_ylim([0, 7])

plt.show()

           
A = np.loadtxt('output.txt',skiprows=1)
var11=1*np.array(A[:,0])
var21=1*np.array(A[:,1])

XRMS=1*np.array(A[:,7])
YRMS=1*np.array(A[:,9])
emitxRMS=(np.array(A[:,19]))
emityRMS=(np.array(A[:,22]))
alive=(np.array(A[:,70]))       


fig2w=plt.figure()
ax=fig2w.add_subplot(111,title="z vs RMS", rasterized=True)
plt.plot(var11, XRMS,label='X')
plt.plot(var11, YRMS,label='Y')
ax.set_xlabel('FActor')
ax.set_ylabel('1 R.M.S [m] ')

figew=plt.figure()
ax=figew.add_subplot(111,title="z vs emit", rasterized=True)
plt.plot(var11, 1e6*emitxRMS)
plt.plot(var11, 1e6*emityRMS)

figealive=plt.figure()
ax=figealive.add_subplot(111,title="alive", rasterized=True)
ax.set_ylabel('Transmission  ')
plt.scatter(var11, alive)

RXR=np.sqrt(1.3*XRMS**2+YRMS**2)

figerms=plt.figure()
ax=figerms.add_subplot(111,title="alive", rasterized=True)
ax.set_ylabel('Radius  ')
plt.scatter(var11, RXR)

figerms2=plt.figure()
ax=figerms2.add_subplot(111,title="alive", rasterized=True)
ax.set_ylabel('Radius  ')
plt.scatter(var21, RXR)

plt.show()
