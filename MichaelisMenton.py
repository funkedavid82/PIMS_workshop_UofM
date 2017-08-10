
# coding: utf-8

# In[84]:

# Here we import all the code we need. And plot inline

get_ipython().magic(u'matplotlib inline')
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint # This is the numerical solver


# In[85]:

import csv
import pandas as pd
## Reading data from csv file
EXP = []
Time = []
TRT = []
mTOR_total = []
pmTOR = []
B_total = []
AMPK_total = []
pAMPK = []
with open('Data.csv') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    count = 0
    for row in csvReader:
        if(count>0):
            EXP.append(row[0])
            Time.append(int(row[1]))
            TRT.append(row[2])
            mTOR_total.append(float(row[3]))
            pmTOR.append(float(row[4]))
            B_total.append(float(row[5]))
            AMPK_total.append(float(row[6]))
            pAMPK.append(float(row[7]))
        count += 1
#EXP_data = [EXP,Time,TRT,mTOR_total,pmTOR,B_total,AMPK_total,pAMPK]

EXP_data = {'EXP' : pd.Series(EXP),'Time' : pd.Series(Time),'TRT':pd.Series(TRT),
           'mTOR_total': pd.Series(mTOR_total), 'pmTOR': pd.Series(pmTOR), 'B_total': pd.Series(B_total),
           'AMPK_total': pd.Series(AMPK_total), 'pAMPK': pd.Series(pAMPK)}

df = pd.DataFrame(EXP_data)


# The system we model is:
# $$\vec{C} = (pB,B,pAMPK,AMPK,pmTOR,mTOR,C_3,C_4,D_1,D_2) $$
# 
# $$\kappa_1 = k_7, \kappa_2 = k_8, \kappa_3 = k_9, \kappa_4 = k_{10}$$
# 
# $$ V_1,\cdots, V_7 $$
# $$ K_1,\cdots, K_7 $$

# In[86]:

## function for sum of squares:
def sum_of_squares(k,kappa,v):
    # Define the system
    def rhs(c,t,k,kappa,v):
        alpha = 0.
        beta = 0.
        dydt = np.zeros((10,),dtype=float)
        dydt[0] = -v[0]*c[0]/(k[0]+c[0]) + v[1]*c[1]*c[4]/(k[1]+c[1])
        dydt[1] = - dydt[0]
        dydt[2] = v[2]*c[3]*c[1]/(k[2]+c[3]) - v[3] * c[2] / (k[3] + c[2])
        dydt[3] = - dydt[2]
        dydt[5] = alpha - v[4] * c[5] / (k[4] + c[5]) - v[5] * c[5] * c[2] / (k[5] + c[5]) + v[6] * c[4] / (k[6] + c[4]) - kappa[0] * c[5] * c[8] - kappa[2] * c[5] * c[9] +  kappa[1] * c[6] + kappa[3] * c[7]
        dydt[4] = -beta * c[4] + v[4] * c[5]/(k[4] + c[5]) - v[6] * c[4] /(k[6]+c[4])
        dydt[6] = kappa[0] * c[5] * c[8] - kappa[1] * c[6]
        dydt[7] = kappa[2] * c[5] * c[9] - kappa[3] * c[7]
        dydt[8] = 0
        dydt[9] = 0
        return dydt

    # Now we set up the inital time range, and initial values for coefficients c1, c2, c3
    t_arr = np.linspace(0.,1.5,1501)
    
    ###### control group number 1
    c_init = np.array([0.0669/100,0.0669,0.9796,0.5517,1.0512, 0.4944,0.0,0.0,0.0,0.0])
    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    

    # Now calculate sum of squares
    ## enzymeB + pAMPK + AMPK + mTOR + pmTOR
    sum_of_squares = sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '1')&(df.TRT=='control')].B_total)**2)/0.003197438**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '1')&(df.TRT=='control')].pAMPK)**2)/0.040478822**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '1')&(df.TRT=='control')].AMPK_total-df[(df.EXP == '1')&(df.TRT=='control')].pAMPK))**2)/0.087347097**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '1')&(df.TRT=='control')].mTOR_total-df[(df.EXP == '1')&(df.TRT=='control')].pmTOR))**2)/0.053604527**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '1')&(df.TRT=='control')].pmTOR)**2)/0.05754211**2 
    #print(sum_of_squares)
    
    ###### control group number 2
    c_init = np.array([0.0709/100,0.0709,1.0992,0.5497,0.9949,0.2395,0.,0.,0.,0.])
    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    # Now calculate sum of squares
    sum_of_squares = sum_of_squares +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '2')&(df.TRT=='control')].B_total)**2)/0.006871944**2 +                     sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '2')&(df.TRT=='control')].pAMPK)**2)/0.064590711**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '2')&(df.TRT=='control')].AMPK_total-df[(df.EXP == '2')&(df.TRT=='control')].pAMPK))**2)/0.140098309**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '2')&(df.TRT=='control')].mTOR_total-df[(df.EXP == '2')&(df.TRT=='control')].pmTOR))**2)/0.118362312**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '2')&(df.TRT=='control')].pmTOR)**2)/0.068750929**2 


    ###### drug 1 group number 1
    c_init = np.array([0.0682/100,0.0682,0.9756,0.7973,1.0753,0.614,0.,0.,1.,0.])

    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    # Now calculate sum of squares
    sum_of_squares = sum_of_squares +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '1')&(df.TRT=='drug1')].B_total)**2)/0.055060696**2 +                     sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '1')&(df.TRT=='drug1')].pAMPK)**2)/0.082961946**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '1')&(df.TRT=='drug1')].AMPK_total-df[(df.EXP == '1')&(df.TRT=='drug1')].pAMPK))**2)/0.131127712**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '1')&(df.TRT=='drug1')].mTOR_total-df[(df.EXP == '1')&(df.TRT=='drug1')].pmTOR))**2)/1.237248112**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '1')&(df.TRT=='drug1')].pmTOR)**2)/0.302311222**2 

                        
    ###### drug 1 group number 2
    c_init = np.array([0.0743/100,0.0743,1.064,0.278,0.9491,0.431,0.,0.,1.,0.])

    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    # Now calculate sum of squares
    sum_of_squares = sum_of_squares +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '2')&(df.TRT=='drug1')].B_total)**2)/0.060240788**2 +                     sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '2')&(df.TRT=='drug1')].pAMPK)**2)/0.087360446**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '2')&(df.TRT=='drug1')].AMPK_total-df[(df.EXP == '2')&(df.TRT=='drug1')].pAMPK))**2)/0.125163173**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '2')&(df.TRT=='drug1')].mTOR_total-df[(df.EXP == '2')&(df.TRT=='drug1')].pmTOR))**2)/1.429151195**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '2')&(df.TRT=='drug1')].pmTOR)**2)/0.273576638**2

    ###### drug 2 group number 2
    c_init = np.array([0.0692/100,0.0692,0.8812,0.6064,1.0343,0.3271,0.,0.,0.,1.])

    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    # Now calculate sum of squares
    sum_of_squares = sum_of_squares +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '2')&(df.TRT=='drug2')].B_total)**2)/0.050584108**2 +                     sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '2')&(df.TRT=='drug2')].pAMPK)**2)/0.07437139**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '2')&(df.TRT=='drug2')].AMPK_total-df[(df.EXP == '2')&(df.TRT=='drug2')].pAMPK))**2)/0.191161848**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '2')&(df.TRT=='drug2')].mTOR_total-df[(df.EXP == '2')&(df.TRT=='drug2')].pmTOR))**2)/0.117766178**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '2')&(df.TRT=='drug2')].pmTOR)**2)/0.275367037**2 
                        
    ###### drug 2 group number 1
    c_init = np.array([0.0612/100,0.0612,0.9967,0.6336,1.132,0.4228,0.,0.,0.,1.])
    # Here is the call to the ODE solver (it is really that simple)
    c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))
    # Now calculate sum of squares
    sum_of_squares = sum_of_squares +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),1] -                           df[(df.EXP == '1')&(df.TRT=='drug2')].B_total)**2)/0.050997258**2 +                     sum((c_arr[(0,4,9,29,59,179,359,719,1439),2] -                           df[(df.EXP == '1')&(df.TRT=='drug2')].pAMPK)**2)/0.036948985**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),3] -                           (df[(df.EXP == '1')&(df.TRT=='drug2')].AMPK_total-df[(df.EXP == '1')&(df.TRT=='drug2')].pAMPK))**2)/0.12030113**2 +                      sum((c_arr[(0,4,9,29,59,179,359,719,1439),5] -                           (df[(df.EXP == '1')&(df.TRT=='drug2')].mTOR_total-df[(df.EXP == '1')&(df.TRT=='drug2')].pmTOR))**2)/0.11674912**2 +                       sum((c_arr[(0,4,9,29,59,179,359,719,1439),4] -                           df[(df.EXP == '1')&(df.TRT=='drug2')].pmTOR)**2)/0.304263863**2
    #print(sum_of_squares)
    return(sum_of_squares)


# In[87]:

#print(sum_of_squares(k=np.linspace(1.,1.,7),kappa=np.linspace(.1,.1,4),v=np.linspace(1.,1.,7)))


# In[88]:

from scipy.optimize import fmin
## define function that is optimized:
def logk_SS(log_all):
    p_all = np.array([np.exp(i) for i in log_all])
    k = p_all[0:7]
    v = p_all[7:14]
    kappa = p_all[14:]
    SS = sum_of_squares(k=k,kappa=kappa,v=v)
    return(SS)


# In[89]:

## start at random point:
log_all0 = np.random.randn(18)/10
log_all0[7:] = log_all0[7:] - 2
print(logk_SS(log_all0),log_all0)
## optimize over sum of squares
SS_opt = fmin(logk_SS, log_all0, xtol=0.01,full_output=1)


# In[90]:

## getting optimal parameter
opt_par = np.array([np.exp(i) for i in SS_opt[0]])


# In[91]:

print(sum_of_squares(k=opt_par[:7],v=opt_par[7:14],kappa=opt_par[14:]))


# In[92]:

## generate the solution:
def rhs(c,t,k,kappa,v):
        alpha = 0.
        beta = 0.
        dydt = np.zeros((10,),dtype=float)
        dydt[0] = -v[0]*c[0]/(k[0]+c[0]) + v[1]*c[1]*c[4]/(k[1]+c[1])
        dydt[1] = - dydt[0]
        dydt[2] = v[2]*c[3]*c[1]/(k[2]+c[3]) - v[3] * c[2] / (k[3] + c[2])
        dydt[3] = - dydt[2]
        dydt[5] = alpha - v[4] * c[5] / (k[4] + c[5]) - v[5] * c[5] * c[2] / (k[5] + c[5]) + v[6] * c[4] / (k[6] + c[4]) - kappa[0] * c[5] * c[8] - kappa[2] * c[5] * c[9] +  kappa[1] * c[6] + kappa[3] * c[7]
        dydt[4] = -beta * c[4] + v[4] * c[5]/(k[4] + c[5]) - v[6] * c[4] /(k[6]+c[4])
        dydt[6] = kappa[0] * c[5] * c[8] - kappa[1] * c[6]
        dydt[7] = kappa[2] * c[5] * c[9] - kappa[3] * c[7]
        dydt[8] = 0
        dydt[9] = 0
        return dydt
t_arr = np.linspace(0.,1.5,1501)
###### control group number 1
c_init = np.array([0.0669/100,0.0669,0.9796,0.5517,1.0512, 0.4944,0.0,0.0,0.0,0.0])
# Here is the call to the ODE solver (it is really that simple)
k,kappa,v =opt_par[:7],opt_par[14:],opt_par[7:14]
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))


# In[93]:

data_control1 = df[(df.EXP=='1')&(df.TRT=="control")]
ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]
plt.ion()

#plt.plot(pAMPK)
#plt.plot(AMPK)
#plt.plot(mTOR)
#plt.plot(pmTOR)
#print(sum_of_squares(k=k,alpha=0.,beta=0.))
#print(data_control1)
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_control1.Time,y=data_control1.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_control1.Time,y=data_control1.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_control1.Time,y=data_control1.AMPK_total-data_control1.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_control1.Time,y=data_control1.mTOR_total-data_control1.pmTOR)


# In[102]:

#plt.plot(mTOR)
#plt.scatter(x=data_control1.Time,y=data_control1.mTOR_total-data_control1.pmTOR,color="r")
#plt.ylabel('mTOR')
#plt.ylim((0,1))


# In[69]:




# In[103]:

###### control group number 2
c_init = np.array([0.0709/100,0.0709,1.0992,0.5497,0.9949,0.2395,0.,0.,0.,0.])
# Here is the call to the ODE solver (it is really that simple)
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))


# In[114]:




# In[104]:

data_control2 = df[(df.EXP=='2')&(df.TRT=="control")]
ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]
plt.ion()

#plt.plot(pAMPK)
#plt.plot(AMPK)
#plt.plot(mTOR)
#plt.plot(pmTOR)
#print(sum_of_squares(k=k,alpha=0.,beta=0.))
#print(data_control1)
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_control2.Time,y=data_control2.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_control2.Time,y=data_control2.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_control2.Time,y=data_control2.AMPK_total-data_control2.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_control2.Time,y=data_control2.mTOR_total-data_control2.pmTOR)


# In[115]:

###### drug 1 group number 1
c_init = np.array([0.0682/100,0.0682,0.9756,0.7973,1.0753,0.614,0.,0.,1.,0.])

# Here is the call to the ODE solver (it is really that simple)
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))

data_drug11 = df[(df.EXP=='1')&(df.TRT=="drug1")]

ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]
plt.ion()
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_drug11.Time,y=data_drug11.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_drug11.Time,y=data_drug11.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_drug11.Time,y=data_drug11.AMPK_total-data_drug11.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_drug11.Time,y=data_drug11.mTOR_total-data_drug11.pmTOR)


# In[126]:




# In[128]:


###### drug 1 group number 2
c_init = np.array([0.0743/100,0.0743,1.064,0.278,0.9491,0.431,0.,0.,1.,0.])

# Here is the call to the ODE solver (it is really that simple)
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))

data_drug12 = df[(df.EXP=='2')&(df.TRT=="drug1")]

ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]


plt.ion()
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_drug12.Time,y=data_drug12.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_drug12.Time,y=data_drug12.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_drug12.Time,y=data_drug12.AMPK_total-data_drug12.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_drug12.Time,y=data_drug12.mTOR_total-data_drug12.pmTOR)    


# In[136]:




# In[138]:

###### drug 2 group number 1
c_init = np.array([0.0612/100,0.0612,0.9967,0.6336,1.132,0.4228,0.,0.,0.,1.])
# Here is the call to the ODE solver (it is really that simple)
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))


data_drug21 = df[(df.EXP=='1')&(df.TRT=="drug2")]

ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]


plt.ion()
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_drug21.Time,y=data_drug21.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_drug21.Time,y=data_drug21.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_drug21.Time,y=data_drug21.AMPK_total-data_drug21.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_drug21.Time,y=data_drug21.mTOR_total-data_drug21.pmTOR)    


# In[146]:




# In[ ]:




# In[147]:

###### drug 2 group number 2
c_init = np.array([0.0692/100,0.0692,0.8812,0.6064,1.0343,0.3271,0.,0.,0.,1.])

# Here is the call to the ODE solver (it is really that simple)
c_arr = odeint(rhs,c_init,t_arr, args=(k,kappa,v,))

data_drug22 = df[(df.EXP=='2')&(df.TRT=="drug2")]

ezmB,pAMPK,AMPK,mTOR,pmTOR = c_arr[:,1],c_arr[:,2],c_arr[:,3],c_arr[:,5],c_arr[:,4]


plt.ion()
plt.subplot(2, 2, 1)
plt.plot(pmTOR)
plt.scatter(x=data_drug22.Time,y=data_drug22.pmTOR)
plt.subplot(2, 2, 2)
plt.plot(pAMPK)
plt.scatter(x=data_drug22.Time,y=data_drug22.pAMPK)
plt.subplot(2, 2, 3)
plt.plot(AMPK)
plt.scatter(x=data_drug22.Time,y=data_drug22.AMPK_total-data_drug22.pAMPK)
plt.subplot(2, 2, 4)
plt.plot(mTOR)
plt.scatter(x=data_drug22.Time,y=data_drug22.mTOR_total-data_drug22.pmTOR)    


# In[156]:




# In[ ]:



