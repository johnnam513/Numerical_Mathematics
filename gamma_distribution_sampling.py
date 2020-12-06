import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.stats as stats

a=4.85
nsim=10000
X1=np.array([0.0]*nsim)
X2=np.array([0.0]*nsim)
X1[0]=np.random.gamma(a,1)
X2[0]=np.random.gamma(a,1)
for i in range(1,nsim):
    Y=np.random.gamma(math.floor(a),a/math.floor(a))
    rhoAR=(math.exp(1)*Y*math.exp(-Y/a)/a)**(a-math.floor(a))
    rhoMH=(stats.gamma.pdf(Y,a,scale=1)/stats.gamma.pdf(X2[i-1],a,scale=1))/(stats.gamma.pdf(Y,math.floor(a),scale=a/math.floor(a))/stats.gamma.pdf(X2[i-1],math.floor(a),scale=a/math.floor(a)))
    rhoMH=min(rhoMH,1)
    randnum=np.random.uniform()
    if randnum<rhoAR:
        X1[i]=Y
    if randnum>=rhoAR:
        X1[i]=0
    if randnum<rhoMH:
        X2[i]=Y
    if randnum>=rhoMH:
        X2[i]=X2[i-1]
X1=np.delete(X1,np.where(X1==0)[0])

fig,ax=plt.subplots(2,2,sharex=False,figsize=(20,20)) 
ax[0,0].hist(X1,bins=125,density=True)
ax[0,0].set_title('Accept-Reject')
ax[0,0].set_xlim([0,15])
t=np.linspace(0,15,100)
ax[0,0].plot(t,stats.gamma.pdf(t,a,scale=1),'r')
ax[0,1].hist(X2[2499:nsim],bins=125,density=True)
ax[0,1].set_title('Metropolis-Hastings')
ax[0,1].set_xlim([0,15])
ax[0,1].plot(t,stats.gamma.pdf(t,a,scale=1),'r')

v=X1
variance=v.var()
v=v-v.mean()
r=np.correlate(v,v,mode='full')[-len(X1):]
result=r/(variance*(np.arange(len(X1),0,-1)))
ax[1,0].bar([i for i in range(50)],result[0:50])
ax[1,0].set_ylabel('ACF')
ax[1,0].set_xlabel('Lag')

v=X2
variance=v.var()
v=v-v.mean()
r=np.correlate(v,v,mode='full')[-len(X2):]
result=r/(variance*(np.arange(len(X2),0,-1)))
ax[1,1].bar([i for i in range(50)],result[0:50])
ax[1,1].set_ylabel('ACF')
ax[1,1].set_xlabel('Lag')

plt.show()