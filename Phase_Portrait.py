import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def f(X,t):
    x1,x2=X
    return [x1*(2-x1-x2),x2*(-2+4*x1-2*x2)]

x1=np.array([0.1,0.5,1,2,4])
x2=np.array([0.1,0.5,1,2,4])

X1,X2=np.meshgrid(x1,x2)

t=0

u,v=np.zeros(X1.shape),np.zeros(X2.shape)

NI,NJ=X1.shape

for i in range(NI):
    for j in range(NJ):
        x=X1[i,j]
        y=X2[i,j]
        yprime=f([x,y],t)
        u[i,j]=yprime[0]
        v[i,j]=yprime[1]
        
Q=plt.quiver(X1,X2,u,v,color='r')

plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.xlim([0,4])
plt.ylim([0,4])
plt.title('Phase Portrait for Predator-Prey Model')

for y1 in x1:
    for y2 in x2:
        tspan=np.linspace(0,50,200)
        y0=[y1,y2]
        ys=odeint(f,y0,tspan)
        plt.plot(ys[:,0],ys[:,1],'b-')

plt.show()