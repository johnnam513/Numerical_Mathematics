import numpy as np
import copy
import math
from sympy import *
from matplotlib import pyplot as plt


# LU factorization
def slu(A):
    m=len(A)
    L=np.array([[0]*m for i in range(m)],dtype=np.float64)
    U=copy.deepcopy(A)
    for k in range(0,m-1):
        for i in range(k+1,m):
            xmult=U[i][k]/U[k][k]
            L[i][k]=xmult
            for j in range(k,m):  
                U[i][j]=U[i][j]-xmult*U[k][j]
    for k in range(0,m):
        L[k][k]=1
    return (L,U)

# PLU factorization
def plu(A):
    m=len(A)
    n=len(A[0])
    P=[[float(i==j) for i in range(m)] for j in range(m)]
    U=copy.deepcopy(A)
    for k in range(0,m-1):
        r=max(range(k,m),key=lambda i:abs(U[i][k]))
        q=r
        P[k],P[q]=P[q],P[k]
        U[k],U[q]=U[q],U[k]
        if U[k][k]!=0:
            for j in range(k+1,m):
                U[j][k]=U[j][k]/U[k][k]
                for i in range(k+1,n):
                    U[j][i]=U[j][i]-U[j][k]*U[k][i]
    L=np.tril(U,-1)
    for k in range(min(m,n)):
        L[k][k]=1
    if n>m:
        for i in range(m,n):
            L=np.delete(L,i,1)
    U=np.triu(U)
    if m>n:
        for i in range(n,m):
            U=np.delete(U,i,0)
    return (P,L,U)

# Solving linear equation using PLU factorization
def plusol(A,b):
    P,L,U=plu(A)
    n=len(A[0])
    y=[[0] for i in range(n)]
    c=np.dot(P,b)
    for j in range(1,n):
        y[0][0]=c[0][0]
        y[j][0]=c[j][0]
        for i in range(0,j):
            y[j][0]=y[j][0]-L[j,i]*y[i][0]
    x=[[0] for i in range(n)]
    for j in range(n-2,-1,-1):
        x[n-1][0]=y[n-1][0]/U[n-1][n-1]
        x[j][0]=y[j][0]
        for i in range(j+1,n):
            x[j][0]=x[j][0]-x[i][0]*U[j][i]
        x[j][0]=x[j][0]/U[j][j]
    return x

# Determinant
def determ(A):
    P,L,U=plu(A)
    d=np.linalg.det(P)
    for i in range(len(U)):
        d=d*np.diag(U)[i]
    return d

# Solving linear equation using Cramer's Rule
def cramer(A,b):
    m=len(A)
    n=len(A[0])
    x=[[0] for i in range(n)]
    for j in range(0,n):
        B=copy.deepcopy(A)
        for i in range(0,m):
            B[i][j]=b[i][0]
        x[j][0]=determ(B)/determ(A)
    return x

# Many ways to compute matrix product
# C(i,j)=A(i,1)B(1,j)+...+A(i,p)B(p,j)
def prod(A,B):
    n=len(B[0])
    m=len(A)
    p=len(B)
    C=np.array([[0]*n for i in range(m)])
    for j in range(0,n):
        for i in range(0,m):
            for k in range(0,p):
                C[i][j]=C[i][j]+A[i][k]*B[k][j]
    return C

# C(i,j)=row i of A times column j of B
def dot(A,B):
    n=len(B[0])
    m=len(A)
    C=np.array([[0]*n for i in range(m)])
    for j in range(0,n):
        for i in range(0,m):
            C[i][j]=np.dot(A[i],B.T[j])
    return C

#column j of C=A times column j of B   
def col(A,B):
    n=len(B[0])
    m=len(A)
    C=np.array([[0]*n for i in range(m)])
    for j in range(0,n):
        C.T[j]=C.T[j]+np.dot(A,B.T[j])
    return C

# row i of C=row i of A times B
def row(A,B):
    n=len(B[0])
    m=len(A)
    C=np.array([[0]*n for i in range(m)])
    for i in range(0,m):
        C[i]=C[i]+np.dot(A[i],B)
    return C
    
# AB=A(:,1)B(1,:)+...+A(:,p)B(p,:)
def outer(A,B):
    n=len(B[0])
    m=len(A)
    p=len(B)    
    C=np.array([[0]*n for i in range(m)])
    for k in range(0,p):
        C=C+np.dot(A.T[k][:,None],B[k][None,:])
    return C

# AB=A*B
def direct(A,B):
    C=np.dot(A,B)
    return C

# Find particular solution of linear equation
def partic_sol(A,b):
    m=len(A)
    n=len(A[0])
    Ab=np.c_[A,b]
    R=copy.deepcopy(Ab)
    R=Matrix(R)
    R,pivcol=R.rref()
    R=np.array(R)
    if max(pivcol)==n:
        x=[]
    else:
        x=[[0] for j in range(n)]
        d=[[0] for j in range(m)]
        for i in range(m):
            d[i][0]=R[i][n]
        for i in pivcol:
            x[i][0]=d[i][0]       
    return x

# Norm
def norm(x):
    n=len(x)
    sqrt_sum=0
    for i in range(n):
        sqrt_sum=sqrt_sum+(x[i][0])**2
    norm=math.sqrt(sqrt_sum)
    return norm

# Classical Gram-Smidtz Orthogonalization
def clgs(A):
    m,n=np.shape(A)
    V=copy.deepcopy(A)
    Q=np.eye(m,n)
    R=np.zeros([n,n])
    for j in range(n):
        for i in range(j):
            R[i,j]=Q[:,i].T@A[:,j]
            V[:,j]=V[:,j]-R[i,j]*Q[:,i]
        R[j,j]=np.linalg.norm(V[:,j])
        Q[:,j]=V[:,j]/R[j,j]
    return Q,R

# Modified Gram-Smidtz Orthogonalization
def mgs(A):
    m,n=np.shape(A)
    Q=copy.deepcopy(A)
    R=np.zeros([n,n])
    for i in range(n-1):
        R[i,i]=np.linalg.norm(Q[:,i])
        Q[:,i]=Q[:,i]/R[i,i]
        R[i,i+1:n]=Q[:,i]@Q[:,i+1:n]
        Q[:,i+1:n]=Q[:,i+1:n]-np.array([Q[:,i]]).T@np.array([R[i,i+1:n]])
    R[n-1,n-1]=np.linalg.norm(Q[:,n-1])
    Q[:,n-1]=Q[:,n-1]/R[n-1,n-1]
    return Q,R

# Householder QR factorization
def qrhouse(A):
    m,n=np.shape(A)
    R=copy.deepcopy(A)
    V=np.zeros([m,n])
    for k in range(0,min(m-1,n)):
        x=R[k:m,k]
        v=x+np.sign(x[0])*np.linalg.norm(x)*np.eye(1,len(x))
        V[k:m,k]=v
        R[k:m,k:n]=R[k:m,k:n]-2*v.T@v@np.array([R[k:m,k:n]])/np.linalg.norm(v)**2
    R=np.triu(R[0:n,0:n])
    return V,R

def formQ(A):
    V,R=qrhouse(A)
    m,n=np.shape(V)
    Q=np.eye(m)
    for j in range(min(m-1,n)-1,-1,-1):
        v=np.array([V[:,j]])
        Q=Q-2*v.T@v@Q/np.linalg.norm(v)**2
    return Q

# QR factorization using Givens Rotation
def Givens(i,k,m,t):
    G=np.eye(m)
    G[i,k]=-math.sin(t)
    G[i,i]=math.cos(t)
    G[k,k]=math.cos(t)
    G[k,i]=math.sin(t)
    return G

def qrgivens(A):
    R=copy.deepcopy(A)
    m,n=np.shape(A)
    Q=np.eye(m)
    if m<=n:
        k=n-1
    else:
        k=n
    for j in range(k):
        for i in range(m-1,j,-1):
            t=math.atan(R[i,j]/R[j,j])
            G=Givens(i,j,m,t)
            R=G@R
            Q=Q@G.T

    return Q,R

# Coefficients of Interpolating Polynomial using Newton Form
def coef(x,y):
    n=len(x)-1
    a=[0]*(n+1)
    for i in range(0,n+1):
        a[i]=y[i]
    for j in range(1,n+1):
        for i in range(n,j-1,-1):
            a[i]=(a[i]-a[i-1])/(x[i]-x[i-j])
    return a

# Trapezoid Rule
def Trapezoid_Uniform(f,a,b,n):
    h=(b-a)/n
    sum=(f(a)+f(b))/2
    for i in range(1,n):
        x=a+i*h
        sum=sum+f(x)
    sum=sum*h
    return sum

# Romberg Algorithm
def Romberg(f,a,b,n):
    r=np.array([[0.0]*n for i in range(n)])
    h=b-a
    r[0][0]=(h/2)*(f(a)+f(b))
    for i in range(1,n):
        h=h/2
        sum=0
        for k in range(1,2**i,2):
            sum=sum+f(a+k*h)
        r[i][0]=r[i-1][0]/2+sum*h
        for j in range(1,i+1):
            r[i][j]=r[i][j-1]+(r[i][j-1]-r[i-1][j-1])/(4**j-1)
    return r

# Simpson Rule
def Simpson(f,a,b,epsilon,level,level_max):
    level=level+1
    h=b-a
    c=(a+b)/2
    one_simpson=h*(f(a)+4*f(c)+f(b))/6
    d=(a+c)/2
    e=(c+b)/2
    two_simpson=h*(f(a)+4*f(d)+2*f(c)+4*f(e)+f(b))/12
    if level>=level_max:
        simpson_result=two_simpson
    else:
        if abs(two_simpson-one_simpson)<15*epsilon:
            simpson_result=two_simpson+(two_simpson-one_simpson)/15
        else:
            left_simpson=Simpson(f,a,c,epsilon/2,level,level_max)
            right_simpson=Simpson(f,c,b,epsilon/2,level,level_max)
            simpson_result=left_simpson+right_simpson
    return simpson_result

# Newton Cote Rule
def Newton_Cote(f,a,b,n,h):
    if n==1:
        return 2*h*f(a+h)
    if n==2:
        return 3*h*(f(a+h)+f(a+2*h))/2
    if n==3:
        return 4*h*(2*f(a+h)-f(a+2*h)+2*f(a+3*h))/3
    if n==4:
        return 5*h*(11*f(a+h)+f(a+2*h)+f(a+3*h)+11*f(a+4*h))/24
    if n==5:
        return 6*h*(11*f(a+h)-14*f(a+2*h)+26*f(a+3*h)-14*f(a+4*h)+11*f(a+5*h))/20

# Spline Curve
def Spline(f,a,b,epsilon):
    n=15
    t=np.linspace(a,b,n+1)
    y=[0.0]*(n+1)
    x=[[0.0]*10 for i in range(n)]
    t_best=[]
    y_best=[]
    for i in range(0,n+1):
        y[i]=f(t[i])
    for i in range(0,n):
        for j in range(10):
            x[i][j]=t[i]+j*(t[i+1]-t[j])/10
            def S(x):
                return y[i]+(x-t[i])*(y[i+1]-y[i])/(t[i+1]-t[i])
            if abs(S(x[i][j])-f(x[i][j]))<=epsilon:
                t_best+=[x[i][j]]
                y_best+=[S(x[i][j])]
    return t_best, y_best

# Natrual Spline Curve
def Spline3(t,y):
    n=len(t)-1
    h=[0.0]*n
    b=[0.0]*n
    u=[0.0]*n
    v=[0.0]*n
    z=[0.0]*(n+1)
    for i in range(n):
        h[i]=t[i+1]-t[i]
        b[i]=(y[i+1]-y[i])/h[i]
    u[1]=2*(h[0]+h[1])
    v[1]=6*(b[1]-b[0])
    for i in range(2,n):
        u[i]=2*(h[i]+h[i-1])-h[i-1]**2/u[i-1]
        v[i]=6*(b[i]-b[i-1])-h[i-1]*v[i-1]/u[i-1]
    z[n]=0
    for i in range(n-1,0,-1):
        z[i]=(v[i]-h[i]*z[i+1])/u[i]
    z[0]=0
    return z

# B-Spline Curve
def BSpline2_Coef(t,y):
    n=len(t)-1
    a=[0.0]*(n+2)
    h=[0.0]*(n+2)
    for i in range(1,n+1):
        h[i]=t[i]-t[i-1]
    h[0]=h[1]
    h[n+1]=h[n]
    delta=-1
    gamma=2*y[0]
    p=delta*gamma
    q=2
    for i in range(1,n+1):
        r=h[i+1]/h[i]
        delta=-r*delta
        gamma=-r*gamma+(r+1)*y[i]
        p=p+gamma*delta
        q=q+delta**2
    a[0]=-p/q
    for i in range(1,n+2):
        a[i]=((h[i-1]+h[i])*y[i-1]-h[i]*a[i-1])/h[i-1]
    return a,h

# Taylor
def Taylor(a,b,x0,n):
    x=x0
    h=(b-a)/n
    mult=0
    results=[]
    results+=[x]
    for i in range(11):
        mult=mult+h**(i)/math.factorial(i)
    for k in range(1,n+1):
        x=x*mult
        results+=[x]
    return x,results

# Runge Kutta method of order 4
def RK4(f,t,x,h,n):
    t_a=t
    for j in range(1,n+1):
        K1=h*f(t,x)
        K2=h*f(t+h/2,x+K1/2)
        K3=h*f(t+h/2,x+K2/2)
        K4=h*f(t+h,x+K3)
        x=x+(K1+2*K2+2*K3+K4)/6
        t=t_a+j*h
    return x

# Runge Kutta Fehlberg method
def RK45(f,t,x,h):
    c20=0.25
    c21=0.25
    c30=0.375
    c31=0.09375
    c32=0.28125
    c40=12/13
    c41=1932/2197
    c42=-7200/2197
    c43=7296/2197
    c51=439/216
    c52=-8
    c53=3680/513
    c54=-845/4104
    c60=0.5
    c61=-8/27
    c62=2
    c63=-3544/2565
    c64=1859/4104
    c65=-0.275
    a1=25/216
    a3=1408/2565
    a4=2197/4104
    a5=-0.2
    b1=16/135
    b3=6656/12825
    b4=28561/56340
    b5=-0.18
    b6=2/55
    K1=h*f(t,x)
    K2=h*f(t+c20*h,x+c21*K1)
    K3=h*f(t+c30*h,x+c31*K1+c32*K2)
    K4=h*f(t+c40*h,x+c41*K1+c42*K2+c43*K3)
    K5=h*f(t+h,x+c51*K1+c52*K2+c53*K3+c54*K4)
    K6=h*f(t+c60*h,x+c61*K1+c62*K2+c63*K3+c64*K4+c65*K5)
    x4=x+a1*K1+a3*K3+a4*K4+a5*K5
    x=x+b1*K1+b3*K3+b4*K4+b5*K5+b6*K6
    t=t+h
    e=abs(x-x4)
    return t,x,e

# Adative Runge Kutta method
def RK45_Adaptive(f,t,x,h,t_b,itmax,e_max,e_min,h_min,h_max):
    delta=10**(-9)/2
    iflag=1
    k=0
    while k<=itmax:
        k=k+1
        if abs(h)<h_min:
            h=np.sign(h)*h_min
        if abs(h)>h_max:
            h=np.sign(h)*h_max
        d=abs(t_b-t)
        if d<=abs(h):
            iflag=0
            if d<=delta*max(abs(t_b),abs(t)):
                break
            h=np.sign(h)*d
        t_save=t
        x_save=x
        t,x,e=RK45(f,t,x,h)
        if iflag==0:
            break
        if e<e_min:
            h=2*h
        if e>e_max:
            h=h/2
            x=x_save
            t=t_save
            k=k-1
    return x

# Many alogrithms to get eigenvalues
# power method
def power(A):
    n=len(A)
    x=np.array([[1.0]*n]).T
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=A@x
        r=f(y)/f(x)
        x=y
    return r

# inverse power method
def invpower(A):
    n=len(A)
    x=np.array([[1.0]*n]).T
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=np.linalg.solve(A,x)
        r=f(y)/f(x)
        x=y
    return 1/r

# shifted power method
def shiftpower(A):
    mu=3
    n=len(A)
    x=np.array([[1.0]*n]).T
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=(A-mu*np.eye(n))@x
        r=f(y)/f(x)
        x=y
    return r+mu

# shifted inverse power method
def shiftinvpower(A):
    mu=3
    n=len(A)
    x=np.array([[1.0]*n]).T
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=np.linalg.solve(A-mu*np.eye(n),x)
        r=f(y)/f(x)
        x=y
    return 1/r+mu

# power method accelaration
def power_with_accel(A):
    n=len(A)
    x=np.array([[1.0]*n]).T
    real=[]
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=A@x
        r=f(y)/f(x)
        real.append(r)
        x=y
        if k>=2:
            s=real[k]-(real[k]-real[k-1])**2/(real[k]-2*real[k-1]+real[k-2])
    return s

# modified power method
def modified_power(A):
    n=len(A)
    x=np.array([[1.0]*n]).T
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=A@x
        r=f(y)/f(x)
        x=y/max(y)
    return r

# modified power method with acceleration
def modified_power_with_accel(A):
    n=len(A)
    x=np.array([[1.0]*n]).T
    real=[]
    def f(x):
        return x[0][0]
    kmax=10
    for k in range(kmax):
        y=A@x
        r=f(y)/f(x)
        real.append(r)
        x=y/max(y)
        if k>=2:
            s=real[k]-(real[k]-real[k-1])**2/(real[k]-2*real[k-1]+real[k-2])
    return s

# iteration method using Jacobi, Gauss-Seidel, SOR method
def iteration(A,b,s,w):
    kmax=100
    delta=10**(-10)
    e=10**(-5)
    n=len(A)
    x=np.array([0.0]*n)
    for k in range(kmax):
        y=copy.deepcopy(x)
        for i in range(n):
            sum=b[i]
            diag=A[i][i]
            if abs(diag)<delta:
                print("diagonal element too small")
            if s=="Jacobi":
                for j in range(n):
                    if j!=i:
                        sum=sum-A[i][j]*y[j]
                x[i]=sum/diag
            else:
                for j in range(i):
                    sum=sum-A[i][j]*x[j]
                for j in range(i+1,n):
                    sum=sum-A[i][j]*x[j]
                if s=="G-S":
                    x[i]=sum/diag
                else:
                    x[i]=sum/diag
                    x[i]=w*x[i]+(1-w)*y[i]
        if np.linalg.norm(x-y)<e:
            return k,x

# linear interpolation
def best_line(x,y):
    A=np.vander(x,2)
    t=np.linalg.pinv(A)@y
    k=np.linspace(min(x),max(x),100)
    l=np.array([t[0]*k[i]+t[1] for i in range(len(k))])
    plt.scatter(x,y)
    plt.plot(k,l,'r')

# quadratic interpolation
def Quad_poly(x,y):
    a=min(x)
    b=max(x)
    m=len(x)-1
    n=2
    z=np.array([0.0]*(m+1))
    for k in range(m+1):
        z[k]=(2*x[k]-a-b)/(b-a)
    def Cheb(i,x):
        if i==0:
            return 1
        if i==1:
            return x
        if i==2:
            return 2*(x**2)-1
    T=np.zeros((n+1,m+1))
    for k in range(m+1):
        for j in range(n+1):
            T[j][k]=Cheb(j,z[k])
    for k in range(m+1):
        T[0][k]=1
        T[1][k]=z[k]
        for j in range(2,n+1):
            T[j][k]=2*z[k]*T[j-1][k]-T[j-2][k]
    A=np.zeros((n+1,n+1))
    b=np.array([0.0]*(n+1))
    for i in range(n+1):
        s=0
        for k in range(m+1):
            s=s+y[k]*T[i][k]
        b[i]=s
        for j in range(i,n+1):
            s=0
            for k in range(m+1):
                s=s+T[i][k]*T[j][k]
            A[i][j]=s
            A[j][i]=s
    c=np.linalg.solve(A,b)
    plt.scatter(x,y)
    t=np.linspace(min(x),max(x),100)
    l=np.array([c[0]+c[1]*Cheb(1,t[i])+c[2]*Cheb(2,t[i]) for i in range(len(t))])
    plt.plot(t,l,'r')

# solve parabolic problems
def Parabolic1():
    n=10
    m=128
    h=2**(-4)
    k=2**(-10)
    u=[0.0]*(n+1)
    v=[0.0]*(n+1)
    u[0]=0
    v[0]=0
    u[n]=0
    v[n]=0
    for i in range(1,n):
        u[i]=i*h*(1-i*h)
    for j in range(1,m+1):
        for i in range(1,n):
            v[i]=(u[i-1]+u[i+1])/2
        t=j*k
        for i in range(1,n):
            u[i]=v[i]
    return(t,u)
    
# solve hyperbolic problems
def Hyperbolic():
    n=10
    m=20
    h=0.1
    k=0.05
    u=[0.0]*(n+1)
    v=[0.0]*(n+1)
    w=[0.0]*(n+1)
    u[0]=0
    v[0]=0
    w[0]=0
    u[n]=0
    v[n]=0
    w[n]=0
    rho=(k/h)**2
    
    def f(x):
        return abs(x)-1
    
    for i in range(1,n):
        x=-1+i*h
        w[i]=f(x)
        v[i]=rho*(f(x-h)+f(x+h))/2+(1-rho)*f(x)
    
    for j in range(2,m+1):
        for i in range(1,n):
            u[i]=rho*(v[i+1]+v[i-1])+2*(1-rho)*v[i]-w[i]
            
    return u

# solving elliptic problem using Gauss-Seidel routine
def g(x,y):
    return 2*math.exp(x+y)

def f(x,y):
    return 0

def True_Solution(x,y):
    return math.exp(x+y)

def Bndy(x,y):
    return True_Solution(x,y)

def Ustart(x,y,i):
    if i==0:
        return x*y
    elif i==1:
        return 0
    elif i==2:
        return (1+x)*(1+y)
    elif i==3:
        return (1+x+x**2/2)*(1+y+y**2/2)
    elif i==4:
        return 1+x*y
    
def Norm(u,n_x,n_y):
    t=0
    for i in range(1,n_x):
        for j in range(1,n_y):
            t=t+u[i][j]**2
    return math.sqrt(t)
    
def Seidel(a_x,a_y,n_x,n_y,h,itmax,u):
    for k in range(1,itmax+1):
        for j in range(1,n_y):
            y=a_y+j*h
            for i in range(1,n_x):
                x=a_x+i*h
                v=u[i+1][j]+u[i-1][j]+u[i][j+1]+u[i][j-1]
                u[i][j]=(v-h**2*g(x,y))/(4-h**2*f(x,y))
    return u

# Finding minimum point using golden section algorithm
def golden_section_algorithm(F,a,b):
    itmax=100
    e=10**(-20)
    r=(-1+math.sqrt(5))/2
    x=a+r*(b-a)
    y=a+(r**2)*(b-a)
    u=F(x)
    v=F(y)
    for i in range(itmax):
        if u>v:
            b=x
            x=y
            u=v
            y=a+(r**2)*(b-a)
            v=F(y)
        elif u<=v:
            a=y
            y=x
            v=u
            x=a+r(b-a)
            u=F(x)
        if r**(i)*(b-a)/2<e:
            break
    return (b+a)/2