import numpy as np
import copy

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