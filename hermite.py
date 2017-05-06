import numpy as np
from scipy.special import factorial
import fitstars

def coeff(ncoeff):
    hcoeff=np.zeros([ncoeff+1,ncoeff+1])
    hcoeff[0,0]=1.
    hcoeff[1,0]=0.
    hcoeff[1,1]=2.

    for i in range(2,ncoeff+1):
        hcoeff[i,0]=-hcoeff[i-1,1]
        for j in range(1,i+1):
            hcoeff[i,j]=2*hcoeff[i-1,j-1]-2*(i-1)*hcoeff[i-2,j]
    return hcoeff

def phi(ncoeff,xx):
    phicube=np.zeros([ncoeff+1]+list(xx.shape))
    for i in range(ncoeff+1):
        for k in range(i+1):
            phicube[i,:]=phicube[i,:]+coeff(ncoeff)[i,k]*xx**k*np.exp(-xx**2/2)
        phicube[i,:]=phicube[i,:]/np.sqrt(2**i*np.pi**0.5*factorial(i))
    return phicube

def Bfunc(ncoeff,rgbeta):
    nx,ny=17,17
    x = np.linspace(-8, 8, nx)
    y = np.linspace(-8, 8, ny)
    xx,yy=np.meshgrid(x,y)
    Bbasis=np.zeros([ncoeff+1,ncoeff+1]+list(xx.shape))
    phi_i_x=phi(ncoeff,xx/rgbeta)
    phi_i_y=phi(ncoeff,yy/rgbeta)
    for i in range(ncoeff+1):
        for j in range(ncoeff+1):
            if i+j <= ncoeff :
                Bbasis[i,j,:,:]=phi_i_x[i,:]*phi_i_y[j,:]/rgbeta

    return xx,Bbasis

def getcoeff(fprofile,BB):
    n1=BB[1].shape[0]
    n2=BB[1].shape[1]
    coeff=np.zeros((n1,n2))
    for i in range(n1+1):
        for j in range(n2+1):
            coeff[i-1,j-1]=np.sum(BB[1][i-1,j-1,:,:]*fprofile)

    return coeff

def recon(coeffs,BB):
    n1=coeffs.shape[0]
    n2=coeffs.shape[1]
    galrecon=np.zeros_like(BB[1][0,0,:,:])
    for i in range(n1+1):
        for j in range(n2+1):
            galrecon=galrecon+coeffs[i-1,j-1]*BB[1][i-1,j-1,:,:]

    return galrecon

def findrgbeta(rgbeta,ncoeff,dataimage,catstars,indexes):
    BB=Bfunc(ncoeff,rgbeta)
    LL=fitstars.chip(dataimage,catstars,BB)
    chi2=0
    for ind in enumerate(indexes):
        istar=ind[1]
        galrecon=recon(LL[istar][1],BB)
        chi2=chi2+sum(map(sum,(galrecon-LL[istar][0])**2))

    return chi2

def Cmatrix(ncoeff,g,a,b):
    Lmat=np.zeros([ncoeff+1,ncoeff+1,ncoeff+1])
    Bmat=np.zeros([ncoeff+1,ncoeff+1,ncoeff+1])
    Cmat=np.zeros([ncoeff+1,ncoeff+1,ncoeff+1])

    nu=1/np.sqrt(g**2+a**2+b**2)
    a1=np.sqrt(2)*nu*g
    a2=np.sqrt(2)*nu*a
    a3=np.sqrt(2)*nu*b

    Lmat[0,0,0]=1
    #  Calculate the L00i,L0i0,Li00 terms
    for i in range(2,ncoeff+1,2):
        Lmat[0,0,i]=2*(i-1)*(a3**2-1)*Lmat[0,0,i-2]
        Lmat[0,i,0]=2*(i-1)*(a2**2-1)*Lmat[0,i-2,0]
        Lmat[i,0,0]=2*(i-1)*(a1**2-1)*Lmat[i-2,0,0]
    #  Calculate the L0ij,Li0j,Lij0 terms
    for i in range(1,ncoeff+1):
        for j in range(1,ncoeff+1):
            if ((i+j) % 2 == 0):
                Lmat[0,i,j]=2*(i-1)*(a2**2-1)*Lmat[0,i-2,j]+2*j*a2*a3*Lmat[0,i-1,j-1]
                Lmat[i,0,j]=2*(i-1)*(a1**2-1)*Lmat[i-2,0,j]+2*j*a1*a3*Lmat[i-1,0,j-1]
                Lmat[i,j,0]=2*(i-1)*(a1**2-1)*Lmat[i-2,j,0]+2*j*a1*a2*Lmat[i-1,j-1,0]
    #  Calculate the L1ij terms
    for i in range(1,ncoeff+1):
        for j in range(1,ncoeff+1):
            if ((1+i+j) % 2 == 0):
                Lmat[1,i,j]=2*i*a1*a2*Lmat[0,i-1,j]+2*j*a1*a3*Lmat[0,i,j-1]
    #  Calculate the Li+1jk terms
    for i in range(1,ncoeff):
        for j in range(1,ncoeff+1):
            for k in range(1,ncoeff+1):
                if ((i+1+j+k) % 2 == 0):
                    Lmat[i+1,j,k]=2*i*(a1**2-1)*Lmat[i-1,j,k]+2*a1*a2*j*Lmat[i,j-1,k]+2*a1*a3*k*Lmat[i,j,k-1]

    #  Calculate the Bijk terms
    for i in range(ncoeff+1):
        for j in range(ncoeff+1):
            for k in range(ncoeff+1):
                if ((i+j+k) % 2 == 0):
                    Bmat[i,j,k]=nu*np.sqrt(g*a*b)/np.sqrt(2**(i+j+k-1)*np.sqrt(np.pi)*factorial(i)*factorial(j)*factorial(k))*Lmat[i,j,k]
                    Cmat[i,j,k]=np.sqrt(2*np.pi)*(-1)**i*(-1)**((i+j+k)/2)*Bmat[i,j,k]

    return Cmat

