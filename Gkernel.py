from astropy.io import fits
import numpy as np
import hermite
import math
import copy
from scipy import ndimage,signal
from astropy.convolution import convolve, convolve_fft

def map(ncoeff,LL,Cmat):
    htarget=np.zeros([ncoeff+1,ncoeff+1])
    htarget[0,0]=1/(2.*np.sqrt(math.pi))
    hflat=htarget.flatten()  ## this is the target PSF, flattened in 1-dimension

    Klist=[]

    for element in enumerate(LL):
        kkernel=np.zeros([ncoeff+1,ncoeff+1])
        kflat=kkernel.flatten()
        starcoeff=element[1][1]  ## this is the shapelet coefficients of current star size (ncoeff+1,ncoeff+1)
        Pmat=Pmatrix(ncoeff,Cmat,starcoeff)
        Pmatflat=Pmat.reshape(((ncoeff+1)**2,(ncoeff+1)**2))

        kflat, r, rank, s = np.linalg.lstsq(Pmatflat, hflat)
        ker=kflat.reshape((ncoeff+1,ncoeff+1))
        Klist.append((ker,Pmat))

    return Klist

#def Pmatrix(ncoeff,Cmat,S):
#    Pmat=np.zeros([ncoeff+1,ncoeff+1,ncoeff+1,ncoeff+1])
#    for i in range(ncoeff+1):
#        for j in range(ncoeff+1):
#            for k in range(ncoeff+1):
#                for l in range(ncoeff+1):
#                    Pmat[i,j,k,l]=np.einsum('i,j,ij->',Cmat[i,k,:],Cmat[j,l,:],S)
#
#    return Pmat

def Pmatrix(ncoeff,Cmat,S):
    Pmat=np.zeros([ncoeff+1,ncoeff+1,ncoeff+1,ncoeff+1])
    Pmat=np.einsum('npi,mqj,ij->nmpq',Cmat,Cmat,S)

    return Pmat

def convol(ncoeff,Pmat,K):
    hcoeffs=np.zeros([ncoeff+1,ncoeff+1])
    for i in range(ncoeff+1):
        for j in range(ncoeff+1):
            hcoeffs[i,j]=np.einsum('ij,ij->',Pmat[i,j,:,:],K[:,:])
    return hcoeffs

def getKlist(ncoeff,LL,KK):
    LK=copy.copy(LL)
    for element in enumerate(LL):  #  loops over the individual stars
        starposit=element[1][2]
        xstar=starposit[0]
        ystar=starposit[1]
        ii=element[0]
        starcoeff=KK[ii][0]  ## this is the shapelet coefficients of the kernel (ncoeff+1,ncoeff+1) for star ii
        LK[ii]=(LL[ii][0],starcoeff,LL[ii][2],LL[ii][3])

    return LK

def Tmatrix(ncoeff,ooK,BB):
    npoly=ooK[3].shape[2]
    nx=BB[1].shape[2]
    ny=BB[1].shape[3]
    Tmat=np.zeros([npoly,nx,ny])
    Tmat=np.einsum('ijk,ijrs->krs',ooK[3],BB[1])

    return Tmat

def gaussianize(ncoeff,Tmat,dataimage,datahdr,weight=None):
#    hdulist = fits.open(image_file)
    scidata = dataimage

    nx=datahdr['NAXIS1']
    ny=datahdr['NAXIS2']
    x=np.arange(1,nx+1)
    y=np.arange(1,ny+1)
    xx,yy=np.meshgrid(x,y)

#    if weight is not None:
    if weight is None:
#        rnorm=np.ones([6,ny,nx])
        weight=np.ones([ny,nx])
        print('Computing weights....')
        datanorm = signal.fftconvolve(weight,Tmat[0,:,:],mode='same') + \
            xx*signal.fftconvolve(weight,Tmat[1,:,:],mode='same') + \
            yy*signal.fftconvolve(weight,Tmat[2,:,:],mode='same') + \
            xx*xx*signal.fftconvolve(weight,Tmat[3,:,:],mode='same') + \
            xx*yy*signal.fftconvolve(weight,Tmat[4,:,:],mode='same') + \
            yy*yy*signal.fftconvolve(weight,Tmat[5,:,:],mode='same') + \
            xx*yy**2*signal.fftconvolve(weight,Tmat[6,:,:],mode='same') + \
            xx**2*yy*signal.fftconvolve(weight,Tmat[7,:,:],mode='same') + \
            xx**3*signal.fftconvolve(weight,Tmat[8,:,:],mode='same') + \
            yy**3*signal.fftconvolve(weight,Tmat[9,:,:],mode='same')

    dataout = signal.fftconvolve(scidata,Tmat[0,:,:],mode='same') + \
              xx*signal.fftconvolve(scidata,Tmat[1,:,:],mode='same') + \
              yy*signal.fftconvolve(scidata,Tmat[2,:,:],mode='same') + \
              xx*xx*signal.fftconvolve(scidata,Tmat[3,:,:],mode='same') + \
              xx*yy*signal.fftconvolve(scidata,Tmat[4,:,:],mode='same') + \
              yy*yy*signal.fftconvolve(scidata,Tmat[5,:,:],mode='same') + \
            xx*yy**2*signal.fftconvolve(scidata,Tmat[6,:,:],mode='same') + \
            xx**2*yy*signal.fftconvolve(scidata,Tmat[7,:,:],mode='same') + \
            xx**3*signal.fftconvolve(scidata,Tmat[8,:,:],mode='same') + \
            yy**3*signal.fftconvolve(scidata,Tmat[9,:,:],mode='same')

    return datanorm,dataout/datanorm




