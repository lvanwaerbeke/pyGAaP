from astropy.io import fits
import numpy as np
import hermite
import copy

def chip(dataimage,catstars,BB):
    stars=catstars
#    hdulist = fits.open(image)
#    hdulist.info()
#    hdr=hdulist[0].header

    LL=[]
    for i in np.where(stars[:,2] !=0)[0]:
        ix,iy=stars[i,0:2].astype(int)
        scidata = dataimage[iy-8:iy+9,ix-8:ix+9]
        rnorm=np.sum(scidata)
#        print('total flux: ',i,ix,iy,rnorm)
        scidata=scidata/rnorm
        CC=hermite.getcoeff(scidata,BB)
        LL.append((scidata,CC,stars[i,0:2],stars[i,2]))

    return LL

def fittedfunc(x,y,coeff):

    one=np.ones_like(x)
#    z=coeff[0]*one+coeff[1]*x+coeff[2]*y+coeff[3]*x**2+coeff[4]*x*y+coeff[5]*y**2
    z=coeff[0]*one+coeff[1]*x+coeff[2]*y+coeff[3]*x**2+coeff[4]*x*y+coeff[5]*y**2+coeff[6]*x*y**2+coeff[7]*x**2*y+coeff[8]*x**3+coeff[9]*y**3

    return z

def fittedcoeff(ncoeff,LL):
    nstars=len(LL)
#    npoly=6
    npoly=10
    CCpoly=np.zeros([ncoeff+1,ncoeff+1,npoly])  #  polynomial coefficients for each shapelet coefficient. 6 is the number of polynomial coefficients, hard coded in fittedfunct() and A defined below
    CCdata=np.zeros([ncoeff+1,ncoeff+1,nstars])  #  contains the shapelet coefficients for all stars
    for i in range(ncoeff+1):
        for j in range(ncoeff+1):
            cc0=[]
            x0=[]
            y0=[]
            for el in enumerate(LL):  #  loops over the individual stars
#                scidata=el[1][0]
#                galrecon=hermite.recon(CC,BB)
                CC=el[1][1]  #  for each star, extract the shapelet coefficients (ncoeff+1,ncoeff+1)
                cc0.append(CC[i,j])  #  from the shapelet coefficients extract the (i,j) element
                x0.append(el[1][2][0])  #  for each star extract its position x0
                y0.append(el[1][2][1])  #  for each star extract its position y0

            x0=np.array(x0)  #  x0 is an array of all stars x-positions. size nstars
            y0=np.array(y0)  #  y0 is an array of all stars y-positions. size nstars
            cc0=np.array(cc0)  #  cc0 is an array of the shapelet coefficients (i,j) for all stars. size nstars
#            A = np.array([x0*0+1, x0, y0, x0**2, x0*y0, y0**2]).T
            A = np.array([x0*0+1, x0, y0, x0**2, x0*y0, y0**2, x0*y0**2, x0**2*y0, x0**3, y0**3]).T
            B = cc0
            polycoeff, res, rank, s = np.linalg.lstsq(A, B)  ##  solves A.polycoeff=B
            CCpoly[i,j,:]=polycoeff  #  polynomial fit coefficients for shapelet element (i,j)
            CCdata[i,j,:]=cc0
#            print(i,j,len(polycoeff),len(cc0))

    return  x0,y0,CCdata,CCpoly

def getcoeff(ncoeff,LL,oo):
    LLfit=copy.copy(LL)
    for element in enumerate(LL):  #  loops over the individual stars
        starcoeff=element[1][1]  ## this is the shapelet coefficients of current star size (ncoeff+1,ncoeff+1)
        starposit=element[1][2]
        xstar=starposit[0]
        ystar=starposit[1]
        for i in range(ncoeff+1):
            for j in range(ncoeff+1):
                starcoeff[i,j]=fittedfunc(xstar,ystar,oo[3][i,j,:])
        ii=element[0]
        LLfit[ii]=(LL[ii][0],starcoeff,LL[ii][2],LL[ii][3])

    return LLfit


