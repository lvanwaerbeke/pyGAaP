from astropy.io import fits
from astropy.stats import sigma_clip
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import fmin
import numpy as np
import math
import hermite
import fitstars
import Gkernel
import utils
import copy
import os
import sys
os.environ['MKL_NUM_THREADS'] = '12'

arg1 = sys.argv[1]
arg2 = sys.argv[2]
arg3 = sys.argv[3]
arg4 = sys.argv[4]
arg5 = sys.argv[5]

image_file=arg1
hdudata = fits.open(image_file)
hdudata.info()
datahdr=hdudata[0].header
dataimage=hdudata[0].data

stars_file=arg2
stars=np.loadtxt(stars_file)
select=np.loadtxt(arg3)
bs=np.array([i for i, item in enumerate(stars[:,8]) if item in select])
catstars=stars[bs,:]
catstars[:,0:2]=catstars[:,0:2]-0.5  # adjust for half integer shift in stars position

ncoeff=10
rg=np.loadtxt(arg4)
rgbeta=np.float(rg)

# #### Read weight image (optional)

#weight_file='1612552p_15C.weight.fits'
#weight_file='v20100210_00246_st_1_bsub_r.weight.fits'
#hdulist = fits.open(weight_file)
#hdulist.info()
#weighthdr=hdulist[0].header
#weightimage=hdulist[0].data
#weightimage[np.where(weightimage != 0)]=1
#dataimage[np.where(weightimage == 0)]=0

BB=hermite.Bfunc(ncoeff,rgbeta)

g,a,b=rgbeta,rgbeta,rgbeta
Cmat=hermite.Cmatrix(ncoeff,g,a,b)

LL=fitstars.chip(dataimage,catstars,BB)  ##  returns list with [scidata,CC,[x,y],weight] for each valid stars at [x,y]

oo=fitstars.fittedcoeff(ncoeff,LL)

LLfit=fitstars.getcoeff(ncoeff,LL,oo)

KK=Gkernel.map(ncoeff,LLfit,Cmat)
LK=Gkernel.getKlist(ncoeff,LL,KK)

ooK=fitstars.fittedcoeff(ncoeff,LK)

Tmat=Gkernel.Tmatrix(ncoeff,ooK,BB)

out=Gkernel.gaussianize(ncoeff,Tmat,dataimage,datahdr)
#out=Gkernel.gaussianize(ncoeff,Tmat,dataimage,datahdr,weight=weightimage)

outfile=arg5
hdu = fits.PrimaryHDU(out[1])
hdu.header=datahdr
hdu.writeto(outfile, clobber=True)

