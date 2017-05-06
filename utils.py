import numpy as np
from astropy.stats import sigma_clip

def radial_profile(data, center):
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr

    return radialprofile


def extractcoeff(ncoeff,LL):
    S=[]
    for element in enumerate(LL):
        S.append(element[1][1])

    return S 

def selectstars(dataimage,catstars):
    nstars=len(catstars)
    iselect=np.ones(nstars)
    for i in range(nstars):
        ix,iy=catstars[i,0:2].astype(int)
        scidata = dataimage[iy-8:iy+9,ix-8:ix+9]
#        print('total star flux: ',i,ix,iy,sum(map(sum,scidata)))
        if scidata.shape != (17,17) :
            iselect[i]=0
        if sum(map(sum,scidata)) == 0 :
            iselect[i]=0

    catstars=catstars[np.where(iselect == 1)]
    return catstars
        
def cleanstars(ncoeff,catstars,LL,sigmaval=3):
    nstars=len(LL)
    flag=np.zeros(nstars, dtype=bool)
    for i in range(ncoeff+1):
        for j in range(ncoeff+1):
            val=[]
            for k in range(nstars):
                val.append(LL[k][1][i,j])
            clipid=sigma_clip(val,sigma=sigmaval)
            flag=[flag[l] or clipid.recordmask[l] for l in range(nstars)]

    return flag


