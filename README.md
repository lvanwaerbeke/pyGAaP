# pyGAaP
PSF homogeneization and aperture photometry

python implementation of shapelet-based gaussianization described in Appendix A1 of https://arxiv.org/pdf/1507.00738

pyGAaP_v0 is a python notebook which contains lot of hard coded features, e.g. shapelet order, shapelet size, postage stamp size and the PSF variation across the image (polynomial variation of a certain order).

instructions: running the notebook "as is" will read the test image "teststar.fits" and the list of stars "teststar.log" and generate a gaussianized version of the image "teststar_gauss.fits".
