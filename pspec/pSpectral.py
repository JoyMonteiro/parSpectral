import numpy as np;
from numpy import arange, pi, zeros, exp;
from spectralTransform import specTrans;

#from numpy.fft.helper import ifftshift;
#from numpy.fft import rfft, ifft, fft2,ifft2;

class parSpectral(object):

    def __init__(self, numPointsX, numPointsY, length=2*pi,
                xType='Fourier', yType='Fourier'):

        self.xn = numPointsX;
        self.yn = numPointsY;

        self.trans = specTrans(numPointsX, numPointsY, xType, yType);

        #Prepare the wavenumber arrays
        self.kx = 2*pi*1j*(arange(numPointsX/2+1))/length;
        self.ky = 2*pi*1j*(arange(numPointsY/2+1))/length;

        #Prepare the filters

        preserveX = self.xn/3;
        truncateX = self.xn/2 - preserveX;

        filterX = zeros((self.xn/2+1, 1));
        filterX[0:preserveX, 0] = 1.;

        i = arange(preserveX, self.xn/2);
        filterX[i, 0] = exp((preserveX-i)/3.);

        self.filterX = filterX;

 
        preserveY = self.yn/3;
        truncateY = self.yn/2 - preserveY;

        filterY = zeros((1, self.yn/2+1));
        filterY[0, 0:preserveY] = 1.;

        i = arange(preserveY, self.yn/2);
        filterY[0, i] = exp((preserveY-i)/3.);

        self.filterY = filterY;


    def partialX(self,field, order=1):

        self.trans.fwdxTrans(field);
        temp = self.trans.interxArr;


        multiplier = (self.kx)**order;

        temp[:] = multiplier[:, np.newaxis] * temp[:];

        self.trans.invxTrans();

        return self.trans.outArr.real.copy();
        

    def partialY(self,field, order=1):

        self.trans.fwdyTrans(field);
        temp = self.trans.interyArr;


        multiplier = (self.ky)**order;

        temp[:] = multiplier[np.newaxis, :] * temp[:];

        self.trans.invyTrans();

        return self.trans.outArr.real.copy();
       
    def gradient(self, field):

       return( [self.partialX(field), self.partialY(field)]);

    def divergence(self, u, v):

       return(self.partialX(u) + self.partialY(v));

    def curl(self, u, v):

       return(self.partialX(v) - self.partialY(u));

    def laplacian(self, field):

       return(self.partialX(field, 2) + self.partialY(field, 2));




