import numpy as np;
from numpy import arange, pi, zeros, exp;
from spectralTransform import specTrans2d;
from pylab import *;

class specInv(object):

    def __init__(self, numPointsX, numPointsY, length=2*pi,
                xType='Fourier', yType='Fourier'):

        self.xn = numPointsX;
        self.yn = numPointsY;
        
        self.trans = specTrans2d(numPointsX, numPointsY, xType, yType);

        #Prepare the wavenumber arrays
        kxx = 2*pi*(concatenate(( arange(0,numPointsX/2),\
                arange(-numPointsX/2,0)))) /length;

        kyy = 2*pi*(concatenate((arange(0,numPointsY/2),\
                arange(-numPointsY/2,0)))) /length;

        [self.kx,self.ky] = meshgrid(kxx,kyy);

        

    def laplacian(self, field):

        self.trans.fwdTrans(field);
        temp = self.trans.intArr;
        

        delsq = -(self.kx**2+self.ky**2);
        #delsq[0,0] = 1;

        temp = temp*delsq;

        # Filter
        temp[sqrt(self.kx**2+self.ky**2) > min(self.xn, self.yn)/3.] = 0;

        self.trans.invTrans();
        return self.trans.outArr.real.copy();
       

    def invLaplacian(self, field):

        self.trans.fwdTrans(field);
        temp = self.trans.intArr;
        

        delsq = -(self.kx**2+self.ky**2);
        delsq[0,0] = 1;

        temp = temp/delsq;
        
        # Filter
        temp[sqrt(self.kx**2+self.ky**2) > min(self.xn, self.yn)/3.] = 0;

        self.trans.invTrans();
        return self.trans.outArr.real.copy();

