from numpy import arange, pi, zeros, exp
#from numpy.fft.helper import ifftshift
from scipy import *
from spectralTransform import specTrans2d;

class specForcing(object):
    
    # Forcing is between kmin and kmax
    # correlation function = 0(white noise) and 1(infinitely correlated)
    # default value = 0.5 (Ref. Maltrud (1990)).
    
    def __init__(self, numPointsX, numPointsY, length=2*pi,
                kmin=20.,kmax=30.,magnitude=1e4, correlation=0.5,
                xType='Fourier', yType='Fourier'):

        self.xn = numPointsX;
        self.yn = numPointsY;
        self.xType = xType;
        self.yType = yType;
        self.kmin = kmin;
        self.kmax = kmax;
        self.magnitude = magnitude;
        self.corr = correlation;

        self.trans = specTrans2d(numPointsX, numPointsY, xType, yType);


        #Prepare the wavenumber arrays
        self.kxx = (2*pi/length)*concatenate((arange(0,numPointsX/2),arange(-numPointsX/2,0)));
        self.kyy = (2*pi/length)*concatenate((arange(0,numPointsY/2),arange(-numPointsY/2,0)));
        
# Forcing is defined in wavenumber space and later transformed to real space 
    def forcingFn(self,F0):
        
        [kx,ky]=meshgrid(self.kxx,self.kyy);
        
        # Forcing defined as a shell in wavenumber space
        A = zeros((self.yn,self.xn));
        A[sqrt(kx**2+ky**2) < self.kmax] = 1.0;
        A[sqrt(kx**2+ky**2) < self.kmin] = 0.0;
        
        signal  = self.magnitude * A * exp(rand(self.yn,self.xn)*1j*2*pi);
        
        
        # Markovian forcing
        F = (sqrt(1-self.corr**2))*signal + self.corr*F0
       
        self.trans.invTrans(F);
        return self.trans.outArr.real.copy();
    

