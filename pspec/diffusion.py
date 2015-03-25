from numpy import arange, pi, zeros, exp
from scipy import *
from spectralTransform import specTrans2d;

class specDiffusion(object):
    
    def __init__(self, numPointsX, numPointsY,alpha, nu, 
	    order = 8., length = 2*pi, xType='Fourier', yType='Fourier'):
        # nu is the higher "order" dissipation coefficient
        # alpha is the linear drag

        self.xn = numPointsX;
	self.yn = numPointsY;
	self.xType = xType;
	self.yType = yType;
	self.alpha = alpha;
	self.nu = nu;
	self.order = order;

	self.trans = specTrans2d(numPointsX, numPointsY, xType, yType);

        #Prepare the wavenumber arrays
        self.kxx = (2*pi/length)*concatenate((arange(0,numPointsX/2),arange(-numPointsX/2,0)));
        self.kyy = (2*pi/length)*concatenate((arange(0,numPointsY/2),arange(-numPointsY/2,0)));

    def diffusionFn(self, dt, field):


        [kx,ky] = meshgrid(self.kxx,self.kyy);
	self.trans.fwdTrans(field);
	temp = self.trans.outArr;
	
	temp = temp * exp(-(self.nu*(kx**self.order+ky**self.order) + self.alpha) * dt)

	self.trans.invTrans(temp);
	return self.trans.outArr.real.copy();
        
