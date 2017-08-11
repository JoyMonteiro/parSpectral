import numpy as np
from numpy import arange, pi, zeros, exp, cos, ones
from spectralTransform import specTrans
from numpy import vstack, transpose, newaxis, linspace, sum
from numpy import hstack, dot, eye, tile, diag
#from numpy.fft.helper import ifftshift
from numpy.linalg import matrix_power
from numpy.fft import rfft, irfft, fft2,ifft2

class parSpectral(object):

    def __init__(self, numPointsX, numPointsY, lengthX = 2*pi, lengthY = 2*pi , xType='Fourier', yType='Fourier'):

        self.xn = numPointsX
        self.yn = numPointsY

        numCPU = 4

        self.trans = specTrans(numPointsX, numPointsY, xType, yType, numCPU)

        #Prepare the wavenumber arrays
        self.kx = 2*pi*1j*(arange(numPointsX/2+1))/lengthX
        self.ky = 2*pi*1j*(arange(numPointsY/2+1))/lengthY

        self.cby = cos(linspace(0,pi, numPointsY))
        self.lengthY = lengthY

        #Prepare the filters

        preserveX = self.xn/3
        truncateX = self.xn/2 - preserveX

        filterX = zeros((1, self.xn/2+1))
        filterX[0, 0:preserveX] = 1.

        i = arange(preserveX, self.xn/2)
        filterX[0, i] = exp((preserveX-i)/1.)

        self.filterX = filterX

 
        preserveY = self.yn/3
        truncateY = self.yn/2 - preserveY

        filterY = zeros((self.yn/2+1, 1))
        filterY[0:preserveY, 0] = 1.

        i = arange(preserveY, self.yn/2)
        filterY[i, 0] = exp((preserveY-i)/1.)

        self.filterY = filterY
        
        def chebMatrix(N):
            m = arange(0,N)
            c = (hstack(( [2.], ones(N-2), [2.]))*(-1)**m).reshape(N,1)
            X = tile(cos(pi*m/(N-1)).reshape(N,1),(1,N))
            dX=X-(X.T)
            D=dot(c,1./c.T)/(dX+eye(N))
            D -= diag(sum(D.T,axis=0))
            return D 

        D = chebMatrix(numPointsY)
        D[0,:] = np.zeros(self.yn)
        D[-1,:] = np.zeros(self.yn)
        self.D = D/(self.lengthY/2.)


    def partialX(self,field, order=1):

        self.trans.fwdxTrans(field)
        temp = self.trans.interxArr


        multiplier = (self.kx)**order

        temp[:] = multiplier[np.newaxis, :] * temp[:]
        temp[:] = self.filterX * temp[:]

        self.trans.invxTrans()

        return self.trans.outArr.real.copy()
        

    def partialY(self,field, order=1):

        self.trans.fwdyTrans(field)
        temp = self.trans.interyArr


        multiplier = (self.ky)**order

        temp[:] = multiplier[:, np.newaxis] * temp[:]
        temp[:] = self.filterY* temp[:]

        self.trans.invyTrans()

        return self.trans.outArr.real.copy()

    def partialChebY(self,field):

        N = self.yn - 1
        z = cos(arange(0, N+1)*pi/N)
        mf = arange(0, N+1)
        m = mf[:, newaxis]
        ans = zeros((N+1, self.xn))

        newField = vstack((field, field[N-1:0:-1]))
        self.trans.fwdyTrans(newField)
        temp = self.trans.interCbyArr

        multiplier = 1j*m
        mtemp = multiplier* temp

        self.trans.invyTrans(mtemp)
        out = self.trans.outCbyArr.real.copy()

        ans[1:N,:] = -out[1:N,:]/ (1- self.cby[1:N, np.newaxis]**2)**0.5 

        ans[0,:] = sum(m**2 * temp[mf,:], axis=0)/N + 0.5*N*temp[N,:];

        ans[N,:] = sum( (-1)**(m+1) * m**2 * temp[mf,:], axis=0)/N + 0.5*(-1**(N+1))* N * temp[N];

        return ans;

    def ChebY(self,field):

        N = self.yn - 1;
        z = cos(arange(0, N+1)*pi/N) /self.lengthY;
        mf = arange(0, N+1);
        m = mf[:, newaxis];
        ans = zeros((N+1, self.xn));

        newField = vstack((field, field[N-1:0:-1]));

        temp = rfft(newField, axis=0); 

        multiplier = 1j*m;
        mtemp = multiplier* temp;

        out = irfft(mtemp, axis=0);

        ans[1:N,:] = -out[1:N,:]/ (1- self.cby[1:N, np.newaxis]**2)**0.5; 

        ans[0,:] = sum(m**2 * temp[mf,:], axis=0)/N + 0.5*N*temp[N,:];

        ans[N,:] = sum( (-1)**(m+1) * m**2 * temp[mf,:], axis=0)/N + 0.5*(-1**(N+1))* N * temp[N];

        return ans;

    def ChebMatY(self, field, order=1):

        return (dot(matrix_power(self.D, order), field))

    def gradient(self, field):

       return( [self.partialX(field), self.partialY(field)]);

    def divergence(self, u, v):
        
       return(self.partialX(u) + self.partialY(v));

    def curl(self, u, v):

       return(self.partialX(v) - self.partialY(u));

    def laplacian(self, field):
        
        return(self.partialX(field, 2) + self.partialY(field, 2));

    def jacobian(self, a, b ):

        return(self.partialX(a)*self.partialY(b) - self.partialX(b)*self.partialY(a));
