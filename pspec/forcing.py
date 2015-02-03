import pyfftw
import os.path
import pickle
import multiprocessing
from numpy import arange, pi, zeros, exp
from numpy.fft.helper import ifftshift
from scipy import *

class specForcing(object):
    
    def __init__(self, numPointsX, numPointsY, length=2*pi,
                kmin=20.,kmax=30.,magnitude=1e4, correlation=0.5,
                xType='Fourier', yType='Fourier'):

        self.inpArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');

        self.outArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');
                
        self.prevForcing = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');

        self.xn = numPointsX;
        self.yn = numPointsY;
        self.wisdomExists = False;
        self.isOneDimensional = False;
        self.numCpus = multiprocessing.cpu_count();
        self.xType = xType;
        self.yType = yType;
        self.kmin = kmin;
        self.kmax = kmax;
        self.magnitude = magnitude;
        self.corr = correlation;


        fname = str(numPointsX)+'x'+str(numPointsY)+'.wis'

# Check if wisdom files exist for this combination

        if(os.path.isfile(fname)):

            self.wisdomExists = True;
            print 'Wisdom available, loading...';
            
            fp = open(fname);
            wisdom = pickle.load(fp);
            
            pyfftw.import_wisdom(wisdom);

        print 'Shapes: ', self.inpArr.shape;

        print 'Estimating optimal FFT, this may'
        print 'take some time...';





        self.invTrans = pyfftw.FFTW(\
            self.inpArr, self.outArr, axes=[0,1],\
            direction='FFTW_BACKWARD',
            flags=['FFTW_PATIENT',],threads=self.numCpus);


        print 'Done estimating!';

        if not self.wisdomExists:

            wisdom = pyfftw.export_wisdom();

            fp = open(fname,'w');
            pickle.dump(wisdom, fp);


        #Prepare the wavenumber arrays
        self.kx = (2*pi/length)*concatenate((arange(0,numPointsX/2),arange(-numPointsX/2,0)));
        self.ky = (2*pi/length)*concatenate((arange(0,numPointsY/2),arange(-numPointsY/2,0)));
        
        
    def forcingFn(self,F0):
        
        [kxx,kyy]=meshgrid(self.kx,self.ky);
        
        A = zeros((self.yn,self.xn));
        A[sqrt(kxx**2+kyy**2) < self.kmax] = 1.0;
        A[sqrt(kxx**2+kyy**2) < self.kmin] = 0.0;
        
        signal  = self.magnitude * A * exp(rand(self.yn,self.xn)*1j*2*pi);
        
        F = (sqrt(1-self.corr**2))*signal + self.corr*F0
       
        self.invTrans(F);
        return self.outArr.real.copy();
    

