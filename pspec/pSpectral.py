import pyfftw;
import os.path;
import pickle;
import multiprocessing;
import numpy as np;
from numpy import arange, pi, zeros, exp;
from numpy.fft.helper import ifftshift;
from numpy.fft import rfft, ifft, fft2,ifft2;

class parSpectral(object):

    def __init__(self, numPointsX, numPointsY, length=2*pi,
                xType='Fourier', yType='Fourier'):

        self.inpArr = pyfftw.n_byte_align_empty(\
                (numPointsX, numPointsY),\
                pyfftw.simd_alignment,
                dtype='float64');

        self.interxArr = pyfftw.n_byte_align_empty(\
                (numPointsX/2 + 1,numPointsY),\
                pyfftw.simd_alignment,
                dtype='complex128');

        self.interyArr = pyfftw.n_byte_align_empty(\
                (numPointsX, numPointsY/2 + 1),\
                pyfftw.simd_alignment,
                dtype='complex128');


        self.outArr = pyfftw.n_byte_align_empty(\
                (numPointsX, numPointsY),\
                pyfftw.simd_alignment,
                dtype='float64');

        self.xn = numPointsX;
        self.yn = numPointsY;
        self.wisdomExists = False;
        self.isOneDimensional = False;
        self.numCpus = multiprocessing.cpu_count();
        self.xType = xType;
        self.yType = yType;


        fname = str(numPointsX)+'x'+str(numPointsY)+'.wis'

# Check if wisdom files exist for this combination

        if(os.path.isfile(fname)):

            self.wisdomExists = True;
            print 'Wisdom available, loading...';
            
            fp = open(fname);
            wisdom = pickle.load(fp);
            
            pyfftw.import_wisdom(wisdom);

        print 'Shapes: ', self.inpArr.shape, self.interxArr.shape;

        print 'Estimating optimal FFT, this may'
        print 'take some time...';



        if (numPointsX == 1): # 1-d transform

            self.isOneDimensional = True;

            self.fwdxTrans = pyfftw.FFTW(\
                self.inpArr, self.interxArr, axes=[0,],\
                flags=['FFTW_PATIENT',],threads=self.numCpus);

            self.invxTrans = pyfftw.FFTW(\
                self.interxArr, self.outArr, axes=[0,],\
                direction='FFTW_BACKWARD',
                flags=['FFTW_PATIENT',],threads=self.numCpus);


        else:


            self.fwdxTrans = pyfftw.FFTW(\
                self.inpArr, self.interxArr, axes=[0,],\
                flags=['FFTW_PATIENT',],threads=self.numCpus);

            self.invxTrans = pyfftw.FFTW(\
                self.interxArr, self.outArr, axes=[0,],\
                direction='FFTW_BACKWARD',
                flags=['FFTW_PATIENT',],threads=self.numCpus);


            self.fwdyTrans = pyfftw.FFTW(\
                self.inpArr, self.interyArr, axes=[1,],\
                flags=['FFTW_PATIENT',],threads=self.numCpus);

            self.invyTrans = pyfftw.FFTW(\
                self.interyArr, self.outArr, axes=[1,],\
                direction='FFTW_BACKWARD',
                flags=['FFTW_PATIENT',],threads=self.numCpus);


        print 'Done estimating!';

        if not self.wisdomExists:

            wisdom = pyfftw.export_wisdom();

            fp = open(fname,'w');
            pickle.dump(wisdom, fp);


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

        self.fwdxTrans(field);
        temp = self.interxArr;


        multiplier = (self.kx)**order;

        temp[:] = multiplier[:, np.newaxis] * temp[:];

        self.invxTrans();

        return self.outArr.real.copy();
        

    def partialY(self,field, order=1):

        self.fwdyTrans(field);
        temp = self.interyArr;


        multiplier = (self.ky)**order;

        temp[:] = multiplier[np.newaxis, :] * temp[:];

        self.invyTrans();

        return self.outArr.real.copy();
       
    def gradient(self, field):

       return( [self.partialX(field), self.partialY(field)]);

    def divergence(self, u, v):

       return(self.partialX(u) + self.partialY(v));

    def curl(self, u, v):

       return(self.partialX(v) - self.partialY(u));

    def laplacian(self, field):

       return(self.partialX(field, 2) + self.partialY(field, 2));




