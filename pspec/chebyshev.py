import pyfftw;
import pickle;
import multiprocessing;
import sys;
import os.path;
import numpy as np;


class chebTrans(object):

    def __init__(self, numPointsX, numPointsY):
        
        self.xn = numPointsX;
        self.yn = numPointsY;
        self.wisdomExists = False;
        self.isOneDimensional = False;
        self.numCpus = multiprocessing.cpu_count();

        self.inpArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='float64');

        self.outArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='float64');

        self.intArr = pyfftw.n_byte_align_empty(\
                (numPointsY/2 + 1, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');


        fname = str(numPointsX)+'x'+str(numPointsY)+'_cby'+'.wis'

# Check if wisdom files exist for this combination

        if(os.path.isfile(fname)):

            self.wisdomExists = True;
            print 'Wisdom available, loading...';
            
            fp = open(fname);
            wisdom = pickle.load(fp);
            
            pyfftw.import_wisdom(wisdom);

        print 'Shapes: ', self.inpArr.shape, self.intArr.shape;

        print 'Estimating optimal FFT, this may'
        print 'take some time...';
        
        
        self.fwdTrans = pyfftw.FFTW(\
            self.inpArr, self.intArr, axes=[0,],\
            flags=['FFTW_PATIENT',],threads=self.numCpus);


        self.invTrans = pyfftw.FFTW(\
            self.intArr, self.outArr, axes=[0,],\
            direction='FFTW_BACKWARD',
            flags=['FFTW_PATIENT',],threads=self.numCpus);



        print 'Done estimating!';

        if not self.wisdomExists:

            wisdom = pyfftw.export_wisdom();

            fp = open(fname,'w');
            pickle.dump(wisdom, fp);

        self.N = self.yn-1;

        z = np.cos(np.arange(0,self.N +1)*pi/ self.N);
        self.m = np.hstack((arange(0,N),[0.], arange(1-N, 0) ));

    def partial(self, field):

        newField = np.hstack((self.field, self.field[N-1:0:-1]));

        self.fwdTrans(field);
        temp = self.intArr;

        self.invTrans();

        return self.outArr.real.copy();

