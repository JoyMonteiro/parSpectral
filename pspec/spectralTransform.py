import pyfftw;
import pickle;
import multiprocessing;
import sys;
import os.path;

class specTrans(object):

    def __init__(self, numPointsX, numPointsY, \
                 xType='Fourier', yType='Fourier', numCPU = None):


        self.xn = numPointsX;
        self.yn = numPointsY;
        self.wisdomExists = False;
        self.isOneDimensional = False;
        if numCPU:
                self.numCpus = numCPU;
        else:
	        self.numCpus = multiprocessing.cpu_count();
        self.xType = xType;
        self.yType = yType;


        self.inpArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='float64');

        self.outArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='float64');


        if (xType is 'Fourier'):

            self.interxArr = pyfftw.n_byte_align_empty(\
                    (numPointsY, numPointsX/2 + 1),\
                    pyfftw.simd_alignment,
                    dtype='complex128');
        else:
            
            print ('Non-Fourier options not implemented\
               along x-axis!');
            sys.exit();
            
        if (yType is 'Fourier'):

            self.interyArr = pyfftw.n_byte_align_empty(\
                    (numPointsY/2 + 1, numPointsX),\
                    pyfftw.simd_alignment,
                    dtype='complex128');
 
            fname = str(numPointsX)+'x'+str(numPointsY)+'.wis'
        
        elif (yType is 'Cosine'):

            self.inpCbyArr = pyfftw.n_byte_align_empty(\
                    (2*numPointsY - 2, numPointsX),\
                    pyfftw.simd_alignment,
                    dtype='float64');

            self.outCbyArr = pyfftw.n_byte_align_empty(\
                    (2*numPointsY - 2, numPointsX),\
                    pyfftw.simd_alignment,
                    dtype='float64');

            self.interCbyArr = pyfftw.n_byte_align_empty(\
                    (numPointsY, numPointsX),\
                    pyfftw.simd_alignment,
                    dtype='complex128');


            fname = str(numPointsX)+'x'+str(numPointsY)+'_cby'+'.wis'

# Check if wisdom files exist for this combination

        if(os.path.isfile(fname)):

            self.wisdomExists = True;
            print ('Wisdom available, loading...');
            
            fp = open(fname);
            wisdom = pickle.load(fp);
            
            pyfftw.import_wisdom(wisdom);

        print ('Shapes: ', self.inpArr.shape, self.interxArr.shape);

        print( 'Estimating optimal FFT, this may')
        print( 'take some time...');



        if (numPointsX == 1): # 1-d transform

            self.isOneDimensional = True;

            self.fwdxTrans = pyfftw.FFTW(\
                self.inpArr, self.interxArr, axes=[1,],\
                flags=['FFTW_PATIENT',],threads=self.numCpus);

            self.invxTrans = pyfftw.FFTW(\
                self.interxArr, self.outArr, axes=[1,],\
                direction='FFTW_BACKWARD',
                flags=['FFTW_PATIENT',],threads=self.numCpus);


        else:


            if (xType is 'Fourier'):

                self.fwdxTrans = pyfftw.FFTW(\
                    self.inpArr, self.interxArr, axes=[1,],\
                    flags=['FFTW_PATIENT',],threads=self.numCpus);

                self.invxTrans = pyfftw.FFTW(\
                    self.interxArr, self.outArr, axes=[1,],\
                    direction='FFTW_BACKWARD',
                    flags=['FFTW_PATIENT',],threads=self.numCpus);
            
            if (yType is 'Fourier'):

                self.fwdyTrans = pyfftw.FFTW(\
                    self.inpArr, self.interyArr, axes=[0,],\
                    flags=['FFTW_PATIENT',],threads=self.numCpus);

                self.invyTrans = pyfftw.FFTW(\
                    self.interyArr, self.outArr, axes=[0,],\
                    direction='FFTW_BACKWARD',
                    flags=['FFTW_PATIENT',],threads=self.numCpus);

            elif (yType is 'Cosine'):

                self.fwdyTrans = pyfftw.FFTW(\
                    self.inpCbyArr, self.interCbyArr, axes=[0,],\
                    flags=['FFTW_PATIENT',],threads=self.numCpus);

                self.invyTrans = pyfftw.FFTW(\
                    self.interCbyArr, self.outCbyArr, axes=[0,],\
                    direction='FFTW_BACKWARD',
                    flags=['FFTW_PATIENT',],threads=self.numCpus);

        print ('Done estimating!');

        if not self.wisdomExists:

            wisdom = pyfftw.export_wisdom();

            fp = open(fname,'w');
            pickle.dump(wisdom, fp);


class specTrans2d(object):

    def __init__(self, numPointsX, numPointsY, \
            xType='Fourier', yType='Fourier', type='1d'):


        self.xn = numPointsX;
        self.yn = numPointsY;
        self.wisdomExists = False;
        self.isOneDimensional = False;
        self.numCpus = multiprocessing.cpu_count();
        self.xType = xType;
        self.yType = yType;

        self.inpArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');

        self.intArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');

        self.outArr = pyfftw.n_byte_align_empty(\
                (numPointsY, numPointsX),\
                pyfftw.simd_alignment,
                dtype='complex128');


        fname = str(numPointsX)+'x'+str(numPointsY)+'_2d'+'.wis'

# Check if wisdom files exist for this combination

        if(os.path.isfile(fname)):

            self.wisdomExists = True;
            print ('Wisdom available, loading...');
            
            fp = open(fname);
            wisdom = pickle.load(fp);
            
            pyfftw.import_wisdom(wisdom);

        print ('Shapes: ', self.inpArr.shape);
        print ('Estimating optimal FFT, this may')
        print ('take some time...');


        self.fwdTrans = pyfftw.FFTW(\
            self.inpArr, self.intArr, axes=[0,1],\
            flags=['FFTW_PATIENT',],threads=self.numCpus);


        self.invTrans = pyfftw.FFTW(\
            self.intArr, self.outArr, axes=[0,1],\
            direction='FFTW_BACKWARD',
            flags=['FFTW_PATIENT',],threads=self.numCpus);



        print ('Done estimating!');

        if not self.wisdomExists:

            wisdom = pyfftw.export_wisdom();

            fp = open(fname,'w');
            pickle.dump(wisdom, fp);


