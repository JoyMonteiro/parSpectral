import numpy as np;
import logData;

xdim = 10;
ydim = 10;
filename = 'test.nc';
fields = ['pv', 'vort'];

logger = logData.logData(filename, fields, ['xdim','ydim'], [xdim, ydim]);

pvField = np.random.rand(xdim,ydim);
vortField = np.random.rand(xdim,ydim);
logger.writeData([pvField, vortField]);

pvField = np.random.rand(xdim,ydim);
vortField = np.random.rand(xdim,ydim);
logger.writeData([pvField, vortField]);

pvField = np.random.rand(xdim,ydim);
vortField = np.random.rand(xdim,ydim);
logger.writeData([pvField, vortField]);

pvField = np.random.rand(xdim,ydim);
vortField = np.random.rand(xdim,ydim);
logger.writeData([pvField, vortField]);

pvField = np.random.rand(xdim,ydim);
vortField = np.random.rand(xdim,ydim);
logger.writeData([pvField, vortField]);


logger.finishLogging();
