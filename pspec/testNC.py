import numpy as np;
import logData;

xdim = 10;
ydim = 10;
filename = 'test.nc';
fields = ['pv', 'vort'];

logger = logData.logData(filename, fields, ['xdim','ydim'], [xdim, ydim]);

pvField = np.zeros((xdim,ydim));
vortField = np.zeros((xdim,ydim));

logger.writeData([pvField, vortField]);
logger.writeData([pvField, vortField]);
logger.writeData([pvField, vortField]);
logger.writeData([pvField, vortField]);
logger.writeData([pvField, vortField]);


logger.finishLogging();
