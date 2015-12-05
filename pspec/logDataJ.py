from netCDF4 import Dataset;


class logData(object):

    """
    This class is a wrapper over the netCDF4 library to simplify
    writing fields to a nc file
    """

    def __init__(self, filename, fieldnames, dimNames, dt=1, currTime=0, \
            overwrite=False):


        self.name = filename;
        self.fields = fieldnames;
        self.dims = dimNames;
        self.dt = dt;
        self.currTime = currTime;


        self.ncFile = Dataset(filename, 'w', clobber=overwrite);

        # create a time dimension
        if 'time' not in dimNames:
            self.ncFile.createDimension('time', None);

        # Create dimensions
        for dimension in dimNames.keys():
            self.ncFile.createDimension(dimension, dimNames[dimension]);

        # Create variables

        self.ncFile.createVariable('time', 'u8', ('time',));

        for field in fieldnames.keys():
            self.ncFile.createVariable(field,'f8', \
                    fieldnames[field]);


        self.ncFile.description = 'Simulation data';

        print 'Created file ' + filename;
        print self.ncFile


    def writeData(self, fields):

        assert len(fields) == len(self.fields), \
            "all fields must be written at the same time.";


        t = self.currTime;

        print 'Writing data at time: ', t;

        variable = self.ncFile.variables.keys();
        variable.remove('time');

        self.ncFile.variables['time'][t] = self.currTime;
        for i in range(len(variable)):

            temp = self.ncFile.variables[variable[i]];
            temp[t,:] = fields[i];


        self.currTime += self.dt;


    def finishLogging(self):


        print 'Finished logging data to ', self.name;

        print 'number of time steps stored: ', \
            len(self.ncFile.variables['time']);

        self.ncFile.close();
