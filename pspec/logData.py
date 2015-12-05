from netCDF4 import Dataset;


class logData(object):

    """
    This class is a wrapper over the netCDF4 library to simplify
    writing fields to a nc file
    """

    def __init__(self, filename, fieldnames, dimNames, dims, cords, time_step, currTime=0, \
            overwrite=False):

        assert len(dimNames) == len(dims), \
            "number of dimensions must match dimension names";

        self.name = filename;
        self.fields = fieldnames;
        self.dims = dims;
        self.currTime = currTime;
        self.time_step = time_step
        self.ii = 0
        self.lats = cords[0]
        self.lons = cords[1]

        self.ncFile = Dataset(filename, 'w', clobber=overwrite)



        # create a time dimension
        if 'time' not in dimNames:
            self.ncFile.createDimension('time', None);

        # Create dimensions
        for i in range(len(dims)):
            self.ncFile.createDimension(dimNames[i], dims[i]);

        # Create variables

        self.ncFile.createVariable('time', 'f4', ('time',))



        for i in range(len(fieldnames)):
            self.ncFile.createVariable(fieldnames[i],'f8', \
                    self.ncFile.dimensions.keys());

        self.ncFile.createVariable('latitude', 'f8', (dimNames[0],))
        self.ncFile.createVariable('longitude', 'f8', (dimNames[1],))

        self.ncFile.variables['latitude'][:] = self.lats
        self.ncFile.variables['longitude'][:] = self.lons

        self.ncFile.description = 'Simulation data';


        print 'Created file ' + filename;





    def writeData(self, fields):

        assert len(fields) == len(self.fields), \
            "all fields must be written at the same time.";


        j = self.ii
        t = self.currTime
        print 'Writing data at time: ', t;

        variable = self.ncFile.variables.keys();
        variable.remove('time')
        variable.remove('latitude')
        variable.remove('longitude')




        self.ncFile.variables['time'][j] = t 
        for i in range(len(variable)):

            temp = self.ncFile.variables[variable[i]];
            temp[j,:] = fields[i];


        self.currTime += self.time_step/(24*3600.)
        self.ii +=1


    def finishLogging(self):


        print 'Finished logging data to ', self.name;

        print 'number of time steps stored: ', \
            len(self.ncFile.variables['time']);

        self.ncFile.close();
