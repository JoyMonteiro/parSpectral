import numpy as np;
from scipy import amax, shape;

class RungeKutta4(object):

    def __init__(self, delta, dfdt, diffusion, maxVel):

        """ Init routine for the integrator.
        
        delta: stores the spatial grid width

        dfdt: function which stores the functions to be used 
               to calculate tendencies

        diffusion: function which is called
                   to do diffusion after the advection step

        maxVel: used to give the integrator
                information about the maximum
                velocity for calculating the time step

        """

        self.delta = delta;
        self.cflConstant = 0.5;

        self.dfdt = dfdt;
        self.diffusion = diffusion;
        self.calcMaxVel = maxVel;

    
    def integrate(self, t, fields, args=None, dt=0):

        """
        Integrator routine

        t: current simulation time

        fields: list of fields to be stepped forward

        args: additional arguments dfdt might require

        dt: force time step to be this instead of that
            calculated by CFL.

        """

        if (dt > 0):
            dT = dt;

        else:

            maxVelocity = self.calcMaxVel(fields, args);
            timeStep = self.delta/maxVelocity;

            dT = self.cflConstant*timeStep;
            dT = min(dT, 0.1);

        print 'Time step: ', dT;

        dt = dT/2.;

        f = np.array(fields);

        k1 = np.array(self.dfdt(t, f, args));
        k1n = (dt/2.)*k1 + f;

        k2 = np.array(self.dfdt(t+(dt/2.), k1n, args));
        k2n = (dt/2.)*k2 + f;

        k3 = np.array(self.dfdt(t+(dt/2.), k2n, args));
        k3n = dt*k3 + f;

        k4 = np.array(self.dfdt(t+dt, k3n, args));

        k = (k1/6. + k2/3. + k3/3. + k4/6.);

        fnew = f + dt*k;

        fnew = self.diffusion(dt, fnew, args);

        tnew = t + dT;

        return (tnew, fnew);
