import numpy as np
from scipy import amax,shape

class AdamBash(object):

    def __init__(self, dfdt, diffusion, ncycle = 0):

        
        
        #self.delta = delta
        self.cflConstant = 0.5
        self.ncycle = ncycle
        self.dfdt = dfdt
        self.diffusion = diffusion
   
        #self.calcMaxVel = maxVel


    def integrate(self, t, fields,dt, args = None):


        F = np.array(fields)
        F1 = F.copy()
        #Fnew = f.copy()
        fnew = np.array(self.dfdt(t,F))

        #forward euler, then 2nd order adams-bashforth time stepping

        if self.ncycle==0:
            self.fnow = fnew.copy()
            self.fold = fnew.copy()

        elif self.ncycle==1:
            self.fold = fnew.copy()

        f1 = F1 + dt*((23/12.)*fnew - (16/12.)*self.fnow + (5/12.)*self.fold)

        fq = self.diffusion(dt,f1)

        tnew = t+dt

        self.fold = self.fnow.copy()
        self.fnow = fnew.copy()
        
        self.ncycle += 1

        #print 'Time:', tnew


        return (tnew,fq)










