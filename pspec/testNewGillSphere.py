import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import AdamsBashforth


nlons = 256  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats = int(nlons/2)   # for gaussian grid.
 
# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity
hbar = 100. # resting depth
hamp = 10. # amplitude of height perturbation to zonal jet
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis

# Relaxation time scales

tau = 3.*24*3600.


tau_vrt = tau
tau_div = tau
tau_ht = tau

# height perturbation

lats_ref = np.radians(0)
lons_ref = np.radians(180)
delta_lats = np.radians(15)
delta_lons = np.radians(30)

S = hamp*np.exp(-((lats-lats_ref)/delta_lats)**2 - ((lons-lons_ref)/delta_lons)**2 )
#S = hamp* np.cos(lats)**4
#S = hamp*(np.pi/2-abs(lats))
ht0 = S
htsp = x.grdtospec(ht0)

vrtsp = divsp = (htsp)*0

def dfdt(t,fn,args=None):

    [vrtsp, divsp, htsp] = fn
    
    vrt = x.spectogrd(vrtsp)
    div = x.spectogrd(divsp)
    u,v = x.getuv(vrtsp,divsp)
    ht = x.spectogrd(htsp)
    
    tmp1 = u*(vrt+f)
    tmp2 = v*(vrt+f)

    tmpa, tmpb = x.getvrtdivspec(tmp1, tmp2)
    dvrtsp = -tmpb - x.grdtospec(vrt/(tau_vrt))

    tmpc = x.spectogrd(tmpa)

    tmp3 = u*(ht+hbar)
    tmp4 = v*(ht+hbar)

    tmpd, tmpe = x.getvrtdivspec(tmp3,tmp4)
    dhtsp = -tmpe + x.grdtospec(ht0/tau_ht - ht/(tau_ht))

    tmpf = x.grdtospec(grav*(ht+hbar)+ 0.5*(u**2+v**2))
    ddivsp = tmpa - x.lap*tmpf - x.grdtospec(div/tau_div)

    return [dvrtsp, ddivsp, dhtsp] 


def diffusion(dt,F):

    [vrtsp,divsp, htsp] = F
    vrtsp *= hyperdiff_fact
    divsp *= hyperdiff_fact
    htsp *= hyperdiff_fact
    return [vrtsp, divsp, htsp]


stepfwd = AdamsBashforth.AdamBash(dfdt,diffusion, ncycle=0)

tmax = 100*86400
t=0
dt = 50.
plt.ion()
ii = -1

# create hyperdiffusion factor
ndiss = 8. # order for hyperdiffusion
hyperdiff_fact = np.exp((5e-3*-dt)*(x.lap/x.lap[-1])**(ndiss/2))


while(t< tmax):
    
    t,[vrtsp,divsp,htsp] = stepfwd.integrate(t,[vrtsp,divsp,htsp], dt)
#    print 'Time in hrs:', t/3600.

   
    
    ht = x.spectogrd(htsp)

    ii = ii+1
    
    if np.mod(ii,1000)==0:

        print 'Time in hrs:', t/3600.
        vrt = x.spectogrd(vrtsp)
        div = x.spectogrd(divsp)
        ht  = x.spectogrd(htsp)
#        u,v = x.getuv(vrtsp,divsp)


        plt.clf()
    #    pv = (0.5*hbar*grav/omega)*(vrt+f)/phi
        plt.contourf(lons,lats,ht+hbar,10, extend='both')
#        plt.pcolormesh(lons,lats,phi)
        plt.xlim(0,2*np.pi)
        plt.ylim(-np.pi/2.,np.pi/2.)
        plt.colorbar(orientation='horizontal',extend="both")
        plt.pause(1e-3)



vrt = x.spectogrd(vrtsp)
div = x.spectogrd(divsp)
u,v = x.getuv(vrtsp,divsp)
ht = x.spectogrd(htsp)


