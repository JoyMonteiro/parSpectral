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
hbar = 10.e3 # resting depth
hamp = 100. # amplitude of height perturbation to zonal jet
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis

# Relaxation time scales

tau = 10*24*3600.
tau_vrt = tau
tau_div = tau
tau_phi = tau

# height perturbation

lats_ref = np.radians(0)
lons_ref = np.radians(180)
delta_lats = np.radians(15)
delta_lons = np.radians(30)

S = hamp*np.exp(-((lats-lats_ref)/delta_lats)**2 - ((lons-lons_ref)/delta_lons)**2 )

# initial vorticity, divergence in spectral space



#vrtg = x.spectogrd(vrtspec)
#tmpg1 = ug*(vrtg+f); tmpg2 = vg*(vrtg+f)
#tmpspec1, tmpspec2 = x.getvrtdivspec(tmpg1,tmpg2)
#tmpspec2 = x.grdtospec(0.5*(ug**2+vg**2))
#phispec = x.invlap*tmpspec1 - tmpspec2
#phig = grav*(hbar + hbump) + x.spectogrd(phispec)
#phispec = x.grdtospec(phig)

phi0 = grav*(S)/tau_phi
phisp = x.grdtospec(phi0)

vrtsp = divsp = (phisp)*0

def dfdt(t,fn,args=None):

    [vrtsp, divsp, phisp] = fn
    
    vrt = x.spectogrd(vrtsp)
    div = x.spectogrd(divsp)
    u,v = x.getuv(vrtsp,divsp)
    phi = x.spectogrd(phisp)
    
    tmp1 = u*(vrt+f)
    tmp2 = v*(vrt+f)

    tmpa, tmpb = x.getvrtdivspec(tmp1, tmp2)
    dvrtsp = -tmpb - x.grdtospec(vrt/(tau_vrt))
#    dvrtsp = -tmpb


    tmpc = x.spectogrd(tmpa)

    tmp3 = u*phi
    tmp4 = v*phi

    tmpd, tmpe = x.getvrtdivspec(tmp3,tmp4)
    dphisp = -tmpe + x.grdtospec(phi0 - phi/(tau_phi))
#    dphisp = -tmpe


    tmpf = x.grdtospec(phi+ 0.5*(u**2+v**2))
    ddivsp = tmpa - x.lap*tmpf - x.grdtospec(div/tau_div)
#    ddivsp = tmpa - x.lap*tmpf


    return [dvrtsp, ddivsp, dphisp] 


def diffusion(dt,F):

    [vrtsp,divsp, phisp] = F
    vrtsp *= hyperdiff_fact
    divsp *= hyperdiff_fact
    return [vrtsp, divsp, phisp]


stepfwd = AdamsBashforth.AdamBash(dfdt,diffusion, ncycle=0)

tmax = 100*86400
t=0
dt = 50
plt.ion()
ii = 0

# create hyperdiffusion factor
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 6. # order for hyperdiffusion
hyperdiff_fact = np.exp((1.*-dt)*(x.lap/x.lap[-1])**(ndiss/2))


while(t< tmax):
    
    t,[vrtsp,divsp,phisp] = stepfwd.integrate(t,[vrtsp,divsp,phisp], dt)
    #print 'Time:', t

   
    
    phi = x.spectogrd(phisp)
    phi_max = np.amax(phi)

    ii = ii+1
    
    if np.mod(ii,1000)==0:

        print 'Time in hrs:', t/3600.
        vrt = x.spectogrd(vrtsp)
        div = x.spectogrd(divsp)
        phi = x.spectogrd(phisp)


        plt.clf()
    #    pv = (0.5*hbar*grav/omega)*(vrt+f)/phi
        plt.contourf(lons,lats,phi,10)
        plt.colorbar()
        plt.pause(1e-3)



vrt = x.spectogrd(vrtsp)
div = x.spectogrd(divsp)
u,v = x.getuv(vrtsp,divsp)
phi = x.spectogrd(phisp)


