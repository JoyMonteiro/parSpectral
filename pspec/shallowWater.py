import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import AdamsBashforth


nlons = 256  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats = int(nlons/2)   # for gaussian grid.
dt = 150 # time step in seconds
itmax = 6*int(86400/dt) # integration length in days
  
# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity
hbar = 10.e3 # resting depth
umax = 80. # jet speed
phi0 = np.pi/7.; phi1 = 0.5*np.pi - phi0; phi2 = 0.25*np.pi
en = np.exp(-4.0/(phi1-phi0)**2)
alpha = 1./3.; beta = 1./15.
hamp = 120. # amplitude of height perturbation to zonal jet
efold = 3.*3600. # efolding timescale at ntrunc for hyperdiffusion
ndiss = 8 # order for hyperdiffusion

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis



# zonal jet.
vg = np.zeros((nlats,nlons),np.float)
u1 = (umax/en)*np.exp(1./((x.lats-phi0)*(x.lats-phi1)))
ug = np.zeros((nlats),np.float)
ug = np.where(np.logical_and(x.lats < phi1, x.lats > phi0), u1, ug)
ug.shape = (nlats,1)
ug = ug*np.ones((nlats,nlons),dtype=np.float) # broadcast to shape (nlats,nlonss)

# height perturbation.
hbump = hamp*np.cos(lats)*np.exp(-(lons/alpha)**2)*np.exp(-(phi2-lats)**2/beta)

# initial vorticity, divergence in spectral space
vrtspec, divspec =  x.getvrtdivspec(ug,vg)
vrtg = x.spectogrd(vrtspec)
divg = x.spectogrd(divspec)

# create hyperdiffusion factor
hyperdiff_fact = np.exp((-dt/efold)*(x.lap/x.lap[-1])**(ndiss/2))

# solve nonlinear balance eqn to get initial zonal geopotential,
# add localized bump (not balanced).
vrtg = x.spectogrd(vrtspec)
tmpg1 = ug*(vrtg+f); tmpg2 = vg*(vrtg+f)
tmpspec1, tmpspec2 = x.getvrtdivspec(tmpg1,tmpg2)
tmpspec2 = x.grdtospec(0.5*(ug**2+vg**2))
phispec = x.invlap*tmpspec1 - tmpspec2
phig = grav*(hbar + hbump) + x.spectogrd(phispec)
phispec = x.grdtospec(phig)

vrtsp = vrtspec
divsp = divspec
phisp = phispec
u = ug
v = vg



def dfdt(t,fn,args=None):

    [vrtsp, divsp, phisp] = fn
    
    vrt = x.spectogrd(vrtsp)
    u,v = x.getuv(vrtsp,divsp)
    phi = x.spectogrd(phisp)
    
    tmp1 = u*(vrt+f)
    tmp2 = v*(vrt+f)

    tmpa, tmpb = x.getvrtdivspec(tmp1, tmp2)
    dvrtsp = -tmpb

    tmpc = x.spectogrd(tmpa)

    tmp3 = u*phi
    tmp4 = v*phi

    tmpd, tmpe = x.getvrtdivspec(tmp3,tmp4)
    dphisp = -tmpe

    tmpf = x.grdtospec(phi+ 0.5*(u**2+v**2))
    ddivsp = tmpa - x.lap*tmpf

    return [dvrtsp, ddivsp, dphisp] 


def diffusion(dt,F):

    [vrtsp,divsp, phisp] = F
    vrtsp *= hyperdiff_fact
    divsp *= hyperdiff_fact
    return [vrtsp, divsp, phisp]


stepfwd = AdamsBashforth.AdamBash(dfdt,diffusion, ncycle=0)

tmax = 6*86400
t=0
dt = 150
plt.ion()

while(t< tmax):

    t,[vrtsp,divsp,phisp] = stepfwd.integrate(t,[vrtsp,divsp,phisp], dt)
    print 'Time:', t
    
vrt = x.spectogrd(vrtsp)
div = x.spectogrd(divsp)
phi = x.spectogrd(phisp)

plt.clf()
pv = (0.5*hbar*grav/omega)*(vrt+f)/phi
plt.imshow(pv)
plt.pause(1e-3)

