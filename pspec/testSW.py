from pylab import *
import diffusion
import forcing
import pSpectral
import RungeKutta
import inversion

ion()

Nx =256 
Ny = 256

xx = linspace(-pi,pi-2*pi/Nx,Nx)
yy = linspace(-pi,pi-2*pi/Ny,Ny)
[x,y]=  meshgrid(xx,yy)


diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=1e-15);
p = pSpectral.parSpectral(Nx,Ny)
inv = inversion.specInv(Nx,Ny)


def dfdt(t,f, arge=None):

    [u,v,n] = f
    u1 = -g*p.partialX(n) -u*p.partialX(u) -v*p.partialY(u) +f0*v
    v1 = -g*p.partialY(n) -u*p.partialX(v) -v*p.partialY(v) -f0*u
    n1 = -(H+n)*( p.partialX(u) + p.partialY(v)) -u*p.partialX(n) 
    -v*p.partialY(n)

    return u1,v1,n1

def diffusion(dt,f):
    
    [u,v,n] = f
    u1 = diff.diffusionFn(dt,u)
    v1 = diff.diffusionFn(dt,v)
    n1 = diff.diffusionFn(dt,n)

    return u1,v1,n1

delta = 2*pi/max(Nx,Ny)

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diffusion ,1)

t=0

u0 = zeros((Ny,Nx))
v0 = zeros((Ny,Nx))
n0 = 0.001*exp(- (x**2/0.1 + y**2/0.1))

# n is the height perturbation
u = u0
v = v0
n = n0
f0 = y

g=1
H=0.01
c = sqrt(g*H)

dt=0.5*delta/c

ii = 0

while (t<100):
    tnew,[unew,vnew,nnew] = stepfwd.integrate(t,[u,v,n],dt)
    t = tnew
    [u,v,n] = [unew,vnew,nnew]
    ii = ii+1
    
    if mod(ii,10)==0:
        clf();
        #pcolormesh(x,y,u)
        #pcolormesh(x,y,v)
        pcolormesh(x,y,n)
        xlim(-pi,pi)
        ylim(-pi,pi)
        colorbar()
        pause(1e-3)

