from pylab import *;
import diffusion;
import forcing;
import pSpectral;
import RungeKutta;
import inversion;

ion();

Nx = 256;
Ny = 256;

xx = linspace(-pi,pi-2*pi/Nx,Nx);
yy = linspace(-pi,pi-2*pi/Ny,Ny);
[x,y]=  meshgrid(xx,yy);




diff = diffusion.specDiffusion(Nx,Ny, alpha=2e-3, nu=1e-16);
p = pSpectral.parSpectral(Nx,Ny);
inv = inversion.specInv(Nx,Ny);
fr = forcing.specForcing(Nx,Ny,38., 42.);

F0 = fr.forcingFn(0);

beta = 10.;

def dfdt(t,f, args=None):

    omega = p.laplacian(f);
    rhs = -beta * p.partialX(f) -p.jacobian(f,omega) + fr.forcingFn(omega);
    out = inv.invLaplacian(rhs);
    
    return out;


def diffusion(dt, f):

    omega = p.laplacian(f);
    out = diff.diffusionFn(dt, omega);

    return inv.invLaplacian(out);

delta = 2*pi/max(Nx,Ny);

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diffusion ,1);

t=0;
a = inv.invLaplacian(F0);
f = a;
dt=0.02;
ii = 0;

while (t<1000):
    tnew,fnew = stepfwd.integrate(t,f,dt);
    t = tnew;
    f = fnew;
    ii = ii+1;
    
    if mod(ii,25)==0:
        clf();
        imshow(p.laplacian(f));
        colorbar();
        pause(1e-3);
