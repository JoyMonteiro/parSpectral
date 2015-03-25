from pylab import *;
import diffusion;
import forcing;
import pSpectral;
import RungeKutta;
import inversion;

ion();

Nx = 256;
Ny = 256;

xx = linspace(0,2*pi-2*pi/Nx,Nx);
yy = linspace(0,2*pi-2*pi/Ny,Ny);
[x,y]=  meshgrid(xx,yy);

a = sin(x)+ cos(y)+ sin(2*x)*cos(2*y);


diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=4.5e-16);
p = pSpectral.parSpectral(Nx,Ny);
inv = inversion.specInv(Nx,Ny);

def dfdt(t,f, args=None):

    omega = p.laplacian(f);
    rhs = -p.jacobian(f,omega) ;
    out = inv.invLaplacian(rhs);
    
    return out;

delta = 2*pi/max(Nx,Ny);

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diff.diffusionFn ,1);

t=0;
f=a;
dt=0.1;

while (t<50):
	tnew,fnew = stepfwd.integrate(t,f,dt);
	t = tnew;
	f = fnew;
	imshow(p.laplacian(fnew));
	colorbar();
	pause(1e-3);
	clf();
