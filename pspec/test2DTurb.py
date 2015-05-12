from pylab import *;
import diffusion;
import forcing;
import pSpectral;
import RungeKutta;
import inversion;

ion();

Nx = 256*4;
Ny = 256*4;

xx = linspace(0,2*pi-2*pi/Nx,Nx);
yy = linspace(0,2*pi-2*pi/Ny,Ny);
[x,y]=  meshgrid(xx,yy);

a = sin(x)+ cos(y)+ sin(2*x)* cos(2*y);
#a = sin(50*x)+cos(40*y)*sin(50*x);

diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=1e-15, order =  8.);
p = pSpectral.parSpectral(Nx,Ny);
inv = inversion.specInv(Nx,Ny);

def dfdt(t,f, args=None):


    omega = p.laplacian(f);
    rhs = -p.jacobian(f,omega) ;
    out = inv.invLaplacian(rhs);
    return out;

def diffusion(dt, f):

    #omega = p.laplacian(f);
    out = diff.diffusionFn(dt, f);
    #print amax(abs(omega - out));

    #return inv.invLaplacian(out);
    return (out);

delta = 2*pi/max(Nx,Ny);

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diffusion ,1);

t=0;
f=a;
dt=0.02;

while (t<50):
	tnew,fnew = stepfwd.integrate(t,f,dt);
	t = tnew;
	f = fnew;
	imshow(p.laplacian(fnew));
	colorbar();
	pause(1e-3);
	clf();
