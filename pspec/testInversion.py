from pylab import *;
import diffusion;
import inversion; 
import pSpectral;
import inversion;

ion();


Nx = 128;
Ny = 256;

xx = linspace(0,2*pi-2*pi/Nx,Nx);
yy = linspace(0,2*pi-2*pi/Ny,Ny);

[x,y]=  meshgrid(xx,yy);
a = sin(x)*sin(y)+cos(x)*sin(y);

p = pSpectral.parSpectral(Nx,Ny);
ip = inversion.specInv(Nx,Ny);

c = p.laplacian(a);
d = ip.invLaplacian(c);
