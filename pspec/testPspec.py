from pylab import *
import pSpectral as p;


ion();

Nx = 128;
Ny = 256;

a = p.parSpectral(Nx,Ny);
kx = linspace(0,2*pi-2*pi/Nx,Nx);
ky = linspace(0,2*pi-2*pi/Ny,Ny);
[x,y]=  meshgrid(kx,ky);

b = sin(x)*cos(y);
bx = cos(x)*cos(y);
bxx = -b;
by = -sin(x)*sin(y);

outx = a.partialX(b);	
outxx = a.partialX(b,2);	
outy = a.partialY(b);
