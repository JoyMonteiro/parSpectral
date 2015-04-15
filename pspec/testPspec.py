from pylab import *
import pSpectral as p;


ion();

Nx = 128;
Ny = 256;

lengthX = 400*pi
lengthY = 2*pi

a = p.parSpectral(Nx,Ny, lengthX, lengthY);
kx = (lengthX/(2*pi))*linspace(-pi,pi-2*pi/Nx,Nx);
ky = (lengthY/(2*pi))*linspace( -pi,pi-2*pi/Ny,Ny);


[x,y]=  meshgrid(kx,ky);

b = sin(x/10.)*cos(y);
bx = 0.1*cos(x/10.)*cos(y);
bxx = -b;
by = -sin(x)*sin(y);

outx = a.partialX(b);	
outxx = a.partialX(b,2);	
outy = a.partialY(b);

print amax(bx-outx)
