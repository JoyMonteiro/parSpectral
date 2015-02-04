from pylab import *
import pSpectral as p;


ion();

a = p.parSpectral(1024,512);
ky = linspace(0,2*pi-2*pi/1024,1024);
kx = linspace(0,2*pi-2*pi/512,512);
[y,x]=  meshgrid(kx,ky);

b = sin(x)*cos(y);
bx = cos(x)*cos(y);
bxx = -b;
by = -sin(x)*sin(y);

outx = a.partialX(b);	
outxx = a.partialX(b,2);	
outy = a.partialY(b);

imshow(b.transpose());
figure();
imshow((outx-bx).transpose());
colorbar();
figure();
imshow((outy-by).transpose());
colorbar();
figure();
imshow((outxx-bxx).transpose());
colorbar();

