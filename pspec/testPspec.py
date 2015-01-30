import pSpectral as p
from pylab import *

ion()

a = p.parSpectral(1024,512)
kx = linspace(0,2*pi-2*pi/1024,1024)
ky = linspace(0,2*pi-2*pi/512,512)
[x,y]=  meshgrid(kx,ky)

b = sin(x)*cos(y)
bpx = a.partialX(b)

d = cos(x)*cos(y)
imshow(bpx)

