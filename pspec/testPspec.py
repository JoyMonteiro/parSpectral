import pSpectral as p
from pylab import *

a = p.parSpectral(1024,512)
x = linspace(0,2*pi,1024)
y = linspace(0,2*pi,512)
[kx,ky]=  meshgrid(y,x)

b = sin(kx)*sin(ky)
bpx = a.partialX(b)
c = cos(kx)*sin(ky)
imshow(bpx)
show()
