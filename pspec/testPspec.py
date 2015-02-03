from pylab import *
import pSpectral as p
import forcing as spf


ion()

#~ a = p.parSpectral(1024,512)
#~ kx = linspace(0,2*pi-2*pi/1024,1024)
#~ ky = linspace(0,2*pi-2*pi/512,512)
#~ [x,y]=  meshgrid(kx,ky)
#~ 
#~ b = sin(x)*cos(y)
#~ out = a.partialX(b)	
#~ 
#~ d = cos(x)*cos(y)
#~ imshow(out)

a = spf.specForcing(256,256);
F0=0;
b = a.forcingFn(F0);
imshow(b)
