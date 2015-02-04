from pylab import *
import forcing as spf;

ion();

a = spf.specForcing(256,256);
F0=0;
b = a.forcingFn(F0);
imshow(b);
