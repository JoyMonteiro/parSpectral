from pylab import *
import forcing as spf;

ion();

a = spf.specForcing(256,256, 20., 30., 10.);
F0=0;
b = a.forcingFn(F0);
imshow(b);
colorbar();
