from pylab import *
import pSpectral as p;

ion()

Nx=512;
Ny=256;

a = p.parSpectral(Nx,Ny,2*pi,'Fourier', 'Cosine');

tx = linspace(0,2*pi,Nx);
ty = linspace(0,pi,Ny);

xx = cos(tx);
yy = cos(ty);

[x,y] = meshgrid(xx,yy);



test = exp(-x*y**10,dtype=float);
actual = -x*10*y**9* test;
#L=2.;
#test = 1 + 0.1*tanh(y/L,dtype=float)
#actual = (0.1/L)*(1.0 - pow(tanh(y/L,dtype=float),2));
 
#test = y**15 + 10*(y**6);
#actual = 15*(y**14) + 60*(y**5);
#~ test = sin(x,dtype=float);
#~ actual = cos(x,dtype=float);

#test = (x**10)*(y**10);
#actual = 10*(y**9)*(x**10);

ans1 = a.partialChebY(test);
#ans 1 uses fftw and ans2 rfft
ans2 = a.ChebY(test);

print 'Usinf FFTW:'
print amax(ans1-actual);
print 'Using rfft:'; 
print amax(ans2-actual);

