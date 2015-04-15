from pylab import *
import pSpectral;

ion()

Nx = 256 
Ny = 256


lengthX = 2*pi
lengthY = 60

tx = linspace(-pi,pi-2*pi/Nx,Nx) 
ty = linspace(0,pi,Ny)

xx = (lengthX/ (2*pi))*(tx)
yy = (lengthY/2.)*cos(ty)
[x,y] = meshgrid(xx,yy)

a = pSpectral.parSpectral(Nx,Ny, lengthX, lengthY, 'Fourier', 'Cosine')
#test = exp(-x*y**10,dtype=float);
#actual = -x*10*y**9* test;
#L=2.;
#test = 1 + 0.1*tanh(y/L,dtype=float)
#actual = (0.1/L)*(1.0 - pow(tanh(y/L,dtype=float),2));
 
#test = y**15 + 10*(y**6);
#actual = 15*(y**14) + 60*(y**5);
#test = sin(y)
#actual = cos(y);

#test = (x**1)*(y**10);
#actual = 10*(y**9)*(x**1);

#ans1 = a.partialChebY(test);

#ans 1 uses fftw and ans2 rfft
#ans2 = a.ChebY(test);

#print 'Usinf FFTW:'
#print amax(ans1-actual);
#print 'Using rfft:'; 
#print amax(ans2-actual);


b = ((0.01*y)**4)*sin(4*x)
#bx = 4*cos(4*x)*(y**4)
by = 4*0.01*((0.01*y)**3)*sin(4*x)
#byy = 12*(y**2)*sin(4*x)

outx = a.partialX(b)
outy = a.ChebMatY(b)
outyy = a.ChebMatY(b,2)

print amax(by-outy)
