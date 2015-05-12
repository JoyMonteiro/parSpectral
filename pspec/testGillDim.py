from pylab import *
import diffusion
import forcing
import pSpectral
import RungeKutta
import inversion

ion()

Nx = 256 
Ny = 128


lengthX = 10*pi
lengthY = 4.

tx = linspace(-pi,pi-2*pi/Nx,Nx) 
ty = linspace(0,pi,Ny)
xx = (lengthX)*(tx)/(2*pi)
yy = (lengthY/2.)*cos(ty)
[x,y] = meshgrid(xx,yy)


diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=1e-10);
p = pSpectral.parSpectral(Nx,Ny, lengthX, lengthY, 'Fourier', 'Cosine')



def dfdt(t,f, args=None):

    [u,v,n] = f
    u1 = -g*p.partialX(n) +f0*v -epsilon*u
    v1 = -g*p.ChebMatY(n) -f0*u -epsilon*v
    n1 = -n0 -(H+n)*( p.partialX(u) + p.ChebMatY(v)) - epsilon*n 
    return u1,v1,n1

def diffusion(dt,f):
    
    [u,v,n] = f
    u1 = diff.diffusionFn(dt,u)
    v1 = diff.diffusionFn(dt,v)
    n1 = diff.diffusionFn(dt,n)

    return u1,v1,n1

delta = min(2*pi/Nx, 1./Ny)

epsilon = 0.005

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diffusion ,1)

t=0

u0 = zeros((Ny,Nx))
v0 = zeros((Ny,Nx))
#n0 = 0.001*exp(- (x**2/0.1 + y**2/0.1))

L=2.
F = cos(pi/2./L * x)
F[abs(x)>L]=0
n0 =0.001* F * exp(-10*y**2/4.)
# n is the height perturbation
u = u0
v = v0
n = n0
f0 = y

g=1
H=0.01
c = sqrt(g*H)

dt=0.5*delta/c


ii = 0

while (t<1000):
    tnew,[unew,vnew,nnew] = stepfwd.integrate(t,[u,v,n],dt)
    t = tnew
    [u,v,n] = [unew,vnew,nnew]
    ii = ii+1
    
    if mod(ii,10)==0:
        clf();
        #pcolormesh(x,y,u)
        #pcolormesh(x,y,v)
        contourf(x,y,n)
        #contour(x,y,n,10,colors='k')
        #xlim(-pi,pi)
        #ylim(-pi,pi)
        colorbar()
        pause(1e-3)

