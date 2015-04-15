from pylab import *
import diffusion
import forcing
import pSpectral
import RungeKutta
import inversion

ion()

Nx = 256 
Ny = 128

tx = linspace(-pi,pi-2*pi/Nx,Nx) 
ty = linspace(0,pi,Ny)

lengthX = 10*pi
lengthY = 8.

xx = (lengthX)*(tx)/(2*pi)
yy = (lengthY/2.)*cos(ty)
[x,y] = meshgrid(xx,yy)


diff = diffusion.specDiffusion(Nx,Ny, alpha=0, nu=1e-6, order=8., length =
                               min(lengthX,lengthY)); 
p = pSpectral.parSpectral(Nx,Ny, lengthX, lengthY,'Fourier', 'Cosine')


def dfdt(t,f, args=None):

    [u,v,pr] = f
    u1 = -p.partialX(pr) + y*v - epsilon*u
    v1 = -p.ChebMatY(pr) - y*u  - epsilon*v   
    pr1 = (-Q - p.partialX(u) -p.ChebMatY(v))*tau  -epsilon*pr

    return u1,v1,pr1

def diffusion(dt,f):
    
    [u,v,pr] = f
    u1 = diff.diffusionFn(dt,u)
    v1 = diff.diffusionFn(dt,v)
    pr1 = diff.diffusionFn(dt,pr)
      
    return u1, v1, pr1


delta = min(lengthX/Nx, lengthY/Ny)

stepfwd = RungeKutta.RungeKutta4(delta,dfdt, diffusion ,1)
L = 2.0
epsilon = 0.1
tau = 1.
t=0

F = cos(pi/2./L * x)
F[abs(x)>L]=0
Q = F * exp(-10*y**2/4.)


u0 = zeros((Ny,Nx))
v0 = zeros((Ny,Nx))

u = u0
v = v0
pr = -Q

c=1.
dt=0.5*delta/c
dt=0.003
ii = 0

while (t<10000):
    tnew,[unew,vnew,prnew] = stepfwd.integrate(t,[u,v,pr],dt)
    t = tnew
    [u,v,pr] = [unew,vnew,prnew]
    ii = ii+1
    
    if mod(ii,10)==0:
        clf();
        #pcolormesh(x,y,u)
        #pcolormesh(x,y,v)
        pcolormesh(x,y,pr)

        colorbar()
        pause(1e-3)
