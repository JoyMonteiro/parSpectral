from pylab import *;
import RungeKutta;

def dummy(t, f, args):

    return zeros(f.shape);

def dummyVel(f, args):

    return 1.3;

u = zeros((10,10));
v = zeros((10,10));
z = zeros((10,10));

delta = 0.1;


stepfwd = RungeKutta.RungeKutta4(delta, dummy, dummy, dummyVel);

tnew, fnew = stepfwd.integrate(0, [u,v,z],0.1);

print tnew;
