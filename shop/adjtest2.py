from dolfin import *
from fenics import *
from dolfin_adjoint import *

alpha = Constant (1)
beta = Constant (1)

mesh = RectangleMesh (Point (0, 0), Point (1, 1), 20, 20)
x = SpatialCoordinate (mesh)
V = VectorFunctionSpace (mesh, "CG", 1)

f = project (as_vector ((alpha*x[0], beta*x[1])), V)
psibc = [DirichletBC (V, f, lambda x, onbnd: onbnd)]
psit = TrialFunction (V)
phi = TestFunction (V)
a = - inner (grad (psit), grad (phi)) * dx
L = inner (Constant ((0,0)), phi) * dx
psi = Function (V, name='psi')
solve (a == L, psi, psibc)

vbc = [DirichletBC (V, as_vector ((x[0], x[1])), lambda x, onbnd: onbnd)]

v = Function (V)
vbar = Constant (((1.32, 0), (0, 0.71)))
E = inner (grad (v) * inv (grad (psi)) - vbar, grad (v) * inv (grad (psi)) - vbar) * abs (det (grad (psi))) * dx
F = derivative (E, v)
J = derivative (F, v)
solve (F == 0, v, vbc, J=J)

print (assemble (E))

da, db = compute_gradient (assemble (E), [Control (alpha), Control (beta)])
Ehat = ReducedFunctional (assemble (E), [Control (alpha), Control (beta)])
h = [Constant(0.001),Constant(0.001)]
conv_rateL = taylor_test(Ehat, [alpha,beta], h)
#ra, rb = minimize (Ehat, method = "L-BFGS-B", bounds=[[0.1,0.1],[10,10]], tol=1e-12)
#print (1/float (ra), 1/float (rb)) # should converge to diagonal entries of vbar