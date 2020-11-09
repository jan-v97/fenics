from dolfin import *
from fenics import *
from dolfin_adjoint import *

mesh = RectangleMesh (Point (0, 0), Point (1, 1), 10, 10)
bmesh = BoundaryMesh (mesh, "exterior")
x = SpatialCoordinate (mesh)
bx = SpatialCoordinate (bmesh)
V = VectorFunctionSpace (mesh, "CG", 1)
bV = VectorFunctionSpace (bmesh, "CG", 1)

alpha = Constant (1)
beta = Constant (1)

bf = project (as_vector ((alpha*bx[0], beta*bx[1])), bV)
f = transfer_from_boundary (bf, mesh) # trivial transfer: will be set to zero away from the boundary

psibc = [DirichletBC (V, Constant ((0, 0)), lambda x, onbnd: onbnd)]

psit = TrialFunction (V)
phi = TestFunction (V)
a = - inner (grad (psit), grad (phi)) * dx
L = inner (grad (f), grad (phi)) * dx
psi = Function (V, name='psi')

solve (a == L, psi, psibc)

psi = project (f + psi, V)

vbc = [DirichletBC (V, as_vector ((x[0], x[1])), lambda x, onbnd: onbnd)]

v = Function (V)
vbar = Constant (((1.32, 0), (0, 0.71)))
E = inner (grad (v) * inv (grad (psi)) - vbar, grad (v) * inv (grad (psi)) - vbar) * abs (det (grad (psi))) * dx
F = derivative (E, v)
J = derivative (F, v)
solve (F == 0, v, vbc, J=J)

print (assemble (E))

# da, db = compute_gradient (assemble (E), [Control (alpha), Control (beta)])
Ehat = ReducedFunctional (assemble (E), [Control (alpha), Control (beta)])
ra, rb = minimize (Ehat, method = "L-BFGS-B", bounds=[[0.1,0.1],[10,10]])
print (1/float (ra), 1/float (rb)) # sghould converge to diagonal entries of vbar