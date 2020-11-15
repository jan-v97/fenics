from dolfin import *
from fenics import *
from dolfin_adjoint import *
tol = 1e-14

alpha = Constant (1)
beta = Constant (1)

mesh = RectangleMesh (Point (0, 0), Point (1, 1), 20, 20)
x = SpatialCoordinate (mesh)
V = VectorFunctionSpace (mesh, "CG", 1)

f = project (as_vector ((alpha*x[0], beta*x[1])), V)

#     der Taylor Test funktioniert nicht, wenn die abhängigkeit der Funktion über mehr als eine Randbedingung gegeben ist             
psibc = [DirichletBC (V, f, lambda x, onbnd: (onbnd and x[0]>0.5-tol)),DirichletBC (V, f, lambda x, onbnd: (onbnd and x[0]<0.5+tol))]
#     die folgenden BC sind äquivalent, liefern aber das richtige Ergebnis beim Taylor Test       
#psibc = [DirichletBC (V, f, lambda x, onbnd: onbnd )]
psit = TrialFunction (V)
phi = TestFunction (V)
a = - inner (grad (psit), grad (phi)) * dx
L = inner (Constant ((0,0)), phi) * dx
psi = Function (V, name='psi')
solve (a == L, psi, psibc)

vbc = [DirichletBC (V, as_vector ((x[0], x[1])), lambda x, onbnd: onbnd)]

v = Function (V, name='v')
vbar = Constant (((1.32, 0), (0, 0.71)))


#    hier eine nichtlineare elastische Energie, die das gleiche Problem liefert:      

#a1=11.56; a2=-17.44; a3=10.04; a4=-9.38
#def energy_density (v, psi, a1, a2, a3, a4):
#	F = ( Identity(2) + grad(v)* inv(grad(psi)) -vbar)
#	C = F.T*F
#	return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(grad(psi)))
#Edens = energy_density (v, psi, a1, a2, a3, a4)
#E = Edens*dx


# Total potential energy and derivatives
E = inner (grad (v) * inv (grad (psi)) - vbar, grad (v) * inv (grad (psi)) - vbar) * abs (det (grad (psi))) * dx
F = derivative (E, v)
J = derivative (F, v)
solve (F == 0, v, vbc, J=J)

print (assemble (E))

#file = XDMFFile ("adjtest2/file.xdmf")
#file.parameters["functions_share_mesh"] = True
#file.write(psi, 0)
#file.write(v, 0)

#da, db = compute_gradient (assemble (E), [Control (alpha), Control (beta)])
Ehat = ReducedFunctional (assemble (E), [Control (alpha), Control (beta)])
h = [Constant(0.001),Constant(0.001)]
conv_rateL = taylor_test(Ehat, [alpha,beta], h)
#ra, rb = minimize (Ehat, method = "L-BFGS-B", bounds=[[0.1,0.1],[10,10]], tol=1e-12)
#print (1/float (ra), 1/float (rb)) # should converge to diagonal entries of vbar