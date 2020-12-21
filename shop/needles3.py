from dolfin import *
from mshr import *
from math import sqrt
tol= 1E-14

L1 = sqrt(2)
L1 += 1e-9 #if L1 is bigger than sqrt(2) the periodicBoundary stops working
resolution = 2**5

# define the complete domain
domain = Polygon([dolfin.Point(0., L1),dolfin.Point(0., 0.),dolfin.Point(L1, 0.),dolfin.Point(L1+1., 1.),dolfin.Point(1., 1.),dolfin.Point(1., L1+1.)])
mesh = generate_mesh (domain, resolution)

class PeriodicBoundary (SubDomain):
	# bottom_left boundary is target domain
	def inside (self, x, on_boundary): 
		return (x[0]<tol)or(x[1]<tol)
	# Map top_right boundary to bottom_left boundary
	def map (self, x, y):
		y[0] = x[0]-1.
		y[1] = x[1]-1.

V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
u = Function (V, name='u')
ut = TrialFunction (V)
v = TestFunction(V)
E = inner((Identity(2)+grad(ut)),(Identity(2)+grad(v)))*dx
dbc = [DirichletBC (V, Constant ((0,0)), lambda x,on_boundary: (x[0]<tol)or(x[1]<tol))]
#setting the dirichlet conditions by hand works
#dbc.append(DirichletBC (V, Constant ((0,0)), lambda x,on_boundary: (x[0]>1-tol)and(x[1]>1-tol)))
solve (lhs(E)==rhs(E), u, dbc)

file = XDMFFile ("needles/file.xdmf")
file.write(u, 0)
file.close()
