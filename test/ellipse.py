from fenics import *
#import dolfin
from mshr import *
from ufl import nabla_div
import numpy as np
import dolfin.cpp.mesh

tol= 1E-14
r = 0.5

# defining the geometry of the computational domain
domain =   Rectangle(dolfin.Point(-1., -1.), dolfin.Point(1., 1.))
domain2 =   Circle(dolfin.Point(0.,0.),r)
domainouter =   Circle(dolfin.Point(0.,0.),1)
domain0 = domain -domainouter
domain1 = domainouter - domain2

# set subdomains to ensure that the verticies coincide with the boundaries of the subdomains
domain.set_subdomain(1, domain0)
domain.set_subdomain(2, domain1)
domain.set_subdomain(3, domain2)

# Create mesh
mesh = generate_mesh(domain, 20, "cgal")

# set different coefficients on subdomains
class Omega_0(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0])*(x[0]) +(x[1])*(x[1]) >= 1-tol)
class Omega_1(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0])*(x[0]) +(x[1])*(x[1]) <= 1+tol) and ((x[0])*(x[0]) +(x[1])*(x[1]) >= r*r-tol)
class Omega_2(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0])*(x[0]) +(x[1])*(x[1]) <= r*r+tol)

G = MeshFunction('size_t',mesh, 2)
subdomain_0 = Omega_0() 
subdomain_1 = Omega_1() 
subdomain_2 = Omega_2() 
subdomain_0.mark(G, 0) 
subdomain_1.mark(G, 1)
subdomain_2.mark(G, 2)

class K(UserExpression):
	def __init__(self,materials,k_0,k_1,k_2, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.k_0 = k_0
		self.k_1 = k_1
		self.k_2 = k_2

	def eval_cell(self,values,x,cell):
		if (self.materials[cell.index] == 0):
			values[0] =self.k_0
		elif (self.materials[cell.index] == 1):
			values[0] =self.k_1
		else:
			values[0] =self.k_2

	def value_shape(self):
		return ()


class Phi(UserExpression):
	def __init__(self,materials,a1,a2,r, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.a1 = a1
		self.a2 = a2
		self.r = r

	def eval(self,values,x,cell):
		if (self.materials[cell.index] == 0):
			values[0] = self.a1*x[0]
			values[1] = self.a1*x[1]
		elif (self.materials[cell.index] == 1):
			values[0] = self.a1*x[0]
			values[1] = self.a1*x[1]
		else:
			values[0] = self.a1*x[0]
			values[1] = self.a1*x[1]

	def value_shape(self):
		return ()

inner = K(G,0,0,1)
domainnumber = K(G,0,1,2)


# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

	# Left boundary is "target domain" G
	def inside(self, x, on_boundary):
		return bool(x[1] < -1.+tol and x[1] > -1.-tol and on_boundary)

	# Map top boundary (H) to bottom boundary (G)
	def map(self, x, y):
		y[0] = x[0]
		y[1] = x[1] - 2.0


# define function space with periodic boundary
V = FunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())
W = VectorFunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())


# define dirichlet boundary condition
u_L=Expression('x[1]<0.25? constl*4.*x[1] : constl*(4./3.)*(1-x[1])',degree=1, constl=0.)
u_R=Expression('x[1]<0.75? constr*(4./3.)*x[1] : constr*4.*(1-x[1])',degree=1, constr= 0.)

def boundary_L(x,on_boundary):
	return on_boundary and (x[0] < -1 + tol)
bc_L = DirichletBC(V,u_L,boundary_L)

def boundary_R(x,on_boundary):
	return on_boundary and (x[0] > 1-tol)
bc_R = DirichletBC(V,u_R,boundary_R)
bc = [bc_L,bc_R]




# startvalues
a1 = 1.*r
a2 = 0.

#cppcodePhi1 = 'domainnumber == 0 ? x[0] : domainnumber == 2 ? sqrt(x[0]*x[0]+x[1]*x[1]) * a1 * sin(arctan((x[1]/x[0])-a2)) : (1-sqrt(x[0]*x[0]+x[1]*x[1]))/(1-r) *sqrt(x[0]*x[0]+x[1]*x[1]) * a1 * sin(arctan((x[1]/x[0])-a2)) + (1-(1-sqrt(x[0]*x[0]+x[1]*x[1]))/(1-r))*x[0]'
#cppcodePhi2 = 'domainnumber == 0 ? x[1] : domainnumber == 2 ? sqrt(x[0]*x[0]+x[1]*x[1]) * ((r*r)/a1) * cos(arctan((x[1]/x[0])-a2)) : (1-sqrt(x[0]*x[0]+x[1]*x[1]))/(1-r) *sqrt(x[0]*x[0]+x[1]*x[1]) ** ((r*r)/a1) * cos(arctan((x[1]/x[0])-a2))  +  (1-(1-sqrt(x[0]*x[0]+x[1]*x[1]))/(1-r)) * x[1] '
cppcodePhi1 = '(domainnumber == 0) ? (a1*x[0]) : ((domainnumber == 2) ? x[0] : x[0])'
cppcodePhi2 = '(domainnumber == 0) ? (a1*x[1]) : ((domainnumber == 2) ? x[1] : x[1])'

# solve Euler Lagrange equation and compute cost functional
veka = Constant((a1,a2))
Phi = Expression((cppcodePhi1, cppcodePhi2),degree=2,domainnumber = domainnumber, a1 = 5)
vek10 = Constant((1.,0.))
u=Function(V)
v=TestFunction(V)
# energy = abs(det(grad(Phi)))*(grad(u)-inner*vek10)**2 *dx
energy = (grad(u)-inner*vek10)**2 *dx
a = derivative(energy,u,v)
# a=dot((grad(u)-domainnumber*vek10),grad(v))*dx
# *abs(det(grad(Phi(doamainnumber)))
L=0
solve(a - L == 0, u, bc)
#Compute solution
#u = Function(V)
vtkfile = File('ellipse/G.pvd')
vtkfile << G
vtkfile = File('ellipse/gradient.pvd')
gradu = project(grad(u),W)
vtkfile << gradu
vtkfile = File('ellipse/deformation.pvd')
deformation = Phi(G,a1,a2,r)
vtkfile << deformation
vtkfile = File('ellipse/solution.pvd')
vtkfile << u
