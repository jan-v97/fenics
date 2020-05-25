from fenics import *
#import dolfin
from mshr import *
from ufl import nabla_div
import numpy as np
import dolfin.cpp.mesh

tol= 1E-14
r = 0.2

# defining the geometry of the computational domain
domain =   Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.))
domain1 =   Circle(dolfin.Point(0.5,0.5),r)
domain0 = domain -domain1

# set subdomains to ensure that the verticies coincide with the boundaries of the subdomains
domain.set_subdomain(1, domain0)
domain.set_subdomain(2, domain1)

# Create mesh
mesh = generate_mesh(domain, 50, "cgal")

# set different coefficients on subdomains
class Omega_0(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0]-0.5)*(x[0]-0.5) +(x[1]-0.5)*(x[1]-0.5) >= r*r-tol)
class Omega_1(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0]-0.5)*(x[0]-0.5) +(x[1]-0.5)*(x[1]-0.5) <= r*r+tol)

G = MeshFunction('size_t',mesh, 2)
subdomain_0 = Omega_0() 
subdomain_1 = Omega_1() 
subdomain_0.mark(G, 0) 
subdomain_1.mark(G, 1)

class K(UserExpression):
	def __init__(self,materials,k_0,k_1, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.k_0 = k_0
		self.k_1 = k_1

	def eval_cell(self,values,x,cell):
		if (self.materials[cell.index] == 0):
			values[0] =self.k_0
		else:
			values[0] =self.k_1

	def value_shape(self):
		return ()

coef = K(G,0.,1.)

# define function space
V = FunctionSpace(mesh, 'P', 1)

# define periodic boundary conditions
u_D = Expression('0.1*x[1]', degree=1)
def boundary(x, on_boundary):
	return on_boundary
bc = DirichletBC(V, u_D, boundary)


# solve Euler Lagrange equation and compute cost functional
Vek = Constant((1.,0.))
u=Function(V)
v=TestFunction(V)
a=dot(grad(u)-coef*Vek,grad(v))*dx
L=0
solve(a - L == 0, u, bc)
#Compute solution
#u = Function(V)
vtkfile = File('energy/G.pvd')
vtkfile << G
vtkfile = File('energy/solution.pvd')
vtkfile << u
