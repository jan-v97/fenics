from fenics import *
#import dolfin
from mshr import *
from math import *
import warnings
from ufl import nabla_div
import numpy as np
import dolfin.cpp.mesh
warnings.simplefilter(action='ignore', category=FutureWarning)

tol= 1E-14
r = 0.5

# defining the geometry of the computational domain
domain =   Rectangle(dolfin.Point(-1.1, -1.1), dolfin.Point(1.1, 1.1))
domain2 =   Circle(dolfin.Point(0.,0.),r)
domainouter =   Circle(dolfin.Point(0.,0.),1)
domain0 = domain -domainouter
domain1 = domainouter - domain2

# set subdomains to ensure that the verticies coincide with the boundaries of the subdomains
domain.set_subdomain(1, domain0)
domain.set_subdomain(2, domain1)
domain.set_subdomain(3, domain2)

# Create mesh
mesh = generate_mesh(domain, 30, "cgal")

# set different coefficients on subdomains
class Omega_0(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0])*(x[0]) +(x[1])*(x[1]) >= 1-tol)
class Omega_1(SubDomain): 
	def inside(self,x,on_boundary): 
		return ((x[0])*(x[0]) +(x[1])*(x[1]) <= 1+tol) and ((x[0])*(x[0]) +(x[1])*(x[1]) >= r*r-0.01*r)
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



class u_exact(UserExpression):
	def __init__(self,materials,r, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.r = r

	def eval_cell(self,values,x,cell):
		if ((self.materials[cell.index] == 0) | (self.materials[cell.index] == 1)):
			values[0] = (0.5*r*r*x[0])/(x[0]*x[0]+x[1]*x[1])
		else:
			values[0] = 0.5*x[0]

	def value_shape(self):
		return ()


interior = K(G,0,0,1)


# define function space with periodic boundary
V = FunctionSpace(mesh, 'P', 1)
W = VectorFunctionSpace(mesh, 'P', 1)


# define dirichlet boundary condition
u=Expression('(0.5*R*R*x[0])/(x[0]*x[0]+x[1]*x[1])',degree=2, R=r)

def boundary(x,on_boundary):
	return on_boundary
bc = DirichletBC(V,u,boundary)

starta1 = 1.
starta2 = 0.


# solve Euler Lagrange equation and compute cost functional
a1 = Constant(starta1)
a2 = Constant(starta2)
vek10 = Constant((1.,0.))
exact_sol = u_exact(G,r,element=V.ufl_element())
exact_sol_int = interpolate(exact_sol, V)
u=Function(V)
v=TestFunction(V)
energy = (grad(u)-interior*vek10)**2 *dx
a = derivative(energy,u,v)
L=0
# Compute solution
solve(a - L == 0, u,bc)
r = project(abs(u - exact_sol_int),V)

# save output
vtkfile = File('circle_exact_sol/gradient.pvd')
gradu = project(grad(u),W)
vtkfile << gradu
vtkfile = File('circle_exact_sol/solution.pvd')
vtkfile << u
vtkfile = File('circle_exact_sol/diff.pvd')
vtkfile << r
vtkfile = File('circle_exact_sol/exact.pvd')
vtkfile << exact_sol_int
vtkfile = File('circle_exact_sol/grad_exact.pvd')
grad_exact = project(grad(exact_sol_int),W)
vtkfile << grad_exact
