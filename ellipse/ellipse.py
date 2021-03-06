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

# returns the distance of a point on the ellipsis for given parameters a,b and the angle theta
def verzerren(a,b,theta):
	return atan(a*tan(theta)/b)

def l(a,b,theta):
	thetav = verzerren(a,b,theta)
	return sqrt(a*a*cos(thetav)*cos(thetav)+b*b*sin(thetav)*sin(thetav))

# returns	0		if |(x,y)| = 1 and 
#			1		if |(x,y)| = r
def rad1(x,y,r):
	return (1.-sqrt(x*x+y*y))/(1-r)

# returns	1		if |(x,y)| = 1 and 
#			0		if |(x,y)| = r
def rad2(x,y,r):
	return (sqrt(x*x+y*y)-r)/(1-r)


class Phi(UserExpression):
	def __init__(self,materials,a1,a2,r, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.a1 = a1
		self.a2 = a2
		self.r = r

	def eval_cell(self,values,x,cell):
		if (self.materials[cell.index] == 0):
			values[0] = x[0]
			values[1] = x[1]
		elif (self.materials[cell.index] == 1):
			values[0] = (rad1(x[0],x[1],self.r) * l(self.a1,(1./(self.a1)),atan2(x[1],x[0])-a2) + rad2(x[0],x[1],self.r)) * x[0]
			values[1] = (rad1(x[0],x[1],self.r) * l(self.a1,(1./(self.a1)),atan2(x[1],x[0])-a2) + rad2(x[0],x[1],self.r)) * x[1]
		else:
			values[0] = (l(self.a1,(1./(self.a1)),atan2(x[1],x[0])-a2))*x[0]
			values[1] = (l(self.a1,(1./(self.a1)),atan2(x[1],x[0])-a2))*x[1]

	def value_shape(self):
		return (2,)


class Displacement(UserExpression):
	def __init__(self,materials,a1,a2,r, **kwargs):
		super().__init__(**kwargs)
		self.materials = materials
		self.a1 = a1
		self.a2 = a2
		self.r = r

	def eval_cell(self,values,x,cell):
		if (self.materials[cell.index] == 0):
			values[0] = 0
			values[1] = 0
		elif (self.materials[cell.index] == 1):
			values[0] = (rad1(x[0],x[1],self.r) * l(self.a1(x),(1./(self.a1(x))),atan2(x[1],x[0])-self.a2) + rad2(x[0],x[1],self.r) -1.) * x[0]
			values[1] = (rad1(x[0],x[1],self.r) * l(self.a1(x),(1./(self.a1(x))),atan2(x[1],x[0])-self.a2) + rad2(x[0],x[1],self.r) -1.) * x[1]
		else:
			values[0] = (l(self.a1(x),(1./(self.a1(x))),atan2(x[1],x[0])-self.a2)-1)*x[0]
			values[1] = (l(self.a1(x),(1./(self.a1(x))),atan2(x[1],x[0])-self.a2)-1)*x[1]

	def value_shape(self):
		return (2,)



interior = K(G,0,0,1)


# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

	# Left boundary is "target domain" G
	def inside(self, x, on_boundary):
		return bool(x[1] < -1.1+tol and x[1] > -1.1-tol and on_boundary)

	# Map top boundary (H) to bottom boundary (G)
	def map(self, x, y):
		y[0] = x[0]
		y[1] = x[1] - 2.2


# define function space with periodic boundary
V = FunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())
W = VectorFunctionSpace(mesh, 'P', 1,constrained_domain=PeriodicBoundary())


# define dirichlet boundary condition
#u_L=Expression('x[1]<0.25? constl*4.*x[1] : constl*(4./3.)*(1-x[1])',degree=1, constl=0.)
#u_R=Expression('x[1]<0.75? constr*(4./3.)*x[1] : constr*4.*(1-x[1])',degree=1, constr= 0.)

#def boundary_L(x,on_boundary):
#	return on_boundary and (x[0] <= -1.1 + tol)
#bc_L = DirichletBC(V,u_L,boundary_L)

#def boundary_R(x,on_boundary):
#	return on_boundary and (x[0] >= 1.1-tol)
#bc_R = DirichletBC(V,u_R,boundary_R)

def boundary(x, on_boundary):
	return False
bc = DirichletBC(V, '0', boundary)

#bc = [bc_L,bc_R]

starta1 = 1.5
starta2 = 0.3 * pi


# solve Euler Lagrange equation and compute cost functional
a1 = Constant(starta1)
a2 = Constant(starta2)
vek10 = Constant((1.,0.))
deformation = Phi(G,a1,a2,r,element=W.ufl_element())
displacement = Displacement(G,a1,a2,r,element=W.ufl_element())
defu = interpolate(deformation, W)
displ = interpolate(displacement,W)
u=Function(V)
v=TestFunction(V)
energy = abs(det(grad(defu)))*((inv(grad(defu).T)*grad(u))-interior*vek10)**2 *dx
a = derivative(energy,u,v)
L=0
# Compute solution
solve(a - L == 0, u)
print ("energie: ", assemble(abs(det(grad(defu)))*((inv(grad(defu).T)*grad(u))-interior*vek10)**2 *dx))
#print ("dE/da1: ", assemble(derivative(energy,a1)))
#print ("dE/da2: ", assemble(derivative(energy,a2)))

#difa1 = diff(energy,a1)
#print (difa1)

# save output
vtkfile = File('ellipse/G.pvd')
vtkfile << G
vtkfile = File('ellipse/gradient.pvd')
gradu = project(inv(grad(defu).T)*grad(u),W)
vtkfile << gradu
vtkfile = File('ellipse/displacement.pvd')
vtkfile << displ
vtkfile = File('ellipse/solution.pvd')
vtkfile << u
