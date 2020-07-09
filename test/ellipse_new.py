from fenics import *
#import dolfin
from mshr import *
from math import *
import warnings
from ufl import nabla_div
import numpy
import dolfin.cpp.mesh

tol= 1E-14
r = 0.5
resolution = 40
alpha = [1.5,0.3*pi]

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
mesh = generate_mesh(domain, resolution, "cgal")


# defining coefficients on subdomains
X = FunctionSpace (mesh, "DG", 0)
dm = X.dofmap()
sudom = MeshFunction ('size_t', mesh, 2, mesh.domains())
sudom_arr = numpy.asarray (sudom.array(), dtype=numpy.int)
for cell in cells (mesh): sudom_arr [dm.cell_dofs (cell.index())] = sudom [cell]

def sudom_fct (sudom_arr, vals, fctspace):
    f = Function (fctspace)
    f.vector()[:] = numpy.choose (sudom_arr, vals)
    return f

chi_inner_circle = sudom_fct (sudom_arr, [0,0,0,1], X)
chi_outer_circle = sudom_fct (sudom_arr, [0,0,1,0], X)
chi_outside = sudom_fct (sudom_arr, [0,1,0,0], X)



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


class deformation_inner_circle(UserExpression):
	def __init__(self,a0,a1,r, **kwargs):
		super().__init__(**kwargs)
		self.a0 = a0
		self.a1 = a1
		self.r = r

	def eval_cell(self,values,x,cell):
		values[0] = (l(self.a0,(1./(self.a0)),atan2(x[1],x[0])-a1))*x[0]
		values[1] = (l(self.a0,(1./(self.a0)),atan2(x[1],x[0])-a1))*x[1]

	def value_shape(self):
		return (2,)

	
class deformation_outer_circle(UserExpression):
	def __init__(self,a0,a1,r, **kwargs):
		super().__init__(**kwargs)
		self.a0 = a0
		self.a1 = a1
		self.r = r

	def eval_cell(self,values,x,cell):
		values[0] = (rad1(x[0],x[1],self.r) * l(self.a0,(1./(self.a0)),atan2(x[1],x[0])-a1) + rad2(x[0],x[1],self.r)) * x[0]
		values[1] = (rad1(x[0],x[1],self.r) * l(self.a0,(1./(self.a0)),atan2(x[1],x[0])-a1) + rad2(x[0],x[1],self.r)) * x[1]

	def value_shape(self):
		return (2,)
	
	
class Identity(UserExpression):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def eval_cell(self,values,x,cell):
		values[0] = x[0]
		values[1] = x[1]

	def value_shape(self):
		return (2,)


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


a0 = Constant(alpha[0])
a1 = Constant(alpha[1])
vek10 = Constant((1.,0.))


# solve Euler Lagrange equation and compute cost functional
deformation_inner = deformation_inner_circle(a0,a1,r,element=W.ufl_element())
deformation_outer = deformation_outer_circle(a0,a1,r,element=W.ufl_element())
def_inner = project(deformation_inner_circle(a0,a1,r,element=W.ufl_element()),W)
def_outer = project(deformation_outer_circle(a0,a1,r,element=W.ufl_element()),W)
#id = project(Identity(),W)



#w = TrialFunction(W)
#d = w.geometric_dimension()
#I = Identity(d)

defu = project(chi_inner_circle*def_inner+chi_outer_circle*def_outer+chi_outside*Identity(), W)
u=Function(V)
v=TestFunction(V)
#energy = (chi_inner_circle * ((grad(u))-vek10)**2 + (chi_outer_circle+chi_outside)*grad(u)**2) *dx
energy = (chi_inner_circle*(abs(det(grad(def_inner)))*((inv(grad(def_inner)).T*grad(u))-vek10)**2)+ chi_outer_circle*(abs(det(grad(def_outer)))*(inv(grad(def_outer)).T*grad(u))**2)+ chi_outside*(grad(u)**2)) *dx
a = derivative(energy,u,v)
L=0
# Compute solution
solve(a - L == 0, u)
print ("energie: ", assemble((chi_inner_circle*(abs(det(grad(def_inner)))*((inv(grad(def_inner)).T*grad(u))-vek10)**2)+ chi_outer_circle*(abs(det(grad(def_outer)))*(inv(grad(def_outer)).T*grad(u))**2)+ chi_outside*(grad(u)**2)) *dx))


# compute the displacement
displacement = project(chi_inner_circle*def_inner+chi_outer_circle*def_outer+chi_outside*Identity()-Identity(),W)

# save output
vtkfile = File('ellipse_neu/G.pvd')
vtkfile << chi_inner_circle
vtkfile = File('ellipse_neu/gradient.pvd')
gradu = project((inv(grad(defu)).T)*grad(u),W)
vtkfile << gradu
vtkfile = File('ellipse_neu/displacement.pvd')
vtkfile << displacement
vtkfile = File('ellipse_neu/solution.pvd')
vtkfile << u
