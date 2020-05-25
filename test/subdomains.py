from fenics import *
#import dolfin
from mshr import *
from ufl import nabla_div
import numpy as np
import dolfin.cpp.mesh

#if not cpp.common.has_cgal():
#	print ("DOLFIN must be compiled with CGAL to run this demo.")
#	exit(0)


# optimization parameter
parL = 5.
parat = 1
parab = 1
pardelta = 0
alpha = [pardelta, parL, parab, parat]

# input parameters for computational domain
Ln = 5.
L = 14.5
Ll = 2.5
Lr = L - Ln - Ll
H = 1.0
theta = 0.25
tol= 1E-14

# defining the geometry of the computational domain
polygon1 = [dolfin.Point(0. , 0.), dolfin.Point(Ln, -0.5*theta), dolfin.Point(Ln+Lr, -0.5*theta), dolfin.Point(Ln+Lr, 0.5*theta), dolfin.Point(Ln, 0.5*theta)]
polygon2 = [dolfin.Point(0. , 0.), dolfin.Point(Ln, 0.5*theta), dolfin.Point(Ln+Lr, 0.5*theta), dolfin.Point(Ln+Lr, 1-0.5*theta), dolfin.Point(Ln, 1-0.5*theta), dolfin.Point(0., 1.)]

domainl =   Rectangle(dolfin.Point(-Ll, 0.), dolfin.Point(0., 1.))
domainu1 =   Polygon(polygon1)
domainu2 =   Polygon(polygon2)
sample = domainl + domainu1 + domainu2

# set subdomains to ensure that the verticies coincide with the boundaries of the subdomains
sample.set_subdomain(1, domainl)
sample.set_subdomain(2, domainu1)
sample.set_subdomain(3, domainu2)
# Create mesh
mesh = generate_mesh(sample, 50, "cgal")

rec = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.))
mesh1 = generate_mesh(rec, 16)
x = SpatialCoordinate(mesh1)
print (x)
#x = mesh1.coordinates()[:,0]
#y = mesh1.coordinates()[:,1]

#def denser(x, y):
#	return [a + (b-a)*((x-a)/(b-a))**s, y]


#def trafo1(x, y , Ln, parL, tol):
#	for i in range(len(x)):
#		if ((x[i] <= Ln + tol) and (x[i] >= -tol)):
#			x[i] = x[i] * parL/Ln
#		elif (x[i] >= Ln - tol):
#			x[i] = x[i] + parL-Ln
#	return [x , y]

#x_bar, y_bar = trafo1(x, y, Ln, parL, tol)
#xy_bar_coor = np.array([x_bar, y_bar]).transpose()
#mesh.coordinates()[:] = xy_bar_coor



# transform the mesh with step (1) - (3) with the design parameters alpha

# defining subdomains to allow to set different parameters on the subdomains
# probably have to adapt the definitions to the deformt area parameters
# e.g.	for		Omega_0		return (x[0] <= -delta*theta*x[1] + tol)
#		for		Omega_1		return ((x[0] >= -delta*theta*x[1] - tol) and (x[0]<= parL-delta*theta*x[1] + tol) and (x[1]>= gammat(x[0]) -tol) and (x[1]<= gammaa(x[0]+delta*theta)+sqrt(1+delta*theta) +tol)) or ((x[0] >= parL-delta*theta*x[1] - tol) and (x[1]>=Delta*sqrt(1-delta*theta)-tol))

class Omega_0(SubDomain): 
	def inside(self,x,on_boundary): 
		return (x[0] <= tol)
class Omega_1(SubDomain): 
	def inside(self,x,on_boundary): 
		return (x[0] >= -tol) and ((x[0]<= Ln+tol and x[1]>= x[0]*theta/(2*Ln) -tol) or ((x[0]>= Ln-tol)and x[1]>=0.5*theta-tol))
class Omega_2(SubDomain): 
	def inside(self,x,on_boundary): 
		return (x[0] >= -tol) and ((x[0]<= Ln+tol and x[1]<= x[0]*theta/(2*Ln) +tol) or ((x[0]>= Ln-tol)and x[1]<=0.5*theta+tol))

materials= MeshFunction('size_t',mesh, 2)
subdomain_0 = Omega_0() 
subdomain_1 = Omega_1() 
subdomain_2 = Omega_2() 
subdomain_0.mark(materials, 0) 
subdomain_1.mark(materials, 1)
subdomain_2.mark(materials, 2)

# define periodic boundary conditions via alpha

# solve Euler Lagrange equation and compute cost functional

# minimize cost functional with the obove steps over alpha


#Compute solution
V = VectorFunctionSpace(mesh, 'P', 1)
#u = Function(V)
vtkfile = File('subdomains/solution.pvd')
vtkfile << materials
