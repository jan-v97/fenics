from fenics import *
from dolfin import *
from mshr import *
import numpy
from math import *

tol= 1E-14

#                                    defining classes for edges and domains                                       

# input of edges
# pointnumber     integer                                 Anzahl der Punkte auf der Kante, Start und Endpunkt
#                                                                   müssen mit angegeben werden
# data            Liste mit punktezahl-Dolfin Punkten     Angabe in Reihenfolge
# deformation     string                                  Information über Deformation auf der Kante
#                                                             '' for no deformation
class edgeinput:
	def __init__(self, data, deformation):
		self.pointnumber = len(data)
		self.data = data;
		self.deformation = deformation
	
	def print(self):
		print ('pointnumber:    ', self.pointnumber)
		print ('data:    ')
		for i in range(self.pointnumber):
			print ('         ( ', self.data[i].x(),' , ', self.data[i].y(),' )')
		print('deformation:   ', self.deformation)


# input der Subdomains
# kantenzahl       integer                                Anzahl der Kanten, die an das Gebiet angrenzen
# kanten           Liste mit kantenzahl-kantenzuginput    in Reihenfolge
# orientierungen   Liste mit kantenzahl-orientierungen    0 für orientierung beibehalten
#                                                         1 für orientierung wechsel
# materialnmbr     integer                                information über Material auf dieser Domain
class subdomaininput:
	def __init__(self, edges, orientations, materialnumber):
		self.edgenumber = len(edges)
		self.edges = edges
		self.orientations = orientations
		self.materialnumber = materialnumber

	def print(self):
		print ('edgenumber: ', int(self.edgenumber))
		for i in range (0,self.edgenumber):
			(self.edges[i]).print()
			print ("  orientations:" , self.orientations[i])
		print ("  materialnumber: ", self.materialnumber)

	def get_polygon(self):
		polygon = []
		for i in range (0,self.edgenumber):
			if (self.orientations[i]==0):
				for j in range (0, (self.edges[i]).pointnumber-1):
					polygon.append((self.edges[i]).data[j])
			else:
				for j in range (0, (self.edges[i]).pointnumber-1):
					polygon.append((self.edges[i]).data[(self.edges[i]).pointnumber-1-j])
		return polygon

	def printpolygon(self):
		for i in range (0,self.edgenumber):
			if (self.orientations[i]==0):
				for j in range (0, (self.edges[i]).pointnumber-1):
					print (' ( ', (self.edges[i]).data[j].x(),' , ', (self.edges[i]).data[j].y(),' )')
			else:
				for j in range (0, (self.edges[i]).pointnumber-1):
					print (' ( ', (self.edges[i]).data[(self.edges[i]).pointnumber-1-j].x(),' , ', (self.edges[i]).data[(self.edges[i]).pointnumber-1-j].y(),' )')





#                                     functions for defining edges                                           

# check if a point x is on the edge
def on_polygon(x,edge):
	for i in range(0,edge.pointnumber-1):
		x0 = ((edge.data)[i]).x()
		y0 = ((edge.data)[i]).y()
		x1 = ((edge.data)[i+1]).x()
		y1 = ((edge.data)[i+1]).y()
		if (near(x[0],x0,tol) and near(x[1],y0,tol) or near(x[0],x1,tol) and near(x[1],y1,tol)):
			return True
		if near(x1,x0,tol):
			r = (x1 - x0) * (x[1] - y0) / (y1 - y0) + x0 - x[0] 
			if (near(r, 0.0, tol) and (((x[1] >= y0) and (x[1] <= y1)) or ((x[1] >= y1) and (x[1] <= y0)))):
				return True
		else:
			r = (y1 - y0) * (x[0] - x0) / (x1 - x0) + y0 - x[1]
			if (near(r, 0.0, tol) and (((x[0] >= x0) and (x[0] <= x1)) or ((x[0] >= x1) and (x[0] <= x0)))):
				return True
		#print ("( ", x0, " , ", y0, " ) ; ( ", x[0], " , ", x[1], " ) ; ( ", x1, " , ", y1, " )           return_val: ", result)
	return False

# define the identity
class Identity(UserExpression):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def eval_cell(self,values,x,cell):
		values[0] = x[0]
		values[1] = x[1]

	def value_shape(self):
		return (2,)



# returns the distance of a point on the ellipsis for given parameters a,b and the angle theta
def verzerren(a,b,theta):
	return atan(a*tan(theta)/b)

def l(a,b,theta):
	thetav = verzerren(a,b,theta)
	return sqrt(a*a*cos(thetav)*cos(thetav)+b*b*sin(thetav)*sin(thetav))

def circle_to_ellipse_x(a0,a1,x):
	return (l(a0,(1./(a0)),atan2(x[1],x[0])-a1))*x[0]

def circle_to_ellipse_y(a0,a1,x):
	return (l(a0,(1./(a0)),atan2(x[1],x[0])-a1))*x[1]

# functions to approximate the circle by a polygon
def polygon_top(circle_points,r,a0,a1):
	polygon = []
	polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[r,0.]),circle_to_ellipse_y(a0,a1,[r,0.])))
	for i in range(1,circle_points-1):
		polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[r*cos(pi*i/(circle_points-1)),r*sin(pi*i/(circle_points-1))]),circle_to_ellipse_y(a0,a1,[r*cos(pi*i/(circle_points-1)),r*sin(pi*i/(circle_points-1))])))
	polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[-r,0.]),circle_to_ellipse_y(a0,a1,[-r,0.])))
	return polygon

def polygon_bottom(circle_points,r,a0,a1):
	polygon = []
	polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[-r,0.]),circle_to_ellipse_y(a0,a1,[-r,0.])))
	for i in range(1,circle_points-1):
		polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[-r*cos(pi*i/(circle_points-1)),-r*sin(pi*i/(circle_points-1))]),circle_to_ellipse_y(a0,a1,[-r*cos(pi*i/(circle_points-1)),-r*sin(pi*i/(circle_points-1))])))
	polygon.append(dolfin.Point(circle_to_ellipse_x(a0,a1,[r,0.]),circle_to_ellipse_y(a0,a1,[r,0.])))
	return polygon


#                                               input parameters, edges, domains                                          

# input parameters for computational domain
circle_points = 35
r = 0.5
L = 1.1
resolution = 40
vek10 = Constant((1.,0.))
alpha=[1.5,0.3*pi]


# input edges
left_bottom = edgeinput([dolfin.Point(-L,0.),dolfin.Point(-L, -L)],Expression(('x[0]','x[1]'),degree=1))
left_top = edgeinput([dolfin.Point(-L,L),dolfin.Point(-L, 0.)],Expression(('x[0]','x[1]'),degree=1))
right_bottom = edgeinput([dolfin.Point(L, -L),dolfin.Point(L,0.)],Expression(('x[0]','x[1]'),degree=1))
right_top = edgeinput([dolfin.Point(L,0),dolfin.Point(L, L)],Expression(('x[0]','x[1]'),degree=1))
left = edgeinput([dolfin.Point(-L,0.),dolfin.Point(circle_to_ellipse_x(alpha[0],alpha[1],[-r,0.]),circle_to_ellipse_y(alpha[0],alpha[1],[-r,0.]))],Expression(('x[0]','x[1]'),degree=1))
right = edgeinput([dolfin.Point(L,0.),dolfin.Point(circle_to_ellipse_x(alpha[0],alpha[1],[r,0.]),circle_to_ellipse_y(alpha[0],alpha[1],[r,0.]))],Expression(('x[0]','x[1]'),degree=1))
bottom = edgeinput([dolfin.Point(-L,-L ),dolfin.Point(L,-L)],Expression(('x[0]','x[1]'),degree=1))
top = edgeinput([dolfin.Point(L,L ),dolfin.Point(-L,L)],Expression(('x[0]','x[1]'),degree=1))
circle_top = edgeinput(polygon_top(circle_points,r,alpha[0],alpha[1]),Expression(('x[0]','x[1]'),degree=1))
circle_bottom = edgeinput(polygon_bottom(circle_points,r,alpha[0],alpha[1]),Expression(('x[0]','x[1]'),degree=1))

# define a vector with the edges
edges = [left_bottom,left_top,right_bottom,right_top,left,right,bottom,top,circle_top,circle_bottom]
edges_number = len(edges)


# define the complete domain
domain_complete = subdomaininput([left_top,left_bottom,bottom,right_bottom,right_top,top],[0,0,0,0,0,0],0)

# input subdomains
domain_bottom = subdomaininput([left_bottom,bottom,right_bottom,right,circle_bottom,left],[0,0,0,0,1,1],0)
domain_top = subdomaininput([right_top,top,left_top,left,circle_top,right],[0,0,0,0,1,1],0)
domain_circle = subdomaininput([circle_top,circle_bottom],[0,0],0)

# define a vector with the subdomains
subdomains = [domain_bottom,domain_top,domain_circle]
subdomain_number = len(subdomains)



#                                                 actual program                                                    

# defining the domain, and the subdomains
domain = Polygon(domain_complete.get_polygon())
for i in range(0, subdomain_number):
	domain.set_subdomain (i+1, Polygon((subdomains[i]).get_polygon()))

# generat the mesh
mesh = generate_mesh (domain, resolution)

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

chi_a = sudom_fct (sudom_arr, [0,0,0,1], X)


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

# compute the deformation u
u=Function(V)
v=TestFunction(V)
energy = ((grad(u))-chi_a*vek10)**2 *dx
a = derivative(energy,u,v)
L=0
# Compute solution
solve(a - L == 0, u)
print ("energie: ", assemble(((grad(u))-chi_a*vek10)**2 *dx))


# save output
vtkfile = File('ellipse_grid/chi_a.pvd')
vtkfile << chi_a
vtkfile = File('ellipse_grid/gradient.pvd')
gradu = project(grad(u),W)
vtkfile << gradu
vtkfile = File('ellipse_grid/solution.pvd')
vtkfile << u