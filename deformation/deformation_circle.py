from fenics import *
from dolfin import *
from mshr import *
import numpy

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

# functions to approximate the circle by a polygon
def polygon_top(circle_points,r):
	polygon = []
	polygon.append(dolfin.Point(r,0.))
	for i in range(1,circle_points-1):
		polygon.append(dolfin.Point(r*cos(pi*i/(circle_points-1)),r*sin(pi*i/(circle_points-1))))
	polygon.append(dolfin.Point(-r,0.))
	return polygon

def polygon_bottom(circle_points,r):
	polygon = []
	polygon.append(dolfin.Point(-r,0.))
	for i in range(1,circle_points-1):
		polygon.append(dolfin.Point(-r*cos(pi*i/(circle_points-1)),-r*sin(pi*i/(circle_points-1))))
	polygon.append(dolfin.Point(r,0.))
	return polygon

	def on_polygon(self,x,on_boundary):
		for i in range(0,edge.pointnumber):
			x1 = ((self.data)[i]).x()
			y1 = ((self.data)[i]).y()
			x2 = ((self.data)[i+1]).x()
			y2 = ((self.data)[i+1]).y()
#check      x-1 parallel to 2-1                             
			if near((x2-x1)/(x[0]-x1),(y2-y1)/(y[0]-y1),tol) & (x2-x1)/(x[0]-x1) < 1+tol & (x2-x1)/(x[0]-x1) > -tol:
				return true
		return false

#                                               input parameters, edges, domains                                          

# input parameters for computational domain
circle_points = 25
r = 0.5
L = 1.1
resolution = 50
alpha=[1.5,1.]

# input edges
left_bottom = edgeinput([dolfin.Point(-L,0.),dolfin.Point(-L, -L)],Expression(('x[0]','x[1]'),degree=1))
left_top = edgeinput([dolfin.Point(-L,L),dolfin.Point(-L, 0.)],Expression(('x[0]','x[1]'),degree=1))
right_bottom = edgeinput([dolfin.Point(L, -L),dolfin.Point(L,0.)],Expression(('x[0]','x[1]'),degree=1))
right_top = edgeinput([dolfin.Point(L,0),dolfin.Point(L, L)],Expression(('x[0]','x[1]'),degree=1))
left = edgeinput([dolfin.Point(-L,0.),dolfin.Point(-r, 0.)],Expression(('x[0]','x[1]'),degree=1))
right = edgeinput([dolfin.Point(L,0.),dolfin.Point(r, 0.)],Expression(('x[0]','x[1]'),degree=1))
bottom = edgeinput([dolfin.Point(-L,-L ),dolfin.Point(L,-L)],Expression(('x[0]','x[1]'),degree=1))
top = edgeinput([dolfin.Point(L,L ),dolfin.Point(-L,L)],Expression(('x[0]','x[1]'),degree=1))
circle_top = edgeinput(polygon_top(circle_points,r),Expression(('x[0]','x[1]'),degree=1))
circle_bottom = edgeinput(polygon_bottom(circle_points,r),Expression(('x[0]','x[1]'),degree=1))
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

mesh = generate_mesh (domain, resolution)


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


W = VectorFunctionSpace(mesh, 'P', 1)

# define dirichlet BC
#bcs = []
#for i in range(0,edges_number):
#	bc = DirichletBC(W,(edges[i]).deformation, edges[i].on_polygon)
#	bcs.append(bc)


#Compute solution
V = VectorFunctionSpace(mesh, 'P', 1)
u = Function(V)
vtkfile = File('grid/const.pvd')
vtkfile << chi_a
