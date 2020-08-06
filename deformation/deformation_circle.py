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

# transformations on the boundaries
def_ell_x = 'sqrt(a0*a0*pow(cos(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)+pow(sin(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)/(a0*a0))*x[0]'
def_ell_y = 'sqrt(a0*a0*pow(cos(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)+pow(sin(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)/(a0*a0))*x[1]'
def_ell_quer = '(((L-abs(x[0]))/(L-r))*sqrt(a0*a0*pow(cos(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)+pow(sin(atan(a0*a0*tan(atan2(x[1],x[0])-a1))),2)/(a0*a0))+((abs(x[0])-r)/(L-r)))*x[0]'



	
class Identity(UserExpression):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def eval_cell(self,values,x,cell):
		values[0] = x[0]
		values[1] = x[1]

	def value_shape(self):
		return (2,)

#                                               input parameters, edges, bc, domains                                          

# input parameters for computational domain
circle_points = 15
r = 0.5
L = 1.1
resolution = 20
alpha=[1.3,0.3*pi]


# input edges
left_bottom = edgeinput([dolfin.Point(-L,0.),dolfin.Point(-L, -L)],Expression(('x[0]','x[1]'),degree=1))
left_top = edgeinput([dolfin.Point(-L,L),dolfin.Point(-L, 0.)],Expression(('x[0]','x[1]'),degree=1))
right_bottom = edgeinput([dolfin.Point(L, -L),dolfin.Point(L,0.)],Expression(('x[0]','x[1]'),degree=1))
right_top = edgeinput([dolfin.Point(L,0),dolfin.Point(L, L)],Expression(('x[0]','x[1]'),degree=1))
left = edgeinput([dolfin.Point(-L,0.),dolfin.Point(-r, 0.)],Expression((def_ell_quer,'x[1]'),degree=1,a0=alpha[0],a1=alpha[1],r=r,L=L))
right = edgeinput([dolfin.Point(L,0.),dolfin.Point(r, 0.)],Expression((def_ell_quer,'x[1]'),degree=1,a0=alpha[0],a1=alpha[1],r=r,L=L))
bottom = edgeinput([dolfin.Point(-L,-L ),dolfin.Point(L,-L)],Expression(('x[0]','x[1]'),degree=1))
top = edgeinput([dolfin.Point(L,L ),dolfin.Point(-L,L)],Expression(('x[0]','x[1]'),degree=1))
circle_top = edgeinput(polygon_top(circle_points,r),Expression((def_ell_x,def_ell_y),degree=1,a0=alpha[0],a1=alpha[1]))
circle_bottom = edgeinput(polygon_bottom(circle_points,r),Expression((def_ell_x,def_ell_y),degree=1,a0=alpha[0],a1=alpha[1]))

# define a vector with the edges
edges = [left_bottom,left_top,right_bottom,right_top,left,right,bottom,top,circle_top,circle_bottom]
edges_number = len(edges)


# define the boundary conditions
def boundary_left_bottom(x, on_boundary):
	return on_polygon(x,left_bottom)
def boundary_left_top(x, on_boundary):
	return on_polygon(x,left_top)
def boundary_right_bottom(x, on_boundary):
	return on_polygon(x,right_bottom)
def boundary_right_top(x, on_boundary):
	return on_polygon(x,right_top)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
def boundary_right(x, on_boundary):
	return on_polygon(x,right)
def boundary_bottom(x, on_boundary):
	return on_polygon(x,bottom)
def boundary_top(x, on_boundary):
	return on_polygon(x,top)
def boundary_circle_top(x, on_boundary):
	return on_polygon(x,circle_top)
def boundary_circle_bottom(x, on_boundary):
	return on_polygon(x,circle_bottom)

# define a vector with the boundary conditions
boundary_edges = [boundary_left_bottom,boundary_left_top,boundary_right_bottom,boundary_right_top,boundary_left,boundary_right,boundary_bottom,boundary_top,boundary_circle_top,boundary_circle_bottom]



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

# create the function space
W = VectorFunctionSpace(mesh, 'P', 1)

# define dirichlet BC
bcs = []
for i in range(0,edges_number):
	bc = DirichletBC(W,(edges[i]).deformation, boundary_edges[i])
	bcs.append(bc)


# compute the deformation u
u=Function(W)
v=TestFunction(W)
a = inner(grad(u),grad(v)) *dx
L = 0
solve(a == L, u, bcs)

# compute the displacement
id = project(Identity(),W)
displacement = project(u-id,W)


# safe displacement
vtkfile = File('deformation_circle/const.pvd')
vtkfile << chi_a
vtkfile = File('deformation_circle/displacement.pvd')
vtkfile << displacement
