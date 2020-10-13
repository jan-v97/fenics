from fenics import *
from dolfin import *
from mshr import *
from math import *
import numpy


tol= 1E-14

#                                    defining classes for edges and domains                                       

# input of edges
# pointnumber     integer                                 Anzahl der Punkte auf der Kante, Start und Endpunkt
#                                                                   müssen mit angegeben werden
# data            Liste mit punktezahl-Dolfin Punkten     Angabe in Reihenfolge
# deformation     string                                  Information über Deformation auf der Kante
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
Ln = 1.5
L = 4.
Ll = 1.
Lr = L - Ln - Ll
resolution = 50

# input parameters for the twin structure
theta = 0.25
delta = 0.2

# parameters for the shape optimization
Delta = 0.05
Ln2 = 2.
ab = 0.15
at = -0.15

# calculations
cosdt = cos(atan2(delta*theta,1.))
sindt = sin(atan2(delta*theta,1.))

# input edges
edge_number = 14
top_left = edgeinput([dolfin.Point(0.,1.0),dolfin.Point(-Ll, 1.0)],Expression(('x[0]-sindt','1.'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
top_mid = edgeinput([dolfin.Point(Ln,1-0.5*theta),dolfin.Point(0., 1.)],Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt))  -  sindt','ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln)  +  1.'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
top_right = edgeinput([dolfin.Point(Ln+Lr,1-0.5*theta),dolfin.Point(Ln, 1-0.5*theta)],Expression(('x[0] + Ln2-Ln - (1.-theta+Delta) * sindt','1. - theta + Delta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
right_top = edgeinput([dolfin.Point(Ln+Lr,0.5*theta),dolfin.Point(Ln+Lr, 1.-0.5*theta)],Expression(('((x[1]-0.5*theta)/(1-theta)) *  (  -sindt*(1-theta) )  +  Ln2+Lr-Delta*sindt','x[1]+ Delta-0.5*theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
right_bottom = edgeinput([dolfin.Point(Ln+Lr,-0.5*theta),dolfin.Point(Ln+Lr, 0.5*theta)],Expression(('((x[1]+0.5*theta)/(theta)) *  (  -sindt*(theta) )  +  Ln2+Lr-(Delta-theta)*sindt','x[1]+ Delta-0.5*theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
mid_right = edgeinput([dolfin.Point(Ln,0.5*theta ),dolfin.Point(Ln+Lr,0.5*theta )],Expression(('x[0] + Ln2-Ln - Delta * sindt','Delta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
mid_left = edgeinput([dolfin.Point(0.,0.),dolfin.Point(0., 1.)],Expression(('-x[1]*sindt','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
mid_bottom = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln, 0.5*theta)],Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta) * sindt)) ','at*(x[0]/Ln)*(x[0]/Ln)  +  (Delta  - at)  *  (x[0]/Ln)'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
left = edgeinput([dolfin.Point(-Ll,1.),dolfin.Point(-Ll, 0.)],Expression(('x[0]-x[1]*sindt','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
bottom_left = edgeinput([dolfin.Point(-Ll,0.),dolfin.Point(0., 0.)],Expression(('x[0]','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
bottom_mid = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln,-0.5*theta )],Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt))','ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln)'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))
bottom_right = edgeinput([dolfin.Point(Ln,-0.5*theta ),dolfin.Point(Ln+Lr,-0.5*theta )],Expression(('x[0] + Ln2-Ln - (Delta-theta) * sindt','Delta-theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at))

# define a vector with the edges
edges = [top_left,top_mid,top_right,right_top,right_bottom,mid_right,mid_left,mid_bottom,left,bottom_left,bottom_mid,bottom_right]
edges_number = len(edges)


# define the boundary conditions
def boundary_top_left(x, on_boundary):
	return on_polygon(x,top_left)
def boundary_top_mid(x, on_boundary):
	return on_polygon(x,top_mid)
def boundary_top_right(x, on_boundary):
	return on_polygon(x,top_right)
def boundary_right_top(x, on_boundary):
	return on_polygon(x,right_top)
def boundary_right_bottom(x, on_boundary):
	return on_polygon(x,right_bottom)
def boundary_mid_right(x, on_boundary):
	return on_polygon(x,mid_right)
def boundary_mid_left(x, on_boundary):
	return on_polygon(x,mid_left)
def boundary_mid_bottom(x, on_boundary):
	return on_polygon(x,mid_bottom)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
def boundary_bottom_left(x, on_boundary):
	return on_polygon(x,bottom_left)
def boundary_bottom_mid(x, on_boundary):
	return on_polygon(x,bottom_mid)
def boundary_bottom_right(x, on_boundary):
	return on_polygon(x,bottom_right)

# define a vector with the boundary conditions
boundary_edges = [boundary_top_left,boundary_top_mid,boundary_top_right,boundary_right_top,boundary_right_bottom,boundary_mid_right,boundary_mid_left,boundary_mid_bottom,boundary_left,boundary_bottom_left,boundary_bottom_mid,boundary_bottom_right]


# input domain
domain_complete = subdomaininput([top_left,left,bottom_left, bottom_mid, bottom_right, right_bottom, right_top, top_right, top_mid],[0,0,0,0,0,0,0,0,0],0)

# input subdomains
domain_left = subdomaininput([top_left,left,bottom_left,mid_left],[0,0,0,0],0)
domain_top = subdomaininput([mid_bottom,mid_right,right_top,top_right,top_mid,mid_left],[0,0,0,0,0,1],0)
domain_bottom = subdomaininput([bottom_mid,bottom_right,right_bottom,mid_right,mid_bottom],[0,0,0,1,1],0)

# define vector of subdomains
subdomains = [domain_left,domain_top,domain_bottom]
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

chi_l = sudom_fct (sudom_arr, [0,1,0,0], X)
chi_t = sudom_fct (sudom_arr, [0,0,1,0], X)
chi_b = sudom_fct (sudom_arr, [0,0,0,1], X)
chi_test = sudom_fct (sudom_arr, [0,1,2,3], X)

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
solve(a == 0, u, bcs)

# compute the displacement
id = project(Identity(),W)
displacement = project(u-id,W)


# safe displacement
vtkfile = File('deformation_needle/const.pvd')
vtkfile << chi_test
vtkfile = File('deformation_needle/displacement.pvd')
vtkfile << displacement