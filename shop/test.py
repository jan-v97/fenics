from scipy import optimize # import before fenics (otherwise L-BFGS-B crashes)
from fenics import *
from dolfin import *
from mshr import *
from math import *
import numpy

nonlin_solve_params = {"nonlinear_solver":"newton", "newton_solver":{"linear_solver":"mumps", "maximum_iterations":100, "relative_tolerance":1e-12}}

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


class Identity2(UserExpression):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def eval_cell(self,values,x,cell):
		values[0] = x[0]
		values[1] = x[1]

	def value_shape(self):
		return (2,)



	
#                                               input parameters, edges, bc, domains                                          


# input parameters for computational domain
L = 1.
resolution = 1


#                                                setting up the mesh                                                      

# input edges
edge_number = 4
left = edgeinput([dolfin.Point(0.,L),dolfin.Point(0., 0.)],Expression(('x[0]','x[1]'), degree=1))
top = edgeinput([dolfin.Point(L,L),dolfin.Point(0., L)],Expression(('x[0]','x[1]'), degree=1))
right = edgeinput([dolfin.Point(L,0.),dolfin.Point(L, L)],Expression(('x[0]','x[1]'), degree=1))
bottom = edgeinput([dolfin.Point(0.,0.),dolfin.Point(L, 0.)],Expression(('x[0]','x[1]'), degree=1))

# define a vector with the edges
edges = [left,top,right,bottom]
edges_number = len(edges)


# input domain
domain_complete = subdomaininput([bottom,right,top,left],[0,0,0,0],0)


# defining the domain, and the subdomains
domain = Polygon(domain_complete.get_polygon())

# generat the mesh
mesh = generate_mesh (domain, resolution)


#                                                 the programm                                                      


# create the function spaces
U = FunctionSpace (mesh, "CG", 1)
W = VectorFunctionSpace(mesh, 'CG', 1)


dx = Measure ('dx', domain=mesh)


#                                                 calculating the deformation                                                    

# initialize parameters for the shape optimization
alpha = Constant (10.)


# define the boundary conditions
left.deformation = Expression(('x[0]','x[1]*alpha/L'),degree=1,alpha=alpha,L=L)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
top.deformation = Expression(('x[0]','x[1]+alpha-L'),degree=1,alpha=alpha,L=L)
def boundary_top(x, on_boundary):
	return on_polygon(x,top)
right.deformation = Expression(('x[0]','x[1]*alpha/L'),degree=1,alpha=alpha,L=L)
def boundary_right(x, on_boundary):
	return on_polygon(x,right)
bottom.deformation = Expression(('x[0]','x[1]'),degree=1,alpha=alpha,L=L)
def boundary_bottom(x, on_boundary):
	return on_polygon(x,bottom)

# define a vector with the boundary conditions
boundary_edges = [boundary_left,boundary_top,boundary_right,boundary_bottom]

bcs = []
for i in range(0,edges_number):
	bc = DirichletBC(W,(edges[i]).deformation, boundary_edges[i])
	b = bc.get_boundary_values()
	print("\n b: \n", b)
	bcs.append(bc)


# compute the deformation psi
u = TrialFunction(W)
v = TestFunction(W)
a = inner(grad(u),grad(v)) *dx
psi = Function(W)
solve(lhs(a)==rhs(a), psi, bcs)

# compute the displacement
id = project(Identity2(),W)
displacement_psi = project(psi-id,W)

#                                                 end of calculating the deformation                       

A = assemble(lhs(a))
b = bc.get_boundary_values()
#print("stiffness_matrix: \n", A.array())


#dalphapsi = derivative(psi,alpha)

#DvE = derivative (E, alpha) - derivative (action (duE, p), alpha) # shape gradient using adjoint




# safe displacement
vtkfile = File('test/displacement_psi.pvd')
vtkfile << displacement_psi