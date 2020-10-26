#from scipy import optimize # import before fenics (otherwise L-BFGS-B crashes)
import sympy as sp
from fenics import *
from dolfin import *
from mshr import *
from fenics_adjoint import *
from pyadjoint.overloaded_type import create_overloaded_object
from math import *
import numpy
import scipy
import sympy as sp

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

def ccode(z):
	return sp.printing.ccode(z)


#                                               input parameters, edges, bc, domains                                          


# input parameters for computational domain
Ln = 6.5
L = 14.5
Ll = 2.5
Lr = L - Ln - Ll
resolution = 150

# input parameters for the twin structure
theta = 0.25
delta = 0.1


#                                                setting up the mesh                                                      


# input edges
edge_number = 14
top_left = edgeinput([Point(0.,1.0),Point(-Ll, 1.0)],Expression(('x[0]','x[1]'), degree=1))
top_mid = edgeinput([Point(Ln,1-0.5*theta),Point(0., 1.)],Expression(('x[0]','x[1]'), degree=1))
top_right = edgeinput([Point(Ln+Lr,1-0.5*theta),Point(Ln, 1-0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
right_top = edgeinput([Point(Ln+Lr,0.5*theta),Point(Ln+Lr, 1.-0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
right_bottom = edgeinput([Point(Ln+Lr,-0.5*theta),Point(Ln+Lr, 0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
mid_right = edgeinput([Point(Ln,0.5*theta ),Point(Ln+Lr,0.5*theta )],Expression(('x[0]','x[1]'), degree=1))
mid_left = edgeinput([Point(0.,0.),Point(0., 1.)],Expression(('x[0]','x[1]'), degree=1))
mid_bottom = edgeinput([Point(0.,0.),Point(Ln, 0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
left = edgeinput([Point(-Ll,1.),Point(-Ll, 0.)],Expression(('x[0]','x[1]'), degree=1))
bottom_left = edgeinput([Point(-Ll,0.),Point(0., 0.)],Expression(('x[0]','x[1]'), degree=1))
bottom_mid = edgeinput([Point(0.,0.),Point(Ln,-0.5*theta )],Expression(('x[0]','x[1]'), degree=1))
bottom_right = edgeinput([Point(Ln,-0.5*theta ),Point(Ln+Lr,-0.5*theta )],Expression(('x[0]','x[1]'), degree=1))

# define a vector with the edges
edges = [top_left,top_mid,top_right,right_top,right_bottom,mid_right,mid_left,mid_bottom,left,bottom_left,bottom_mid,bottom_right]
edges_number = len(edges)

# input domain
domain_complete = subdomaininput([top_left,left,bottom_left, bottom_mid, bottom_right, right_bottom, right_top, top_right, top_mid],[0,0,0,0,0,0,0,0,0],0)

# input subdomains
domain_left = subdomaininput([top_left,left,bottom_left,mid_left],[0,0,0,0],0)
domain_top = subdomaininput([mid_bottom,mid_right,right_top,top_right,top_mid,mid_left],[0,0,0,0,0,1],0)
domain_bottom = subdomaininput([bottom_mid,bottom_right,right_bottom,mid_right,mid_bottom],[0,0,0,1,1],0)

# define vector of subdomains
subdomains = [domain_left,domain_top,domain_bottom]
subdomain_number = len(subdomains)

# defining the domain, and the subdomains
domain = Polygon(domain_complete.get_polygon())
for i in range(0, subdomain_number):
	domain.set_subdomain (i+1, Polygon((subdomains[i]).get_polygon()))

# generat the mesh
mesh = generate_mesh (domain, resolution)
mesh = create_overloaded_object(mesh)

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

chi_a = sudom_fct (sudom_arr, [0,1,0,1], X)
chi_b = sudom_fct (sudom_arr, [0,0,1,0], X)
chi_test = sudom_fct (sudom_arr, [0,1,2,1], X)



#                                                 the programm                                                      

a1=11.56; a2=-17.44; a3=10.04; a4=-9.38

class PeriodicBoundary (SubDomain):
	# bottom boundary is target domain
	def inside (self, x, on_boundary): return bool (near (x[1], 0.) and on_boundary)
	# Map top boundary to bottom boundary
	def map (self, x, y): y[0] = x[0]; y[1] = x[1]-1.0



# create the function spaces
U = FunctionSpace (mesh, "CG", 1)
V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
W = VectorFunctionSpace(mesh, 'CG', 1)


dx = Measure ('dx', domain=mesh)

class DirichletBoundaryOpt (SubDomain):
	def inside (self, x, on_boundary): return bool ((fabs(x[0]) < 4*tol) and (fabs(x[0]) < 4*tol))

zero = Constant ((0,0))
tip = DirichletBoundaryOpt()
dbcopt = DirichletBC (V, zero, tip)
bcsopt = [dbcopt]


u = Function (V, name='displacement')
p = Function (V, name='dual solution')
v = TestFunction(V)
uu = TrialFunction (V) 

#                                                 calculating the deformation                                                    

# define the dependencies of the boundary conditions

Ln2, Delta, ab, at = sp.symbols('Ln2, Delta, ab, at')
x0, x1 = sp.symbols('x[0], x[1]')

exp_top_left_x = x0-sin(atan2(delta*theta,1.))
exp_top_left_y = x1

exp_top_mid_x = (x0/Ln)  *  (Ln2 - (Delta-theta) * sin(atan2(delta*theta,1.)))  -  sin(atan2(delta*theta,1.))
exp_top_mid_y = ab*(x0/Ln)*(x0/Ln)  +  (Delta-theta  - ab)  *  (x0/Ln)  +  1.

exp_top_right_x = x0 + Ln2-Ln - (1.-theta+Delta) * sin(atan2(delta*theta,1.))
exp_top_right_y = 1. - theta + Delta

exp_right_top_x = ((x1-0.5*theta)/(1-theta)) *  (  -sin(atan2(delta*theta,1.))*(1-theta) )  +  Ln2+Lr-Delta*sin(atan2(delta*theta,1.))
exp_right_top_y = x1+ Delta-0.5*theta

exp_right_bottom_x = ((x1+0.5*theta)/(theta)) *  (  -sin(atan2(delta*theta,1.))*(theta) )  +  Ln2+Lr-(Delta-theta)*sin(atan2(delta*theta,1.))
exp_right_bottom_y = x1+ Delta-0.5*theta

exp_mid_right_x = x0 + Ln2-Ln - Delta * sin(atan2(delta*theta,1.))
exp_mid_right_y = Delta

exp_mid_left_x = -x1*sin(atan2(delta*theta,1.))
exp_mid_left_y = x1

exp_mid_bottom_x = (x0/Ln)  *  (Ln2 - (Delta) * sin(atan2(delta*theta,1.)))
exp_mid_bottom_y = at*(x0/Ln)*(x0/Ln)  +  (Delta  - at)  *  (x0/Ln)

exp_left_x = x0-x1*sin(atan2(delta*theta,1.))
exp_left_y = x1

exp_bottom_left_x = x0
exp_bottom_left_y = x1

exp_bottom_mid_x = (x0/Ln)  *  ((Ln2 - (Delta-theta) * sin(atan2(delta*theta,1.))))
exp_bottom_mid_y = ab*(x0/Ln)*(x0/Ln)  +  (Delta-theta  - ab)  *  (x0/Ln)

exp_bottom_right_x = x0 + Ln2-Ln - (Delta-theta) * sin(atan2(delta*theta,1.))
exp_bottom_right_y = Delta-theta


print (ccode(exp_top_left_x))
print (ccode(exp_top_left_y))


# define the dependencies of the boundary values
top_left.deformation = Expression((ccode(exp_top_left_x),ccode(exp_top_left_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_top_left_Ln2 = Expression((ccode(exp_top_left_x.diff(Ln2)),ccode(exp_top_left_y.diff(Ln2))),degree=2)
d_top_left_Delta = Expression((ccode(exp_top_left_x.diff(Delta)),ccode(exp_top_left_y.diff(Delta))),degree=2)
d_top_left_ab = Expression((ccode(exp_top_left_x.diff(ab)),ccode(exp_top_left_y.diff(ab))),degree=2)
d_top_left_at = Expression((ccode(exp_top_left_x.diff(at)),ccode(exp_top_left_y.diff(at))),degree=2)
top_left.deformation.dependencies = [Ln2,Delta,ab,at]
top_left.deformation.user_defined_derivatives = {Ln2: d_top_left_Ln2, Delta: d_top_left_Delta, ab: d_top_left_ab, at: d_top_left_at}

top_mid.deformation = Expression((ccode(exp_top_mid_x),ccode(exp_top_mid_y).replace('pow(x[0], 2)', 'x[0]*x[0]')),degree=3,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_top_mid_Ln2 = Expression((ccode(exp_top_mid_x.diff(Ln2)),ccode(exp_top_mid_y.diff(Ln2)).replace('pow(x[0], 2)', 'x[0]*x[0]')),degree=3)
d_top_mid_Delta = Expression((ccode(exp_top_mid_x.diff(Delta)),ccode(exp_top_mid_y.diff(Delta)).replace('pow(x[0], 2)', 'x[0]*x[0]')),degree=3)
d_top_mid_ab = Expression((ccode(exp_top_mid_x.diff(ab)),ccode(exp_top_mid_y.diff(ab)).replace('pow(x[0], 2)', 'x[0]*x[0]')),degree=3)
d_top_mid_at = Expression((ccode(exp_top_mid_x.diff(at)),ccode(exp_top_mid_y.diff(at)).replace('pow(x[0], 2)', 'x[0]*x[0]')),degree=3)
top_mid.deformation.dependencies = [Ln2,Delta,ab,at]
top_mid.deformation.user_defined_derivatives = {Ln2: d_top_mid_Ln2, Delta: d_top_mid_Delta, ab: d_top_mid_ab, at: d_top_mid_at}

top_right.deformation = Expression((ccode(exp_top_right_x),ccode(exp_top_right_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_top_right_Ln2 = Expression((ccode(exp_top_right_x.diff(Ln2)),ccode(exp_top_right_y.diff(Ln2))),degree=2)
d_top_right_Delta = Expression((ccode(exp_top_right_x.diff(Delta)),ccode(exp_top_right_y.diff(Delta))),degree=2)
d_top_right_ab = Expression((ccode(exp_top_right_x.diff(ab)),ccode(exp_top_right_y.diff(ab))),degree=2)
d_top_right_at = Expression((ccode(exp_top_right_x.diff(at)),ccode(exp_top_right_y.diff(at))),degree=2)
top_right.deformation.dependencies = [Ln2,Delta,ab,at]
top_right.deformation.user_defined_derivatives = {Ln2: d_top_right_Ln2, Delta: d_top_right_Delta, ab: d_top_right_ab, at: d_top_right_at}

right_top.deformation = Expression((ccode(exp_right_top_x),ccode(exp_right_top_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_right_top_Ln2 = Expression((ccode(exp_right_top_x.diff(Ln2)),ccode(exp_right_top_y.diff(Ln2))),degree=2)
d_right_top_Delta = Expression((ccode(exp_right_top_x.diff(Delta)),ccode(exp_right_top_y.diff(Delta))),degree=2)
d_right_top_ab = Expression((ccode(exp_right_top_x.diff(ab)),ccode(exp_right_top_y.diff(ab))),degree=2)
d_right_top_at = Expression((ccode(exp_right_top_x.diff(at)),ccode(exp_right_top_y.diff(at))),degree=2)
right_top.deformation.dependencies = [Ln2,Delta,ab,at]
right_top.deformation.user_defined_derivatives = {Ln2: d_right_top_Ln2, Delta: d_right_top_Delta, ab: d_right_top_ab, at: d_right_top_at}

right_bottom.deformation = Expression((ccode(exp_right_bottom_x),ccode(exp_right_bottom_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_right_bottom_Ln2 = Expression((ccode(exp_right_bottom_x.diff(Ln2)),ccode(exp_right_bottom_y.diff(Ln2))),degree=2)
d_right_bottom_Delta = Expression((ccode(exp_right_bottom_x.diff(Delta)),ccode(exp_right_bottom_y.diff(Delta))),degree=2)
d_right_bottom_ab = Expression((ccode(exp_right_bottom_x.diff(ab)),ccode(exp_right_bottom_y.diff(ab))),degree=2)
d_right_bottom_at = Expression((ccode(exp_right_bottom_x.diff(at)),ccode(exp_right_bottom_y.diff(at))),degree=2)
right_bottom.deformation.dependencies = [Ln2,Delta,ab,at]
right_bottom.deformation.user_defined_derivatives = {Ln2: d_right_bottom_Ln2, Delta: d_right_bottom_Delta, ab: d_right_bottom_ab, at: d_right_bottom_at}

mid_right.deformation = Expression((ccode(exp_mid_right_x),ccode(exp_mid_right_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_mid_right_Ln2 = Expression((ccode(exp_mid_right_x.diff(Ln2)),ccode(exp_mid_right_y.diff(Ln2))),degree=2)
d_mid_right_Delta = Expression((ccode(exp_mid_right_x.diff(Delta)),ccode(exp_mid_right_y.diff(Delta))),degree=2)
d_mid_right_ab = Expression((ccode(exp_mid_right_x.diff(ab)),ccode(exp_mid_right_y.diff(ab))),degree=2)
d_mid_right_at = Expression((ccode(exp_mid_right_x.diff(at)),ccode(exp_mid_right_y.diff(at))),degree=2)
mid_right.deformation.dependencies = [Ln2,Delta,ab,at]
mid_right.deformation.user_defined_derivatives = {Ln2: d_mid_right_Ln2, Delta: d_mid_right_Delta, ab: d_mid_right_ab, at: d_mid_right_at}

mid_left.deformation = Expression((ccode(exp_mid_left_x),ccode(exp_mid_left_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_mid_left_Ln2 = Expression((ccode(exp_mid_left_x.diff(Ln2)),ccode(exp_mid_left_y.diff(Ln2))),degree=2)
d_mid_left_Delta = Expression((ccode(exp_mid_left_x.diff(Delta)),ccode(exp_mid_left_y.diff(Delta))),degree=2)
d_mid_left_ab = Expression((ccode(exp_mid_left_x.diff(ab)),ccode(exp_mid_left_y.diff(ab))),degree=2)
d_mid_left_at = Expression((ccode(exp_mid_left_x.diff(at)),ccode(exp_mid_left_y.diff(at))),degree=2)
mid_left.deformation.dependencies = [Ln2,Delta,ab,at]
mid_left.deformation.user_defined_derivatives = {Ln2: d_mid_left_Ln2, Delta: d_mid_left_Delta, ab: d_mid_left_ab, at: d_mid_left_at}

mid_bottom.deformation = Expression((ccode(exp_mid_bottom_x),ccode(exp_mid_bottom_y).replace('x[0]**2', 'x[0]*x[0]')),degree=3,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_mid_bottom_Ln2 = Expression((ccode(exp_mid_bottom_x.diff(Ln2)),ccode(exp_mid_bottom_y.diff(Ln2)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_mid_bottom_Delta = Expression((ccode(exp_mid_bottom_x.diff(Delta)),ccode(exp_mid_bottom_y.diff(Delta)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_mid_bottom_ab = Expression((ccode(exp_mid_bottom_x.diff(ab)),ccode(exp_mid_bottom_y.diff(ab)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_mid_bottom_at = Expression((ccode(exp_mid_bottom_x.diff(at)),ccode(exp_mid_bottom_y.diff(at)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
mid_bottom.deformation.dependencies = [Ln2,Delta,ab,at]
mid_bottom.deformation.user_defined_derivatives = {Ln2: d_mid_bottom_Ln2, Delta: d_mid_bottom_Delta, ab: d_mid_bottom_ab, at: d_mid_bottom_at}

left.deformation = Expression((ccode(exp_left_x),ccode(exp_left_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_left_Ln2 = Expression((ccode(exp_left_x.diff(Ln2)),ccode(exp_left_y.diff(Ln2))),degree=2)
d_left_Delta = Expression((ccode(exp_left_x.diff(Delta)),ccode(exp_left_y.diff(Delta))),degree=2)
d_left_ab = Expression((ccode(exp_left_x.diff(ab)),ccode(exp_left_y.diff(ab))),degree=2)
d_left_at = Expression((ccode(exp_left_x.diff(at)),ccode(exp_left_y.diff(at))),degree=2)
left.deformation.dependencies = [Ln2,Delta,ab,at]
left.deformation.user_defined_derivatives = {Ln2: d_left_Ln2, Delta: d_left_Delta, ab: d_left_ab, at: d_left_at}

bottom_left.deformation = Expression((ccode(exp_bottom_left_x),ccode(exp_bottom_left_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_bottom_left_Ln2 = Expression((ccode(exp_bottom_left_x.diff(Ln2)),ccode(exp_bottom_left_y.diff(Ln2))),degree=2)
d_bottom_left_Delta = Expression((ccode(exp_bottom_left_x.diff(Delta)),ccode(exp_bottom_left_y.diff(Delta))),degree=2)
d_bottom_left_ab = Expression((ccode(exp_bottom_left_x.diff(ab)),ccode(exp_bottom_left_y.diff(ab))),degree=2)
d_bottom_left_at = Expression((ccode(exp_bottom_left_x.diff(at)),ccode(exp_bottom_left_y.diff(at))),degree=2)
bottom_left.deformation.dependencies = [Ln2,Delta,ab,at]
bottom_left.deformation.user_defined_derivatives = {Ln2: d_bottom_left_Ln2, Delta: d_bottom_left_Delta, ab: d_bottom_left_ab, at: d_bottom_left_at}

bottom_mid.deformation = Expression((ccode(exp_bottom_mid_x),ccode(exp_bottom_mid_y).replace('x[0]**2', 'x[0]*x[0]')),degree=3,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_bottom_mid_Ln2 = Expression((ccode(exp_bottom_mid_x.diff(Ln2)),ccode(exp_bottom_mid_y.diff(Ln2)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_bottom_mid_Delta = Expression((ccode(exp_bottom_mid_x.diff(Delta)),ccode(exp_bottom_mid_y.diff(Delta)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_bottom_mid_ab = Expression((ccode(exp_bottom_mid_x.diff(ab)),ccode(exp_bottom_mid_y.diff(ab)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
d_bottom_mid_at = Expression((ccode(exp_bottom_mid_x.diff(at)),ccode(exp_bottom_mid_y.diff(at)).replace('x[0]**2', 'x[0]*x[0]')),degree=3)
bottom_mid.deformation.dependencies = [Ln2,Delta,ab,at]
bottom_mid.deformation.user_defined_derivatives = {Ln2: d_bottom_mid_Ln2, Delta: d_bottom_mid_Delta, ab: d_bottom_mid_ab, at: d_bottom_mid_at}

bottom_right.deformation = Expression((ccode(exp_bottom_right_x),ccode(exp_bottom_right_y)),degree=2,Ln2=Ln2,Delta=Delta,ab=ab,at=at)
d_bottom_right_Ln2 = Expression((ccode(exp_bottom_right_x.diff(Ln2)),ccode(exp_bottom_right_y.diff(Ln2))),degree=2)
d_bottom_right_Delta = Expression((ccode(exp_bottom_right_x.diff(Delta)),ccode(exp_bottom_right_y.diff(Delta))),degree=2)
d_bottom_right_ab = Expression((ccode(exp_bottom_right_x.diff(ab)),ccode(exp_bottom_right_y.diff(ab))),degree=2)
d_bottom_right_at = Expression((ccode(exp_bottom_right_x.diff(at)),ccode(exp_bottom_right_y.diff(at))),degree=2)
bottom_right.deformation.dependencies = [Ln2,Delta,ab,at]
bottom_right.deformation.user_defined_derivatives = {Ln2: d_bottom_right_Ln2, Delta: d_bottom_right_Delta, ab: d_bottom_right_ab, at: d_bottom_right_at}


# setting the boundary conditions
Ln2 = Constant(6.7)
Delta = Constant(0.1225)
ab = Constant(0.1)
at = Constant(-0.1)


top_left.deformation = Expression((ccode(exp_top_left_x),ccode(exp_top_left_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_left(x, on_boundary):
	return on_polygon(x,top_left)
top_mid.deformation = Expression((ccode(exp_top_mid_x),ccode(exp_top_mid_y).replace('x[0]**2', 'x[0]*x[0]')),degree=3,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_mid(x, on_boundary):
	return on_polygon(x,top_mid)
top_right.deformation = Expression((ccode(exp_top_right_x),ccode(exp_top_right_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_right(x, on_boundary):
	return on_polygon(x,top_right)
right_top.deformation = Expression((ccode(exp_right_top_x),ccode(exp_right_top_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_right_top(x, on_boundary):
	return on_polygon(x,right_top)
right_bottom.deformation = Expression((ccode(exp_right_bottom_x),ccode(exp_right_bottom_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_right_bottom(x, on_boundary):
	return on_polygon(x,right_bottom)
mid_right.deformation = Expression((ccode(exp_mid_right_x),ccode(exp_mid_right_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_right(x, on_boundary):
	return on_polygon(x,mid_right)
mid_left.deformation = Expression((ccode(exp_mid_left_x),ccode(exp_mid_left_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_left(x, on_boundary):
	return on_polygon(x,mid_left)
mid_bottom.deformation = Expression((ccode(exp_mid_bottom_x),ccode(exp_mid_bottom_y).replace('x[0]**2', 'x[0]*x[0]')),degree=3,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_bottom(x, on_boundary):
	return on_polygon(x,mid_bottom)
left.deformation = Expression((ccode(exp_left_x),ccode(exp_left_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
bottom_left.deformation = Expression((ccode(exp_bottom_left_x),ccode(exp_bottom_left_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_left(x, on_boundary):
	return on_polygon(x,bottom_left)
bottom_mid.deformation = Expression((ccode(exp_bottom_mid_x),ccode(exp_bottom_mid_y).replace('x[0]**2', 'x[0]*x[0]')),degree=3,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_mid(x, on_boundary):
	return on_polygon(x,bottom_mid)
bottom_right.deformation = Expression((ccode(exp_bottom_right_x),ccode(exp_bottom_right_y)),degree=2,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_right(x, on_boundary):
	return on_polygon(x,bottom_right)


# define a vector with the boundary conditions
boundary_edges = [boundary_top_left,boundary_top_mid,boundary_top_right,boundary_right_top,boundary_right_bottom,boundary_mid_right,boundary_mid_left,boundary_mid_bottom,boundary_left,boundary_bottom_left,boundary_bottom_mid,boundary_bottom_right]

bcs = []
for i in range(0,edges_number):
	bc = DirichletBC(W,(edges[i]).deformation, boundary_edges[i])
	bcs.append(bc)


# compute the deformation 
print ("******* compute the deformation from computational domain to fundamental cell:")
psit=TrialFunction(W)
vt=TestFunction(W)
a = inner(grad(psit),grad(vt)) *dx
psi=Function(W, name='psi')
solve(lhs(a)==rhs(a), psi, bcs)

# compute the displacement
id = project(Identity2(),W)
displacement_psi = project(psi-id,W)

# T = grad (psi)

#                                                 end of calculating the deformation                       

GA = Constant (((1,delta), (0,1)))
GB = Constant (((1,-delta), (0,1)))
G = chi_a*GA + chi_b*GB

S1 = Constant(((1,-delta*theta/sqrt(1+delta**2+theta**2)),(0,1/sqrt(1+delta**2+theta**2))))
S2 = Constant(((0,(2*delta*theta-delta)/sqrt(1+delta**2+theta**2)),(0,0)))

def energy_density (u, psi, G, a1, a2, a3, a4):
	F = ( Identity(2) + S2 + grad(u)* inv(grad(psi)) ) * inv(G)
	C = F.T*F
	return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(grad(psi)))


# Total potential energy and derivatives
Edens = energy_density (u, psi, G, a1, a2, a3, a4)
E = Edens*dx

# Derivatives (directions are nameless, so they can be test function implicitly, use action() to plug in a trial function)
duE = derivative (E, u)
dpsiduE = derivative (duE,psi)
F = derivative (E, u, v)
duduE = derivative (duE, u)


print ("******* compute the elastic deformation:")
solve (F == 0, u, bcsopt)
startE = assemble(E)
print ("********** E = %f" % startE, flush=True)





# calculating dual solution p
#print ("******* solving the dual equation:")
#solve (action(duduE,uu)==duE,p)


#reduced_functional = ReducedFunctional(E, Ln2)
#m_opt = minimize(reduced_functional, method = 'SLSQP')


dl, dd, db, dt = compute_gradient(assemble(E), [Control(Ln2),Control(Delta),Control(ab),Control(at)])
Ehat = ReducedFunctional(assemble(E), [Control(Ln2),Control(Delta),Control(ab),Control(at)])
#res = minimize (Ehat)



# safe displacement
vtkfile = File('needle/const.pvd')
vtkfile << chi_test
vtkfile = File('needle/displacement_psi.pvd')
vtkfile << displacement_psi
vtkfile = File('needle/u.pvd')
vtkfile << u