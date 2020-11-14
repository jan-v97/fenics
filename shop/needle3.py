#from scipy import optimize # import before fenics (otherwise L-BFGS-B crashes)
from fenics import *
from dolfin import *
from mshr import *
from fenics_adjoint import *
from pyadjoint.overloaded_type import create_overloaded_object
from math import *
import numpy
import scipy

tol= 1E-14

#                                    defining classes for edges and domains                                       

# input of edges
# pointnumber     integer                                 Anzahl der Punkte auf der Kante, Start und Endpunkt
#                                                                   müssen mit angegeben werden
# data            Liste mit punktezahl-Dolfin Punkten     Angabe in Reihenfolge
# deformation     string                                  Information über Deformation auf der Kante
class edgeinput:
	def __init__(self, data, deformation=0.):
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
resolution = 100

# input parameters for the twin structure
theta = 0.25
delta = 0.1

# calculations
cosdt = cos(atan2(delta*theta,1.))
sindt = sin(atan2(delta*theta,1.))

#                                                setting up the mesh                                                      


# input edges
edge_number = 14
top_left = edgeinput([Point(0.,1.0),Point(-Ll, 1.0)])
top_mid = edgeinput([Point(Ln,1-0.5*theta),Point(0., 1.)])
top_right = edgeinput([Point(Ln+Lr,1-0.5*theta),Point(Ln, 1-0.5*theta)])
right_top = edgeinput([Point(Ln+Lr,0.5*theta),Point(Ln+Lr, 1.-0.5*theta)])
right_bottom = edgeinput([Point(Ln+Lr,-0.5*theta),Point(Ln+Lr, 0.5*theta)])
mid_right = edgeinput([Point(Ln,0.5*theta ),Point(Ln+Lr,0.5*theta )])
mid_left = edgeinput([Point(0.,0.),Point(0., 1.)])
mid_bottom = edgeinput([Point(0.,0.),Point(Ln, 0.5*theta)])
left = edgeinput([Point(-Ll,1.),Point(-Ll, 0.)])
bottom_left = edgeinput([Point(-Ll,0.),Point(0., 0.)])
bottom_mid = edgeinput([Point(0.,0.),Point(Ln,-0.5*theta )])
bottom_right = edgeinput([Point(Ln,-0.5*theta ),Point(Ln+Lr,-0.5*theta )])

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
x = SpatialCoordinate(mesh)


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

u = Function (V, name='displacement')
v = TestFunction(V)


#                                                 calculating the deformation                                                    

#********** [Ln2,Delta,ab,at] =  [ 6.70055834  0.02786897  0.09576149 -0.08563622]
Ln2 = Constant(6.7)
Delta = Constant(0.)
ab = Constant(0.1)
at = Constant(-0.1)



# redefine the boundary conditions
top_left.deformation = project(as_vector((x[0]-sindt,1.)),W)
def boundary_top_left(x, on_boundary):
	return on_polygon(x,top_left)
top_mid.deformation = project(as_vector(((x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt))  -  sindt,ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln)  +  1.)),W)
def boundary_top_mid(x, on_boundary):
	return on_polygon(x,top_mid)
top_right.deformation = project(as_vector((x[0] + Ln2-Ln - (1.-theta+Delta) * sindt,1. - theta + Delta)),W)
def boundary_top_right(x, on_boundary):
	return on_polygon(x,top_right)
right_top.deformation = project(as_vector((((x[1]-0.5*theta)/(1-theta)) *  (  -sindt*(1-theta) )  +  Ln2+Lr-Delta*sindt,x[1]+ Delta-0.5*theta)),W)
def boundary_right_top(x, on_boundary):
	return on_polygon(x,right_top)
right_bottom.deformation = project(as_vector((((x[1]+0.5*theta)/(theta)) *  (  -sindt*(theta) )  +  Ln2+Lr-(Delta-theta)*sindt,x[1]+ Delta-0.5*theta)),W)
def boundary_right_bottom(x, on_boundary):
	return on_polygon(x,right_bottom)
mid_right.deformation = project(as_vector((x[0] + Ln2-Ln - Delta * sindt,Delta)),W)
def boundary_mid_right(x, on_boundary):
	return on_polygon(x,mid_right)
mid_left.deformation = project(as_vector((-x[1]*sindt,x[1])),W)
def boundary_mid_left(x, on_boundary):
	return on_polygon(x,mid_left)
mid_bottom.deformation = project(as_vector(((x[0]/Ln)  *  ((Ln2 - (Delta) * sindt)) ,at*(x[0]/Ln)*(x[0]/Ln)  +  (Delta  - at)  *  (x[0]/Ln))),W)
def boundary_mid_bottom(x, on_boundary):
	return on_polygon(x,mid_bottom)
left.deformation = project(as_vector((x[0]-x[1]*sindt,x[1])),W)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
bottom_left.deformation = project(as_vector((x[0],x[1])),W)
def boundary_bottom_left(x, on_boundary):
	return on_polygon(x,bottom_left)
bottom_mid.deformation = project(as_vector(((x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt)),ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln))),W)
def boundary_bottom_mid(x, on_boundary):
	return on_polygon(x,bottom_mid)
bottom_right.deformation = project(as_vector((x[0] + Ln2-Ln - (Delta-theta) * sindt,Delta-theta)),W)
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
dpsi = Function(W, name='dpsi')
dpsi = project(psi-id,W)

#                                                 end of calculating the deformation                       

GA = Constant (((1,delta), (0,1)))
GB = Constant (((1,-delta), (0,1)))
G = chi_a*GA + chi_b*GB

S2 = Constant(((0,(2*delta*theta-delta)/sqrt(1+delta**2+theta**2)),(0,0)))

def energy_density (u, psi, G, a1, a2, a3, a4):
	F = ( Identity(2) + grad(u)* inv(grad(psi)) ) * inv(G)
	C = F.T*F
	return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(grad(psi)))


# Total potential energy and derivatives
Edens = energy_density (u, psi, G, a1, a2, a3, a4)
#E = Edens*dx
E = inner((Identity(2)+grad(u)* inv(grad(psi)))* inv(G),(Identity(2)+grad(u)* inv(grad(psi)))* inv(G))*dx



class DirichletBoundaryOpt (SubDomain):
	def inside (self, x, on_boundary): return bool ((fabs(x[0]) < 4*tol) and (fabs(x[0]) < 4*tol))

zero = Constant ((0,0))
tip = DirichletBoundaryOpt()
dbcopt = DirichletBC (V, zero, tip)
bcsopt = [dbcopt]



F = derivative (E, u, v)
print ("******* compute the elastic deformation:")
solve (F == 0, u, bcsopt)
startE = assemble(E)
print ("********** E_start = %f" % startE, flush=True)

def iter_cb(m):
	print ("m = ", m)

def eval_cb(j, m):
	print ("j = %f, m = %f." % (j, float(m)))
	
def derivative_cb(j, dj, m):
	print ("j = %f, dj = %f, m = %f." % (j, dj, float(m)))

dl, dd, db, dt = compute_gradient(assemble(E), [Control(Ln2),Control(Delta),Control(ab),Control(at)])
Ehat = ReducedFunctional(assemble(E), [Control(Ln2),Control(Delta),Control(ab),Control(at)])
#db = compute_gradient(assemble(E), Control(Ln2))
#Ehat = ReducedFunctional(assemble(E), Control(Ln2))
h = [Constant(1),Constant(1),Constant(1),Constant(1)]
conv_rateL = taylor_test(Ehat, [Ln2,Delta,ab,at], h)
#h = Constant(0.01)
#conv_rateL = taylor_test(Ehat, Ln2, h)
#Computed residuals: [1.6110796255681702e-10, 8.055763048608066e-11, 4.0279957341608826e-11, 2.0140850640401635e-11]
#Computed convergence rates: [0.999934645383346, 0.9999590932213036, 0.9999375392098214]
#rLn2, rDelta, rab, rat = minimize (Ehat, method = 'SLSQP', tol = 1e-10, options = {'disp': True}, callback = iter_cb)

#print (float(rLn2), float(rDelta),float(rab),float(rat)) 
# 6.700851990542797 0.0025949206313023287 0.09556013899617283 -0.00462138746180401



# safe displacement
vtkfile = File('needle/const.pvd')
vtkfile << chi_test
vtkfile = File('needle/dpsi.pvd')
vtkfile << dpsi
vtkfile = File('needle/u.pvd')
vtkfile << u