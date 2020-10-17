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
top_left = edgeinput([dolfin.Point(0.,1.0),dolfin.Point(-Ll, 1.0)],Expression(('x[0]','x[1]'), degree=1))
top_mid = edgeinput([dolfin.Point(Ln,1-0.5*theta),dolfin.Point(0., 1.)],Expression(('x[0]','x[1]'), degree=1))
top_right = edgeinput([dolfin.Point(Ln+Lr,1-0.5*theta),dolfin.Point(Ln, 1-0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
right_top = edgeinput([dolfin.Point(Ln+Lr,0.5*theta),dolfin.Point(Ln+Lr, 1.-0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
right_bottom = edgeinput([dolfin.Point(Ln+Lr,-0.5*theta),dolfin.Point(Ln+Lr, 0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
mid_right = edgeinput([dolfin.Point(Ln,0.5*theta ),dolfin.Point(Ln+Lr,0.5*theta )],Expression(('x[0]','x[1]'), degree=1))
mid_left = edgeinput([dolfin.Point(0.,0.),dolfin.Point(0., 1.)],Expression(('x[0]','x[1]'), degree=1))
mid_bottom = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln, 0.5*theta)],Expression(('x[0]','x[1]'), degree=1))
left = edgeinput([dolfin.Point(-Ll,1.),dolfin.Point(-Ll, 0.)],Expression(('x[0]','x[1]'), degree=1))
bottom_left = edgeinput([dolfin.Point(-Ll,0.),dolfin.Point(0., 0.)],Expression(('x[0]','x[1]'), degree=1))
bottom_mid = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln,-0.5*theta )],Expression(('x[0]','x[1]'), degree=1))
bottom_right = edgeinput([dolfin.Point(Ln,-0.5*theta ),dolfin.Point(Ln+Lr,-0.5*theta )],Expression(('x[0]','x[1]'), degree=1))

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
    def inside (self, x, on_boundary): return bool (near (x[0], 0) and near (x[1], 0))

zero = Constant ((0,0))
tip = DirichletBoundaryOpt()
dbcopt = DirichletBC (V, zero, tip,method='pointwise')
bcsopt = [dbcopt]


u = Function (V, name='displacement')
p = Function (V, name='dual solution')
v = TestFunction(V)
uu = TrialFunction (V) 

#                                                 calculating the deformation                                                    

alpha = Constant((Ln,0.5*theta,0.,0.))
values = alpha.values()
Ln2 = values[0]
Delta = values[1]
ab = values[2]
at = values[3]

#Ln2 = Ln
#Delta = 0.5*theta
#ab = 0.
#at = 0.

# calculations
cosdt = cos(atan2(delta*theta,1.))
sindt = sin(atan2(delta*theta,1.))

# redefine the boundary conditions
top_left.deformation = Expression(('x[0]-sindt','1.'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_left(x, on_boundary):
	return on_polygon(x,top_left)
top_mid.deformation = Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt))  -  sindt','ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln)  +  1.'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_mid(x, on_boundary):
	return on_polygon(x,top_mid)
top_right.deformation = Expression(('x[0] + Ln2-Ln - (1.-theta+Delta) * sindt','1. - theta + Delta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_top_right(x, on_boundary):
	return on_polygon(x,top_right)
right_top.deformation = Expression(('((x[1]-0.5*theta)/(1-theta)) *  (  -sindt*(1-theta) )  +  Ln2+Lr-Delta*sindt','x[1]+ Delta-0.5*theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_right_top(x, on_boundary):
	return on_polygon(x,right_top)
right_bottom.deformation = Expression(('((x[1]+0.5*theta)/(theta)) *  (  -sindt*(theta) )  +  Ln2+Lr-(Delta-theta)*sindt','x[1]+ Delta-0.5*theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_right_bottom(x, on_boundary):
	return on_polygon(x,right_bottom)
mid_right.deformation = Expression(('x[0] + Ln2-Ln - Delta * sindt','Delta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_right(x, on_boundary):
	return on_polygon(x,mid_right)
mid_left.deformation = Expression(('-x[1]*sindt','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_left(x, on_boundary):
	return on_polygon(x,mid_left)
mid_bottom.deformation = Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta) * sindt)) ','at*(x[0]/Ln)*(x[0]/Ln)  +  (Delta  - at)  *  (x[0]/Ln)'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_mid_bottom(x, on_boundary):
	return on_polygon(x,mid_bottom)
left.deformation = Expression(('x[0]-x[1]*sindt','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_left(x, on_boundary):
	return on_polygon(x,left)
bottom_left.deformation = Expression(('x[0]','x[1]'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_left(x, on_boundary):
	return on_polygon(x,bottom_left)
bottom_mid.deformation = Expression(('(x[0]/Ln)  *  ((Ln2 - (Delta-theta) * sindt))','ab*(x[0]/Ln)*(x[0]/Ln)  +  (Delta-theta  - ab)  *  (x[0]/Ln)'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_mid(x, on_boundary):
	return on_polygon(x,bottom_mid)
bottom_right.deformation = Expression(('x[0] + Ln2-Ln - (Delta-theta) * sindt','Delta-theta'),degree=1,sindt=sindt,cosdt=cosdt,Ln=Ln,Ln2=Ln2,Lr=Lr,theta=theta,Delta=Delta,ab=ab,at=at)
def boundary_bottom_right(x, on_boundary):
	return on_polygon(x,bottom_right)

# define a vector with the boundary conditions
boundary_edges = [boundary_top_left,boundary_top_mid,boundary_top_right,boundary_right_top,boundary_right_bottom,boundary_mid_right,boundary_mid_left,boundary_mid_bottom,boundary_left,boundary_bottom_left,boundary_bottom_mid,boundary_bottom_right]

bcs = []
for i in range(0,edges_number):
	bc = DirichletBC(W,(edges[i]).deformation, boundary_edges[i])
	bcs.append(bc)


# compute the deformation 
psit=TrialFunction(W)
vt=TestFunction(W)
a = inner(grad(psit),grad(vt)) *dx
psi=Function(W, name='psi')
solve(lhs(a)==rhs(a), psi, bcs)

# compute the displacement
id = project(Identity2(),W)
displacement_psi = project(psi-id,W)

T = grad (psi)

#                                                 end of calculating the deformation                       

print ("********** L = %f + %f + %f, H = 1, theta = %f" % (Ll, Ln2, Lr, theta), flush=True)
print ("********** a = (%f, %f, %f, %f), delta = %f" % (a1, a2, a4, a4, delta), flush=True)

GA = Constant (((1,delta), (0,1)))
GB = Constant (((1,-delta), (0,1)))
G = chi_a*GA + chi_b*GB

S1 = Constant(((1,-delta*theta/sqrt(1+delta**2+theta**2)),(0,1/sqrt(1+delta**2+theta**2))))
S2 = Constant(((0,(2*delta*theta-delta)/sqrt(1+delta**2+theta**2)),(0,0)))

def energy_density (u, T, G, a1, a2, a3, a4):
	F = ( Identity(2) + S2 + grad(u)* inv(T) ) * inv(G)
	C = F.T*F
	return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(T))


# Total potential energy and derivatives
Edens = energy_density (u, T, G, a1, a2, a3, a4)
E = Edens*dx

# Derivatives (directions are nameless, so they can be test function implicitly, use action() to plug in a trial function)
duE = derivative (E, u)
dpsiduE = derivative (E,psi)
F = derivative (E, u, v)
duduE = derivative (duE, u)

solve (F == 0, u, bcsopt)
startE = assemble(E)
print ("********** E = %f" % startE, flush=True)



#Ln2 = Function(U)
#Delta = Constant (0.5*theta)
#ab = Constant (0.)
#at = Constant (0.)
#DvE = derivative (E, alpha) - derivative (action (duE, p), alpha) # shape gradient using adjoint



file = XDMFFile ("iterates.xdmf")
file.parameters ["flush_output"] = True
file.parameters ["functions_share_mesh"] = True
file.parameters ["rewrite_function_mesh"] = False

# Save a minimization step
#def output (u, deform, g, it):
#	j = assemble (J) # only for output
#	ng = norm (g)
#	if (it % save_each == 0):
#		file.write (deform, it)
#		file.write (u, it)
#	print ("**** iteration %03d, objective = %016.12g, gradient = %016.12g" % (it, j, ng))
#	sys.stdout.flush()
#	return

# Target functional and shape derivative    
#def Jhat (npv):
#	npa_to_fun (npv, v)
#	solve (action (duduE, uu) == 0, u, ubc)
#	j = assemble (J)
#	return j

#it = 0
#def DJhat (npv):
#	npa_to_fun (npv, v)
#	solve (action (duduE, uu) == 0, u, ubc)
#	p.assign (u)
#	g = Function (V)
#	assemble (DvJ, tensor=g.vector())
#	npg = fun_to_npa (g)
#	global it
#	it = it + 1
#	output (u, v, g, it)
#	return npg



# Minimization
# res = optimize.minimize (Jhat, npv, method="L-BFGS-B", jac=DJhat, bounds=numpy.transpose ([nplb, npub]), options={'disp' : True, 'ftol' : ftol, 'gtol' : gtol})



# safe displacement
vtkfile = File('deformation_needle/const.pvd')
vtkfile << chi_test
vtkfile = File('deformation_needle/displacement_psi.pvd')
vtkfile << displacement_psi
vtkfile = File('deformation_needle/u.pvd')
vtkfile << u