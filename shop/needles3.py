from fenics import *
from dolfin import *
from mshr import *
from fenics_adjoint import *
from pyadjoint.overloaded_type import create_overloaded_object
from math import *
import numpy
import scipy
import time
tol= 1E-14

#                                    defining classes for edges and domains                                       

# input of edges
# pointnumber     integer                                 Anzahl der Punkte auf der Kante, Start und Endpunkt
#                                                                   müssen mit angegeben werden
# data            Liste mit punktezahl-Dolfin Punkten     Angabe in Reihenfolge
# deformation     string                                  Information über Deformation auf der Kante
#                                                             '' for no deformation
class edgeinput:
	def __init__(self, data, deformation=0):
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
#    closed = 'c' (default)--> both endpoints are included
#    closed = 'h'          --> first endpoint include, second not
#    closed = 'o'          --> no endpoint is included
def on_polygon(x,edge,closed='c'):
	for i in range(0,edge.pointnumber-1):
		x0 = ((edge.data)[i]).x()
		y0 = ((edge.data)[i]).y()
		x1 = ((edge.data)[i+1]).x()
		y1 = ((edge.data)[i+1]).y()
		# x near (x0,y0)
		if (near(x[0],x0,tol) and near(x[1],y0,tol)):
			if (closed=='c'):
				return True
			elif (closed =='h'):
				return True
			else: 
				return False
		# x near(x1,y1)
		elif (near(x[0],x1,tol) and near(x[1],y1,tol)):
			if (closed=='c'):
				return True
			elif (closed =='h'):
				return False
			else: 
				return False
		elif near(x1,x0,tol):
			r = (x1 - x0) * (x[1] - y0) / (y1 - y0) + x0 - x[0] 
			if (near(r, 0.0, tol) and (((x[1] >= y0) and (x[1] <= y1)) or ((x[1] >= y1) and (x[1] <=y0)))):
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

def array_to_const_mat(G):
	return Constant(((G[0][0],G[0][1]),(G[1][0],G[1][1])))


def m_to_array(m):
	array = []
	for entry in m:
		array.append(float(entry))
	return array

#                                               input parameters, edges, bc, domains                                          

# input parameters for computational domain
L1 = 7.
L2 = 7.
resolution = 2**5

# parameters for elastic energy
a1=11.562724; a2=-17.437087; a3=10.062913; a4=-9.375448

# parameters for Optimization
ftol = 1e-8
gtol = 1e-6
method = 'L-BFGS-B'
maxiter = 50

# input parameters for the twin structure
theta_t = 0.5
theta_r = 0.5
phi = 0.25*pi
alpha = 0.5*pi
delta = 0.1


G_tl = [[1 + delta*cos(alpha)*sin(alpha) ,-delta*cos(alpha)*cos(alpha)],[delta*sin(alpha)*sin(alpha),1-delta*cos(alpha)*sin(alpha)]]
G_tr = [[1-delta*cos(alpha)*sin(alpha) ,delta*cos(alpha)*cos(alpha)],[-delta*sin(alpha)*sin(alpha),1+delta*cos(alpha)*sin(alpha)]]
G_rt = [[1,-delta],[0,1]]
G_rb = [[1,delta],[0,1]]


#startparameters for the shape optimization
#shift of needles
sDelta_t = 0.
sDelta_r = 0.
#stretching of needle
sLt = L1
sLr = L1
#displacement of midpoint
spt = 0.
spr = 0.
#quadratic parameters
stc1 = 0.
stc2 = 0.
stc3 = 0.
stc4 = 0.
stlb = 0.
stbl = 0.



#parameters for the shape optimization
#shift of needles
Delta_t = Constant(sDelta_t)
Delta_r = Constant(sDelta_r)
#stretching of needle
Lt = Constant(sLt)
Lr = Constant(sLt)
#displacement of midpoint
pt = Constant(spt)
pr = Constant(spr)
#quadratic parameters
tc1 = Constant(stc1)
tc2 = Constant(stc2)
tc3 = Constant(stc3)
tc4 = Constant(stc4)
tlb = Constant(stlb)
tbl = Constant(stbl)

#                                                setting up the mesh                                         


# input edges
lt = edgeinput([dolfin.Point(0., -0.25+L1+L2),dolfin.Point(0., -0.75+L1)])
lb = edgeinput([dolfin.Point(0., -0.75+L1),dolfin.Point(0., 0.)])
tl = edgeinput([dolfin.Point(0.5, 0.25+L1+L2),dolfin.Point(0., -0.25+L1+L2)])
trk = edgeinput([dolfin.Point(1., 0.75+L1+L2),dolfin.Point(0.5, 0.25+L1+L2)])
mt = edgeinput([dolfin.Point(0.5, 0.25+L1),dolfin.Point(0.5, 0.25+L1+L2)])
lt2 = edgeinput([dolfin.Point(1., 0.25+L1),dolfin.Point(1., 0.75+L1+L2)])
lb2 = edgeinput([dolfin.Point(1., 1.),dolfin.Point(1., 0.25+L1)])
c4 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(0.5, 0.25+L1)])
c3 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(1., 1.)])
c2 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(L1, 0.5)])
c1 = edgeinput([dolfin.Point(0., 0.),dolfin.Point(0.75, 0.25)])
bl = edgeinput([dolfin.Point(0., 0.),dolfin.Point(L1, 0.)])
br = edgeinput([dolfin.Point(L1, 0),dolfin.Point(L1+L2, 0)])
mr = edgeinput([dolfin.Point(L1, 0.5),dolfin.Point(L1+L2+0.5, 0.5)])
rb = edgeinput([dolfin.Point(L1+L2, 0.),dolfin.Point(L1+L2+0.5, 0.5)])
rt = edgeinput([dolfin.Point(L1+L2+0.5, 0.5),dolfin.Point(L1+L2+1., 1.)])
br2 = edgeinput([dolfin.Point(L1+L2+1., 1.),dolfin.Point(1.+L1, 1)])
bl2 = edgeinput([dolfin.Point(1+L1, 1.),dolfin.Point(1., 1.)])

# define a vector with the edges
edges = [lt,lb,tl,trk,mt,lt2,lb2,c4,c3,c2,c1,bl,br,mr,rb,rt,br2,bl2]
edges_number = len(edges)

# define the complete domain
domain_complete = subdomaininput([lt,lb,bl,br,rb,rt,br2,bl2,lb2,lt2,trk,tl],[0,0,0,0,0,0,0,0,0,0,0,0],0)

# input subdomains
domain_left = subdomaininput([lt,lb,c1,c4,mt,tl],[0,0,0,0,0,0],0)
domain_top = subdomaininput([c3,lb2,lt2,trk,mt,c4],[0,0,0,0,1,1],0)
domain_right = subdomaininput([c2,mr,rt,br2,bl2,c3],[0,0,0,0,0,1],0)
domain_bottom = subdomaininput([bl,br,rb,mr,c2,c1],[0,0,0,1,1,1],0)

# define a vector with the subdomains
subdomains = [domain_left,domain_top,domain_right,domain_bottom]
subdomain_number = len(subdomains)

# defining the domain, and the subdomains
domain = Polygon(domain_complete.get_polygon())
for i in range(0, subdomain_number):
	domain.set_subdomain (i+1, Polygon((subdomains[i]).get_polygon()))

# generat the mesh
mesh = generate_mesh (domain, resolution)
mesh = create_overloaded_object(mesh)
print ("Anzahl Knoten:", mesh.num_vertices())

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

chi_a = sudom_fct (sudom_arr, [0,1,0,0,0], X)
chi_b = sudom_fct (sudom_arr, [0,0,1,0,0], X)
chi_c = sudom_fct (sudom_arr, [0,0,0,1,0], X)
chi_d = sudom_fct (sudom_arr, [0,0,0,0,1], X)
chi_test = sudom_fct (sudom_arr, [0,1,2,3,4], X)


class PeriodicBoundary (SubDomain):
	# bottom_left boundary is target domain
	def inside (self, x, on_boundary): 
		return (near(x[0],0) or near(x[1],0)) 
	# Map top_right boundary to bottom_left boundary
	def map (self, x, y):
		y[0] = x[0]-1.
		y[1] = x[1]-1.

# create the function spaces
U = FunctionSpace (mesh, "CG", 1)
V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
W = VectorFunctionSpace(mesh, 'CG', 1)

u = Function (V, name='displacement')
v = TestFunction(V)



#                                                 the programm                                        



GA = array_to_const_mat(G_tl)
GB = array_to_const_mat(G_tr)
GC = array_to_const_mat(G_rt)
GD = array_to_const_mat(G_rb)
G = chi_a*GA+chi_b*GB+ chi_c*GC+ chi_d*GD


def energy_density (u,  G, a1, a2, a3, a4):
	F = ( Identity(2) +  grad(u) ) * inv(G)
	D = F.T*F
	return (a1*(tr(D))**2 + a2*det(D) - a3*ln(det(D)) + a4*(D[0,0]**2+D[1,1]**2) - (4*a1+a2+2*a4))


# Total potential energy and derivatives
Edens = energy_density (u, G, a1, a2, a3, a4)
E = Edens*dx
#E = inner((Identity(2)+grad(u)* inv(grad(psi)))* inv(G),(Identity(2)+grad(u)* inv(grad(psi)))* inv(G))*dx

def boundary_opt(x, on_boundary):
	return (on_polygon(x,edges[0]) and on_boundary) or	(on_polygon(x,edges[1]) and on_boundary)  or (on_polygon(x,edges[11]) and on_boundary) or	(on_polygon(x,edges[12]) and on_boundary) 

zero = Constant ((0,0))
dbcopt = DirichletBC (V, zero, boundary_opt)
bcsopt = [dbcopt]

F = derivative (E, u, v)
print ("******* compute the elastic deformation:")
solve (F == 0, u, bcsopt)
#startE = assemble(E)
#print ("********** E_start = %f" % startE, flush=True)

def iter_cb(m):
	#global it
	#it = it + 1
	print ("m = ", m)

def derivative_cb(j, dj, m):
	vec_m = m_to_array(m)
	print (vec_m)


controls = [Control(Delta_t),Control(Delta_r),Control(Lt),Control(Lr),Control(pt),Control(pr),Control(tc1),Control(tc2),Control(tc3),Control(tc4),Control(tlb),Control(tbl)]


Ehat = ReducedFunctional(assemble(E), controls,derivative_cb_post = derivative_cb)

#rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = minimize (Ehat, method = method, tol = 1e-12, options = {'disp': True, 'ftol':ftol, 'gtol': gtol,'maxiter':maxiter}, callback = iter_cb)


# safe displacement
file = XDMFFile ("needles/file.xdmf")
file.parameters["functions_share_mesh"] = True
chi_test.rename("color","label")
file.write(chi_test, 0)
#dpsi.rename("psi","label")
#file.write(dpsi, 0)
u.rename("u","label")
file.write(u, 0)
file.close()
