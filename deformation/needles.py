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
L1 = 2.
L2 = 2.
theta_t = 0.6
theta_r = 0.4
resolution = 40

#parameters for the deformation
#shift of needles
delta_t = 0.1
delta_r = 0.1
#stretching of needle
Lt = 3.
Lr = 2.5
#displacement of midpoint
pt = -0.1
pr = 0.1
#quadratic parameters
tc1 = -0.15
tc2 = -0.1
tc3 = -0.1
tc4 = 0.4
tlb = -0.2
tbl = 0.2


# input edges
lt = edgeinput([dolfin.Point(0., 0.25+L1+L2),dolfin.Point(0., -0.75+L1)],Expression(('x[0]+delta_t','x[1]+(Lt-L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
lb = edgeinput([dolfin.Point(0., -0.75+L1),dolfin.Point(0., 0.)],Expression(('tlb*((x[1])/(L1-0.75))*((x[1])/(L1-0.75))+(delta_t-tlb)*((x[1])/(L1-0.75))','((x[1])/(L1-0.75))*(Lt-0.75)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tlb=tlb))
tl = edgeinput([dolfin.Point(1-theta_t, 0.25+L1+L2),dolfin.Point(0., 0.25+L1+L2)],Expression(('x[0]+delta_t','x[1]+(Lt-L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
tr = edgeinput([dolfin.Point(1., 0.25+L1+L2),dolfin.Point(1-theta_t, 0.25+L1+L2)],Expression(('x[0]+delta_t','x[1]+(Lt-L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
mt = edgeinput([dolfin.Point(1-theta_t, 0.25+L1),dolfin.Point(1-theta_t, 0.25+L1+L2)],Expression(('x[0]+delta_t','x[1]+(Lt-L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
lt2 = edgeinput([dolfin.Point(1., 0.25+L1),dolfin.Point(1., 0.25+L1+L2)],Expression(('x[0]+delta_t','x[1]+(Lt-L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
lb2 = edgeinput([dolfin.Point(1., 1.),dolfin.Point(1., 0.25+L1)],Expression(('tlb*((x[1]-1.)/(L1-0.75))*((x[1]-1.)/(L1-0.75))+(delta_t-tlb)*((x[1]-1.)/(L1-0.75))+1.','((x[1]-1.)/(L1-0.75))*(Lt-0.75)+1'),degree=1,delta_t=delta_t,L1=L1,Lt=Lt,tlb=tlb))
c4 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(1-theta_t, 0.25+L1)],Expression(('tc4*((x[1]-0.25)/(L1))*((x[1]-0.25)/(L1))+(0.5-theta_t+delta_t-0.25-pr-tc4)*((x[1]-0.25)/(L1))+0.75+pr','((x[1]-0.25)/(L1))*(Lt-pt)+0.25+pt'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tc4=tc4,theta_t=theta_t))
c3 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(1., 1.)],Expression(('tc3*((x[1]-0.25)/0.75)*((x[1]-0.25)/0.75)+(0.25-pr-tc3)*((x[1]-0.25)/0.75)+0.75+pr','((x[1]-0.25)/0.75)*(0.75-pt)+0.25+pt'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tc3=tc3))
c2 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(L1, theta_r-0.25)],Expression(('((x[0]-0.75)/(L1-0.75))*(Lr-0.75-pr)+0.75+pr','tc2*((x[0]-0.75)/(L1-0.75))*((x[0]-0.75)/(L1-0.75))+(theta_r-0.5+delta_r-pt-tc2)*((x[0]-0.75)/(L1-0.75))+0.25+pt'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tc2=tc2,theta_r=theta_r))
c1 = edgeinput([dolfin.Point(0., 0.),dolfin.Point(0.75, 0.25)],Expression(('(x[0]/0.75)*(0.75+pr)','tc1*(x[0]/0.75)*(x[0]/0.75)+(0.25+pt-tc1)*(x[0]/0.75)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tc1=tc1))
bl = edgeinput([dolfin.Point(0., 0.),dolfin.Point(L1, -0.25)],Expression(('(x[0]/L1)*(Lr)','tbl*(x[0]/L1)*(x[0]/L1)+(-0.25+delta_r-tbl)*(x[0]/L1)'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tbl=tbl))
br = edgeinput([dolfin.Point(L1, -0.25),dolfin.Point(L1+L2, -0.25)],Expression(('x[0]+(Lr-L1)','x[1]+delta_r'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
mr = edgeinput([dolfin.Point(L1, theta_r-0.25),dolfin.Point(L1+L2, theta_r-0.25)],Expression(('x[0]+(Lr-L1)','x[1]+delta_r'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
rb = edgeinput([dolfin.Point(L1+L2, -0.25),dolfin.Point(L1+L2, theta_r-0.25)],Expression(('x[0]+(Lr-L1)','x[1]+delta_r'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
rt = edgeinput([dolfin.Point(L1+L2, theta_r-0.25),dolfin.Point(L1+L2, 0.75)],Expression(('x[0]+(Lr-L1)','x[1]+delta_r'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
br2 = edgeinput([dolfin.Point(L1+L2, 0.75),dolfin.Point(1+L1, 0.75)],Expression(('x[0]+(Lr-L1)','x[1]+delta_r'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr))
bl2 = edgeinput([dolfin.Point(1+L1, 0.75),dolfin.Point(1., 1.)],Expression(('((x[0]-1.)/L1)*(Lr)+1.','tbl*((x[0]-1.)/L1)*((x[0]-1.)/L1)+(-0.25+delta_r-tbl)*((x[0]-1.)/L1)+1.'),degree=1,delta_t=delta_t,delta_r=delta_r,L1=L1,Lt=Lt,Lr=Lr,pt=pt,pr=pr,tbl=tbl))

# define a vector with the edges
edges = [lt,lb,tl,tr,mt,lt2,lb2,c4,c3,c2,c1,bl,br,mr,rb,rt,br2,bl2]
edges_number = len(edges)


# define the boundary conditions
def boundary_lt(x, on_boundary):
	return on_polygon(x,lt)
def boundary_lb(x, on_boundary):
	return on_polygon(x,lb)
def boundary_tl(x, on_boundary):
	return on_polygon(x,tl)
def boundary_tr(x, on_boundary):
	return on_polygon(x,tr)
def boundary_mt(x, on_boundary):
	return on_polygon(x,mt)
def boundary_lt2(x, on_boundary):
	return on_polygon(x,lt2)
def boundary_lb2(x, on_boundary):
	return on_polygon(x,lb2)
def boundary_c4(x, on_boundary):
	return on_polygon(x,c4)
def boundary_c3(x, on_boundary):
	return on_polygon(x,c3)
def boundary_c2(x, on_boundary):
	return on_polygon(x,c2)
def boundary_c1(x, on_boundary):
	return on_polygon(x,c1)
def boundary_bl(x, on_boundary):
	return on_polygon(x,bl)
def boundary_br(x, on_boundary):
	return on_polygon(x,br)
def boundary_mr(x, on_boundary):
	return on_polygon(x,mr)
def boundary_rb(x, on_boundary):
	return on_polygon(x,rb)
def boundary_rt(x, on_boundary):
	return on_polygon(x,rt)
def boundary_br2(x, on_boundary):
	return on_polygon(x,br2)
def boundary_bl2(x, on_boundary):
	return on_polygon(x,bl2)

# define a vector with the boundary conditions
boundary_edges = [boundary_lt,boundary_lb,boundary_tl,boundary_tr,boundary_mt,boundary_lt2,boundary_lb2,boundary_c4,boundary_c3,boundary_c2,boundary_c1,boundary_bl,boundary_br,boundary_mr,boundary_rb,boundary_rt,boundary_br2,boundary_bl2]


# define the complete domain
domain_complete = subdomaininput([lt,lb,bl,br,rb,rt,br2,bl2,lb2,lt2,tr,tl],[0,0,0,0,0,0,0,0,0,0,0,0],0)

# input subdomains
domain_left = subdomaininput([lt,lb,c1,c4,mt,tl],[0,0,0,0,0,0],0)
domain_top = subdomaininput([c3,lb2,lt2,tr,mt,c4],[0,0,0,0,1,1],0)
domain_right = subdomaininput([c2,mr,rt,br2,bl2,c3],[0,0,0,0,0,1],0)
domain_bottom = subdomaininput([bl,br,rb,mr,c2,c1],[0,0,0,1,1,1],0)

# define a vector with the subdomains
subdomains = [domain_left,domain_top,domain_right,domain_bottom]
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

chi_l = sudom_fct (sudom_arr, [0,1,0,0,0], X)
chi_t = sudom_fct (sudom_arr, [0,0,1,0,0], X)
chi_r = sudom_fct (sudom_arr, [0,0,0,1,0], X)
chi_b = sudom_fct (sudom_arr, [0,0,0,0,1], X)
chi_test = sudom_fct (sudom_arr, [0,1,2,3,4], X)

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
vtkfile = File('needles/const.pvd')
vtkfile << chi_test
vtkfile = File('needles/displacement.pvd')
vtkfile << displacement
