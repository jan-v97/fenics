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
	for i in range(0,1):
		x0 = ((edge.data)[i]).x()
		y0 = ((edge.data)[i]).y()
		x1 = ((edge.data)[edge.pointnumber-1]).x()
		y1 = ((edge.data)[edge.pointnumber-1]).y()
		# x near (x0,y0)
		if (near(x[0],x0,tol) and near(x[1],y0,tol)):
			if ((closed=='c')):
				return True
			elif (closed =='h'):
				return True
			else: 
				return False
		# x near(x1,y1)
		elif (near(x[0],x1,tol) and near(x[1],y1,tol)):
			if ((closed=='c')):
				return True
			elif (closed =='h'):
				return False
			else: 
				return False
		elif (abs(x1-x0)<abs(y1-y0)):
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

def get_points(L1,L2,resolution):
	L = L1+L2
	h = 4./resolution
	num_points_1 = int(L1/h)
	num_points_2 = int(L2/h)
	num_points_3 = int((L1-0.75)/h)
	lt_points,lb_points,bl_points,br_points,lt2_points,lb2_points,bl2_points,br2_points = [],[],[],[],[],[],[],[]
	for i in range(0,num_points_2):
		lt_points.append(dolfin.Point(0.,L1+L2-0.25-i*h))
		lt2_points.append(dolfin.Point(1.,1.+L1+L2-0.25-i*h))
		br_points.append(dolfin.Point(L1+i*h,0.))
		br2_points.append(dolfin.Point(1.+L1+i*h,1.))
	lt_points.append(dolfin.Point(0.,L1-0.75))
	lt2_points.append(dolfin.Point(1.,1.+L1-0.75))
	br_points.append(dolfin.Point(L1+L2,0.))
	br2_points.append(dolfin.Point(1.+L1+L2,1.))
	for i in range(0,num_points_1):
		bl_points.append(dolfin.Point(i*h,0.))
		bl2_points.append(dolfin.Point(1.+i*h,1.))
	bl_points.append(dolfin.Point(L1,0.))
	bl2_points.append(dolfin.Point(1.+L1,1.))
	for i in range(0,num_points_3):
		lb_points.append(dolfin.Point(0.,L1-0.75-i*h))
		lb2_points.append(dolfin.Point(1.,1.+L1-0.75-i*h))
	lb_points.append(dolfin.Point(0.,0.))
	lb2_points.append(dolfin.Point(1.,1.))
	return lt_points,lb_points,bl_points,br_points,lt2_points,lb2_points,bl2_points,br2_points
	
#                                                setting up the mesh                                         
def get_mesh(L1,L2,theta_t,theta_r,resolution):

	# input edges
	lt_points,lb_points,bl_points,br_points,lt2_points,lb2_points,bl2_points,br2_points = get_points(L1,L2,resolution)
	lt = edgeinput(lt_points)
	lb = edgeinput(lb_points)
	tl = edgeinput([dolfin.Point(0.5, 0.25+L1+L2),dolfin.Point(0., -0.25+L1+L2)])
	tr = edgeinput([dolfin.Point(1., 0.75+L1+L2),dolfin.Point(0.5, 0.25+L1+L2)])
	mt = edgeinput([dolfin.Point(0.5, 0.25+L1),dolfin.Point(0.5, 0.25+L1+L2)])
	lt2 = edgeinput(lt2_points)
	lb2 = edgeinput(lb2_points)
	c4 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(0.5, 0.25+L1)])
	c3 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(1., 1.)])
	c2 = edgeinput([dolfin.Point(0.75, 0.25),dolfin.Point(L1, 0.5)])
	c1 = edgeinput([dolfin.Point(0., 0.),dolfin.Point(0.75, 0.25)])
	bl = edgeinput(bl_points)
	br = edgeinput(br_points)
	mr = edgeinput([dolfin.Point(L1, 0.5),dolfin.Point(L1+L2+0.5, 0.5)])
	rb = edgeinput([dolfin.Point(L1+L2, 0.),dolfin.Point(L1+L2+0.5, 0.5)])
	rt = edgeinput([dolfin.Point(L1+L2+0.5, 0.5),dolfin.Point(L1+L2+1., 1.)])
	br2 = edgeinput(br2_points)
	bl2 = edgeinput(bl2_points)

	# define a vector with the edges
	edges = [lt,lb,tl,tr,mt,lt2,lb2,c4,c3,c2,c1,bl,br,mr,rb,rt,br2,bl2]
	edges_number = len(edges)

	# define the complete domain
	domain_complete = subdomaininput([lt,lb,bl,br,rb,rt,br2,bl2,lb2,lt2,tr,tl],[0,0,0,0,0,0,1,1,1,1,0,0],0)

	# input subdomains
	domain_left = subdomaininput([lt,lb,c1,c4,mt,tl],[0,0,0,0,0,0],0)
	domain_top = subdomaininput([c3,lb2,lt2,tr,mt,c4],[0,1,1,0,1,1],0)
	domain_right = subdomaininput([c2,mr,rt,br2,bl2,c3],[0,0,0,1,1,1],0)
	domain_bottom = subdomaininput([bl,br,rb,mr,c2,c1],[0,0,0,1,1,1],0)
	#domain_left.printpolygon()
	#print("\n")
	#domain_top.printpolygon()
	#print("\n")
	#domain_right.printpolygon()
	#print("\n")
	#domain_bottom.printpolygon()
	#print("\n")
	#lb2.print()

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
	return mesh, edges, edges_number, chi_a, chi_b, chi_c, chi_d, chi_test,mesh.num_vertices()




#                                                 calculating the deformation                                
def get_deformation(L1,L2,theta_t,theta_r,phi,alpha,resolution,delta_t,delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,edges,edges_number,V,x):
	# components of basis vectors
	x1_x = cos(alpha)
	x1_y = sin(alpha)
	x2_x = sin(alpha)
	x2_y = -cos(alpha)
	y1_x = cos(phi)*sqrt(2)
	y1_y = sin(phi)*sqrt(2)
	# geometrical calculations
	A = sqrt((y1_x-(0.75+pr))**2+(y1_y-(0.25+pt))**2)
	C = sin(alpha-phi)*sqrt(2)
	B = sin(alpha-atan2(y1_y-(0.25+pt),y1_x-(0.75+pr)))*A

	point_a_x = 0.75+pr + (Lt+L2) * x1_x - (theta_t*C-(B+delta_t)) * x2_x  + (theta_t-1)*y1_x
	point_a_y = 0.25+pt + (Lt+L2) * x1_y - (theta_t*C-(B+delta_t)) * x2_y  + (theta_t-1)*y1_y
	point_b_x = 0.75+pr + Lt * x1_x + (B+delta_t) * x2_x  -  y1_x
	point_b_y = 0.25+pt + Lt * x1_y + (B+delta_t) * x2_y  -  y1_y
	point_c_x = 0.75+pr + (Lt+L2) * x1_x - (theta_t*C-(B+delta_t)) * x2_x
	point_c_y = 0.25+pt + (Lt+L2) * x1_y - (theta_t*C-(B+delta_t)) * x2_y
	point_d_x = 0.75+pr + (Lt+L2) * x1_x - (theta_t*C-(B+delta_t)) * x2_x  + theta_t*y1_x
	point_d_y = 0.25+pt + (Lt+L2) * x1_y - (theta_t*C-(B+delta_t)) * x2_y  + theta_t*y1_y
	point_e_x = 0.75+pr + Lt * x1_x - (theta_t*C-(B+delta_t)) * x2_x
	point_e_y = 0.25+pt + Lt * x1_y - (theta_t*C-(B+delta_t)) * x2_y
	point_f_x = 0.75+pr + Lt * x1_x + (B+delta_t) * x2_x
	point_f_y = 0.25+pt + Lt * x1_y + (B+delta_t) * x2_y
	point_g_y = 0.25+pt + delta_r - theta_r * sin(phi)*sqrt(2)
	point_h_x = Lr+L2+theta_r*cos(phi)*sqrt(2)
	point_h_y = 0.25+pt + delta_r
	point_i_x = Lr+L2+cos(phi)*sqrt(2)
	point_i_y = 0.25+pt + delta_r+(1-theta_r)*sin(phi)*sqrt(2)
	point_j_x = Lr+cos(phi)*sqrt(2)
	point_j_y = 0.25+pt + delta_r+(1-theta_r)*sin(phi)*sqrt(2)

	edges[0].deformation = project(as_vector((((x[1]-(-0.25+L1+L2))/(-(L2+0.5)))  *   (  point_b_x  -  point_a_x)  +  point_a_x ,((x[1]-(-0.25+L1+L2))/(-(L2+0.5)))  *   (  point_b_y  -  point_a_y)  +  point_a_y)),V)
	def boundary_lt(x, on_boundary):
		return on_polygon(x,edges[0], closed='c')

	edges[1].deformation = project(as_vector((tlb*((x[1])/(L1-0.75))*((x[1])/(L1-0.75))  +  (point_b_x-tlb)  *  ((x[1])/(L1-0.75)) , ((x[1])/(L1-0.75))*point_b_y)),V)
	def boundary_lb(x, on_boundary):
		return on_polygon(x,edges[1], closed='o')

	edges[2].deformation = project(as_vector(((x[0]/0.5)  *  (  point_c_x  -  point_a_x  )  +  point_a_x , (x[0]/0.5)  *  (  point_c_y  -  point_a_y  )  +  point_a_y)),V)
	def boundary_tl(x, on_boundary):
		return on_polygon(x,edges[2], closed='o')

	edges[3].deformation = project(as_vector((((x[0]-0.5)/0.5)  *  (  point_d_x  -  point_c_x  )  +  point_c_x,((x[0]-0.5)/0.5)  *  (  point_d_y  -  point_c_y  )  +  point_c_y)),V)
	def boundary_tr(x, on_boundary):
		return on_polygon(x,edges[3], closed='h')

	edges[4].deformation = project(as_vector((((x[1]-(0.25+L1+L2))/(-L2))  *   (  point_e_x  -  point_c_x)  +  point_c_x,((x[1]-(0.25+L1+L2))/(-L2))  *   (  point_e_y  -  point_c_y)  +  point_c_y)),V)
	def boundary_mt(x, on_boundary):
		return on_polygon(x,edges[4], closed='c')

	edges[5].deformation = project(as_vector((((x[1]-(0.75+L1+L2))/(-(L2+0.5)))  *   (  point_f_x  -  point_d_x)  +  point_d_x,((x[1]-(0.75+L1+L2))/(-(L2+0.5)))  *   (  point_f_y  -  point_d_y)  +  point_d_y)),V)
	def boundary_lt2(x, on_boundary):
		return on_polygon(x,edges[5], closed='o')

	edges[6].deformation = project(as_vector((tlb*((x[1]-1.)/(L1-0.75))*((x[1]-1.)/(L1-0.75))   +   (point_f_x  -  cos(phi)*sqrt(2)  -  tlb)*((x[1]-1.)/(L1-0.75))  +  cos(phi)*sqrt(2) , ((x[1]-1.)/(L1-0.75))  *  (point_f_y  -  sin(phi)*sqrt(2))  +  sin(phi)*sqrt(2))),V)
	def boundary_lb2(x, on_boundary):
		return on_polygon(x,edges[6], closed='c')

	edges[7].deformation = project(as_vector((tc4*((x[1]-0.25)/(L1))*((x[1]-0.25)/(L1))  +  (point_e_x  -  (0.75+pr)  -  tc4)*((x[1]-0.25)/(L1))  +  0.75+pr , ((x[1]-0.25)/(L1))  *  (  point_e_y  -  (0.25+pt)  )  +  0.25+pt)),V)
	def boundary_c4(x, on_boundary):
		return on_polygon(x,edges[7], closed='o')

	edges[8].deformation = project(as_vector((tc3*((x[1]-0.25)/0.75)*((x[1]-0.25)/0.75)  +  (cos(phi)*sqrt(2)  -  (0.75+pr)  -  tc3)*((x[1]-0.25)/0.75)  +  0.75+pr,((x[1]-0.25)/0.75)  *  (  sin(phi)*sqrt(2)  -  (0.25+pt)  )  +  0.25+pt)),V)
	def boundary_c3(x, on_boundary):
		return on_polygon(x,edges[8], closed='o')

	edges[9].deformation = project(as_vector((((x[0]-0.75)/(L1-0.75))  *  (  Lr  -  (0.75+pr)  )  +  0.75+pr,tc2*((x[0]-0.75)/(L1-0.75))*((x[0]-0.75)/(L1-0.75))  +  (0.25+pt+delta_r - (0.25+pt) - tc2)*((x[0]-0.75)/(L1-0.75))  +  0.25+pt)),V)
	def boundary_c2(x, on_boundary):
		return on_polygon(x,edges[9], closed='o')

	edges[10].deformation = project(as_vector(((x[0]/0.75)*(0.75+pr),tc1*(x[0]/0.75)*(x[0]/0.75)+(0.25+pt-tc1)*(x[0]/0.75))),V)
	def boundary_c1(x, on_boundary):
		return on_polygon(x,edges[10], closed='c')

	edges[11].deformation = project(as_vector(((x[0]/L1)*(Lr),tbl*(x[0]/L1)*(x[0]/L1)  +  (point_g_y  -  tbl)*(x[0]/L1))),V)
	def boundary_bl(x, on_boundary):
		return on_polygon(x,edges[11], closed='o')

	edges[12].deformation = project(as_vector((x[0]+(Lr-L1),point_g_y)),V)
	def boundary_br(x, on_boundary):
		return on_polygon(x,edges[12], closed='h')

	edges[13].deformation = project(as_vector((((x[0]-L1)/(L2+0.5))  *  (L2+theta_r*(cos(phi)*sqrt(2)))  +  Lr,point_h_y)),V)
	def boundary_mr(x, on_boundary):
		return on_polygon(x,edges[13], closed='c')

	edges[14].deformation = project(as_vector(((x[1]/(0.5))  *   theta_r*cos(phi)*sqrt(2)+ Lr+L2,((x[1])/(0.5))  *  (point_h_y  - point_g_y)  + point_g_y)),V)
	def boundary_rb(x, on_boundary):
		return on_polygon(x,edges[14], closed='h')

	edges[15].deformation = project(as_vector((((x[1]-0.5)/(0.5))  * (1-theta_r)*(cos(phi)*sqrt(2))  +  Lr+L2+theta_r*(cos(phi)*sqrt(2)),((x[1]-0.5)/(0.5))  *  (point_i_y - point_h_y) + point_h_y)),V)
	def boundary_rt(x, on_boundary):
		return on_polygon(x,edges[15], closed='o')

	edges[16].deformation = project(as_vector((((x[0]-(L1+1.))/(L2))  *  L2  +  Lr+cos(phi)*sqrt(2),point_j_y)),V)
	def boundary_br2(x, on_boundary):
		return on_polygon(x,edges[16], closed='c')

	edges[17].deformation = project(as_vector((((x[0]-1.)/L1)*(Lr)+cos(phi)*sqrt(2),tbl*((x[0]-1.)/L1)*((x[0]-1.)/L1)  +  (point_j_y  -  (sin(phi)*sqrt(2))  -  tbl)*((x[0]-1.)/L1)  +  sin(phi)*sqrt(2))),V)
	def boundary_bl2(x, on_boundary):
		return on_polygon(x,edges[17], closed='o')

	
	# define a vector with the boundary conditions
	boundary_edges = [boundary_lt,boundary_lb,boundary_tl,boundary_tr,boundary_mt,boundary_lt2,boundary_lb2,boundary_c4,boundary_c3,boundary_c2,boundary_c1,boundary_bl,boundary_br,boundary_mr,boundary_rb,boundary_rt,boundary_br2,boundary_bl2]

	bcs = []
	for i in range(0,edges_number):
		bc = DirichletBC(V,(edges[i]).deformation, boundary_edges[i])
		bcs.append(bc)


	# compute the deformation 
	print ("******* compute the deformation from computational domain to fundamental cell:")
	psit=TrialFunction(V)
	vt=TestFunction(V)
	a = inner(grad(psit),grad(vt)) *dx
	psi=Function(V, name='psi')
	solve(lhs(a)==rhs(a), psi, bcs)

	# compute the displacement
	id = project(Identity2(),V)
	dpsi = Function(V, name='dpsi')
	dpsi.assign(project(psi-id,V))
	return psi,dpsi


#                                                 the programm                                        
def do_shape_opt(L1,L2,theta_t,theta_r,phi,alpha,resolution,a1,a2,a3,a4,sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl,G_tl,G_tr,G_rt,G_rb,taylor_testing,ftol,gtol,method,maxiter,mesh, edges, edges_number, chi_a, chi_b, chi_c, chi_d):

	x = SpatialCoordinate(mesh)

	class PeriodicBoundary (SubDomain):
		# bottom_left boundary is target domain
		def inside (self, x, on_boundary): 
			return bool ( (on_polygon(x,edges[0]) and on_boundary) or	(on_polygon(x,edges[1]) and on_boundary)  or (on_polygon(x,edges[11]) and on_boundary) or	(on_polygon(x,edges[12]) and on_boundary) )
		# Map top_right boundary to bottom_left boundary
		def map (self, x, y):
			y[0] = x[0]-1.
			y[1] = x[1]-1.

	# create the function spaces
	U = FunctionSpace (mesh, "CG", 1)
	V = VectorFunctionSpace (mesh, "CG", 1, constrained_domain=PeriodicBoundary())
	W = VectorFunctionSpace(mesh, 'CG', 1)

	u = Function (V)
	v = TestFunction(V)


	#parameters for the shape optimization
	#shift of needles
	Delta_t = Constant(sDelta_t)
	Delta_r = Constant(sDelta_r)
	#stretching of needle
	Lt = Constant(sLt)
	Lr = Constant(sLr)
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

	psi, dpsi = get_deformation(L1,L2,theta_t,theta_r,phi,alpha,resolution,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,edges,edges_number,W,x)
	

	GA = array_to_const_mat(G_tl)
	GB = array_to_const_mat(G_tr)
	GC = array_to_const_mat(G_rt)
	GD = array_to_const_mat(G_rb)
	G = chi_a*GA+chi_b*GB+ chi_c*GC+ chi_d*GD


	def energy_density (u, psi, G, a1, a2, a3, a4):
		F = ( Identity(2) +  grad(u)* inv(grad(psi)) ) * inv(G)
		C = F.T*F
		return (a1*(tr(C))**2 + a2*det(C) - a3*ln(det(C)) + a4*(C[0,0]**2+C[1,1]**2) - (4*a1+a2+2*a4))*abs(det(grad(psi)))


	# Total potential energy and derivatives
	Edens = energy_density (u, psi, G, a1, a2, a3, a4)
	E = Edens*dx
	#E = inner((Identity(2)+grad(u)* inv(grad(psi)))* inv(G),(Identity(2)+grad(u)* inv(grad(psi)))* inv(G))*dx

	def boundary_opt(x, on_boundary):
		return near(x[0],0,tol) and (x[1]< 9/resolution)

	zero = Constant ((0,0))
	dbcopt = DirichletBC (V, zero, boundary_opt)
	bcsopt = [dbcopt]

	F = derivative (E, u, v)
	print ("******* compute the elastic deformation:")
	solve (F == 0, u, bcsopt)
	#startE = assemble(E)
	#print ("********** E_start = %f" % startE, flush=True)
	u = project (u,V)
	Energy = assemble(E)

	def iter_cb(m):
		#global it
		#it = it + 1
		print ("m = ", m)

	def derivative_cb(j, dj, m):
		vec_m = m_to_array(m)
		global it
		global string_time
		print ("dj_cb, it: ",it, " m: ",vec_m)
		if (it%1==0):
			print_and_write_sol(True,True,j,vec_m[0],vec_m[1],vec_m[2],vec_m[3],vec_m[4],vec_m[5],vec_m[6],vec_m[7],vec_m[8],vec_m[9],vec_m[10],vec_m[11],0,0,0,it,string_time,0)
		it = it + 1

	def derivative_cb_1(j, dj, m):
		vec_m = m_to_array(m)
		global it
		global string_time
		print ("dj_cb, it: ",it, " m: ",vec_m)
		if (it%1==0):
			print_and_write_sol(True,True,j,vec_m[0],0,0,0,0,0,0,0,0,0,0,0,0,0,0,it,string_time,0)
		it = it + 1


	controls = [Control(Delta_t),Control(Delta_r),Control(Lt),Control(Lr),Control(pt),Control(pr),Control(tc1),Control(tc2),Control(tc3),Control(tc4),Control(tlb),Control(tbl)]
	h = [Constant(1.5),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1),Constant(0.1)]
	test = [Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl]

	if (taylor_testing == 0):
		Ehat = ReducedFunctional(Energy, controls)
		conv_rateL = taylor_test(Ehat, test, h)
		datei = open('needles/taylor.txt','a')
		datei.write('\n taylor_testing: {:2d} \n'.format(taylor_testing))
		datei.write(str(conv_rateL))
	
		rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl

	elif (1<=taylor_testing<=12):
		Ehat = ReducedFunctional(Energy, [controls[taylor_testing-1]])
		conv_rateL = taylor_test(Ehat, [test[taylor_testing-1]], h[taylor_testing-1])
		datei = open('needles/taylor.txt','a')
		datei.write('\n taylor_testing: {:2d} \n'.format(taylor_testing))
		datei.write(str(conv_rateL))
		
		rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl

	elif (14<=taylor_testing<=25):
		Ehat = ReducedFunctional(Energy, [controls[taylor_testing-14]],derivative_cb_post = derivative_cb_1)
		min = minimize (Ehat, method = method, tol = 1e-12, options = {'disp': True, 'ftol':ftol, 'gtol': gtol,'maxiter':maxiter}, callback = iter_cb)
		returns = [Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl]
		returns[taylor_testing-14] = min
		rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = returns
		

	elif (taylor_testing==13):
		
		rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl
		return Energy,float(rDelta_t),float(rDelta_r),float(rLt),float(rLr),float(rpt),float(rpr),float(rtc1),float(rtc2),float(rtc3),float(rtc4),float(rtlb),float(rtbl),dpsi,u

	else:
		bounds = [[-1000,-1000,0.75,0.75,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000],[1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000]]
		Ehat = ReducedFunctional(assemble(E), controls,derivative_cb_post = derivative_cb)
		rDelta_t,rDelta_r,rLt,rLr,rpt,rpr,rtc1,rtc2,rtc3,rtc4,rtlb,rtbl = minimize (Ehat, method = method, tol = 1e-12,bounds=bounds, options = {'disp': True, 'ftol':ftol, 'gtol': gtol,'maxiter':maxiter}, callback = iter_cb)

	return assemble(E),float(rDelta_t),float(rDelta_r),float(rLt),float(rLr),float(rpt),float(rpr),float(rtc1),float(rtc2),float(rtc3),float(rtc4),float(rtlb),float(rtbl),dpsi,u

#                     print and write                            

def print_and_write(prin,write,L1,L2,theta_t,theta_r,phi,alpha,resolution,a1,a2,a3,a4,sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl,G_tl,G_tr,G_rt,G_rb,string_time):

	if prin:
		print("***************    ")
		print(time.time())
		print("     *************** ")
		print('\n  parameters for computational domain and elasticity')
		print('\n{:^5} {:^5} {:^9} {:^10} {:^9} {:^10}'.format('L1','L2','a1', 'a2', 'a3', 'a4'))
		print('\n{:1.3f} {:1.3f} {:2.6f} {:2.6f} {:2.6f} {:2.6f}'.format(L1, L2, a1, a2, a3, a4))
		print('\n  parameters for the twin structure')
		print('\n{:^7} {:^7} {:^7} {:^7}'.format('theta_t','theta_r','phi', 'alpha'))
		print('\n{:1.5f} {:1.5f} {:1.5f} {:1.5f}'.format(theta_t,theta_r,phi,alpha))
		print('\n{:^16}   {:^16}   {:^16}   {:^16}'.format('G_tl','G_tr','G_rt', 'G_rb'))
		print('\n[[{:1.3f}, {:1.3f}],    [[{:1.3f}, {:1.3f}],    [[{:1.3f}, {:1.3f}],    [[{:1.3f}, {:1.3f}],'.format(G_tl[0][0],G_tl[0][1],G_tr[0][0],G_tr[0][1],G_rt[0][0],G_rt[0][1],G_rb[0][0],G_rb[0][1]))
		print('\n [{:1.3f}, {:1.3f}]]   [{:1.3f}, {:1.3f}]]   [{:1.3f}, {:1.3f}]]   [{:1.3f}, {:1.3f}]]'.format(G_tl[1][0],G_tl[1][1],G_tr[1][0],G_tr[1][1],G_rt[1][0],G_rt[1][1],G_rb[1][0],G_rb[1][1]))
		print('\n  starting parameters for shape optimization')
		print('\n{:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}'.format('Delta_t','Delta_r','Lt', 'Lr', 'pt', 'pr','c1','c2','c3', 'c4', 'tlb', 'tbl'))
		print('\n{:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} '.format(sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl))

	if write:
		datei = open("needles/"+string_time+"_log.txt",'a')
		datei.write("***************    ")
		datei.write(time.ctime())
		datei.write("     *************** ")
		datei.write('\n\n  parameters for computational domain and elasticity')
		datei.write('\n    {:^5} {:^5} {:^9} {:^10} {:^9} {:^10}'.format('L1','L2','a1', 'a2', 'a3', 'a4'))
		datei.write('\n    {:1.3f} {:1.3f} {:2.6f} {:2.6f} {:2.6f} {:2.6f}'.format(L1, L2, a1, a2, a3, a4))
		datei.write('\n\n  parameters for the twin structure')
		datei.write('\n    {:^7} {:^7} {:^7} {:^7}'.format('theta_t','theta_r','phi', 'alpha'))
		datei.write('\n    {:1.5f} {:1.5f} {:1.5f} {:1.5f}'.format(theta_t,theta_r,phi,alpha))
		datei.write('\n    {:^16}   {:^16}   {:^16}   {:^16}'.format('G_tl','G_tr','G_rt', 'G_rb'))
		datei.write('\n    [[{:1.3f}, {:1.3f}],   [[{:1.3f}, {:1.3f}],   [[{:1.3f}, {:1.3f}],   [[{:1.3f}, {:1.3f}],'.format(G_tl[0][0],G_tl[0][1],G_tr[0][0],G_tr[0][1],G_rt[0][0],G_rt[0][1],G_rb[0][0],G_rb[0][1]))
		datei.write('\n     [{:1.3f}, {:1.3f}]]    [{:1.3f}, {:1.3f}]]    [{:1.3f}, {:1.3f}]]    [{:1.3f}, {:1.3f}]]'.format(G_tl[1][0],G_tl[1][1],G_tr[1][0],G_tr[1][1],G_rt[1][0],G_rt[1][1],G_rb[1][0],G_rb[1][1]))
		datei.write('\n\n  starting parameters for shape optimization')
		datei.write('\n    {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15}'.format('Delta_t','Delta_r','Lt', 'Lr', 'pt', 'pr','c1','c2','c3', 'c4', 'tlb', 'tbl'))
		datei.write('\n    {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} '.format(sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl))
		datei.write('\n\n  results of shape optimization and details')
		datei.write('\n        {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^10} {:^10} {:^10} {:^3}'.format('E_end', 'Delta_t','Delta_r','Lt', 'Lr', 'pt', 'pr','c1','c2','c3', 'c4', 'tlb', 'tbl', 'time', 'res', 'verts', 'it'))
		datei.close()


def print_and_write_sol(prin,write,E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,time,resolution,verts,it,string_time,i):
	if prin:
		if not (verts==0):
			print('\n  results of shape optimization and details')
			print('\n\n{:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^15} {:^10} {:^10} {:^10} {:^3}'.format('E_end', 'Delta_t','Delta_r','Lt', 'Lr', 'pt', 'pr','c1','c2','c3', 'c4', 'tlb', 'tbl', 'time', 'res', 'verts', 'it'))
			print('\n{:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:9.0f} {:10d} {:10d} {:3d}'.format(E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,time,resolution,verts,it))

	if write:
		if not (verts==0):
			datei = open("needles/"+string_time+"_log.txt",'a')
			datei.write('\n {:3d}    {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:9.0f} {:10d} {:10d} {:3d}'.format(i,E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,time,resolution,verts,it))
		else:
			datei = open("needles/"+string_time+'_log.txt','a')
			datei.write('\n    {:3d} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e} {:.9e}'.format(it,E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl))
		datei.close()

#                                               input parameters, edges, bc, domains                                          

# input parameters for computational domain
L1 = 1.5
L2 = 5.
resolution = 2**8

# parameters for elastic energy
a1=11.562724; a2=-17.437087; a3=10.062913; a4=-9.375448

# parameters for Optimization
ftol = 1e-12
gtol = 1e-8
# method = 'CG'
method = 'L-BFGS-B'
maxiter = 30

# input parameters for the twin structure
phi = 0.25*pi
alpha = 0.5*pi
theta_t = 0.5
theta_r = 0.5
# theta_r = (theta_t-0.5)*(cos(alpha)*sin(alpha)*(cos(phi)/sin(phi))+cos(alpha)*cos(alpha))+0.5
delta = 0.1


G_tr = [[1 + delta*cos(alpha)*sin(alpha) ,-delta*cos(alpha)*cos(alpha)],[delta*sin(alpha)*sin(alpha),1-delta*cos(alpha)*sin(alpha)]]
G_tl = [[1-delta*cos(alpha)*sin(alpha) ,delta*cos(alpha)*cos(alpha)],[-delta*sin(alpha)*sin(alpha),1+delta*cos(alpha)*sin(alpha)]]
G_rt = [[1,delta],[0,1]]
G_rb = [[1,-delta],[0,1]]


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

prin = True
write = True

#sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl = 2.598051445e-01, 4.337388497e-02, 9.866990118e-01, 1.402109923e+00, 1.962614530e-01, 1.156560173e-01, -3.106693171e-01, -1.856464106e-02, -1.103661512e-01, 2.180654962e-01, -4.144822456e-01, -2.783244785e-02

string_time = time.strftime("%b_%d_%H_%M", time.gmtime())

print_and_write(prin,write,L1,L2,theta_t,theta_r,phi,alpha,resolution,a1,a2,a3,a4,sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl,G_tl,G_tr,G_rt,G_rb,string_time)

file = XDMFFile ("needles/"+string_time+"_file.xdmf")
file.parameters ["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.parameters ["rewrite_function_mesh"] = False
mesh, edges, edges_number, chi_a, chi_b, chi_c, chi_d, chi_test, verts = get_mesh(L1,L2,theta_t,theta_r,resolution)

maxi = 15
#delta_E = 1.
#E_alt=1
it = 0
i = 0
while ((i<maxi)&(not(it==2))):
	it = 0
	taylor_testing = -1
	set_working_tape(Tape())
	# 0 for taylor test in all controls
	# 1-12 for taylor test in control 1-12
	# 13 just computing deformation
	# 14-25 optimization in just one control
	# else shape opt
	start = time.time()
	E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl, dpsi, u = do_shape_opt(L1,L2,theta_t,theta_r,phi,alpha,resolution,a1,a2,a3,a4,sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl,G_tl,G_tr,G_rt,G_rb,taylor_testing,ftol,gtol,method,maxiter,mesh, edges, edges_number, chi_a, chi_b, chi_c, chi_d)
	end = time.time()

	# computing the deformation at the ende
	#if not (0 <= taylor_testing <= 14):
	#	set_working_tape(Tape())
	#	E_end,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl, chi_test, dpsi, u,verts = do_shape_opt(L1,L2,theta_t,theta_r,phi,alpha,resolution,a1,a2,a3,a4,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,G_tl,G_tr,G_rt,G_rb,13,ftol,gtol,method,maxiter)
	

	print_and_write_sol(prin,write,0.,Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl,end-start,resolution,verts,it,string_time,i)
	print("E_end:",E_end)

	sDelta_t,sDelta_r,sLt,sLr,spt,spr,stc1,stc2,stc3,stc4,stlb,stbl = Delta_t,Delta_r,Lt,Lr,pt,pr,tc1,tc2,tc3,tc4,tlb,tbl

	# safe displacement
	chi_test.rename("color","label")
	file.write(chi_test, i)
	dpsi.rename("psi","label")
	file.write(dpsi, i)
	u.rename("u","label")
	file.write(u, i)
	file.close()

	#delta_E = abs(E_alt - E_end)
	#E_alt = E_end
	i = i+1
