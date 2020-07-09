from fenics import *
from dolfin import *
from mshr import *
import numpy

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



# input parameters for computational domain
Ln = 1.5
L = 4.
Ll = 1.
Lr = L - Ln - Ll
H = 1.0
theta = 0.25
tol= 1E-14
resolution = 50

# input kantenzüge
edge_number = 14
top_left = edgeinput([dolfin.Point(0.,1.0),dolfin.Point(-Ll, 1.0)],'hier wird deformiert')
top_mid = edgeinput([dolfin.Point(Ln,1-0.5*theta),dolfin.Point(0., 1.)],'')
top_right = edgeinput([dolfin.Point(Ln+Lr,1-0.5*theta),dolfin.Point(Ln, 1-0.5*theta)],'')
right_top = edgeinput([dolfin.Point(Ln+Lr,0.5*theta),dolfin.Point(Ln+Lr, 1.-0.5*theta)],'')
right_bottom = edgeinput([dolfin.Point(Ln+Lr,-0.5*theta),dolfin.Point(Ln+Lr, 0.5*theta)],'')
mid_right = edgeinput([dolfin.Point(Ln,0.5*theta ),dolfin.Point(Ln+Lr,0.5*theta )],'')
mid_left = edgeinput([dolfin.Point(0.,0.),dolfin.Point(0., 1.)],'')
mid_top = edgeinput([dolfin.Point(Ln,0.5*theta ),dolfin.Point(Ln,1-0.5*theta )],'')
mid_bottom = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln, 0.5*theta)],'')
mid_bottom2 = edgeinput([dolfin.Point(Ln,-0.5*theta),dolfin.Point(Ln, 0.5*theta)],'')
left = edgeinput([dolfin.Point(-Ll,1.),dolfin.Point(-Ll, 0.)],'')
bottom_left = edgeinput([dolfin.Point(-Ll,0.),dolfin.Point(0., 0.)],'')
bottom_mid = edgeinput([dolfin.Point(0.,0.),dolfin.Point(Ln,-0.5*theta )],'')
bottom_right = edgeinput([dolfin.Point(Ln,-0.5*theta ),dolfin.Point(Ln+Lr,-0.5*theta )],'')

# input domain and subdomains
domain_complete = subdomaininput([top_left,left,bottom_left, bottom_mid, bottom_right, right_bottom, right_top, top_right, top_mid],[0,0,0,0,0,0,0,0,0],0)
domain_left = subdomaininput([top_left,left,bottom_left,mid_left],[0,0,0,0],0)
domain_mid_top = subdomaininput([mid_bottom,mid_top,top_mid,mid_left],[0,0,0,1],0)
domain_mid_bottom = subdomaininput([bottom_mid,mid_bottom2,mid_bottom],[0,0,1],0)
domain_right_top = subdomaininput([mid_right,right_top,top_right,mid_top],[0,0,0,1],0)
domain_right_bottom = subdomaininput([bottom_right,right_bottom,mid_right,mid_bottom2],[0,0,1,1],0)
subdomains = [domain_left,domain_mid_top,domain_mid_bottom,domain_right_top,domain_right_bottom]
subdomain_number = len(subdomains)


# top_left.print()

# domain_left.print()

#print('domain_left: ')
#domain_left.printpolygon()
#print('domain_mid_top: ')
#domain_mid_top.printpolygon()
#print('domain_mid_bottom: ')
#domain_mid_bottom.printpolygon()
#print('domain_right_top: ')
#domain_right_top.printpolygon()
#print('domain_right_bottom: ')
#domain_right_bottom.printpolygon()

# defining the domain, and the subdomains
domain = Polygon(domain_complete.get_polygon())
for i in range(0, subdomain_number):
	domain.set_subdomain (i+1, Polygon((subdomains[i]).get_polygon()))

mesh = generate_mesh (domain, resolution)


#Compute solution
V = VectorFunctionSpace(mesh, 'P', 1)
u = Function(V)
vtkfile = File('deformation_needle/const.pvd')
vtkfile << u