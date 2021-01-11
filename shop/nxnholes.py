from fenics import *
from fenics_adjoint import *
from pyadjoint.overloaded_type import create_overloaded_object
from mshr import *
import numpy as np
import math

class Identity2(UserExpression):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	def eval_cell(self,values,x,cell):
		values[0] = x[0]
		values[1] = x[1]

	def value_shape(self):
		return (2,)


# this function computes the deformation psi and the displacement dpsi for a given vector of parameters alpha
def get_deformation(alpha,n,U,char_funcs):
	# define the deformation on the boundaries
	boundary_function = as_vector((0,0))
	for i in range(0,n):
		for j in range(0,n):
			boundary_function =as_vector(boundary_function+as_vector(((char_funcs[n*i+j]*(((x[0]*n-i-0.5)*alpha[n*i+j])+i+0.5)/n),(((x[1]*n-j-0.5)*alpha[n*i+j]+j+0.5)/n))))
	boundary_psi = []
	boundary_psi.append(DirichletBC(U,project(boundary_function,U),lambda x, on_boundary : on_boundary and not(near(x[0],0) or near(x[1],0) or near(x[0],1) or near(x[1],1))))
	boundary_psi.append(DirichletBC(U,project(as_vector((x[0],x[1])),U),lambda x, on_boundary: on_boundary and (near(x[0],0) or near(x[1],0) or near(x[0],1) or near(x[1],1))))

	# solve linear equation to get complete deformation
	psit=TrialFunction(U)
	phi=TestFunction(U)
	a = inner(grad(psit),grad(phi)) *dx
	psi = Function(U)
	solve(lhs(a)==rhs(a), psi, boundary_psi)
	
	id = project(Identity2(),U)
	dpsi = Function(U)
	dpsi = project(psi-id,U)
	return dpsi, psi


# equations of elasticity
# Strain and stress
def epsilon (Du):
	return 0.5 * (Du + Du.T)

def sigma (F):
	return la*(F[0,0]+F[1,1])*Identity(2) + mu*(F+F.T)

def energy_density (u, psi):
	F = (Identity(2) +  grad(u)* inv(grad(psi)))
	return inner(sigma(F), 0.5*(F+F.T))

def ausgabe(u,dpsi,psi,k):
	F = (Identity(2) +  grad(u)* inv(grad(psi)))
	density = 0.5 * inner(sigma(F), epsilon(F))
	density = project(density, V)
	density.rename("dens","label")
	file.write(density,k)
	dpsi.rename("psi","label")
	file.write(dpsi,k)
	u.rename("u","label")
	file.write(u,k)
	file.close()


def m_to_array(m):
	alpha = []
	for entry in m:
		alpha.append(float(entry))
	return alpha

def volume(m):
	alpha = m_to_array(m)
	square_sum = 0
	for i in range (n*n):
		square_sum += (alpha[i]*(1/(4*n)))**2
	return 1-pi*square_sum

def derivative_cb(j, dj, m):
	alpha = m_to_array(m)
	print (alpha)
	global k
	k = k + 1
	if (k%15==0):
		dpsi,psi,u,J = get_elast_def(alpha,n,U)
		print ("ausgabe: ",k)
		print ("volume:", volume(m))
		ausgabe(u,dpsi,psi,k)


def get_elast_def(alpha,n,U,char_funcs):
	dpsi,psi = get_deformation(alpha,n,U,char_funcs)
	g = Constant((-0.5*weight,0))
	
	C = dot (g,u) * ds(1)
	S = energy_density(u,psi) * dx
	P = volweight * abs(det(grad(psi))) * dx

	J = C + P
	E = S - C

	duE = derivative (E, u)
	duduE = derivative (duE, u)
	duC = derivative (C, u)

	# Solve state equation once (adjoint is done automatically)
	# solve (duE == 0, u, ubc, J=duduE) # here unnecessary, state equation is linear
	solve (action (duduE, uu) == duC, u, bc_bottom)
	return dpsi,psi,u,J



#         start of programm                                    

# parameter for number of holes
n = 8
resolution = 20

# parameter for elsasticity
la = 5.
mu = 5.
weight = 1.5
volweight = 0.4
ftol = 1e-10
gtol = 1e-8

#         defining the domain and the function spaces               
domain = Rectangle(dolfin.Point(0., 0.), dolfin.Point(1., 1.)) 
for i in range(0,n):
	for j in range(0,n):
		domain = domain - Circle(dolfin.Point((0.5+i)/n, (0.5+j)/n), 1/(4*n))

for i in range(0,n):
	for j in range(0,n):
		#print(i*n+j+1)
		domain.set_subdomain (i*n+j+1, Polygon([dolfin.Point(i/n,j/n),dolfin.Point((i+1)/n,j/n),dolfin.Point((i+1)/n,(j+1)/n),dolfin.Point(i/n,(j+1)/n)]))


# define mesh and function space
mesh = generate_mesh (domain, resolution)
mesh = create_overloaded_object(mesh)
x = SpatialCoordinate(mesh)


# defining coefficients on subdomains
X = FunctionSpace (mesh, "DG", 0)
dm = X.dofmap()
sudom = MeshFunction ('size_t', mesh, 2, mesh.domains())
print(sudom.array())
sudom_arr = np.asarray (sudom.array(), dtype=np.int)
print(sudom_arr.size)
for cell in cells (mesh): sudom_arr [dm.cell_dofs (cell.index())] = sudom [cell]

const = project(Constant(0),X)
file2 = XDMFFile('nxnholes/const.xdmf')
file2.write(const,0)

def sudom_fct (sudom_arr, vals, fctspace):
	f = Function (fctspace)
	f.vector()[:] = np.choose (sudom_arr, vals)
	return f

# creat list with nxn+1 zeros
zeros = [0]
colorful = [0]
for i in range(0,n):
	for j in range(0,n):
		zeros.append(0)
		colorful.append(n*i+j+1)

char_funcs = []
for i in range(0,n):
	for j in range(0,n):
		zeros[n*i+j+1] = 1
		print(zeros)
		print(colorful)
		char_funcs.append(sudom_fct (sudom_arr,zeros, X))
		zeros[n*i+j+1] = 0

# Function spaces
U = VectorFunctionSpace (mesh, 'CG', 1)
V = FunctionSpace (mesh, 'CG', 1)
density = Function(V)
u = Function (U)
uu = TrialFunction (U) 


#             define subdomain for top boundary                  
subdomains = MeshFunction ("size_t", mesh, mesh.topology().dim() - 1) 
subdomains.set_all (0)
class Top (SubDomain):
	def inside (self, x, on_boundary):
		return on_boundary and near(x[1],1.)
top = Top ()
top.mark (subdomains, 1)

dx = Measure ('dx', domain=mesh)
ds = Measure ('ds', domain=mesh, subdomain_data=subdomains)

#          define vector of parameters, bounds and controls           
alpha = []
lbound = []
ubound = []
calpha = []
for i in range(0,n):
	for j in range(0,n):
		alpha.append(Constant(1.))
		lbound.append(0.1)
		ubound.append(1.9)
		calpha.append(Control(alpha[i*n+j]))


#            define dirichlet boundary conditions for elastic deformation u               
def boundary_bottom(x,on_boundary):
	return on_boundary and near(x[1],0)
bc_bottom = DirichletBC(U,Constant((0,0)),boundary_bottom)

file = XDMFFile('nxnholes/Sol.xdmf')
file.parameters ["flush_output"] = True
file.parameters["functions_share_mesh"] = True
file.parameters ["rewrite_function_mesh"] = False


dpsi,psi,u,J = get_elast_def(alpha,n,U,char_funcs)
ausgabe(u,dpsi,psi,0)

Jhat = ReducedFunctional(assemble(J),calpha,derivative_cb_post = derivative_cb)

boundaries = [lbound,ubound]
k = 0
sol = np.copy(minimize (Jhat, method = 'L-BFGS-B', tol = 1e-12, bounds=boundaries, options={'disp' : True, 'ftol' : ftol, 'gtol' : gtol}))

ralpha = m_to_array(sol)

print(ralpha)


dpsi,psi,u,J = get_elast_def(ralpha,n,U)
ausgabe(u,dpsi,psi,k)
file.close()
