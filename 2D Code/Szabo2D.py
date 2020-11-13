#!/usr/bin/env python
# coding: utf-8

# Approximates the solution to Szabos fractional wave equation using schemes with and without correction terms within the approximation of the fractional derivatives, corresponding schemes are seen in Eqs 3.9 and 3.8 respectively. This notebook contains 2 examples, one with a smooth solution (used to make Fig 3) and the other nonsmooth. 

# Run the support that contains start up packages and weights codes:

# In[1]:


#For .py:
#exec(open("supportCorrections.py").read())

#For .ipynb:
get_ipython().run_line_magic('run', 'supportCorrections.ipynb')


# Set Parameters

# In[2]:


dx = 0.125 #space meshsize
dt = dx/10 #time step size

tend = 2 #end time
N = int(tend/dt) #number of time steps
dt = tend/N

gamma = 0.7 #order of fractional derivative
a0 = 1 

#constant in front of FD
#A=0 #set =0 to remove fd
A = -a0*(4/math.pi)*math.gamma(-gamma-1)*math.gamma(gamma+2)*cos((gamma+1)*(math.pi/2))


B=1 #B=0 to remove corrections

p=2 #p=1 for BDF1, p=2 for BDF2 within CQ

#Controls extra terms required for higher order derivatives
if gamma<0:
    X = 0
else:
    X = 1-B

cgam = ceil(gamma)


# Set up the problem with the exact solution:

# In[3]:


#Example 1: Smooth case
c1 = 24
c2 = 12

def exact_time(tn):
    return sin(c1*tn) + cos(c2*tn)

def dexact_time(tn):
    return c1*cos(c1*tn) - c2*sin(c2*tn)

def ddexact_time(tn):
    return -c1**2*sin(c1*tn) - c2**2*cos(c2*tn)

#Example 2: Non-Smooth case
#def exact_time(tn):
 #   return 1+ tn + tn**2 - 2*tn**(2+cgam-gamma)/(math.gamma(2-gamma)*(3-gamma)*(2-gamma))

#def dexact_time(tn):
 #   return 1+ 2*tn - 2*(2+cgam-gamma)*tn**(1+cgam-gamma)/(math.gamma(2-gamma)*(3-gamma)*(2-gamma))

#def ddexact_time(tn):
 #   return 2 - 2*(2+cgam -gamma)*(1+cgam-gamma)*tn**(cgam-gamma)/(math.gamma(2-gamma)*(3-gamma)*(2-gamma))
    
    
#Full term
def exact(tn):
    return exact_time(tn)*sin(math.pi * x)*sin(math.pi * y)

def dexact(tn):
    return dexact_time(tn)*sin(math.pi * x)*sin(math.pi * y)


# Generate mesh and finite element materials

# In[4]:


geo = SplineGeometry()
geo.AddRectangle( (-1, -1), (1, 1), bcs = ("bottom", "right", "top", "left"))
mesh = Mesh( geo.GenerateMesh(maxh=dx))
fes = H1(mesh, order=1, dirichlet="bottom|right|left|top")
u = fes.TrialFunction()
v = fes.TestFunction()

#generate stiffness and mass matrices
a=BilinearForm(fes,symmetric=False)
a+=SymbolicBFI(grad(u)*grad(v))
a.Assemble()

m=BilinearForm(fes,symmetric=False)
m+=SymbolicBFI(u*v)
m.Assemble()


# Define the function f, which forces u to be an exact solution

# In[5]:


def f_time(t):
    if ceil(gamma) == 0:
        ans= ddexact_time(t) + (2*(math.pi)**2)*(exact_time(t)) + A*simple_fint(t,dexact_time,ceil(gamma)-gamma)
    else:
        ans= ddexact_time(t) + (2*(math.pi)**2)*(exact_time(t)) + A*simple_fint(t,ddexact_time,ceil(gamma)-gamma)
    return float(ans)

f_space_exact = sin(math.pi * x) *sin(math.pi *y)

f_space = LinearForm(fes)
f_space += SymbolicLFI(f_space_exact*v)
f_space.Assemble()


# Define initial conditions

# In[6]:


un1=GridFunction(fes) #u0 / u_{n-1}
un1.Set(exact_time(0)*sin(math.pi*x)*sin(math.pi*y))

un=GridFunction(fes) #u1 / u_n
un.Set((exact_time(0)+dt*dexact_time(0)+0.5*dt**2 *ddexact_time(0))*sin(math.pi*x)*sin(math.pi*y))
Draw(un,mesh,"u")

V0=GridFunction(fes) #v0
V0.Set(dexact_time(0)*sin(math.pi*x)*sin(math.pi*y))

V1=GridFunction(fes) #v1 placeholder
V1.Set(0*x*y)
V1 = V1.vec


# Gather weights for CQ

# In[7]:


stored_weights = weights(N,tend,p-1, lambda s: s**gamma)
w0 = float(stored_weights[0])


# Invert the matrices that scale $u_{n+1}$

# In[8]:


#make matrix that will be inverted in the first step only - to help calculate v1
mstar_initial=m.mat.CreateMatrix()
mstar_initial.AsVector().data=(1 + ((A * dt * w0)/2) + (A * B * (dt * corr_weights1(1,gamma))/2) )* m.mat.AsVector()
invmstar_initial=mstar_initial.Inverse(freedofs=fes.FreeDofs())

#make matrix that will be inverted during every other time step
mstar=m.mat.CreateMatrix()
mstar.AsVector().data=(1 + ((A * dt * w0)/2)) * m.mat.AsVector()
invmstar=mstar.Inverse(freedofs=fes.FreeDofs())


# Create stores for $u$s and $v$s

# In[9]:


#store u0 and u1 as numpy arrays with a vector of zeros as a placeholder for u2
u_storage = np.array([un1.vec.FV().NumPy()[:], un.vec.FV().NumPy()[:], np.zeros(len(un1.vec))])

#create a storage space for v's
v_storage = np.zeros((N,len(V0.vec)))
v_storage[0,:] = V0.vec.FV().NumPy()[:]


# Create a vector to store the new solution

# In[10]:


res = un.vec.CreateVector()


# Initiate error

# In[11]:


error = 0


# Do the time stepping

# In[12]:


for n in range(1,N):
    
    #so we can calulate the first step differently to get V1
    if n==1:
        D=0
    else:
        D=1

    #compute the approximation of the fractional derivative
    CQ = np.zeros(len(un.vec))
    for k in range (0, n):
            CQ +=stored_weights[n-k] * (v_storage[k] - X*v_storage[0])
    
    #make this useable - via temporary vector
    temp = un.vec.CreateVector()
    temp.FV().NumPy()[:] = CQ 
    
    #run scheme
    res.data = dt * dt * f_time(n*dt) * f_space.vec
    res.data += 2 * m.mat * un.vec 
    res.data -= m.mat * un1.vec 
    res.data -= dt * dt * a.mat * un.vec 
    res.data -= A * dt**2 * m.mat * temp
    res.data += A * 0.5 * dt * w0 * m.mat * un1.vec
    res.data += A * dt**2 * X *w0* m.mat * V0.vec 
    
    res.data -= A * B * dt**2 * corr_weights0(n,gamma) * m.mat * V0.vec 
    res.data -= A * B * D * dt**2 * corr_weights1(n,gamma) * m.mat * V1 
    res.data += A * B * (1-D)  * dt * 0.5 * corr_weights1(1,gamma) * m.mat * un1.vec 
    
    
    #redefine u_n and u_{n-1}
    un1.vec.data = un.vec
     
    if n==1:
        un.vec.data = invmstar_initial * res
    else:
        un.vec.data = invmstar * res
    
    #add new data into u_storage
    if n==1:
        u_storage[2] = un.vec.FV().NumPy()[:]
    else:
        u_storage[0] = u_storage[1]
        u_storage[1] = u_storage[2]
        u_storage[2] = un.vec.FV().NumPy()[:]
    
    #calculate new v   
    v_new = np.array([(u_storage[-1] - u_storage[-3])/(2*dt)])
    
    #for the first step replace the v1 placeholder with v1
    if n==1:
        temp2 = un.vec.CreateVector()
        temp2.FV().NumPy()[:] = v_new 
        V1 = temp2
    
    #add this to the v array    
    v_storage[n,:] = v_new 
    
    Redraw(blocking=True)
    
    #error calculation
    error = max(error,sqrt (Integrate ( (un-exact((n+1)*dt))*(un-exact((n+1)*dt)), mesh)))


# Print error

# In[13]:


print("Max l2 error:" , error)
print ("End L2 error:", sqrt (Integrate ( (un-exact((n+1)*dt))*(un-exact((n+1)*dt)), mesh)))


# In[ ]:




