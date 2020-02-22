import numpy as onp
from jax.ops import index,index_update,index_add
import jax
import jax.numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False


class Target:
    def __init__(self,neglog,d,u):
        self.__neglog = neglog
        self.__hessian_fun = jax.jit(jax.jacfwd(jax.jacrev(self.__neglog)))
        self.__d = d
        if u.ndim == 1 and u.shape[0]==self.__d:
            self.__u = u
        else:
            raise ValueError(u)
        
    @property
    def u(self):
        return self.__u

    @u.setter
    def u(self,u):
        if u.ndim == 1 and u.shape[0]==self.__d:
            self.__u = u
        else:
            raise ValueError(u)
    
    @property
    def d(self):
        return self.__d

    @property
    def hessian_at(self):
        return self.__hessian_fun
    
      
    def neglog(self,x):
        return self.__neglog(x)   
   
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __metric(self,x):
        H = self.hessian_at(x)
        return modifiedCholesky(H,self.u)

    
    def metric(self,x):
        return self.__metric(x)
    
        
class Hamiltonian:
    def __init__(self,target,x,p):
        if isinstance(target,Target):
            self.__target = target
        else:
            raise ValueError(target)

        if x.shape[0] == self.__target.d:
            self.__x = x
        else:
            raise ValueError(x)
        
        if p.shape[0] == self.__target.d:
            self.__p = p
        else:
            raise ValueError(p)

        self.__jac_fun = jax.jit(jax.grad(self.__fun)) #by default the derivative is against the first argument
        

    #Hamiltonian at x_star and p_star
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __fun(self,x_star,p_star):
        L = self.__target.metric(x_star)
        p_temp = np.linalg.solve(L,p_star)
        return self.__target.neglog(x_star) + 0.5*np.sum(p_temp*p_temp) + np.linalg.slogdet(L)[1]


    @property
    def x(self):
        return self.__x
    
    @x.setter
    def x(self,x):
        if x.shape[0] == self.__target.d:
            self.__x = x
        else:
            raise ValueError(x)
    
    @property
    def p(self):
        return self.__p
    
    @p.setter
    def p(self,p):
        if p.shape[0] == self.__target.d:
            self.__p = p
        else:
            raise ValueError(p)

    @property
    def value(self):
        return self.__fun(self.__x,self.__p)
    
    @property
    def jacobian(self):
        return self.__jac_fun(self.__x,self.__p)

    @property
    def jacobian_at(self):
        return self.__jac_fun

    def value_at(self,x_star,p_star):
        return self.__fun(x_star,p_star)
    
    
class Leapfrog:
    def __init__(self,epsilon,l,omega,target,hamiltonian):
        if isinstance(target,Target):
            self.__target = target
        else:
            raise ValueError(target)

        if isinstance(hamiltonian,Hamiltonian):
            self.__hamiltonian = hamiltonian
        else:
            raise ValueError(hamiltonian)

        if epsilon>0:
            self.__epsilon = epsilon
        else:
            raise ValueError(epsilon)

        if l>0:
            self.__l = int(l)
        else:
            raise ValueError(l)

        if omega>0:
            self.__omega = int(omega)
        else:
            raise ValueError(omega)

        self.__cos = np.cos(2*self.__omega*self.__epsilon)
        self.__sin = np.sin(2*self.__omega*self.__epsilon)

    
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __phi_H_1(self,x,p,xtilde,ptilde):
        p = p - 0.5*self.__epsilon*self.__hamiltonian.jacobian_at(x,ptilde)
        L = self.__target.metric(x)
        xtilde = xtilde + 0.5*self.__epsilon*np.linalg.solve(L@L.T,ptilde)
        return x,p,xtilde,ptilde
    
    
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __phi_H_2(self,x,p,xtilde,ptilde):
        ptilde = ptilde - 0.5*self.__epsilon*self.__hamiltonian.jacobian_at(xtilde,ptilde)
        L = self.__target.metric(x)
        x = x + 0.5*self.__epsilon*np.linalg.solve(L@L.T,ptilde)
        return x,p,xtilde,ptilde
    
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __phi_omega_h(self,x,p,xtilde,ptilde):
        x = (x+xtilde + self.__cos*(x-xtilde) + self.__sin*(p-ptilde))/2
        p = (p+ptilde - self.__sin*(x-xtilde) + self.__cos*(p-ptilde))/2
        xtilde = (x+xtilde - self.__cos*(x-xtilde) - self.__sin*(p-ptilde))/2
        ptilde = (p+ptilde + self.__sin*(x-xtilde) - self.__cos*(p-ptilde))/2
        return x,p,xtilde,ptilde
    
    # @jax.jit
    def leap(self):
        x = self.__hamiltonian.x.copy()
        p = self.__hamiltonian.p.copy()
        xtilde = x.copy()
        ptilde = p.copy()

        for _ in range(self.__l):
            x,p,xtilde,ptilde = self.__phi_H_1(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_2(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_omega_h(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_2(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_1(x,p,xtilde,ptilde)

        return x,p


class HMC:
    def __init__(self,nsamples,target,x_init,p_init):
        if nsamples>0:
            self.__nsamples = int(nsamples)
        else:
            raise ValueError(nsamples)

        if isinstance(target,Target):
            self.__target = target
        else:
            raise ValueError(target)

        self.__epsilon = 0.5*self.__target.d**(-0.25)
        self.__l = 1.5/self.__epsilon
        self.__omega = 100
        
        self.__hamiltonian = Hamiltonian(self.__target,x_init,p_init)

        self.__leapfrog = Leapfrog(self.__epsilon,self.__l,self.__omega,self.__target,self.__hamiltonian)
        
        self.__samples = onp.empty((self.__nsamples,target.d),dtype=onp.float32)
        self.__samples[0,:] = x_init

    @property
    def nsamples(self):
        return self.__nsamples

    @property
    def samples(self):
        return self.__samples

    def run(self):

        #draw new p
        L = self.__target.metric(self.__hamiltonian.x)
        p_star = np.linalg.solve(L@L.T,np.random.)

    




'''
The Sabs Function
'''
@jax.jit
def sabs(x,u):
    log2 = np.log(2)
    temp = np.exp(x*log2/u)
    ret = np.log(temp+1/temp)*u/log2
    return ret

'''
Implementation of Modified Cholesky Decomposition routine
taken from :
`Modified Cholesky Riemann Manifold Hamiltonian Monte Carlo:
exploiting sparsity for fast sampling of high-dimensional targets`
    by Tore Selland Kleppe. DOI 10.1007/s11222-017-9763-5
'''
@jax.jit
def modifiedCholesky(A,u,K=0):
    
    d = A.shape[0]
    Id = np.eye(d)
    Lhat = Id
    D = np.diag(A)
    for j in range(d):
        if j>0:
            Lhat = index_update(Lhat,index[j,:j-1],Lhat[j,:j-1]/D[:j-1])
            if j<d-1:
                Lhat = index_update(Lhat,index[j+1:,j],A[j+1:,j])
                Lhat = index_update(Lhat,index[j+1:,j],Lhat[j+1:,j] - Lhat[j+1:,:j-1]*Lhat[j,:j-1])
        else:
            if j<d:
                Lhat = index_update(Lhat,index[j+1:,j],A[j+1:,j])
        if j>K:
            D = index_update(D,index[j],sabs(D[j],u[j]) )
        
        if j<d:
            D = index_update(D,index[j+1:],D[j+1:] - Lhat[j+1:,j]*Lhat[j+1:,j]/D[j] )
    L = Lhat@np.diag(D)
    return L


'''
Example of target density
'''
@jax.jit
def funnel_neglog(x):
    nLP = (x[0]**2/(2*np.exp(x[1])) + x[1]/2 + x[1]**2/18)
    return nLP


