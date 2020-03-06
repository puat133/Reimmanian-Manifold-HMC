import numpy as onp
from jax.ops import index,index_update,index_add
import jax
import jax.numpy as np
import jax.scipy.linalg as sla
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['image.interpolation'] = 'nearest'
rcParams['image.cmap'] = 'viridis'
rcParams['axes.grid'] = False
from tqdm import tqdm
import scipy.fftpack as FFT

#Disable this if you not compiling code
# from jax.config import config
# config.update("jax_debug_nans", True)


class Target:
    def __init__(self,neglog,d,metric_fun):
        self.__neglog = neglog
        self.__hessian_fun = jax.jit(jax.jacfwd(jax.jacrev(self.__neglog)))
        self.__metric_fun = metric_fun
        self.__d = d
        self.__softabs_const = 1e0
        # if u.ndim == 1 and u.shape[0]==self.__d:
        #     self.__u = u
        # else:
        #     raise ValueError(u)
        
    # @property
    # def u(self):
    #     return self.__u

    # @u.setter
    # def u(self,u):
    #     if u.ndim == 1 and u.shape[0]==self.__d:
    #         self.__u = u
    #     else:
    #         raise ValueError(u)
    

    @property
    def softabs_const(self):
        return self.__softabs_const

    @softabs_const.setter
    def softabs_const(self,softabs_const):
        if softabs_const>0:
            self.__softabs_const = softabs_const
        else:
            raise ValueError(softabs_const)
    
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
        return self.__metric_fun(H,self.__softabs_const)
        # return modifiedCholesky(H,self.u)
        # return softabs(H,self.__softabs_const)
        # return metric_expm(H,self.__softabs_const)

    
    def metric(self,x):
        # return np.eye(self.__d)
        return self.__metric(x)
    
    @property
    def metric_fun(self):
        return self.__metric_fun

    @metric_fun.setter
    def metric_fun(self,value):
        self.__metric_fun = value

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
        # p_temp = np.linalg.solve(L,p_star)
        p_temp = sla.solve_triangular(L,p_star,lower=False)
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
    def __init__(self,epsilon,l,omega,target,hamiltonian,track=False):

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
        
        if isinstance(target,Target):
            self.__target = target
        else:
            raise ValueError(target)

        if isinstance(hamiltonian,Hamiltonian):
            self.__hamiltonian = hamiltonian
        else:
            raise ValueError(hamiltonian)

        self.__track = track
        self.__path = onp.empty((self.__l,self.__target.d))

        self.__cos = np.cos(2*self.__omega*self.__epsilon)
        self.__sin = np.sin(2*self.__omega*self.__epsilon)

    @property
    def track(self):
        return self.__track

    @track.setter
    def track(self,value):
        self.__track=value

    
    @property
    def l(self):
        return self.__l

    @l.setter
    def l(self,value):
        if value>0:
            self.__l = value
            self.__path = onp.empty((self.__l,self.__target.d))
        else:
            raise ValueError(value)

    @property
    def epsilon(self):
        return self.__epsilon

    @epsilon.setter
    def epsilon(self,value):
        if value>0:
            self.__epsilon = value
        else:
            raise ValueError(value)

    @property
    def omega(self):
        return self.__omega

    @omega.setter
    def omega(self,value):
        if value>0:
            self.__omega = value
        else:
            raise ValueError(value)


    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __phi_H_1(self,x,p,xtilde,ptilde):
        p = p - 0.5*self.__epsilon*self.__hamiltonian.jacobian_at(x,ptilde)
        L = self.__target.metric(x)
        # xtilde = xtilde + 0.5*self.__epsilon*np.linalg.solve(L@L.T,ptilde)
        xtilde = xtilde + 0.5*self.__epsilon*sla.solve_triangular(L.T,sla.solve_triangular(L,ptilde,lower=False),lower=True)
        return x,p,xtilde,ptilde
    
    
    @jax.partial(jax.jit, static_argnums=(0,))#https://github.com/google/jax/issues/1251
    def __phi_H_2(self,x,p,xtilde,ptilde):
        ptilde = ptilde - 0.5*self.__epsilon*self.__hamiltonian.jacobian_at(xtilde,p)
        L = self.__target.metric(xtilde)
        # x = x + 0.5*self.__epsilon*np.linalg.solve(L@L.T,p)
        x = x + 0.5*self.__epsilon*sla.solve_triangular(L.T,sla.solve_triangular(L,p,lower=False),lower=True)
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

        for i in range(self.__l):
            x,p,xtilde,ptilde = self.__phi_H_1(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_2(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_omega_h(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_2(x,p,xtilde,ptilde)

            x,p,xtilde,ptilde = self.__phi_H_1(x,p,xtilde,ptilde)

            if self.__track:
                self.__path[i,:] = x

        if self.__check_nan(x,p):
            print('There is nan in the result! revert back to the original position')
            x = self.__hamiltonian.x.copy()
            p = self.__hamiltonian.p.copy()
        
        if self.__track:
            return x,p,self.__path
        else:
            return x,p
        

    def __check_nan(self,x,p):
        ret=False
        if np.any(np.isnan(x)) or np.any(np.isnan(p)):
            ret = True
        return ret
            
        
class RMHMC:
    def __init__(self,nsamples,target,x_init,p_init,seed=0):
        if nsamples>0:
            self.__nsamples = int(nsamples)
        else:
            raise ValueError(nsamples)

        if isinstance(target,Target):
            self.__target = target
        else:
            raise ValueError(target)

        if seed>0:
            self.__seed = int(seed)
        else:
            raise ValueError(seed)

        self.__r_key = jax.random.PRNGKey(self.__seed)
        self.__d = self.__target.d

        #This values are set based on the suggested value mentioned in the paper
        epsilon = 0.5*self.__target.d**(-0.25)
        l = int(1.5/epsilon)
        omega = 100
        self.__hamiltonian = Hamiltonian(self.__target,x_init,p_init)
        self.__leapfrog = Leapfrog(epsilon,l,omega,self.__target,self.__hamiltonian)
        
        self.__samples = onp.zeros((self.__nsamples,self.__target.d),dtype=np.float32)
        self.__path = onp.empty((self.__leapfrog.l*self.__nsamples,self.__target.d))

        # index_update(self.__samples,index[0,:],self.__hamiltonian.x)
        self.__samples[0,:] = self.__hamiltonian.x
        


    @property
    def leapfrog(self):
        return self.__leapfrog

    @property
    def l(self):
        return self.__leapfrog.l 

    @l.setter
    def l(self,value):
        if value>0:
            self.__leapfrog.l = value
            self.__path = onp.empty((self.__leapfrog.l *self.__nsamples,self.__target.d))
        else:
            raise ValueError(value)

    @property
    def path(self):
        return self.__path
    
    @property
    def track(self):
        return self.__leapfrog.track

    @track.setter
    def track(self,value):
        self.__leapfrog.track=value

    @property
    def epsilon(self):
        return self.__leapfrog.epsilon

    @epsilon.setter
    def epsilon(self,value):
        if value>0:
            self.__leapfrog.epsilon = value
        else:
            raise ValueError(value)

    @property
    def omega(self):
        return self.__leapfrog.omega

    @omega.setter
    def omega(self,value):
        if value>0:
            self.__leapfrog.omega = value
        else:
            raise ValueError(value)
    
    @property
    def nsamples(self):
        return self.__nsamples
    
    @nsamples.setter
    def nsamples(self,value):
        if value>0:
            self.__nsamples = int(value)
            self.__samples = onp.zeros((self.__nsamples,self.__target.d),dtype=np.float32)
            self.__path = onp.empty((self.__leapfrog.l*self.__nsamples,self.__target.d))
            # index_update(self.__samples,index[0,:],self.__hamiltonian.x)
            self.__samples[0,:] = self.__hamiltonian.x
        else:
            raise ValueError(value)


    @property
    def samples(self):
        return self.__samples

    def __get_randn(self):
        self.__r_key,subkey = jax.random.split(self.__r_key)
        return jax.random.normal(subkey,shape=(self.__d,))

    def __get_randu(self):
        self.__r_key,subkey = jax.random.split(self.__r_key)
        return jax.random.uniform(subkey,shape=(1,))

    
    def run(self):
        # self.__samples = np.empty((nsamples,self.__target.d),dtype=np.float32)
        # index_update(self.__samples,index[0,:],self.__hamiltonian.x)
        

        #Initial hamiltonian
        for i in tqdm(range(1,self.__nsamples)):
            #draw new p
            L = self.__target.metric(self.__hamiltonian.x)
            w = self.__get_randn()
            p_rand = L.T@w
            # print('w = {}, p_rand = {}'.format(w,p_rand))
            self.__hamiltonian.p = p_rand
            if not self.track:
                x_new,p_new = self.__leapfrog.leap()
            else:
                x_new,p_new,path = self.__leapfrog.leap()
            
            eval = min(1,np.exp(self.__hamiltonian.value - self.__hamiltonian.value_at(x_new,p_new)))
            if self.__get_randu() < eval:                
                # print('accepted! x_old = {}, x_new = {}'.format(self.__hamiltonian.x,x_new))
                self.__hamiltonian.x = x_new
                self.__hamiltonian.p = p_new
                if self.track:
                    self.__path[i*self.__leapfrog.l:(i+1)*self.__leapfrog.l,:] = path
                # print('updated! x = {}, p = {}'.format(self.__hamiltonian.x,self.__hamiltonian.p))
            else:
                if self.track:
                    self.__path[i*self.__leapfrog.l:(i+1)*self.__leapfrog.l,:] = x_new

            self.__samples[i,:] = self.__hamiltonian.x
            # index_update(self.__samples,index[i,:],self.__hamiltonian.x)


    

        


#These are some metrics
#generally it takes H, and constant as an argument
'''
The softabs function
'''
@jax.jit
def softabs(H,softabs_const=1e0):
    spec,T = np.linalg.eigh(H)
    abs_spec = (1./np.tanh(softabs_const * spec)) * spec
    G = ((T*abs_spec)@T.T)
    L = sla.cholesky(G)
    return L


'''
The softabs function
'''
@jax.jit
def metric_expm(H,softabs_const=1e0):
    temp = sla.expm(softabs_const*H)+sla.expm(-softabs_const*H)
    return sla.cholesky(temp)



    




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
                Lhat = index_update(Lhat,index[j+1:,j],Lhat[j+1:,j] - Lhat[j+1:,:j-1]@Lhat[j,:j-1].T)
        else:
            if j<d:
                Lhat = index_update(Lhat,index[j+1:,j],A[j+1:,j])
        if j>K:
            s = sabs(D[j],u[j])
            D = index_update(D,index[j],s)
        
        if j<d:
            D = index_update(D,index[j+1:],D[j+1:] - Lhat[j+1:,j]*Lhat[j+1:,j]/D[j] )
    L = Lhat@np.diag(np.sqrt(D))
    return L


'''
Example of target density
'''
@jax.jit
def funnel_neglog(x):
    nLP = (x[0]**2/(2*np.exp(x[1])) + x[1]/2 + x[1]**2/18)
    return nLP

@jax.jit
def fourth_order_neglog(x):
    nLP = np.linalg.norm(x)
    return nLP

@jax.jit
def two_dimensionalGaussian(x):
    return (x[0]*x[0]+x[1]*x[1])


if __name__=='__main__':
    
    target = Target(funnel_neglog,2,metric_fun=softabs)
    x_init = np.array([0.,0.])
    p_init = np.array([0.,0.])
    hmc = RMHMC(100,target,x_init,p_init,seed=onp.random.randint(1,1000))
    hmc.track=True
    target.metric_fun = softabs
    target.softabs_const = 1e0
    hmc.epsilon *= 0.05
    hmc.l *=100
    hmc.run()
    print('is there any nan here? {}'.format(onp.any(onp.isnan(hmc.samples))))
    plt.figure(figsize=(25,5))
    plt.plot(hmc.path[:,0],hmc.path[:,1],alpha=0.3,linewidth=0.3)
    plt.scatter(hmc.samples[:,0],hmc.samples[:,1],alpha=0.5)
    plt.show()


# def autocorrelation(x):
#     xp = FFT.ifftshift((x - onp.average(x))/onp.std(x))
#     n, = xp.shape
#     xp = onp.r_[xp[:n//2], np.zeros_like(xp), xp[n//2:]]
#     f = FFT.fft(xp)
#     p = onp.absolute(f)**2
#     pi = FFT.ifft(p)
#     return onp.real(onp.pi)[:n//2]/(onp.arange(n//2)[::-1]+n//2)