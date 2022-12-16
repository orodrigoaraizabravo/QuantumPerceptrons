# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 11:22:03 2022

@author: HP
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import tlquantum as tlq
from tlquantum.tt_precontraction import layers_contract
import tensorly as tl
import torch as t
from math import factorial 
from torch import nn, pi, rand, optim 
from torch.optim import Adam
from tenpy.linalg.np_conserved import Array
from tenpy.networks.site import SpinHalfSite
from tenpy.networks.mps import MPS
import scipy.sparse as ss
import numpy as np 
import qutip as qt
from scipy.sparse.linalg import lobpcg, expm
import pickle

class Perceptron(tlq.Unitary):
    def __init__(self, N, ncontraq=2, approx=1,dt=0.01,Js=None, hs=None, device=None):
        super().__init__([], N+1, ncontraq, dtype=t.complex128, device=device)
        self.N, self.approx = N, approx
        self.ncontraq=ncontraq
        self.dt, self.device =dt, device
        
        #print('Getting parameters')
        if Js is None:self.Js = nn.Parameters(t.pi*(2*t.rand([N], device=device)-1))
        else: self.Js = Js 
        if hs is None: self.hs = nn.Parameters(t.pi*(2*t.rand([2], device=device)-1))
        else: self.hs = hs
        
        #print('Setting gates')
        self._set_gates([perceptron(approx=approx,dt=dt,device=device,J=Js[0], h=None, end=0)]+\
        [perceptron(approx=approx,dt=dt,device=device,J=Js[i], h=None, end=i) for i in range(1,N)]+\
        [perceptron(approx=approx,dt=dt,device=device,J=None, h=hs, end=-1)])

def core_addition(c1, c2, end=0):
    if end==0: 
        return tl.concatenate((c1, c2), axis=3)
    elif end==-1: 
        return tl.concatenate((c1, c2), axis=0)
    else: 
        pc1 = tl.concatenate(
                (c1, tl.zeros((c2.shape[0], c1.shape[1], c1.shape[2], c1.shape[3]), device=c1.device))
                 , axis=0)
        pc2 = tl.concatenate(
                (tl.zeros((c1.shape[0], c1.shape[1], c1.shape[2], c2.shape[3]), device=c2.device), c2),
                axis=0)
        return tl.concatenate((pc1, pc2), axis=3)

def core_multiplication(f, core, i):
    return layers_contract([[f*core]]+[[core]*(i)], i+1)[0]

class perceptron(nn.Module):
    """This class produces the unitary evolution of a perceptron by calculating 
    the core for site 'end' for the approximation 1-iH-H^2/2+... up to order 'approx'.
    approx (int): order of the approximation
    'dt (float): time of the evolution 
     J (float): coupling constant between the input and the output
     h (list of floats): list of fields on the output qubit. h[0] is the Rabi drive
     device: device in which the core is to be stored
     end (int): a number that indicates the site of the core. -1 corresponds to the output qubit"""
    def __init__(self, approx=1, dt=0.01, J=None, h=None, device=None, end=0): 
        super().__init__()
        self.dt= dt
        self.end, self.approx, self.device= end, approx, device
        if end != -1:
            if J is None: self.J = nn.Parameter(t.pi*(2*t.rand(1, device=device)-1))
            else: self.J = J
        else: 
            if h is None: self.J = nn.Parameter(2*t.pi*t.rand(2, device=device))
            else: self.J = h
    
    def forward(self):
        self.core = tlq.IDENTITY(device=self.device).forward()
        if self.end==0:
            _core = tl.zeros((1,2,2,2),  device=self.device, dtype=t.complex128)
            _core[0,:,:,0] = tl.eye(2, device=self.device, dtype=t.complex128)
            _core[0,:,:,1] = self.J*tl.tensor([[1,0],[0,-1]], dtype=t.complex128, device=self.device)
            for i in range(self.approx):
                f = (-1j*self.dt)**(i+1)/factorial(i+1)
                self.core = core_addition(self.core,core_multiplication(f, _core, i), end=self.end)
        elif self.end==-1: 
            _core = tl.zeros((2,2,2,1), device=self.device, dtype=t.complex128)
            _core[1,:,:,0]=tl.tensor([[1,0],[0,-1]], dtype=t.complex128, device=self.device)
            _core[0,:,:,0]=self.J[1]*tl.tensor([[1,0],[0,-1]], dtype=t.complex128, device=self.device)
            _core[0,:,:,0]+=self.J[0]*tl.tensor([[0,1],[1,0]], dtype=t.complex128, device=self.device) 
            for i in range(self.approx):
                self.core = core_addition(self.core,core_multiplication(1., _core, i), end=self.end)
        else: 
            _core = tl.zeros((2,2,2,2), device=self.device, dtype=t.complex128)
            _core[0,:,:,0] = tl.eye(2, device=self.device, dtype=t.complex128)
            _core[1,:,:,1] = tl.eye(2, device=self.device, dtype=t.complex128)
            _core[0,:,:,1] = self.J*tl.tensor([[1,0],[0,-1]], dtype=t.complex128, device=self.device)
            for i in range(self.approx): 
                self.core = core_addition(self.core,core_multiplication(1., _core, i), end=self.end)
        return self.core

def exp_pauli_x(dtype=t.complex128, device=None):
    """Matrix for sin(theta) component of X-axis rotation in tt-tensor form.
    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) X-rotation component.
    """
    return tl.tensor([[[[0],[-1j]],[[-1j],[0]]]], dtype=dtype, device=device)

def exp_pauli_z(dtype=t.complex128, device=None):
    """Matrix for sin(theta) component of X-axis rotation in tt-tensor form.
    Parameters
    ----------
    device : string, device on which to run the computation.

    Returns
    -------
    tt-tensor core, sin(theta) X-rotation component.
    """
    return tl.tensor([[[[-1j],[0]],[[0],[1j]]]], dtype=dtype, device=device)

class single_Rx(nn.Module):
    """Qubit rotations about the X-axis with randomly initiated theta.
    Parameters
    ----------
    device : string, device on which to run the computation.
    Returns
    theta : float, parameter
    -------
    RotX
    """
    def __init__(self, dtype=t.complex128, device=None, theta=None):
        super().__init__()
        if theta is None: self.theta = nn.Parameter(2*t.pi*t.rand(1, device=device))
        else: self.theta= theta
        self.iden, self.epx = tlq.identity(dtype=dtype, device=device), exp_pauli_x(dtype=dtype, device=device)
    
    def forward(self):
        """Prepares the RotX gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).
        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*t.cos(self.theta/2)+self.epx*t.sin(self.theta/2)

class single_Rz(nn.Module):
    """Qubit rotations about the z-axis with randomly initiated theta.
    Parameters
    ----------
    device : string, device on which to run the computation.
    Returns
    theta : float, parameter
    -------
    RotX
    """
    def __init__(self, dtype=t.complex128, device=None, theta=None):
        super().__init__()
        if theta is None: self.theta = nn.Parameter(2*t.pi*t.rand(1, device=device))
        else: self.theta= theta
        self.iden, self.epz = tlq.identity(dtype=dtype, device=device), exp_pauli_z(dtype=dtype, device=device)
    
    def forward(self):
        """Prepares the RotZ gate for forward contraction by calling the forward method
        and preparing the tt-factorized form of rotation matrix depending on theta (which is
        typically updated every epoch through backprop via PyTorch Autograd).
        Returns
        -------
        Gate tensor for general forward pass.
        """
        return self.iden*t.cos(self.theta/2)+self.epz*t.sin(self.theta/2)

class R(tlq.Unitary):
    """A Unitary sub-class that generates a layer of unitary, single-qubit rotations
    along the x-axis

    Parameters
    ----------
    N : int, number of qubits
    ncontraq : int, number of qubits to do pre-contraction over
               (simplifying contraciton path/using fewer indices)
    contrsets : list of lists of ints, the indices of qubit cores to
                merge in the pre-contraction path.
    device : string, device on which to run the computation.
    rotate_last: boolean if the last qubit is rotated
    Returns
    -------
    UnaryGatesUnitary
    """
    def __init__(self, nqubits, ncontraq, axis,contrsets=None, dtype=t.complex128, device=None, ths=None, rotate_last=False):
        super().__init__([], nqubits, ncontraq, contrsets=contrsets, dtype=dtype, device=device)
        if ths is None: ths=[None]*nqubits
        if axis=='x':
            l= [single_Rx(dtype=dtype,device=device, theta=ths[i]) for i in range(nqubits)]
            if rotate_last: l.append(single_Rx(dtype=dtype,device=device, theta=ths[-1]))
        elif axis=='z':
            l= [single_Rz(dtype=dtype,device=device, theta=ths[i]) for i in range(nqubits)]
            if rotate_last: l.append(single_Rz(dtype=dtype,device=device, theta=ths[-1]))
        else: l.append(tlq.IDENTITY(dtype=dtype,device=device))
        
        self._set_gates(l)
    
class Witness(nn.Module):
    """Witnessing class. This class takes in the number of input qubits N and 
     the number of layers L. It will produce the circuits for many different 
     times dts such that sigma_z(t) can be measured on the output
     -----------
     Parameters:
     N: int, number of input qubits
     L: int, number of layers, 
     dts: list of floats with the times 
     ncontraq: int
     nctrsets: int
     approx: int
     device: string"""
    def __init__(self, N, L,dts, ncontraq=2, ncontral=2, approx=1, device=None):
        super().__init__()
        self.N, self.L, self.dts = N, L, dts
        self.ncq,self.cts =ncontraq, ncontral
        self.approx, self.device = approx, device
        self.fix_omega = True
        
        self.Js=t.nn.Parameter(t.pi*(2*t.rand([L,N], device=device)-1))
        fields= [[10.,0.0] for l in range(L)]
        self.hs= nn.Parameter(t.tensor(fields, device=device)).requires_grad_(not self.fix_omega)
        self.ths = nn.Parameter(t.pi*(2*t.rand([L,4,N+1], device=device)-1))
        self.W = nn.Linear(len(dts),1,device=device,dtype=t.double)
        self.nonlinearity= nn.Softsign()
        self.operator = [tlq.identity(dtype=t.complex128)]*N+[tlq.pauli_z(dtype=t.complex128)]
        self.get_circuits()
        self.criterion = t.nn.MSELoss()
        
    def get_circuits(self):
        """This function prepares the circuits for the witness using different times for evolution"""
        self.circuits= []
        for i in range(len(self.dts)):
            c = []
            for l in range(self.L):
                c.append(R(self.N+1,axis='x',ncontraq=self.ncq,
                          ths=self.ths[l,0], device=self.device))
                c.append(R(self.N+1,axis='z',ncontraq=self.ncq,
                           ths=self.ths[l,1], device=self.device))
                c.append(Perceptron(self.N,ncontraq=self.ncq,\
                          approx=self.approx,dt=self.dts[i],Js=self.Js[l], \
                          hs=self.hs[l], device=self.device))
                
                #c.append(tlq.UnaryGatesUnitary(self.N+1, 2))
            self.circuits.append(tlq.TTCircuit(c, self.ncq, self.cts))
        return
        
    def single_time_forward(self, input, time):
        """Single time forward pass"""
        return self.circuits[time].forward_expectation_value(input,self.operator).real
        
    def forward(self, input):
        """forward passes"""
        v = t.zeros(len(self.dts),device=self.device, dtype=t.double)
        for i in range(len(self.dts)):
            v[i] = self.single_time_forward(input, i)
        return self.nonlinearity((self.W(v)))
    
    def train(self, X, Y, Nepochs, batchsize, lr=0.1): 
        """training function, takes in X as MPS and Y as a list of labels"""
        self.opt = Adam(list(self.parameters()), lr=lr)
        Nsamples = len(X)
        
        Y.to(self.device)
        
        losses = []
    
        for epoch in range(Nepochs):
            l=0
            for j in range(0, Nsamples, batchsize):
                outputs = t.zeros([batchsize], device=self.device, dtype=Y.dtype)
            
                for i in range(batchsize): outputs[i] = self.forward(X[j+i])#.to(self.device))

                loss = self.criterion(outputs, Y[j:j+batchsize])
                l+=loss.item()/(Nsamples//batchsize)
                loss.backward()
                self.opt.step()
                self.opt.zero_grad()
                self.get_circuits()
            
            losses.append(l)
            print(f"Epoch: {epoch}. Loss: {l}")
        return losses
    
    def accuracy(self, X, y):
        outputs=[]
        true_positives=0
        true_negatives=0
        false_positives=0
        false_negatives=0
        with t.no_grad():
            for j in range(len(X)):
                output = self.forward(X[j])
                if t.sign(output)==t.sign(y[j]):
                    if (t.sign(output)==-1).item(): 
                        true_positives+=1
                    else: true_negatives+=1
                else:
                    if (t.sign(output)==-1).item(): 
                        false_positives+=1
                    else: false_negatives+=1
                outputs.append(output)
        return true_positives/(len(X)/2), true_negatives/(len(X)/2), false_positives/(len(X)/2), false_negatives/(len(X)/2), outputs

def vect_to_MPS(v, N):
    """Transforms a densed vector to an MPS"""
    v = v.reshape([2]*N)
    sites = [SpinHalfSite(conserve=None)]*N
    b = Array.from_ndarray_trivial(v, labels=['p'+str(i) for i in range(N)])
    B=MPS.from_full(sites, b)
    return [tl.tensor(B.get_B(i).to_ndarray(), dtype=tl.complex128) for i in range(N)]

"""Functions for preparation of the data set"""
I2 = ss.csc_matrix([[1.0, 0.0+0j],[0.0,1.0]])
sx = ss.csc_matrix([[0.0, 1.0+0j],[1.0,0.0]])
sy = ss.csc_matrix([[0.0, -1j],[1j,0.0]])
sz = ss.csc_matrix([[1.0, 0.0+0j],[0.0,-1.0]])

def expz(theta):
    return I2*np.cos(theta)-1j*np.sin(theta)*sz

def expx(theta):
    return I2*np.cos(theta)-1j*np.sin(theta)*sx

def expy(theta):
    return I2*np.cos(theta)-1j*np.sin(theta)*sy

def add_op(ops,sites, L, full=False): 
    if not full: 
        l= [I2 for i in range(L)] 
        for s in range(len(sites)): l[sites[s]]=ops[s]
    else: l = ops
    m=l[0]
    for s in range(1,L): m=ss.kron(m,l[s], format='csc')
    return m

plus = np.array([[1/np.sqrt(2)],[1/np.sqrt(2)]])

def overlap(psi, phi):
    return (abs(psi.conj().T@phi)**2)[0]

def RandomHeisenberg(N, index, top=1): 
    p = States_prep(N, index)
    efinal, initial_overlap, final_overlap=p.train() 
    print('Final overlap = ', final_overlap)
    ground_label= -1
    dm = qt.ket2dm(qt.Qobj(p.ground_state,dims=[[2]*N, [1]*2]))
    if qt.entropy_vn(dm.ptrace(range(N//2)))<1e-6: ground_label=1
    
    a = ss.kron(p.ground_state,plus).toarray()
    b = ss.kron(p.state,plus).toarray()
    a[abs(a)<=1e-8], b[abs(b)<=1e-8] = 0,0

    return (vect_to_MPS(a,N+1), ground_label), (vect_to_MPS(b,N+1), 1)

class States_prep(nn.Module): 
    """Class for preparation of mean-field theory and ground states of models"""
    def __init__(self,N,index,device='cpu'):
        super(States_prep, self).__init__()
        self.N = N
        self.th = nn.Parameter(2*pi*rand((N, 2), device=device)-1)
        self.paulis = [I2,sx,sy,sz]
        self.k0 = ss.csc_matrix([[1.0],[0.0]])
        self.index=index
        self.optimizer = optim.Adam([self.th],lr = 0.1)
        self.criterion = nn.MSELoss()
        self.size=3 
        self.get_h()
        self.get_ground()
    
    def get_h(self, top=1.0):
        """Depending on the index, we produce a model with random couplings"""
        self.H = 0*ss.eye(2**self.N, dtype=I2.dtype, format='csc')
        eps=1e-5
        if self.index==0:
        #Training Hamiltonian!
            J = np.random.uniform(-top,top)
            self.Jx, self.Jy, self.Jz, self.h = J,J,J, eps
            
        elif self.index==1:
        #XY Hamiltonian 
            J = np.random.uniform(-top,top)
            #J = 1.0
            self.Jx, self.Jy, self.Jz, self.h = J, J, 0.0, eps
            
        elif self.index==2: 
            #Ising Hamiltonian
            self.Jz= np.random.uniform(-top,top)
            self.Jx, self.Jy, self.h = 0.0, 0.0, 0.5

        elif self.index==3: 
        #XX-Z Hamiltonian 
            Jxy = np.random.uniform(-top,top)
            self.Jx, self.Jy = Jxy, Jxy
            self.Jz = np.random.uniform(-top,top)
            self.h = eps
            
        else: 
        #XYZ Hamiltonian
            self.Jx = np.random.uniform(-top,top)
            self.Jy = np.random.uniform(-top,top)
            self.Jz = np.random.uniform(-top,top)
            self.h  = eps
            
        for i in range(self.N-1): 
            self.H += add_op([self.Jz*sz,sz],[i,i+1], self.N)
            self.H += add_op([self.Jy*sy,sy],[i,i+1], self.N)
            self.H += add_op([self.Jx*sx,sx],[i,i+1], self.N)
            self.H += add_op([self.h*sz],[i], self.N)
        self.H += add_op([self.h*sz],[-1], self.N)
    
    def Energy(self):
        """Energy of the mean field theory state which is parametrized by 
        two angles per qubit. Angles are to be optimized over to minimize energy"""
        e=0
        for i in range(self.N-1):
            e+=self.Jx*t.sin(self.th[i,0])*t.sin(self.th[i,1])*t.sin(self.th[i+1,0])*t.sin(self.th[i+1,1])
            e+=self.Jy*t.sin(self.th[i,0])*t.cos(self.th[i,1])*t.sin(self.th[i+1,0])*t.cos(self.th[i+1,1])
            e+=self.Jz*t.cos(self.th[i,0])*t.cos(self.th[i+1,0])
            e+=self.h*t.sin(self.th[i,0])*t.sin(self.th[i,1])
        e+=self.h*t.sin(self.th[self.N-1,0])*t.sin(self.th[self.N-1,1])   
        return e
    
    def get_ground(self):
        """Quick function for getting ground eigenspace of a sparse matrix"""
        X = np.random.rand(2**self.N, self.size)
        self.evals, self.vect = lobpcg(-self.H, X)
        self.ground_energy = -self.evals[0]
        self.ground_state = ss.csc_matrix(self.vect[:,0]).T
        return 

    def build_mf(self): 
        """For building the mean-field theory state"""
        self.state = expm(-1j*self.th[0,1].item()*sz/2)@expm(-1j*self.th[0,0].item()*sx/2)@self.k0
        for i in range(1,self.N):
            self.state = ss.kron(self.state,expm(-1j*self.th[i,1].item()*sz/2)@expm(-1j*self.th[i,0].item()*sx/2)@self.k0, format='csc')
        return 
    
    def train(self, EPOCHS=100, verbose=False):
        """Training the MF state"""
        energy_vec = t.zeros((EPOCHS))

        with t.no_grad():
            self.build_mf()
            init_overlap = overlap(self.ground_state, self.state)
        
        for epoch in range(EPOCHS):
            energy = self.Energy()
            if verbose:
                print('Energy (loss) at epoch ' + str(epoch) + ' is ' + str(energy.item()) + '. \n')
            
            energy.backward()
            self.optimizer.step()
            self.optimizer.zero_grad(epoch)
            energy_vec[epoch] = energy.item()
        
        with t.no_grad():
            self.build_mf()
            final_overlap = overlap(self.ground_state, self.state)
        
        #print('Final overlap: ', final_overlap)
        
        return energy_vec, init_overlap, final_overlap

def build_data_set(N,index, Nsamples):
    X= []
    Y= []
    for n in range(Nsamples): 
        a,b = RandomHeisenberg(N,index)
        Y.append(a[1])
        Y.append(b[1])
        X.append(tlq.qubits_contract(a[0], 2))
        X.append(tlq.qubits_contract(b[0], 2))
    return X, t.tensor(Y, dtype=t.float32)

def build_training_set(N, Nsamples=100, index=0):
    X,Y = build_data_set(N, index, Nsamples)
    Ntrain = int(Nsamples*0.8)
    return (X[:Ntrain], Y[:Ntrain]), (X[Ntrain:], Y[Ntrain:])

def build_and_store_data_sets(N, Nsamples):
    models = ['Heis', 'XY', 'Ising','XX-Z', 'XYZ']
    for i in range(len(models)): 
        file = 'Data_N'+str(N)+'_Model_'+models[i]+'_Nsamps'+str(Nsamples)+'_.pt'
        if i==0:
            (Xtrain, ytrain), (Xtest, ytest)=build_training_set(N, Nsamples=Nsamples)
            dic = {'X': Xtrain, 'y': ytrain, 'Xtest': Xtest, 'ytest': ytest}
        else:
            X,y = build_data_set(N, i, Nsamples)
            dic = {'X': Xtrain, 'y': ytrain}
            
        with open(file, 'wb') as outfile: 
            pickle.dump(dic,outfile)
    print('Done building data sets!')
    return 

