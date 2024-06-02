# Filippos Tzimkas-Dakis   Virginia Tech   MARCH 2024
#
# Version V 1.2
# 
# The present script defines a class whose objects behave as Fock/Number states.
# Every object of this class has all the basic features of a quantum Fock state.
# Namely, one can define a state, normalize it, compute overlaps between states, 
# add quantum different Fock states, displacement operators, compute the Wigner 
# and Q-Husimi function, calculate the photon distribution, average photon number.
#
# Please have a look at the example accompanying this class.
#
# I might add more features in the future.

import numpy as np
import scipy
import scipy.linalg
from scipy.special import factorial, comb
import copy 
import collections.abc
import multiprocessing as mp
import src.Library as lib

# The following functions are needed for the optimization of thw Wigner function computation 
# ----------------------------------------------------------------------------------------------------------------------------------
def init_shared_arrays(shared_array, N):
    global W_shared, N_shared
    W_shared = np.frombuffer(shared_array.get_obj(), dtype=np.complex128).reshape((N, N))
    N_shared = N

def compute_wigner_chunk(chunk):
    x, y, st, parity, start_i, end_i, start_j, end_j = chunk
    N = N_shared
    for i in range(start_i, end_i):
        for j in range(start_j, end_j):
            b = x[j] + 1j * y[i]
            D_psi, D_psi_conj = st.D__(-b)
            W_shared[i, j] = np.dot(D_psi_conj, np.dot(parity, D_psi))
# ----------------------------------------------------------------------------------------------------------------------------------


class FockBasis:
    def __init__ (self, nn, N_space = 30):
        # Constructor of the Number Basis class
        # Inputs:
        #   nn     : List or array containing the coefficients of every state number state, e.g., [1, 0, 1j] = |0> + 1j|2>.
        #   N_space: Number to truncate the infinite Hilbert space to N_space levels. Default is 30.
        #
        # Check if the variable is an array (ndarray) or a scalar
        if not isinstance(nn, np.ndarray):                                   # check if the inputs np.array,
            if isinstance(nn, collections.abc.Sequence): nn = np.array(nn)   # if not convert it to one
            else: nn = np.array([nn]) 

        # if k!=None or (not isinstance(k, np.ndarray)):                      # check if the inputs np.array,
        #     if isinstance(k, collections.abc.Sequence): k = np.array(nn)     # if not convert it to one
        #     else: k = np.array([k]) 

        #     if np.min(k)<0 or (not all(lib.isinteger(k))):
        #         raise ValueError("Input number states must me non negative integers!")  
        #     k  = k.reshape(-1,1)          
        
        nn = nn.astype(np.complex128)                                        # coefficients must be complex in general
        # print(nn.shape,*nn)
        # nn = nn.reshape(-1,1)                                                # make sure it is a column vector

        if not np.nonzero(nn) and np.max(np.nonzero(nn)) > N_space:
            nn = nn[0:N_space-1]                                             # truncate the hilbert space 
            nonzero_indices = np.nonzero(nn)[0]
        else:
            nonzero_indices = np.nonzero(nn)[0]                              # find the nonzero elemets, we need them 
            if len(nonzero_indices) != 0:                                    # because we want to know which number states
                nn = nn[:nonzero_indices[-1] + 1]                            # have are multiplied by nonzero coefficients
            else:
                nn = np.array([0])
        
        if len(nonzero_indices) != 0:
            self.Coeff = np.concatenate((nn, np.zeros(N_space - len(nn))))   # assign the input values to class properties
            self.Kets  = np.arange(N_space)                                  # because the basis starts from |0>
            self.n     = np.nonzero(self.Coeff)[0]                           # assign the populated number states
            self.N_Hilbert = N_space
        else:
            self.Coeff = np.array([0])
            self.Kets  = np.array([0]) 
            self.n     = np.array([0])                                   # assign the populated number states
            self.N_Hilbert = 1
        # --------------------------------------------------------------------------------------------------------------
        # Calculate coefficients of the associated Laguerre Polynomials, L_{n}^{k}, up to m-th order
        # Here we calculate coefficients of the associated Laguerre Polynomials,  L_{n}^{k} ,up to  m-th order.
        # for more info check   https://mathworld.wolfram.com/AssociatedLaguerrePolynomial.html
        # We will use then in the Displacement operator. To that end, we calculate them once and then use their values through the matrix .
        # We do so, because they contain factorials, which is a time consuming calculation, and we aim to use factorials a less as possible .
        self.__Lnk = np.zeros((N_space, N_space, N_space))   # the "__" in front of Lnk defines a Private attribute that can be used only within the class
        for k in range(N_space):
            for nn in range(N_space):
                for m in range(nn + 1):
                    self.__Lnk[nn, m, k] = 1 / factorial(m) * comb(k + nn, nn - m) * (-1) ** m

        # -----------------------------------------------------------------------------------------------------------------
        # The following matrix "obj.sqrt_n_over_m" is a lower-triangular matrix that contains the 
        # values of sqrt(n!/m!) needed for the Dispalcement operator written in the the Fock basis, 
        # see https://physics.stackexchange.com/questions/553225/representation-of-the-displacement-operator-in-number-basis
        self.__sqrt_n_over_m = np.tril(np.zeros((N_space, N_space)))
        for m in range(N_space):
            for nn in range(m + 1):
                self.__sqrt_n_over_m[m, nn] = np.sqrt(factorial(nn) / factorial(m))
        
        # Parity matrix
        self.__Parity = np.diag(np.cos(np.arange(N_space) * np.pi))

    def normalize(self):
        # Normalizes the input state so that ⟨ψ|ψ⟩ = 1.
        # It does not affect the global phase. For instance |ψ⟩ = (1 + 1j)|n⟩ ----> |ψ⟩ = (0.707 + 0.707j)|n⟩.
        # If you want to remove or change the global phase you must do using other methods shown below 
        self.Coeff = self.Coeff / np.sqrt(self.braket())

    def braket(self, other=None):
        # Computes the dot product ⟨n_1|n_2⟩. If only one argument is provided, computes ⟨n_1|n_1⟩.
        # Parameters:(FockBasis obj, optional)  The other FockBasis object to compute the dot product with. Defaults to None.
        # 
        # Returns:
        # complex (in general): The dot product  ⟨self|other⟩.
        
        if other is None:
            other = copy.deepcopy(self)
        
        l      = max(len(self.Coeff), len(other.Coeff))
        Coeff1 = np.concatenate((self.Coeff, np.zeros(l - len(self.Coeff))))
        Coeff2 = np.concatenate((other.Coeff, np.zeros(l - len(other.Coeff))))
        
        return np.vdot(Coeff1, Coeff2)
        
    def print_state(self,s=''):
        # This method/function prints the CoherentBasis object in \ket ( |⋅ ⟩ ) notation.
        # This is only for printing/illustration purposes and simple sanity checks!!
        # To get the precise coeeficients and kets use    self.Coeffs and self.Kets !!
        #
        # Inputs: 
        #    s  : string type. This input puts a "subscript" to the printed state. 
        # Outputs:
        #    s1 : string type,  |ψ_k⟩ = Coeff[1] | Kets[1] ⟩ + Coeff[2] | Kets[2] ⟩ + .....
        #    s2 : string type,          Coeff[1] | Kets[1] ⟩ + Coeff[2] | Kets[2] ⟩ + .....
        s1 = f'|n{s}⟩ = '
        s2 = ''
        for ii in self.n:
            coeff_str = lib.compact_complex(self.Coeff[ii])
            ket_str   = str(self.Kets[ii])
           
            if abs(self.Coeff[ii].real) and abs(self.Coeff[ii].imag):
                s2 += f'({coeff_str}) |{ket_str}⟩'
            else:
                s2 += f'{coeff_str} |{ket_str}⟩'

            if ii < max(self.n):
                s2 += '  +  '
        s1 += s2
        return s1, s2
    
    def __add__(self,other):
        # This function adds up two objects,  |ψ_3> = |ψ_1> + |ψ_2>   (it DOES NOT normalize the final vector/object) .
        # If a Number state is included in both |ψ_1>,|ψ_2> we merge its coefficients 
        l1 = len(self.Coeff)
        l2 = len(other.Coeff)
        st1 = copy.deepcopy(self)
        st2 = copy.deepcopy(other)
        
        if l1 == l2:
            coeff = st1.Coeff + st2.Coeff
        elif l1 > l2:
            coeff= st1.Coeff + np.concatenate((st2.Coeff, np.zeros(l1 - l2)))
        else:
            coeff = st2.Coeff + np.concatenate((st1.Coeff, np.zeros(l2 - l1)))

        return FockBasis(coeff, max(l1, l2)) # the final object/vector has dimensions  max(l1,l2)
    
    def __mul__(self, scalar):
        # Multiplies the coefficients by a scalar
        new_coeff = scalar * self.Coeff
        return FockBasis(new_coeff, self.N_Hilbert)

    def __rmul__(self, scalar):
        # This ensures multiplication works if the scalar is on the left
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        # Divides the coefficients by a scalar
        new_coeff = self.Coeff / scalar
        return FockBasis(new_coeff, self.N_Hilbert)

    def D_(self, a):
        # Displacement operator function 
        # We calculate D(a) in the number basis. We truncate the operator in the N_space Hilbert space .
        # For large Hilbert space D(a)|ψ> approaches the exact result but the code becomes slower - KEEP THAT IN MIND.
        # We calculate    <m|D(a)|n> = D_{m,n}(a) using Glauber's formula, 
        # or this link  https://physics.stackexchange.com/questions/553225/representation-of-the-displacement-operator-in-number-basis.
        N = len(self.Coeff)                     
        D = np.zeros((N, N), dtype=complex)     # initialize D
        
        for nn in range(N):
            for m in range(nn, N):
                # This part calculates the      m >= n    elements
                # LaguerreLnk = L_{n}^{k}(x)  where x = |a|^2 in our case.
                # We compute  L_{n}^{k}(x)  using the matrix we defined in the constructor    " function obj = Nbasis(n,N_space) "  .

                LaguerreLnk = np.sum(self.__Lnk[nn, :nn+1, m-nn] * (np.abs(a) ** (2 * np.arange(nn+1))))
                
                if m == nn:
                    # diagonal elements
                    D[m, nn] = self.__sqrt_n_over_m[m, nn] * (a ** (m - nn)) * np.exp(-0.5 * np.abs(a) ** 2) * LaguerreLnk
                else:
                    # lower triangular matrix elements
                    D[m, nn] = self.__sqrt_n_over_m[m, nn] * (a ** (m - nn)) * np.exp(-0.5 * np.abs(a) ** 2) * LaguerreLnk

                    # upper trangular matrix elements
                    D[nn, m] = self.__sqrt_n_over_m[m, nn] * ((-a.conjugate()) ** (m - nn)) * np.exp(-0.5 * np.abs(a) ** 2) * LaguerreLnk
        
        new_coeff = np.dot(D, self.Coeff)
        new_obj = FockBasis(new_coeff, self.N_Hilbert)
        return new_obj, D
    
    def A(self):
        # Annihilation operator a,            A(c|n⟩) = c*sqrt(n)|n-1⟩
        new_coeffs = np.sqrt(self.Kets[1:]) * self.Coeff[1:]
        new_obj    = FockBasis(np.concatenate((new_coeffs, [0])), self.N_Hilbert)
        return new_obj
    
    def A_dagger(self):
        # Creation Operator a^†,   A^†(c|n⟩) = c*sqrt(n+1)|n+1⟩
        new_coeffs = np.sqrt(self.Kets[1:]) * self.Coeff[:-1]
        return FockBasis(np.concatenate(([0], new_coeffs)), self.N_Hilbert)
    
    def PhotonNumber(self):
        # This function calculates the average number of photons in the state and the photon distribution.
        # Inputs: 
        #    obj = object of the class  .
        # Outputs:
        #    average_num : scalar, average photon number
        #    Photon_distribution = np.array column vector, number of photons in every Fock State |n⟩.
        self.normalize()
        P = np.abs(self.Coeff) ** 2                          # Photon distribution P(n)
        average_num = np.dot(P, np.arange(len(self.Coeff)))  # <n> average photon number
        return average_num, P
    
    def D__(self, a):                                       # This is used only in the WignerFunction2 calculation
        N = len(self.Coeff)
        D = np.zeros((N, N), dtype=complex)
        exp_factor = np.exp(-0.5 * np.abs(a) ** 2)
        abs_a_sq = np.abs(a) ** 2
        for nn in range(N):
            abs_a_pow = abs_a_sq ** np.arange(nn + 1)
            for m in range(nn, N):
                LaguerreLnk = np.sum(self.__Lnk[nn, :nn+1, m-nn] * abs_a_pow)
                D_factor = self.__sqrt_n_over_m[m, nn] * exp_factor * LaguerreLnk
                if m == nn:
                    D[m, nn] = D_factor * (a ** (m - nn))
                else:
                    D[m, nn] = D_factor * (a ** (m - nn))
                    D[nn, m] = D_factor * ((-a.conjugate()) ** (m - nn))

        new_coeff = np.dot(D, self.Coeff)
        return new_coeff, np.conjugate(new_coeff).T

    def WignerFunction(self, x_max, Nx, x_min = None, y_min=None, y_max = None, Ny = None, num_chunks=16):                       # optimized WignerFunction
        # Wigner Function
        # Inputs : 
        #          obj   : the object/state to calculate the Wigner-function.
        #          x_max : maximum x value of the grid
        #          Nx    : number of points in x direction. 
        #          y_max : maximum y value of the grid
        #          Ny    : number of points in y direction. 
        # Outputs: 
        #          W     : Wigner quasiprobability distribution, Nx x Ny matrix, computed in [(-x_max,x_max),(-y_max,y_max)] .
        #
        if y_max == None:                                 # If y_max is not given, Wigner function is computed in a square grid 
            y_max = x_max                                 # defined by x_max and Nx
            Ny = Nx
        elif Ny == None:                                  # If Ny is not given, then Ny = Nx
            Ny = Nx          
        if x_min == None: x_min = -x_max
        if y_min == None: y_min = -y_max         

        x = np.linspace(x_min, x_max, Nx)                 # creates the x-axis data points
        y = np.linspace(y_min, y_max, Ny)                 # creates the y-axis data point
        # X, Y = np.meshgrid(x, y)                        # assigns the grid to matrices
        # B = X + (1j * Y)                                # independent variable  W = W(b)
        W = np.zeros((Nx, Ny), dtype=np.complex128)       # initializes the matrix, memory allocation

        st = copy.deepcopy(self)
        st.normalize()
        parity = st.__Parity

        # Create a shared memory array
        shared_array = mp.Array('d', 2 * Nx * Ny)  # Using 'd' for double precision (complex numbers need 2 * N * N)
        with mp.Pool(initializer=init_shared_arrays, initargs=(shared_array, Nx)) as pool:
            # Divide the work into chunks
            chunk_size = Ny // num_chunks
            chunks = []
            for i in range(0, Nx, chunk_size):
                for j in range(0, Ny, chunk_size):
                    start_i = i
                    end_i = min(i + chunk_size, Nx)
                    start_j = j
                    end_j = min(j + chunk_size, Ny)
                    chunks.append((x, y, st, parity, start_i, end_i, start_j, end_j))
            
            pool.map(compute_wigner_chunk, chunks)

        W = np.frombuffer(shared_array.get_obj(), dtype=np.complex128).reshape((Nx, Ny))
        W = W * 2 / np.pi
        return W

    def Q_function(self, x_max, Nx, x_min = None, y_min=None, y_max = None, Ny = None):
        # Husimi-Q function
        # Inputs: obj   = the object/state to calculate the Q-function.
        #         x_max = maximum x (and y -- square grid y_max = x_max). Q-function will be computed between [(-x_max,x_max),(-y_max,y_max)].
        #         N     = number of points in each direction. The final matrix Q will have dimensions of NxN.
        # Outputs: Q    = Husimi distribution, NxN matrix
        #
        if y_max == None:                                  # If y_max is not given, Wigner function is computed in a square grid 
            y_max = x_max                                  # defined by x_max and Nx
            Ny = Nx
        elif Ny == None:                                   # If Ny is not given, then Ny = Nx
            Ny = Nx 
        if x_min == None: x_min = -x_max
        if y_min == None: y_min = -y_max         

        st = copy.deepcopy(self)
        st.normalize()
        k   = np.arange(len(st.Coeff))                     # size of Hilbert space
        x   = np.linspace(x_min, x_max, Nx)                # creates the x-axis data points
        y   = np.linspace(y_min, y_max, Ny)                # creates the y-axis data points
        X, Y = np.meshgrid(x, y)                           # assigns the grid to matrices
        B = X + 1j * Y                                     # independent variable  W = W(b)
        Q = np.zeros((Nx, Ny), dtype=complex)              # initializes the matrix, memory allocation

        for i in range(Nx):
            for j in range(Ny):
                Q[i, j] = np.sum(((B[i, j]) ** (k)) / np.sqrt(factorial(k)) * st.Coeff)

        Q = 1/np.pi * np.abs(np.exp(-1/2 * np.abs(B) ** 2) * Q) ** 2  # Q function (ready for plot)

        return Q
    
    def S_(self,z, N_space = None):
        #  Squeeze operator function
        if N_space == None: N_space = self.N_Hilbert
        coeffs = self.Coeff
        kets   = self.Kets
        if N_space>self.N_Hilbert:
            coeffs = np.concatenate((coeffs,np.zeros(N_space-self.N_Hilbert)))
            kets   = np.array([ii for ii in range(N_space)])

        a = np.diag(np.sqrt(kets[1:]), k=1)
        a_dag = np.diag(np.sqrt(kets[1:]), k=-1)

        # Compute the squeeze operator
        squeeze_operator = scipy.linalg.expm(0.5 * (np.conj(z) * np.dot(a, a) - z * np.dot(a_dag, a_dag)))
        new_Coeffs = np.dot(squeeze_operator, coeffs)
        return FockBasis(new_Coeffs,N_space)

