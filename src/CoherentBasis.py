# Filippos Dakis      Virginia Tech   May 2024
#
# Version V 1.0.0
# 
# The present script defines a class whose objects behave as coherent states.
# Every object of this class has all the basic features of a quantum coherent state.
# Namely, one can define a state, normalize it, compute overlaps between states, 
# add quantum states, apply creation and annihilation operators, perform displacement 
# operators, compute the Wigner and Q-Husimi function, calculate he photon distribution,
# and the average photo number.
#
#
# This is a continuous personal project where I will be adding more features every once in a while.
# 
# Stay tuned !!

import numpy as np
# import scipy as sc
# import pandas as pd
import copy
import collections.abc
import src.Library as lib
from src.FockBasis import FockBasis


class CoherentBasis:
    def __init__(self, c, s):
        # Constructor of the CoherentBasis class
        # Inputs:
        #   c : numpy array, column vector containing the coefficients of every coherent state,
        #       i.e., c = [1, 1j] = |s[0]⟩ + 1j|s[1]⟩.
        #   s : numpy array, column vector containing the arguments of input coherent states,
        #       i.e., s= [0, 2.4, 1+1j]  ------->  |0⟩ , |2.4⟩ ,  |1+1j⟩.
        if not isinstance(c, np.ndarray):                 # check if the input is a numpy.array 
            if isinstance(c, collections.abc.Sequence):   # check if it is a scalar or a vector
                c = np.array(c)                           # convert vector to numpy.array 
            else: c = np.array([c])                       # convert scalar to numpy.array 
        if not isinstance(s, np.ndarray):                 # check if the input is a numpy.array 
            if isinstance(s, collections.abc.Sequence):   # check if it is a scalar or a vector
                s = np.array(s)                           # convert vector to numpy.array 
            else: s = np.array([s])                       # convert scalar to numpy.array

        if len(c) == len(s):
            self.Coeff = c[abs(c)> 0]                         # keep only the non-zero elements
            self.Kets  = s[abs(c)> 0]                         # keep only the kets that have non-zero coeficient in front
            self.Coeff = self.Coeff.astype(np.complex128)     # make sure both arrays are complex
            self.Kets  = self.Kets.astype(np.complex128)      # 
        else:
            raise ValueError("Coefficients and arguments must have the same length.")
        self.Coeff.reshape(-1,1)  
        self.Kets.reshape(-1,1)  

    def print_state(self, s ='',s_dgts = 3):
        # This method/function prints the CoherentBasis object in \ket ( |⋅ ⟩ ) notation.
        # This is only for printing/illustration purposes and simple sanity checks!!
        # To get the precise coeeficients and kets use    self.Coeffs and self.Kets !!
        #
        # Inputs: 
        #    s  : string type. This input puts a "subscript" to the printed state.
        #    s_dgts : scalar > 0, sets the significant digits accuracy for the printed numbers  
        # Outputs:
        #    s1 : string type,  |ψ_k⟩ = Coeff[1] | Kets[1] ⟩ + Coeff[2] | Kets[2] ⟩ + .....
        #    s2 : string type,          Coeff[1] | Kets[1] ⟩ + Coeff[2] | Kets[2] ⟩ + .....
        #
        s1 = f'|ψ{s}⟩ = '
        s2 = ''
        for ii in range(len(self.Coeff)):                      # This loop goes through self.Coeff and self.Kets and transform them
            coeff_str = lib.compact_complex(self.Coeff[ii],sign_digits = 3)    # into string type compact complex complex numbers. For instance,      
            ket_str   = lib.compact_complex(self.Kets[ii],sign_digits = s_dgts)     # (1.000002 + 3.17563333j) |2 + 0j⟩ ---->  (1 + 3.175j) |2 ⟩

            if abs(self.Coeff[ii].real) and abs(self.Coeff[ii].imag):   # The following commands assemble the final strings
                s2 += f'({coeff_str}) |{ket_str}⟩'
            else:
                s2 += f'{coeff_str} |{ket_str}⟩'

            if ii < len(self.Coeff)-1:
                s2 += '  +  '
        s1 += s2
        return s1, s2


    def normalize(self):
        # Normalizes the input state so that ⟨ψ|ψ⟩ = 1.
        # It does not affect the global phase. For instance |ψ⟩ = (1 + 1j)|a⟩ ----> |ψ⟩ = (0.707 + 0.707j)|a⟩.
        # If you want to remove or change the global phase you must do using other methods shown below  
        self.Coeff = self.Coeff / np.sqrt(self.braket())


    def braket(self,obj = None):
        # Calculate the braket (inner product) of the state and returns  ⟨ψ|φ⟩ = np.vdot(ψ,φ)
        # Inputs:
        #   obj : CoherentBasis object. state vector to be the |φ⟩ of the inner product
        #         If there is no input then the function returns the norm of the self object, namely ⟨ψ|ψ⟩ = N 
        if obj is None:
            a = np.conj(self.Coeff)
            b = self.Kets
        else:
            a = np.conj(obj.Coeff)
            b = obj.Kets
        q = 0
        for m in range(len(self.Coeff)):
            for n in range(len(a)):
                q = q + self.Coeff[m] * a[n] * np.exp(-1/2 * (np.abs(self.Kets[m])**2 + np.abs(b[n])**2 - 2*np.conj(self.Kets[m]) * b[n] ) )
        return q
    

    def A(self):
        # Annihilation operator a,            A(c|α⟩) = (α + c)|α⟩
        new_Coeff = np.multiply(self.Coeff,self.Kets)  # α + c
        return CoherentBasis(new_Coeff, self.Kets)


    def A_dagger(self, m = 1, N_hilbert = 35):
        # Creation Operator a^†
        # The final output will be a state written in the FOCK/NUMBER  basis !!!!!
        # This is due to the peculiarity of the creation operator (a^†) that does not have eigen-KET.
        #
        #  For more information please see   ---- Phys. Rev. A 43, 492 ----  especially Eq. (2.9)
        #
        # Inputs:
        #        m          = the power at which we raise a^†,  (a^†)^m
        #        N_hilbert  = hilbert space of the output state (written in the FOCK/NUMBER basis)
        #                     the higher the number of average photons  n a coherent state the larger the
        #                     Hilbert space should be
        # Outputs:
        #        FockBasis object. The result is also known as "Agrawal State", see Phys. Rev. A 43, 492 .
        
        # Create zero number state 1|0⟩ in FOCK basis
        zero_number_state = FockBasis(nn = 1.,N_space = N_hilbert)

        # Generate a^† matrix representation in FOCK basis
        a_dagger = np.diag(np.sqrt(np.arange(1, N_hilbert)), k=-1)

        # Memory allocation
        new_coeffs = 0*zero_number_state.Coeff  

        for ii in range(len(self.Kets)):
            #
            # (a^†)^m |a_i⟩ = (a^†)^m D(a_i)|0⟩,   for each coherent state of the input object
            zer0       = zero_number_state.D_(self.Kets[ii])[0]    # create the D(a_i)|0⟩ in Fock basis 
            new_coeffs = new_coeffs + np.dot(np.linalg.matrix_power(a_dagger, m),zer0.Coeff) * self.Coeff[ii]
    
        return FockBasis(new_coeffs, N_hilbert)
          

    def D_(self,z):      
        # This method performs a displacement on the input state      
        # Displacement operator   D(z)|a⟩ = exp(i*Im(z·a^*))|a+z⟩
        Coeff = np.multiply(self.Coeff, np.exp(1j * np.imag(z * np.conj(self.Kets))))
        Kets  = self.Kets + z
        return CoherentBasis(Coeff, Kets)


    def __add__(self, other):
        # This operation overload defines the addition of two CoherentBasis objects
        # This allows for actions like |ψ_c⟩ = |ψ_a⟩ + |ψ_b⟩ = c_a[1]|s_a[1]⟩ + ..... + c_b[1]|s_b[1]⟩ + ....
        # If the same coherent exist in |ψ_a⟩ and |ψ_b⟩, then the coefficients infront 
        # of them are added and the final state contains only one copy of it (as it should).
        # 
        #
        # The output a CoherentBasis object

        st1 = copy.deepcopy(self)                  # create copies so the original objects are not affected
        st2 = copy.deepcopy(other)                 #

        if len(st1.Coeff.shape) == 1:
            st1.Coeff = st1.Coeff.reshape(st1.Coeff.shape[0],1)
            st1.Kets  = st1.Kets.reshape(st1.Coeff.shape[0],1)
        if len(other.Coeff.shape) == 1:
            st2.Coeff = st2.Coeff.reshape(st2.Coeff.shape[0],1)
            st2.Kets  = st2.Kets.reshape(st2.Coeff.shape[0],1)

        coeff = np.vstack((st1.Coeff, st2.Coeff))
        kets  = np.vstack((st1.Kets, st2.Kets))

        unique_kets, indices = np.unique(kets, axis=0, return_index=True)
        coeff_sum = np.array( [np.sum(coeff[np.where(kets == ket)[0]]) for ket in unique_kets] ).reshape(-1, 1)

        idx = np.argsort(np.abs(unique_kets[:, 0]))     # we sort "kets" and "coeffss" for coding purposes only
        unique_kets = unique_kets[idx]                  # 
        coeff_sum   = coeff_sum[idx]                    #

        return CoherentBasis(coeff_sum, unique_kets)
    

    def __mul__(self, scalar):
        # This operation overload defines the scalar multiplication with CoherentBasis objects
        # The output is: |ψ`⟩ = f * |ψ⟩ = (f*c[1])|s[1]⟩ + (f*c[2])|s[2]⟩ + (f*c[3])|s[3]⟩ + ....
        new_coeff = scalar * self.Coeff
        return CoherentBasis(new_coeff, self.Kets)


    def __rmul__(self, scalar):
        # This ensures the scalar multiplication works if the scalar is on the right 
        # The output is: |ψ`⟩ = |ψ⟩ * f = (f*c[1])|s[1]⟩ + (f*c[2])|s[2]⟩ + (f*c[3])|s[3]⟩ + ....
        return self.__mul__(scalar)
    
    def __truediv__(self, scalar):
        # Divides the coefficients by a scalar
        new_coeff = self.Coeff / scalar
        return CoherentBasis(new_coeff,self.Kets)


    def PhotonNumber(self,N_hilbert = 30):
        # This function calculates the average number of photons in the state and the photon distribution.
        # Inputs: 
        #    obj = object of the class  .
        #    N_hilbert : scalar, truncates the infinite Hilbert to first N_hilbert Fock States
        # Outputs:
        #    average_num : scalar, average photon number
        #    Photon_distribution = np.array column vector, number of photons in every Fock State |n⟩.
        obj = copy.deepcopy(self)
        obj.normalize()                                      # normalizes |ψ⟩
        Fock_state = np.zeros(N_hilbert, dtype=complex)
        for i in range(N_hilbert):
            # projection on Fock states   ⟨n|a⟩= exp(-1/2 |a|^2) (a^n)/sqrt(n!)
            Fock_state[i] = (np.sum( obj.Coeff * np.exp(-1 / 2 * np.abs(obj.Kets) ** 2) * (obj.Kets ** (i)))
                                                    / np.sqrt(float(np.math.factorial(i))))
        Photon_Distribution = np.abs(Fock_state) ** 2                               # P(n) = |<n|ψ⟩|^2
        average_num         = np.sum(Photon_Distribution * np.arange(N_hilbert))    # <n> = Sum_n P(n)*n
        
        return average_num, Photon_Distribution.reshape(-1,1)


    def Q_function(self, x_max, Nx, y_max = None, Ny = None):
        # Husimi-Q function
        # Inputs : 
        #          obj   : the object/state to calculate the Q-function.
        #          x_max : maximum x value of the grid
        #          Nx    : number of points in x direction. 
        #          y_max : maximum y value of the grid
        #          Ny    : number of points in y direction. 
        # Outputs: 
        #          Q     = Husimi distribution, Nx x Ny matrix,  Q-function is computed in [(-x_max,x_max),(-y_max,y_max)] .
        
        if y_max == None:         # If y_max is not given, Wigner function is computed in a square grid 
            y_max = x_max         # defined by x_max and Nx
            Ny = Nx
        elif Ny == None:          # If Ny is not given, then Ny = Nx
            Ny = Nx          

        x_max = abs(x_max)        # make sure x_max is always positive
        y_max = abs(y_max)        # make sure y_max is always positive

        selff = copy.deepcopy(self)                        
        selff.normalize()                             # normalizes input state |ψ⟩
    
        x = np.linspace(-x_max, x_max, Nx)            # creates the x-axis data points
        y = np.linspace(-y_max, y_max, Ny)            # creates the y-axis data point
        X, Y = np.meshgrid(x, y)                      # assigns the grid to matrices
        B = X + (1j * Y)                              # independent variable  W = W(b)
        Q = np.zeros((Nx, Ny),dtype=complex)          # initializes the matrix, memory allocation

        for m in range(len(selff.Coeff)):
            Q += selff.Coeff[m] * np.exp( -1 / 2 * (np.abs(B) ** 2 + np.abs(selff.Kets[m]) ** 2 - 2 * np.conj(B) * (selff.Kets[m])) )

        Q = 1 / np.pi * np.abs(Q) ** 2                # Q function (ready for plot)
        return Q.real

    def WignerFunction(self, x_max, Nx, y_max = None, Ny = None):
        # Wigner Function
        # Inputs : 
        #          obj   : the object/state to calculate the Wigner-function.
        #          x_max : maximum x value of the grid
        #          Nx    : number of points in x direction. 
        #          y_max : maximum y value of the grid
        #          Ny    : number of points in y direction. 
        # Outputs: 
        #          W     : Wigner quasiprobability distribution, Nx x Ny matrix, computed in [(-x_max,x_max),(-y_max,y_max)] .

        if y_max == None:       # If y_max is not given, Wigner function is computed in a square grid 
            y_max = x_max       # defined by x_max and Nx
            Ny = Nx
        elif Ny == None:        # If Ny is not given, then Ny = Nx
            Ny = Nx          
    
        selff = copy.deepcopy(self)                        
        selff.normalize()                                  # normalizes input state |ψ⟩

        x = np.linspace(-x_max, x_max, Nx)                 # creates the x-axis data points
        y = np.linspace(-y_max, y_max, Ny)                 # creates the y-axis data point
        X, Y = np.meshgrid(x, y)                           # assigns the grid to matrices
        B = X + (1j * Y)                                   # independent variable  W = W(b)
        W = np.zeros((Nx, Ny), dtype=np.complex128)        # initializes the matrix, memory allocation

        for m in range(len(selff.Coeff)):
            for n in range(len(selff.Coeff)):
                # We compute all the possible combinations of coeff(a_2^*) * coeff(a_1) <a_2|D†(-b) Π D(-b)|a_1⟩.
                # In the following formula, we have already done some algebra, in detail we used Parity's properties to obtain:
                # ⟨a_2|D†(-b) Π D(-b)|a_1⟩ = ⟨a_2|D†(-b) D(b) Π |a_1⟩ = ⟨a_2|D†(-b) D(b)|- a_1⟩ =
                # ⟨a_2|D†(-b) = exp(1j * imag(conj(a_2) * b)) ⟨a_2 - b|
                # D(b)|- a_1⟩ = exp(1j * imag(conj(-a_1) * b)) |b - a_1⟩
                # ⟨a_2 - b|b - a_1⟩ = exp(-1/2 * (|a_2 - b|^2 + |b - a_1|^2 - 2 * conj(a_2 - b) * (a_1 - b)))
                phase    = np.exp(1j * (np.imag(np.conj(selff.Kets[m]) * B) + np.imag(-np.conj(selff.Kets[n]) * B)))
                argument = np.exp(-1/2 * (np.abs(selff.Kets[m] - B)**2 + np.abs(selff.Kets[n] - B)**2 -
                                           2 * np.conj(selff.Kets[m] - B) * (B - selff.Kets[n])))
                W += selff.Coeff[m].conj() * selff.Coeff[n] * phase * argument

        W *= 2 / np.pi        # Wigner function (ready for plot)
        return W
