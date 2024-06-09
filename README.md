# Quantum Optics Lab
Quantum Optics Lab is a side project that I work on my free time. The main goal is to incorporate as many quantum optics tools and features as possible. 

## Requirements & Installation
You can download and install the repo using the following commands. Open a terminal window in the directory you want to save the files and type:
```
git clone https://github.com/Filippos-Dakis/Quantum-Optics-Lab.git
cd Quantum-Optics-Lab/
```
It is recommend to install dependencies in a virtual environment to avoid messing your system-wide Python installation. Notice that you need to also install Jupyter notebook to execute the examples. Please, follow the instructions below conditional to your system:
- Unix
  ```
  python -m venv venv
  . venv/bin/activate
  pip install -r requirements.txt
  pip install jupyter notebook
  python -m jupyter notebook
  ```
- Windows
  ```
  py -(version, ie 3.11) -m venv .venv
  .\.venv\Scripts\activate
  py -m pip install -r requirements.txt
  py -m pip install jupyter notebook
  python -m jupyter notebook
  ```
  ## Information and Guidelines
  All the scripts were developed with Python 3.11.9, but they are expected to work for all 3.0+ python versions.
  
  This reciprocity contains two classes:
  - src/CoherentBasis.py
  - src/FockBasis.py
  
  which define two types of objects. The first one, CoherentBasis.py , refers to Coherent states basis of the Qunatum Harmonic Oscillator (QHO), while the second, FockBasis.py, refers to the number/Fock states 
  of the QHO.
  
  The reciprocity also includes two jupyter notebook examples:
  - CoherentBasis_Example_1.ipynb
  - FockBasis_Example_1.ipynb
  
  that help the user explore all the features of the two main classes step by step.

  The key feature of these two classes (CoherentBasis, FockBasis) is the calculation of the **Wigner Quasi-Probability distribution** and the **Q-Husimi Distribution** for any given (pure) state! However, they also include other useful features such as quantum state addition, normalization, annihilation and creation operators, Displacement operations, Squeezing operations, etc.

  For instance, **CoherentBasis_Example_1.ipynb** produces the following Wigner Quasi-Probability distributions

  ![Wigner_Coherentbasis](https://github.com/Filippos-Dakis/Quantum-Optics-Lab/assets/114699564/9321c021-48da-45e7-b1c8-04c42e4dcebe)


  and **FockBasis_Example_1.ipynb** produces these three Wigner Distributions

  ![Wigner_Fockbasis](https://github.com/Filippos-Dakis/Quantum-Optics-Lab/assets/114699564/d1ed3e61-2c66-4bb4-a104-997a32807ef4)

  Similar figures are produced for the Q-Husimi function. Also, you can get the photon number distribution for any kind of state. Here is an example of squeezed states

  ![Squeezed_states](https://github.com/Filippos-Dakis/Quantum-Optics-Lab/assets/114699564/b2a510dc-f83b-4a65-a710-4bc26fee797e)

## Future extensions
  This is the very first version and I intend to incorporate more features. Some of them are:  
  - density matrix representation 
  - state vector evolution for a given Hamiltonian
  - switch between the two bases (connect the two classes)
  - your suggestions
 
Suggested textbooks:
- Use "Cat Qubtis.pdf" as a quick reference (especially the appendix)
- Exploring the Quantum Atoms, Cavities and Photons.  Serge Haroche Jean-Michel Raimond
- Introductory Quantum Optics. Christopher Gerry Peter Knight

I am open to other suggestion. Please feel free to contact me at dakisfilippos@vt.edu

If you found my code useful, please cite is as  [https://github.com/Filippos-Dakis/Wigner-Function-Quantum-Optics](https://github.com/Filippos-Dakis/Quantum-Optics-Lab)

Any feedback and suggestions are much appreciated! 

More features are on the way! Stay tuned !!

