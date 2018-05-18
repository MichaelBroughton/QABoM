# QABoM
Simple demo of some logic used in https://arxiv.org/abs/1712.05304 This Code is intended to showcase some baseline structure for using a quantum computer to do the training of a Restricted Boltzmann Machine. (Note: Training more general connectivity Boltzmann machines is possible by extending this code.)

# Abstract
The question has remained open if near-term gate model quantum computers will offer a quantum
advantage for practical applications in the pre-fault tolerance noise regime. A class of algorithms
which have shown some promise in this regard are the so-called classical-quantum hybrid variational
algorithms. Here we develop a low-depth quantum algorithm to train quantum Boltzmann
machine neural networks using such variational methods. We introduce a method which employs
the quantum approximate optimization algorithm as a subroutine in order to approximately sample
from Gibbs states of Ising Hamiltonians

# Requirements
To run this code you will need python 2.7. However the code can be easily modified to work on python3. You will need numpy, scipy, requests[security], pyquil, and grove from Rigetti. This can be done with

  ```$ pip install --user requests[security] numpy scipy pyquil quantum-grove```
  
pyquil and grove require python3 however it can be made to work with python2 by installing requests[security].

Finally you will also need to head over the Rigetti website: https://www.rigetti.com/forest and get yourself an API key (it's free) to be able to use their quantum simulator.

# Getting Started

Once you have signed up for Forest, registered your Forest API key with you config, and installed everything you should be able to run a simple test program such as:

```python

from pyquil.quil import Program
import pyquil.api as api
from pyquil.gates import *
qvm = api.QVMConnection()
p = Program()
p.inst(H(0), CNOT(0, 1))
    <pyquil.pyquil.Program object at 0x101ebfb50>
wavefunction = qvm.wavefunction(p)
print(wavefunction)
    (0.7071067812+0j)|00> + (0.7071067812+0j)|11>
 ```
 
 Now to start with your RBM:
 Begin wih creating a qvm connection.
 Then specify the number of visible and hidden units you would like to have. NOTE: (num_visible + num_hidden) * 2 <= number of simulable qubits = 19 (So no you definitly can't run this on MNIST). As well you can choose how many measurements of your circuit to do (a value of None calculates things partially analytically so the simulations will be much faster!):
 
 ```python
qvm = api.SyncConnection()
r = qRBM(qvm, num_visible=4, num_hidden=1, n_quantum_measurements=None, verbose=True)
```

Then we will create a very simple artificially high dimensional dataset

```python
simple_data = [[1,1,-1,-1], [1,1,-1,-1], [-1,-1,1,1], [-1,-1,1,1]]
```

Then We can train our RBM, specifying the number of epochs, learning rate as well as what ratio of classical to quantum statistics we want to use in our update rule. The following uses 70% quantum and 30% classical (CD-1):
```python
r.train(simple_data, n_epochs=100, quantum_percentage=0.7, classical_percentage=0.3)
```
Finally we can transform our data using the RBM to see what we get and If all went well it should be able to distinguish the two and you should see something like:

```python
print r.transform(simple_data)
[[0.99965316]
 [0.99653161]
 [0.00034683]
 [0.00046839]]

```
# Notes
As is mentioned in the ealier it is also possible to use this methodology and extend this code to train more general connectivity boltzmann machines rather than just having to stick with the RBM. This can be done by using an entirely quantum update rule featuring the data clamping QAOA we propose in the paper, along with whatever connectivity you are after that you will define in your driver/mixer + cost hamiltonians.

Also don't hesitate to share if you find any improvements/bugs in the code (This version was mostly rewritten from the original to be as clear as possible) to make it more user friendly :)
