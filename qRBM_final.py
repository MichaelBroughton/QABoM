import numpy as np
from scipy.optimize import minimize
from scipy.optimize import fmin_bfgs


import pyquil.quil as pq
import pyquil.api as api
from pyquil.paulis import *
from pyquil.gates import *


from grove.pyqaoa.qaoa import QAOA
from grove.pyvqe.vqe import VQE


import json
import copy


class qRBM:

	"""
	Quantum Classical Hybrid RBM implementation.
	"""


	def __init__(self, QM, num_visible, num_hidden, n_quantum_measurements=None, verbose=False):
		"""
		create an RBM with the specified number of visible and hidden units

		Params
		-------------------------------------------------------------

		QM:						(rigetti QVM connection) QVM connection for which to use for quantum circuit simulation
		num_visible:			(int) Number of visible units in RBM
		num_hidden				(int) Number of hidden units in RBM
		n_quantum_measurements: (int) Number of measuremants to use for Quantum expectation estimation (default to None which does analytical)
		verbose:				(bool) Verbosity of qRBM

		--------------------------------------------------------------
		"""
		self.n_visible = num_visible
		self.n_hidden = num_hidden
		self.qvm = QM
		self.verbose = verbose
		self.n_quantum_measurements = n_quantum_measurements


		#tweak at your leisure.
		self.n_qaoa_steps = 1

		self.beta_temp = 2.0



		#don't tweak below here unless you know what you're doing.


		# only want this for built in expectation calculations... 
		self.vqe_inst = VQE(minimizer=minimize,
         			        minimizer_kwargs={'method': 'nelder-mead'})


		self.state_prep_angle = np.arctan(np.e**(-self.beta_temp/2.0)) * 2.0
		self.WEIGHTS = np.asarray(np.random.uniform(
    						low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                        	high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
                        	size=(num_visible, num_hidden)))

		# IN THIS VERSION BIAS IS UNUSED!
		self.BIAS = np.asarray(np.random.uniform(
  		  					low=-0.1 * np.sqrt(6. / (num_hidden + num_visible)),
  		                      high=0.1 * np.sqrt(6. / (num_hidden + num_visible)),
  		                      size=(num_hidden)))

		#BIASES ARE ON HIDDENS.

		# W[i][j] = ith visible to jth hidden
		# Bias[j] = bias on jth hidden.

	def make_unclamped_QAOA(self):
		"""
		Internal helper function for building QAOA circuit to get RBM expectation
		using Rigetti Quantum simulator

		Returns
		---------------------------------------------------
		nus:		(list) optimal parameters for cost hamiltonians in each layer of QAOA
		gammas:		(list) optimal parameters for mixer hamiltonians in each layer of QAOA
		para_prog:  (fxn closure) fxn to return QAOA circuit for any supplied nus and gammas
		---------------------------------------------------

		"""

		visible_indices = [i for i in range(0, self.n_visible)]
		hidden_indices = [i + self.n_visible for i in range(0, self.n_hidden)]


		full_cost_operator = []
		full_mixer_operator = []
		for i in visible_indices:
			for j in hidden_indices:
				full_cost_operator.append(PauliSum([PauliTerm("Z", i, -1.0 * self.WEIGHTS[i][j - self.n_visible]) * PauliTerm("Z", j, 1.0)]))

		# UNCOMMENT THIS TO ADD BIAS IN *untested* in this version of code*
		# for i in hidden_indices:
		# 	full_cost_operator.append(PauliSum([PauliTerm("Z", i, -1.0 * self.BIAS[i - self.n_visible])]))

		for i in hidden_indices + visible_indices:
			full_mixer_operator.append(PauliSum([PauliTerm("X", i, 1.0)]))



		n_system = len(visible_indices) + len(hidden_indices)

		state_prep = pq.Program()
		for i in visible_indices + hidden_indices:
			tmp = pq.Program()
			tmp.inst(RX(self.state_prep_angle, i + n_system), CNOT(i + n_system, i))
			state_prep += tmp


		full_QAOA = QAOA(self.qvm, 
						 n_qubits=n_system,
						 steps=self.n_qaoa_steps, 
						 ref_hamiltonian=full_mixer_operator, 
						 cost_ham=full_cost_operator,
						 driver_ref=state_prep,
						 store_basis=True,
						 minimizer=fmin_bfgs,
						 minimizer_kwargs={'maxiter':50},
				 		 vqe_args={'samples': self.n_quantum_measurements},
						 rand_seed=1234)


		nus, gammas = full_QAOA.get_angles()

		if self.verbose:
			print 'Found following for nus and gammas from QAOA'
			print nus
			print gammas
			print '-'*80

		program = full_QAOA.get_parameterized_program()
		return nus, gammas, program, 0 #full_QAOA.result['fun']


	def sigmoid(self, x):
		"""
		simple helper function to compute sigmoid across data matrix
		where the rows are samples

		Params
		-----------------
		DATA: (array) matrix of data where rows are samples
		-----------------

		Returns
		-----------------
		result: (array) that same matrix but with sigmoid applied to entries
		-----------------

		"""
		return 1.0 / (1.0 + np.exp(-x))



	def train(self, DATA, learning_rate=0.1, n_epochs=100, quantum_percentage=1.0, classical_percentage=0.0):
		"""
		Train an RBM with mixture of quantum and classical update rules

		Params
		-------------------------------------------------------------------------
		DATA: 				  (list) matrix with rows as data samples
		learning_rate:  	  (float) the learning rate used in the update rule by the rbm good value is 0.1
		n_epochs: 			  (int) number of weight update loops to do over RBM weights
		quantum_percentage:   (float) fraction of update rule to be dictated by quantum circuit
		classical_percentage: (float) fraction of update rule to be dictated by classical CD-1
		--------------------------------------------------------------------------

		NOTE: quantum_percentage + classical_percentage =1.0 must hold!!!

		"""

		assert(quantum_percentage + classical_percentage == 1.0)

		DATA = np.asarray(DATA)

		for epoch in range(n_epochs):

			print 'Beginning epoch', epoch

			visible_indices = [i for i in range(0, self.n_visible)]
			hidden_indices = [i + self.n_visible for i in range(0, self.n_hidden)]

			new_weights = copy.deepcopy(self.WEIGHTS)
			new_bias = copy.deepcopy(self.BIAS)

			model_nus, model_gammas, model_para_prog, _ = self.make_unclamped_QAOA()
			model_sampling_prog = model_para_prog(np.hstack((model_nus, model_gammas)))

			print 'Found model expectation program....'


			neg_phase_quantum = np.zeros_like(self.WEIGHTS)


			# UNCOMMENT FOR BIAS
			# neg_phase_quantum_bias = np.zeros_like(self.BIAS)


			for a in range(self.n_visible):
				for b in range(self.n_hidden):
					model_expectation = self.vqe_inst.expectation(model_sampling_prog, 
															sZ(visible_indices[a]) * sZ(hidden_indices[b]),
															self.n_quantum_measurements,
															self.qvm)

					neg_phase_quantum[a][b] = model_expectation


			# UNCOMMENT THIS FOR NEGATIVE PHASE COMPONENT OF BIAS
			# for b in range(self.n_hidden):
			# 	model_expectation = self.vqe_inst.expectation(model_sampling_prog, 
			# 											sZ(hidden_indices[b]),
			# 											n_measurements,
			# 											self.qvm)
			# 	neg_phase_quantum_bias[b] = model_expectation




			#IF ADDING BIAS MODIFY THIS AS WELL!!!
			#follow all standard conventions...
			hidden_probs = self.sigmoid(np.dot(DATA, self.WEIGHTS))
			pos_phase = np.dot(DATA.T, hidden_probs) * (1./float(len(DATA)))


			pos_hidden_states = hidden_probs > np.random.rand(len(DATA), self.n_hidden)


			neg_visible_activations = np.dot(pos_hidden_states, self.WEIGHTS.T)
			neg_visible_probs = self.sigmoid(neg_visible_activations)

			neg_hidden_activations = np.dot(neg_visible_probs, self.WEIGHTS)
			neg_hidden_probs = self.sigmoid(neg_hidden_activations)


			neg_phase_classical = np.dot(neg_visible_probs.T, neg_hidden_probs) * 1./len(DATA)

			if self.verbose:
				print 'POSITIVE PHASE,'
				print pos_phase
				print 'NEGATIVE PHASE (QUANTUM)'
				print neg_phase_quantum
				print 'Negative PHASE(classical)'
				print neg_phase_classical
				print 'WEIGHTS'
				print self.WEIGHTS
				print '-'*80


			# can update weights with weighted avg of quantum and classical.
			new_weights += learning_rate * (pos_phase - (classical_percentage*neg_phase_classical + quantum_percentage*neg_phase_quantum))

			# UNCOMMENT HERE TO DO BIAS UPDATES
			#can update bias with weighted avg of quantum and classical.
			# new_bias += learning_rate * (pos_expect_bias - (0.0*neg_associations_bias + 1.0 * neg_phase_quantum_bias))


			self.WEIGHTS = copy.deepcopy(new_weights)
			self.BIAS = copy.deepcopy(new_bias)


			with open("RBM_info.txt", "w") as myfile:
				myfile.write(json.dumps(list(self.WEIGHTS.tolist()))+'\n')
				myfile.write(json.dumps(list(self.BIAS.tolist()))+'\n')

			with open("RBM_history.txt", "a") as myfile:
				myfile.write(json.dumps(list(self.WEIGHTS.tolist()))+'\n')
				myfile.write(json.dumps(list(self.BIAS.tolist()))+'\n')
				myfile.write(str('-'*80) + '\n')		

		print 'Training Done!'



	def transform(self, DATA):
		"""
		Transforms vectors from visible to hidden

		Params
		-----------------
		DATA: (list) matrix containing rows that are data samples
		-----------------

		Returns
		----------------
		result: (list) the hidden layers invoked from the samples in data matrix
		----------------
		"""
		# MODIFY THIS IF INCLUDING BIAS
		return self.sigmoid(np.dot(DATA, self.WEIGHTS))



""" Simple use case example """

# Setup a Rigetti qvm connection
qvm = api.SyncConnection()


#Creat an instance of a qRBM
r = qRBM(qvm, num_visible=4, num_hidden=1, n_quantum_measurements=None, verbose=False)

# simple artificially high dimensional data
simple_data = [[1,1,-1,-1], [1,1,-1,-1], [-1,-1,1,1], [-1,-1,1,1]]

#train for 100 epochs using only quantum calculated negative phase in update rule.
r.train(simple_data, n_epochs=100, quantum_percentage=1.0, classical_percentage=0.0)

# transorm down to 1 dimension to see how we did.
print r.transform(simple_data)

