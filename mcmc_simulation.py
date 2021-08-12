''' 

Python Source code for the simulations described in the paper:

On the numerical approximation of geodesic distances between pairwise isotropic Gaussian-Markov random fields

Journal of Computational Science

Author: Alexandre L. M. Levada

'''

import numpy as np
import scipy.misc as spm
import matplotlib.pyplot as plt
import pickle							
import time
import warnings
from mpl_toolkits.mplot3d import Axes3D
from imageio import imwrite
from numpy import log
from skimage.io import imsave

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

start = time.time()

MAX_IT = 1000      	# maximum number of iterations
MEIO = MAX_IT//2 	
SIZE = 256			# random field dimensions (512 x 512 particles)
METADE = MEIO + 50  
BURN_IN = 5 		# Initial samples (discarded)


# Initial parameter values
beta = 0				
# Mean
media = 0				
# Variance
sigma = 5

# Fisher information matrices
tensor1 = np.zeros((3,3))	# This is the metric tensor!!!
tensor2 = np.zeros((3,3))	# Not the metric tensor (used only to comarison purposes)

# Creates the initial configuration (iid Gaussian random variables)
img = np.random.normal(media, sigma, (SIZE, SIZE))

# Stores all random field configurations along the simulation
sequence = np.zeros((MAX_IT, img.shape[0], img.shape[1]))

# Boundary value problem
K = 1
img = np.lib.pad(img, ((K,K), (K,K)), 'symmetric')

# Dimensions of the outcomes
nlin, ncol = (img.shape[0], img.shape[1])

# For the estimation of the initial inverse temperature parameter
numerator = 0
denominator = 0

# These vectors are used to store the information-theoretic quantities along the simulation
vetor_media = np.zeros(MAX_IT)
vetor_variancia = np.zeros(MAX_IT)
vetor_beta = np.zeros(MAX_IT)
vetor_betaMPL = np.zeros(MAX_IT)
vetor_ent_gauss = np.zeros(MAX_IT)
vetor_ent = np.zeros(MAX_IT)
vetor_phi = np.zeros(MAX_IT)
vetor_psi = np.zeros(MAX_IT)
vetor_mu1 = np.zeros(MAX_IT)
vetor_mu2 = np.zeros(MAX_IT)
vetor_sigma1 = np.zeros(MAX_IT)
vetor_sigma2 = np.zeros(MAX_IT)
vetor_sigbeta1 = np.zeros(MAX_IT)
vetor_sigbeta2 = np.zeros(MAX_IT)
vetor_ds2_1 = np.zeros(MAX_IT)
vetor_ds2_2 = np.zeros(MAX_IT)

# 3 x 3 window = 9 x 1 vector
col = 9
centro = col//2		# central position
delta = 8     		# number of neighbors

# Convertion of the 3 x 3 windows into 9 x 1 vectors
for i in range(K, nlin-K):
	for j in range(K, ncol-K):
		neigh = img[i-K:i+K+1, j-K:j+K+1]
		neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
		vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
		central = neigh[neigh.size//2]
		numerator += (central - media)*sum(vizinhanca - media)
		denominator += sum(vizinhanca - media)**2

# Initial inverse temperature
beta_inicial = numerator/denominator
print('Estimated initial beta = ', beta_inicial)

sample = img.copy()

####################################################
################### Main Loop ######################
####################################################
for iteracoes in range(0, MAX_IT):

	# Samples for the computation of the covariance matrix
	amostras = np.zeros(((nlin-2)*(ncol-2), col))
	ind = 0

	print('\nIteration ', iteracoes)
	# Alterar no caso de vizinhança de 3, 4 e 5 ordens
	for i in range(K, nlin-K):
		for j in range(K, ncol-K):

			neigh = img[i-1:i+2, j-1:j+2]
			neigh = np.reshape(neigh, neigh.shape[0]*neigh.shape[1])
			amostras[ind,:] = neigh
			ind += 1
			vizinhanca = np.concatenate((neigh[0:(neigh.size//2)], neigh[(neigh.size//2)+1:neigh.size]))
			central = neigh[neigh.size//2]
			# Calculates the probability of the current central value
			P1 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(central - media - beta*sum(vizinhanca - media))**2)
			# Choose a random number from a Gaussian distribution
			g = np.random.normal(media, sigma)
			# Discard outliers
			while (g < media - 3*np.sqrt(sigma)) or (g > media + 3*np.sqrt(sigma)):	
				g = np.random.normal(media, sigma)
			# Calculates the probability of the new value
			P2 = (1/np.sqrt(2*np.pi*sigma))*np.exp((-1/(2*sigma))*(g - media - beta*(vizinhanca - media).sum())**2)
			# Define the acceptance threshold
			limiar = 1
			razao = P2/P1
			if (razao < 1):
				limiar = razao
			# Acept new value with probability p
			epson = np.random.rand()
			if epson <= limiar:
				sample[i,j] = g

	img = sample.copy()
	nucleo = img[K:nlin-K, K:ncol-K]

	# Estimate the mean and the variance from the current configuration
	media_est = nucleo.mean()	
	variancia = nucleo.var()

	mc = np.cov(amostras.T)		# covariance matrix

	# Comptes the submatrix (Sigma_{p}^{-})
	sigma_minus = mc.copy()
	sigma_minus[:,centro] = 0
	sigma_minus[centro,:] = 0

	# MPL estimator of the inverse temperature parameter
	left_half = mc[centro, 0:centro]
	right_half = mc[centro, centro+1:col]
	rho = np.concatenate((left_half, right_half))	
	
	beta_cov = rho.sum()/sigma_minus.sum()	# MPL estimator
	#beta_cov = beta 	# if want to use real inverse temperature
	
	print('betaMPL = %.6f' % beta_cov)
	print('mean = %.6f' % media_est)
	print('variance = %.6f' % variancia)

	# Stores the initial parameter vector (configuration A)
	if iteracoes == BURN_IN:
		mediaA, varianciaA, betaA = media_est, variancia, beta_cov
		rho_sum_A = rho.sum()
		sigma_minus_sum_A = sigma_minus.sum()
	# Stores the parameter vector of configuration B
	elif iteracoes == METADE:
		mediaB, varianciaB, betaB = media_est, variancia, beta_cov
		rho_sum_B = rho.sum()
		sigma_minus_sum_B = sigma_minus.sum()
		dKL_AB = 0.5*log(varianciaB/varianciaA) - (1/(2*varianciaA))*( varianciaA - 2*betaA*rho_sum_A + (betaA**2)*sigma_minus_sum_A  ) + (1/(2*varianciaB))*( (varianciaA - 2*betaB*rho_sum_A + (betaB**2)*sigma_minus_sum_A) + ((mediaA - mediaB)**2)*(1 - delta*betaB)**2 )
		dKLsym_AB = (1/(4*varianciaA*varianciaB))*( (varianciaA - varianciaB)**2 - 2*(betaB*varianciaA - betaA*varianciaB)*(rho_sum_A - rho_sum_B) + (betaB**2 * varianciaA - betaA**2 * varianciaB)*(sigma_minus_sum_A - sigma_minus_sum_B) + (mediaA - mediaB)**2 * (varianciaA*(1 - delta*betaB)**2 + varianciaB*(1 - delta*betaA)**2 ) )
		print()		
		print('dKL(A, B) = %f' %dKL_AB)
		print()
		print('dKLsym(A, B) = %f' %dKLsym_AB)
		print()
	# Stores the parameter vector of final configuration 
	# For a information cycle must be equal to the initial configuration A
	elif iteracoes == MAX_IT - 1:
		mediaC, varianciaC, betaC = media, variancia, beta_cov
		rho_sum_C = rho.sum()
		sigma_minus_sum_C = sigma_minus.sum()
		dKL_BC = 0.5*log(varianciaC/varianciaB) - (1/(2*varianciaB))*( varianciaB - 2*betaB*rho_sum_B + (betaB**2)*sigma_minus_sum_B  ) + (1/(2*varianciaC))*( (varianciaB - 2*betaC*rho_sum_B + (betaC**2)*sigma_minus_sum_B) + ((mediaB - mediaC)**2)*(1 - delta*betaC)**2 )
		dKLsym_BC = (1/(4*varianciaB*varianciaC))*( (varianciaB - varianciaC)**2 - 2*(betaC*varianciaB - betaB*varianciaC)*(rho_sum_B - rho_sum_C) + (betaC**2 * varianciaB - betaB**2 * varianciaC)*(sigma_minus_sum_B - sigma_minus_sum_C) + (mediaB - mediaC)**2 * (varianciaB*(1 - delta*betaC)**2 + varianciaC*(1 - delta*betaB)**2 ) )
		print()		
		print('dKL(B, C) = %f' %dKL_BC)
		print()
		print('dKLsym(B, c) = %f' %dKLsym_BC)
		print()

	######## Fisher information matrix ###########
	####### Component I_mu_mu^{1}	
	mu_1 = (1/variancia)*((1 - beta_cov*delta)**2)*(1 - (1/variancia)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()))
	tensor1[0,0] = mu_1
	print('mu_1 = %.6f' %mu_1)	

	####### Component I_mu_mu^{2}	
	mu_2 = (1/variancia)*((1 - beta_cov*delta)**2)
	tensor2[0,0] = mu_2
	print('mu_2 = %.6f' %mu_2)	

	####### Component I_sigma_sigma^{1}
	rho_sig = np.kron(rho, sigma_minus)
	sig_sig = np.kron(sigma_minus, sigma_minus)
	sigma_1 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum()) + (1/variancia**4)*(3*(beta_cov**2)*sum(np.kron(rho, rho)) - 3*(beta_cov**3)*rho_sig.sum() + 3*(beta_cov**4)*sig_sig.sum() )
	tensor1[1,1] = sigma_1
	print('sigma_1 = %.6f' %sigma_1)

	####### Component I_sigma_sigma^{2}
	sigma_2 = (1/(2*variancia**2)) - (1/variancia**3)*(2*beta_cov*rho.sum() - (beta_cov**2)*sigma_minus.sum())
	tensor2[1,1] = sigma_2
	print('sigma_2 = %.6f' %sigma_2)

	####### Component I_sigma_beta^{1}
	sigbeta_1 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum()) - (1/(2*variancia**3))*(6*beta_cov*sum(np.kron(rho, rho)) - 9*(beta_cov**2)*rho_sig.sum() + 3*(beta_cov**3)*sig_sig.sum() )
	tensor1[1,2] = sigbeta_1
	tensor1[2,1] = sigbeta_1
	print('sigbeta_1 = %.6f' %sigbeta_1)

	####### Component I_sigma_beta^{2}
	sigbeta_2 = (1/variancia**2)*(rho.sum() - beta_cov*sigma_minus.sum())
	tensor2[1,2] = sigbeta_2
	tensor2[2,1] = sigbeta_2
	print('sigbeta_2 = %.6f' %sigbeta_2)

	####### Component I_beta_beta^{1} (PHI)
	# First term
	T1 = (1/variancia)*sigma_minus.sum()
	# Second term
	T2 = (2/variancia**2)*sum(np.kron(rho, rho))
	# Third term
	T3 = -6*beta_cov*rho_sig.sum()/variancia**2
	# Fourth term
	T4 = 3*(beta_cov**2)*sig_sig.sum()/variancia**2
	# Summation of the individual terms
	phi = (T1+T2+T3+T4)
	tensor1[2,2] = phi
	print('PHI = %.6f' % phi)

	####### Component I_beta_beta^{2} (PSI)
	psi = T1
	tensor2[2,2] = psi
	print('PSI = %.6f' % psi)

	####### Entropy #########
	entropia_gauss = 0.5*(np.log(2*np.pi) + np.log(variancia) + 1)
	entropia = entropia_gauss - ( (beta_cov/variancia)*rho.sum() - 0.5*(beta_cov**2)*psi )
	print('GAUSSIAN ENTROPY = %.6f' % entropia_gauss)
	print('GMRF ENTROPY = %.6f' % entropia)

	# Store the current information-thoretic measures
	vetor_media[iteracoes] = media_est
	vetor_variancia[iteracoes] = variancia
	vetor_betaMPL[iteracoes] = beta_cov
	vetor_phi[iteracoes] = phi
	vetor_psi[iteracoes] = psi
	vetor_ent_gauss[iteracoes] = entropia_gauss
	vetor_ent[iteracoes] = entropia
	vetor_mu1[iteracoes] = mu_1
	vetor_mu2[iteracoes] = mu_2
	vetor_sigma1[iteracoes] = sigma_1
	vetor_sigma2[iteracoes] = sigma_2
	vetor_sigbeta1[iteracoes] = sigbeta_1
	vetor_sigbeta2[iteracoes] = sigbeta_2

	####### Computes the infinitesimal displacements ################
	dmu = vetor_media[iteracoes] - vetor_media[iteracoes-1]
	dsigma = vetor_variancia[iteracoes] - vetor_variancia[iteracoes-1]
	dbeta = vetor_betaMPL[iteracoes] - vetor_betaMPL[iteracoes-1]
	vetor_parametros = np.array([dmu, dsigma, dbeta])
	ds2_1 = np.dot(vetor_parametros, np.dot(tensor1, vetor_parametros))
	vetor_ds2_1[iteracoes] = np.sqrt(ds2_1)
	print('ds (tensor1) = %.15f' %np.sqrt(ds2_1))

	# Store the current configuration of the system
	sequence[iteracoes,:,:] = nucleo

	################# Modulating the system's behavior ###########
	print('beta real = %.3f' % beta)
	# Com beta estimado
	if iteracoes <= MEIO:
		if beta < 0.8:			
			beta += 0.002		
	else:
		if beta > 0:
			beta -= 0.002

	
###################################################

# Discards the first samples (burn-in)
vetor_media = vetor_media[BURN_IN:]
vetor_variancia = vetor_variancia[BURN_IN:]
vetor_betaMPL = vetor_betaMPL[BURN_IN:]
vetor_phi = vetor_phi[BURN_IN:]
vetor_psi = vetor_psi[BURN_IN:]
vetor_ent_gauss = vetor_ent_gauss[BURN_IN:]
vetor_ent = vetor_ent[BURN_IN:]
vetor_mu1 = vetor_mu1[BURN_IN:]
vetor_mu2 = vetor_mu2[BURN_IN:]
vetor_sigma1 = vetor_sigma1[BURN_IN:]
vetor_sigma2 = vetor_sigma2[BURN_IN:]
vetor_sigbeta1 = vetor_sigbeta1[BURN_IN:]
vetor_sigbeta2 = vetor_sigbeta2[BURN_IN:]
vetor_ds2_1 = vetor_ds2_1[BURN_IN:]

print('\n*** END OF THE SIMULATION ***')

print('----- Total elapsed time: %s seconds ----' % (time.time() - start))
print()

# Initial and final configurations (for a closed cycle A must be equal to C)
A = np.array([mediaA, varianciaA, betaA])
B = np.array([mediaB, varianciaB, betaB])
C = np.array([mediaC, varianciaC, betaC])

d_total = np.sum(vetor_ds2_1)
d_AB = np.sum(vetor_ds2_1[:MEIO])
d_BC = np.sum(vetor_ds2_1[MEIO:])

print('Model parameters in A: ')
print('Mean A: %f' %mediaA)
print('Variance A: %f' %varianciaA)
print('Beta A: %f' %betaA)
print()
print('Model parameters in B: ')
print('Mean B: %f' %mediaB)
print('Variance B: %f' %varianciaB)
print('Beta B: %f' %betaB)
print()
print('Model parameters in C: ')
print('Mean C: %f' %mediaC)
print('Variance C: %f' %varianciaC)
print('Beta C: %f' %betaC)
print()
print('Total geodesic distance: %f' %d_total)
print('Geodesic distance from A to B: %f' %d_AB)
print('Geodesic distance form B to C: %f' %d_BC)
print('KL divergence from A to B: %f' %dKL_AB)
print('KL divergência from B to C: %f' %dKL_BC)
print('Symmetrized KL divergence from A to B: %f' %dKLsym_AB)
print('Symmetrized KL divergence from B to C: %f' %dKLsym_BC)
print('Euclidean distance from A to B: %f' %np.linalg.norm(A-B))
print('Euclidean distance from B to C: %f' %np.linalg.norm(B-C))


# PHI plot
plt.figure(1)
plt.plot(vetor_phi, 'b', label='PHI')
plt.xlabel('iteration')
plt.ylabel('Fisher information (beta)')
plt.savefig('PHI.png')

# Entropy plot
plt.figure(2)
plt.plot(vetor_ent, 'k')
plt.xlabel('iteration')
plt.ylabel('Entropy')
plt.savefig('Entropy.png')

# Information cycles
plt.figure(3)
plt.plot(vetor_phi[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_phi[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information')
plt.ylabel('Entropy')
plt.savefig('PHI_Entropy.png')

plt.figure(4)
plt.plot(vetor_mu1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_mu1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (mean)')
plt.ylabel('Entropy')
plt.savefig('Mean_Entropy.png')

plt.figure(5)
plt.plot(vetor_sigma1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_sigma1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (sigma)')
plt.ylabel('Entropy')
plt.savefig('Sigma_Entropy.png')

plt.figure(6)
plt.plot(vetor_sigbeta1[:METADE], vetor_ent[:METADE], 'b')
plt.plot(vetor_sigbeta1[METADE:], vetor_ent[METADE:], 'r')
plt.xlabel('Fisher information (sigma/beta)')
plt.ylabel('Entropy')
plt.savefig('SigmaBeta_Entropy.png')

# Infinitesimal displacements
plt.figure(7)
plt.plot(vetor_ds2_1[BURN_IN:], 'b')
plt.xlabel('iteration')
plt.ylabel('Infinitesimal displacements')
#plt.legend()
plt.savefig('DS.png')

# Mean parameter estimatives
plt.figure(8)
plt.plot(vetor_media, 'b')
plt.xlabel('iteration')
plt.ylabel('Estimated mean')
plt.savefig('Mean_Estimated.png')

# Variance parameter estimatives
plt.figure(9)
plt.plot(vetor_variancia, 'b')
plt.xlabel('iteration')
plt.ylabel('Estimated variance')
plt.savefig('Variance_Estimated.png')

# Inverse temperature parameter estimatives
plt.figure(10)
plt.plot(vetor_betaMPL, 'b')
plt.xlabel('iteration')
plt.ylabel('Estimated inverse temperature')
plt.savefig('Inverse_Temperature__Estimated.png')


# Initial and final configurations
imgA = sequence[BURN_IN, :, :]
saidaA = np.uint8(255*(imgA - imgA.min())/(imgA.max() - imgA.min()))
imsave('A.png', saidaA)

imgB = sequence[METADE, :, :]
saidaB = np.uint8(255*(imgB - imgB.min())/(imgB.max() - imgB.min()))
imsave('B.png', saidaB)

plt.clf()
plt.close('all')
