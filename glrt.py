import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_mpsk_samples(M, N):
    """ Générer des échantillons M-PSK """
    samples = np.random.randint(0, M, N)
    phase = 2 * np.pi * samples / M
    return np.exp(1j * phase), samples

def apply_fading_channel(signal):
    """ Simuler du fading """
    alpha = 0.8 # np.random.uniform(0.25,1)
    theta = np.pi/3 #np.random.uniform(0,2*np.pi)
    return signal * alpha * np.exp(-1j * theta), alpha, theta

def add_awgn(signal, snr_db):
    """ Ajouter du bruit """
    snr_linear = 10 ** (snr_db / 10)
    power_signal = np.mean(np.abs(signal)**2)
    noise_power = power_signal / snr_linear
    noise = np.sqrt(noise_power/2) * (np.random.randn(*signal.shape) + 1j * np.random.randn(*signal.shape))
    sigma2 = np.var(noise)
    return signal + noise, sigma2

def blind_channel_estimation(received_signal):
    """ Estimer les paramètres du canal sans connaître le signal transmis """
    alpha_est = 1 #np.mean(np.abs(received_signal))
    theta_est = 0 #np.angle(np.mean(received_signal))
    sigma2_est = 1 # np.var(received_signal) - alpha_est**2
    return alpha_est, theta_est, sigma2_est

def Joint_Likelihood(r, alpha, theta, sigma2, alphabet):
    """ Calcule Joint likelihood pour des échantillons r et un alphabet 
     \n avec les paramètres estimé du canal """
    fading = alpha*np.exp(-1j*theta)
    cste = 1/(len(alphabet)*2*np.pi*sigma2)
    sum = []
    prod = []
    for n in len(r) :
        for Am in alphabet:
            sum.append(np.exp(-(np.abs(r[n] - fading*Am) ** 2)/(2*sigma2)))
        prod.append(np.sum(sum)*cste)
    return np.prod(prod)

def glrt_Likelihood(r, alphas, thetas, sigma2s, alphabet):
    """ Calcule la GLRT likelihood """
    likelihoodS = []
    for i in range(len(alphas)) :
        likelihoodS.append(Joint_Likelihood(r, alphas, thetas, sigma2s, alphabet))
    return np.max(likelihoodS)

def glrt_classification(received_signal, alphabet, alpha_est, theta_est, sigma2_est ):
    """ Appliquer GLRT pour classifier la modulation """
    # estimation
    # likelihood
    # max likelyhood 
    
    return 

# Paramètres
M_list = [2, 4, 8]  # BPSK, QPSK, 8-PSK
M_true = 4  # Modulation réelle
N = 1000  # Nombre d'échantillon
SNR_dB = 12  # Rapport Signal/Bruit

# Générer un signal M-PSK
signal, true_samples = generate_mpsk_samples(M_true, N)

# Générer l'alphabet
Alphabet=[]
for m in M_list:
    i = np.arange(m)
    am = np.array(np.exp(1j * 2 * np.pi * i / m ))
    Alphabet.append(am)

# Propagation dans un canal de fading
faded_signal, alpha, theta = apply_fading_channel(signal)

# Ajouter du bruit AWGN
received_signal, sigma2 = add_awgn(faded_signal, SNR_dB)

# Estimation aveugle des paramètres du canal
alpha_est, theta_est, sigma2_est = blind_channel_estimation(received_signal)
print(f"Estimation: alpha={alpha_est}, theta={theta_est}, sigma2={sigma2_est}")

# # Classification avec GLRT
M_and_fade_estimated = None # glrt_classification(received_signal, Alphabet) #,alpha_est, theta_est, sigma2_est)
M_estimated          = None # glrt_classification(received_signal, Alphabet) #,alpha    , theta    , sigma2)
print(f"Modulation et fading estimée: {M_and_fade_estimated}-PSK")
print(f"Modulation estimée: {M_estimated}-PSK")

plt.figure(figsize=(8, 8))

plt.scatter(received_signal.real, received_signal.imag, color='blue', alpha=0.5, label=f'Symboles reçus\n Modulation estimée: {M_and_fade_estimated}-PSK (fade) \n Modulation estimée: {M_estimated}-PSK')
plt.scatter(signal.real,signal.imag,color='red', marker='x', label=f'Symboles idéaux {M_true}-PSK')
plt.scatter(Alphabet[1].real,Alphabet[1].imag,color='green', marker='o',alpha=0.8, label=f'Alphabet')

plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(0, color='black', linewidth=1)
plt.axvline(0, color='black', linewidth=1)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.title(f"AMR M-PSK")
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.xlabel("In-Phase (I)", fontsize=16)
plt.ylabel("Quadrature (Q)", fontsize=16)
plt.legend()
plt.show()