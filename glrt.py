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
    alpha = 1 # np.random.uniform(0.25,1)
    theta = 0 # np.pi/3 #np.random.uniform(0,2*np.pi)
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
    return

def Joint_Likelihood(r, alpha, theta, sigma2, symboles_modulation):
    """ Eq (3.3) 
    \n Calcule la Joint likelihood pour des échantillons r et une modulation a tester  
     \n avec les paramètres estimé du canal """
    fading = alpha*np.exp(-1j*theta)
    cste = 1/(len(symboles_modulation)*2*np.pi*sigma2)
    sum = []
    prod = []
    for n in range(len(r)) :
        for Am in symboles_modulation:
            sum.append(np.exp(-(np.abs(r[n] - fading*Am) ** 2)/(2*sigma2)))
        prod.append(np.sum(sum)*cste)
    return np.prod(prod)

def glrt_Likelihood(r, alphas, thetas, sigma2s, symboles_modulation):
    """ Eq (3.22) \n 
    Calcule la GLRT likelihood """    
    Joint_likelihoods = []
    for i in range(len(alphas)) :
        Joint_likelihoods.append(Joint_Likelihood(r, alphas[i], thetas[i], sigma2s[i], symboles_modulation))
    return np.max(Joint_likelihoods)

def glrt_classification(received_signal, alphabet, alpha_est, theta_est, sigma2_est ):
    """Eq (3.13) \n 
    Appliquer GLRT pour classifier la modulation """
    L_gltr = {}
    for mod in alphabet :  # Boucler sur les modulations -> calculer glrt likelihood
        L_gltr[mod] = glrt_Likelihood(received_signal, alpha_est, theta_est ,sigma2_est , alphabet[mod])
    best_match = max(L_gltr, key=L_gltr.get) # Classifier selon 3.13
    return best_match

# Paramètres
Mod_list = {"2PSK": 2, "4PSK" : 4,"8PSK" : 8}  # BPSK, QPSK, 8-PSK
M_true = 4  # Modulation réelle
N = 100  # Nombre d'échantillon
SNR_dB = 16 # Rapport Signal/Bruit

# Générer l'alphabet de références de modulation
Alphabet={}
for mod in Mod_list:
    order = Mod_list[mod]
    i = np.arange(order)
    am = np.array(np.exp(1j * 2 * np.pi * i / order ))
    Alphabet[mod] = am


# # Classification avec GLRT
def test_mono(M_true, N, SNR_dB, plots = False):
    """Test PSK"""
    # Générer un signal M-PSK
    signal, true_samples = generate_mpsk_samples(M_true, N)

    # Propagation dans un canal de fading
    faded_signal, true_alpha, true_theta = apply_fading_channel(signal)

    # Ajouter du bruit AWGN
    received_signal, true_sigma2 = add_awgn(faded_signal, SNR_dB)

    # Estimation aveugle des paramètres du canal
    alpha_est = np.random.uniform(0.25,1, size=10)
    theta_est = np.random.uniform(0,2*np.pi, size=10)
    sigma2_est = np.random.uniform(0.1,2, size=10)

    M_estimated = glrt_classification(received_signal, Alphabet, alpha_est, theta_est, sigma2_est)

    if plots:

        plt.figure(figsize=(8, 8))

        plt.scatter(received_signal.real, received_signal.imag, color='blue', alpha=0.5, label=f'Symboles reçus\n Estimé: {M_estimated}')
        plt.scatter(signal.real,signal.imag,color='red', marker='x', label=f'Symboles idéaux {M_true}-PSK')
        # plt.scatter(Alphabet["2PSK"].real,Alphabet["2PSK"].imag,color='green', marker='o',alpha=0.8, label=f'Alphabet')

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
    if M_estimated == str(M_true)+"PSK":
        return True
    else :
        return False

def proba_erreur(SNR_dB,plots = False):
    return

test_mono(M_true, N, SNR_dB, True)

plt.show()