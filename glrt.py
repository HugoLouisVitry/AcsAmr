import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm
from time import time
from modulation import *

def apply_fading_channel(signal):
    """ Simuler du fading """
    alpha = 1# np.random.uniform(0.25,1)
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

def Log_Joint_Likelihood(r, alpha, theta, sigma2, symboles_modulation):
    """ Eq (3.3) 
    \n Calcule la Log Joint likelihood pour des échantillons r et une modulation a tester  
     \n avec les paramètres estimé du canal """
    fading = alpha*np.exp(-1j*theta)
    log_cste = -np.log(len(symboles_modulation)*2*np.pi*sigma2)
    log_L = []
    for n in range(len(r)) :
        sum = []
        for Am in symboles_modulation:
            sum.append( np.exp( -( np.abs(r[n] - fading*Am) ** 2 )/(2*sigma2) ) )
        log_L.append(np.log(np.sum(sum))+log_cste)
    return np.sum(log_L)

def glrt_Likelihood(r, alphas, thetas, sigma2s, symboles_modulation):
    """ Eq (3.22) \n 
    Calcule la GLRT likelihood """    
    Joint_likelihoods = []
    for i in range(len(alphas)) :
        Joint_likelihoods.append(Log_Joint_Likelihood(r, alphas[i], thetas[i], sigma2s[i], symboles_modulation))
    return np.max(Joint_likelihoods)

def glrt_classification(received_signal, alphabet, alpha_est, theta_est, sigma2_est ):
    """Eq (3.13) \n 
    Appliquer GLRT pour classifier la modulation """
    L_gltr = {}
    for mod in alphabet :  # Boucler sur les modulations -> calculer glrt likelihood
        L_gltr[mod] = glrt_Likelihood(received_signal, alpha_est, theta_est ,sigma2_est , alphabet[mod])
    best_match = max(L_gltr, key=L_gltr.get) # Classifier selon 3.13
    return best_match



# # Classification avec GLRT
def test_GLRT(Mod_true, N, SNR_dB,alphabet, plots = False):

    if Mod_true.find("PSK") != -1:
        m = int(Mod_true[:-3])
        lim = 2
        signal, true_samples = generate_mpsk_samples(m, N)
    
    if Mod_true.find("QAM") != -1 :
       m = int(Mod_true[:-3])
       lim = np.log2(m)
       signal, true_samples = generate_mqam_samples(m, N)

    # Propagation dans un canal de fading
    faded_signal, true_alpha, true_theta = apply_fading_channel(signal)

    # Ajouter du bruit AWGN
    received_signal, true_sigma2 = add_awgn(faded_signal, SNR_dB)

    # Estimation aveugle des paramètres du canal
    alpha_est = [true_alpha] #np.random.uniform(0.25,1, size=10)
    theta_est = [true_theta] #np.random.uniform(0,2*np.pi, size=10)
    sigma2_est =  [true_sigma2] #np.random.uniform(0.1,2, size=10)

    M_estimated = glrt_classification(received_signal, alphabet, alpha_est, theta_est, sigma2_est)

    if plots:

        plt.figure(figsize=(8, 8))

        plt.scatter(received_signal.real, received_signal.imag, color='blue', alpha=0.5, label=f'Symboles reçus\n Estimé: {M_estimated}')
        plt.scatter(signal.real,signal.imag,color='red', marker='x', label=f'Symboles idéaux {Mod_true}')
        #plt.scatter(qam_Alphabet["64QAM"].real,qam_Alphabet["64QAM"].imag,color='green', marker='o',alpha=0.8, label=f'Alphabet')

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axhline(0, color='black', linewidth=1)
        plt.axvline(0, color='black', linewidth=1)
        plt.xlim(- lim, lim )
        plt.ylim(- lim, lim )
        plt.title(f"AMR M-PSK")
        plt.xticks(fontsize = 14)
        plt.yticks(fontsize = 14)
        plt.xlabel("In-Phase (I)", fontsize=16)
        plt.ylabel("Quadrature (Q)", fontsize=16)
        plt.legend()
    if M_estimated == Mod_true:
        return True
    else :
        return False

def taux_erreur(Ms_true,  SNRs_dB, N_echantillons, N_test, alphabet):
    plt.figure()
    total = time()
    for m_test in Ms_true :
        t = time() 
        Tes = []
        for SNR_dB in SNRs_dB :
            Te = 0
            for _ in range(N_test):
                correct_classification = test_GLRT(m_test, N_echantillons, SNR_dB,alphabet)
                if not correct_classification :
                    Te = Te + 1
            Tes.append(Te / N_test)
        print(f'{m_test} :{time()-t} s')
        if m_test.find("QAM") != -1 :
            Linestyle = ':'
        else : 
            Linestyle = '-'
        plt.plot(SNRs_dB, Tes, marker='o', linestyle=Linestyle , label=f"{m_test}")
    
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Taux d'erreur de classification")
    plt.title(f"Taux d'erreur pour {Ne} echantillons transmis")
    plt.legend()
    plt.grid()
    print(f'Total execution time : {(time()-total)/60} min')
    return Tes



M_true = "8PSK"  
Ne = 30  # Nombre d'échantillon
SNR_dB = 16
a = test_GLRT(M_true, Ne, SNR_dB,ALPHABET, True)
print(a)

taux_erreur(MODLIST, [i for i in range(4,21,1)], Ne, 50, ALPHABET)
plt.show()
