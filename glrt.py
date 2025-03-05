import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm
from time import time
from modulation import *

def Log_Joint_Likelihood(r, alpha, theta, sigma2, symboles_modulation):
    """ Eq (3.3/3.4) 
    \n Calcule la Log Joint likelihood pour des échantillons r et une modulation a tester  
     \n avec les paramètres estimé du canal """
    fading = alpha*np.exp(-1j*theta)
    log_cste = -np.log(len(symboles_modulation)*2*np.pi*sigma2)
    joint_log_L = 0
    for n in range(len(r)) :
        sum = 0
        for Am in symboles_modulation:
            sum += np.exp( -( np.abs(r[n] - fading*Am) ** 2 )/(2*sigma2) )
        logsum= np.log(sum)
        joint_log_L += logsum + log_cste
    return joint_log_L

def glrt_Likelihood(r, alphas, thetas, sigma2s, symboles_modulation):
    """ Eq (3.22) \n 
    Calcule la GLRT likelihood """    
    MaxJointLikelihoods = -np.inf
    for alpha in alphas:
        for sigma2 in sigma2s:
            for theta in thetas:
                Jlikelihood = Log_Joint_Likelihood(r, alpha, theta, sigma2, symboles_modulation)
                if Jlikelihood > MaxJointLikelihoods:
                    MaxJointLikelihoods = Jlikelihood
    return MaxJointLikelihoods

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
    alpha =  np.random.uniform(0.25,1)
    theta =  np.random.uniform(0,2*np.pi)
    faded_signal, true_alpha, true_theta = apply_fading_channel(signal)

    # Ajouter du bruit AWGN
    received_signal, true_sigma2 = add_awgn(faded_signal, SNR_dB)

    # Estimation aveugle des paramètres du canal
    # Utiliser les "true" parameters pour avoir l'effet d'une estimation correcte
    # Utiliser une constante pour simuler une erreur d'estimation 
    
    alpha_est  = np.linspace(0.25,1, 2)
    theta_est  = np.linspace(0,np.pi/2,2)
    sigma2_est = [true_sigma2]

    M_estimated = glrt_classification(received_signal, alphabet, alpha_est, theta_est, sigma2_est)

    if plots:

        plt.figure(figsize=(8, 8))

        plt.scatter(received_signal.real, received_signal.imag, color='blue', alpha=0.5, label=f'Symboles reçus\n Estimé: {M_estimated}')
        plt.scatter(signal.real,signal.imag,color='red', marker='x', label=f'Symboles idéaux {Mod_true} \nalpha:{true_alpha}\ntheta:{true_theta}\nsigma2:{true_sigma2}')

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
    # plt.figure(dpi=300)
    plt.figure()
    total = time()
    print("Started")
    for m_test in Ms_true :
        t = time() 
        PCC = []
        for SNR_dB in SNRs_dB :
            Pc = 0
            for _ in range(N_test):
                # t2 = time()
                correct_classification = test_GLRT(m_test, N_echantillons, SNR_dB,alphabet)
                if correct_classification :
                    Pc = Pc + 1
                # print(f'{_} :{time()-t2} s')
            PCC.append(Pc / N_test * 100)
        print(f'{m_test} :{time()-t} s')
        if m_test.find("QAM") != -1 :
            Linestyle = ':'
        else : 
            Linestyle = '-'
        plt.plot(SNRs_dB, PCC, marker='o', linestyle=Linestyle , label=f"{m_test}")
    
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Pcc %")
    plt.title(f"Taux de réussite de classification (Pcc) en fonction du SNR (no fading)")
    plt.legend()
    plt.grid()
    print(f'Total execution time : {(time()-total)/60} min')
    return PCC


if __name__ == "__main__":
    M_true = "16QAM"  
    Ne = 500  # Nombre d'échantillon
    SNR_dB = 16
    # a = test_GLRT(M_true, Ne, SNR_dB,ALPHABET, True)
    # print(a)

    #taux_erreur(MODLIST, [i for i in range(-20,21,2)], Ne, 150, ALPHABET)
    taux_erreur(psk_Mod_list, [i for i in range(-20,21,2)], Ne, 200, pskAlphabet)
    # taux_erreur(QAM_Mod_list, [i for i in range(-20,21,2)], Ne, 200, qam_Alphabet)

    plt.show()
