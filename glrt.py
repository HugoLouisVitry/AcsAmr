import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_mpsk_samples(M, N):
    """ Générer des échantillons M-PSK """
    samples = np.random.randint(0, M, N)
    phase = 2 * np.pi * samples / M
    return np.exp(1j * phase), samples

def generate_mqam_samples(M, N):
    """ Générer des échantillons M-QAM aléatoires """

    # Vérifier si M est une valeur standard de QAM
    if M not in [4, 8, 16, 32, 64]:
        raise ValueError("M-QAM non supporté. Utiliser 4, 8, 16, 32 ou 64.")

    # Générer l'alphabet M-QAM
    if M in [4, 16, 64]:  # Grille carrée pour 4-QAM, 16-QAM et 64-QAM
        M_side = int(np.sqrt(M))  # Taille du carré QAM (ex: 16-QAM -> 4x4)
        real_part = np.arange(-M_side + 1, M_side, 2)
        imag_part = np.arange(-M_side + 1, M_side, 2)
        alphabet = np.array([x + 1j*y for y in imag_part for x in real_part])

    elif M == 8:  # 8-QAM avec structure spécifique (croix)
        alphabet = np.array([
            -1+1j, 1+1j, -1-1j, 1-1j,  # Carré intérieur
            -2, 2, -2j, 2j  # Points sur les axes
        ])

    elif M == 32:  # 32-QAM avec structure optimisée (croix)
        alphabet = np.array([
            -1+1j, 0+1j, 1+1j,
            -1+0j, 0+0j, 1+0j,
            -1-1j, 0-1j, 1-1j,
            -3, -2, -1, 1, 2, 3,
            -3j, -2j, -1j, 1j, 2j, 3j,
            -2+2j, 2+2j, -2-2j, 2-2j
        ])

    # Normalisation de l'énergie moyenne à 1
    #alphabet /= np.sqrt((np.abs(alphabet) ** 2).mean())

    # Générer des échantillons aléatoires
    indices = np.random.randint(0, len(alphabet), N)
    samples = alphabet[indices]

    return samples, indices  # Retourne aussi les indices pour analyse

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

# Paramètres
psk_Mod_list = {"2PSK": 2, "4PSK" : 4,"8PSK" : 8}  # BPSK, QPSK, 8-PSK

# Générer l'alphabet de références de modulation
pskAlphabet={}
for mod in psk_Mod_list:
    order = psk_Mod_list[mod]
    i = np.arange(order)
    am = np.array(np.exp(1j * 2 * np.pi * i / order ))
    pskAlphabet[mod] = am

# Paramètres pour les différentes modulations QAM
QAM_Mod_list = {"4QAM": 4, "8QAM": 8, "16QAM": 16, "32QAM": 32, "64QAM": 64}

# Générer l'alphabet de référence pour QAM
qam_Alphabet = {}

for mod in QAM_Mod_list:
    order = QAM_Mod_list[mod]

    if order in [4, 16, 64]:  # Grille carrée pour 4-QAM, 16-QAM et 64-QAM
        M = int(np.sqrt(order))  # Dimension du carré QAM (ex: 16-QAM -> M=4)
        real_part = np.arange(-M + 1, M, 2)
        imag_part = np.arange(-M + 1, M, 2)

        # Construire la constellation complète
        Am = np.array([x + 1j*y for y in imag_part for x in real_part])

    elif order == 8:  # Constellation optimisée pour 8-QAM
        Am = np.array([
            -1+1j, 1+1j, -1-1j, 1-1j,  # Carré intérieur
            -2, 2, -2j, 2j  # Points sur les axes
        ])

    elif order == 32:  # Constellation optimisée pour 32-QAM (croix modifiée)
        Am = np.array([
            # Carré intérieur 3x3 (9 points)
            -1+1j, 0+1j, 1+1j,
            -1+0j, 0+0j, 1+0j,
            -1-1j, 0-1j, 1-1j,
            # Points sur les axes (16 points)
            -3, -2, -1, 1, 2, 3,
            -3j, -2j, -1j, 1j, 2j, 3j,
            # Coins de la croix (4 points)
            -2+2j, 2+2j, -2-2j, 2-2j
        ])

    # Normaliser l'énergie moyenne à 1
    #Am /= np.sqrt((np.abs(Am) ** 2).mean())

    qam_Alphabet[mod] = Am



# # Classification avec GLRT
def test_GLRT(Mod_true, N, SNR_dB,alphabet, plots = False):
    """Test PSK"""
    # Générer un signal M-PSK
    if Mod_true.find("PSK"):
        m = int(Mod_true[:-3])
        lim = 2
        signal, true_samples = generate_mpsk_samples(m, N)
    if Mod_true.find("QAM"):
       m = int(Mod_true[:-3])
       lim = np.log2(m)
       signal, true_samples = generate_mqam_samples(m, N)

    # Propagation dans un canal de fading
    faded_signal, true_alpha, true_theta = apply_fading_channel(signal)

    # Ajouter du bruit AWGN
    received_signal, true_sigma2 = add_awgn(faded_signal, SNR_dB)

    # Estimation aveugle des paramètres du canal
    alpha_est = [1] #np.random.uniform(0.25,1, size=10)
    theta_est = [0] #np.random.uniform(0,2*np.pi, size=10)
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

def taux_erreur(Ms_true,  SNRs_dB, N_echantillons = 100, N_test = 100):
    plt.figure()
    for m_test in Ms_true : 
        Tes = []
        for SNR_dB in SNRs_dB :
            Te = 0
            for _ in range(N_test):
                correct_classification = test_GLRT(m_test, N_echantillons, SNR_dB)
                if not correct_classification :
                    Te = Te + 1
            Tes.append(Te / N_test)
        plt.plot(SNRs_dB, Tes, marker='o', linestyle='-' , label=f"{m_test}-PSK")
    
    
    plt.xlabel("SNR (dB)")
    plt.ylabel("Taux d'erreur de classification")
    plt.title(f"Taux d'erreur pour {Ms_true}-PSK")
    plt.legend()
    plt.grid()
    return Tes

M_true = "16QAM"  
Ne = 500  # Nombre d'échantillon
SNR_dB = 16 
test_GLRT(M_true, Ne, SNR_dB,pskAlphabet, True)
#taux_erreur([4], [i for i in range(4,21,3)], Ne, 50)
plt.show()
