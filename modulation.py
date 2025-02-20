import numpy as np
import matplotlib.pyplot as plt

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

def plot_mod(Mod,alphabet):
    
    if Mod.find("PSK") != -1:
        lim = 2
    
    if Mod.find("QAM") != -1 :
       m = int(Mod[:-3])
       lim = np.log2(m)

    plt.figure(figsize=(8, 8))
    plt.scatter(alphabet[Mod].real,alphabet[Mod].imag,color='green', marker='o',alpha=0.8, label=f'{Mod}')
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

ALPHABET = {**pskAlphabet, **qam_Alphabet}
MODLIST = list(psk_Mod_list)+list(QAM_Mod_list)

plot_mod("64QAM",ALPHABET)