import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
from time import time
from modulation import*


# Generate data for different modulations
modulations = ["8QAM", "4QAM", "16QAM"]#, "2PSK", "4PSK"]
data = []
labels = []
N_data = 1000
snr_db = 20

for mod in modulations:
    if "PSK" in mod:
        M = int(mod[:-3])
        samples, _ = generate_mpsk_samples(M, N_data)
    else:
        M = int(mod[:-3])
        samples, _ = generate_mqam_samples(M, N_data)

    samples, _  = add_awgn(samples, snr_db)

    # Use real and imaginary parts as features
    features = np.vstack((samples.real, samples.imag)).T

    data.append(features)
    labels.extend([mod] * N_data)

# Combine data and labels
data = np.vstack(data)
labels = np.array(labels)

# Convert labels to numerical values
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_num = np.array([label_map[label] for label in labels])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels_num, test_size=0.2, random_state=42)

# Create and train the SVM classifier
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=modulations))

# Plot decision boundaries with legend
def plot_decision_boundaries(X, y, classifier, modulations):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Use a consistent colormap
    cmap = plt.cm.Spectral
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=cmap)
    plt.xlabel('In-Phase (I)')
    plt.ylabel('Quadrature (Q)')
    plt.title('SVM Decision Boundaries')

    # Add legend for decision boundaries
    unique_labels = np.unique(Z)
    legend_handles = []
    for label in unique_labels:
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=cmap(label / max(unique_labels)), markersize=10))

    # Correctly map numerical labels back to modulation names
    inv_label_map = {v: k for k, v in label_map.items()}
    plt.legend(legend_handles, [modulations[label] for label in unique_labels], title="Classes")

plot_decision_boundaries(X_train, y_train, clf, modulations)
plt.show()
print("Done")