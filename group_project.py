import os
import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'umist_cropped.mat'))
facedat = mat['facedat']

for i in range(5):
    print(facedat[0,0][:,:,i].shape)

plt.figure(figsize=(10,3))
for i in range(5):
    img = facedat[0,0][:,:,i]
    plt.subplot(1,5,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Img {i+1}")
plt.show()

# === Class balance analysis (images per person) ===
counts = []
for person in range(facedat.shape[1]):
    num_images = facedat[0, person].shape[2]
    counts.append(num_images)
    print(f"Person {person+1}: {num_images} images")

counts = np.array(counts)
print("\nTotal persons:", facedat.shape[1])
print("Total images:", counts.sum())
print("Mean images per person:", counts.mean())
print("Std deviation:", counts.std())
print("Min:", counts.min(), " | Max:", counts.max())

# Bar chart: number of images per person
plt.figure(figsize=(10,4))
plt.bar(range(1, facedat.shape[1] + 1), counts)
plt.xlabel("Person")
plt.ylabel("Number of images")
plt.title("Number of images per person (class balance)")
plt.tight_layout()
plt.show()

# === Build X (images) and y (targets) ===
X = []
y = []
for person in range(facedat.shape[1]):
    imgs = facedat[0, person]  # shape: H x W x N_imgs
    for i in range(imgs.shape[2]):
        X.append(imgs[:, :, i])
        y.append(person)  # label: person index 0..19

# Target composition (how many images per class)
df_y = pd.DataFrame({"persona": y})
print("\nTarget composition (count per person):")
print(df_y["persona"].value_counts().sort_index())

# Percentages per class
target_pct = (df_y["persona"].value_counts(normalize=True).sort_index() * 100).round(2)

print("\nPercentage per person (%):")
print(target_pct)

# === Split & preprocessing ===
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Flatten images
X_flat = [img.flatten() for img in X]
X_flat = np.array(X_flat)
y = np.array(y)

# stratified split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_flat, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("\nSplit sizes:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# count per class in splits
def plot_split_distribution(labels, title):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(8,3))
    plt.bar(unique, counts)
    plt.title(title)
    plt.xlabel("Person")
    plt.ylabel("Count in subset")
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), "split_distributions")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"{title.replace(' ', '_').lower()}.png"))
    plt.show()

plot_split_distribution(y_train, "Distribution in TRAIN")
plot_split_distribution(y_val, "Distribution in VALIDATION")
plot_split_distribution(y_test, "Distribution in TEST")

# === PCA: dimensionality reduction and eigenface visualization ===
from sklearn.decomposition import PCA

# Fit PCA on the scaled training set.
# Use 95% retained variance to choose number of components automatically.
pca = PCA(n_components=0.95, svd_solver='full', random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\n[PCA] Components selected for 95% variance:", pca.n_components_)
print("[PCA] Cumulative explained variance: {:.2f}%".format(100 * np.sum(pca.explained_variance_ratio_)))
print("[PCA] Shapes -> Train:", X_train_pca.shape, "Val:", X_val_pca.shape, "Test:", X_test_pca.shape)

# Output folder for PCA artifacts
pca_dir = os.path.join(os.path.dirname(__file__), "pca_outputs")
os.makedirs(pca_dir, exist_ok=True)

# (1) Scree plot / Cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,3))
plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - Cumulative explained variance")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig(os.path.join(pca_dir, "pca_varianza_explicada_acumulada.png"))
plt.show()

# (2) Visualization of the first eigenfaces (principal components as images)
# Each PCA component has dimension equal to the number of pixels (112*92).
# Reshape to 112x92 to visualize as an "eigenface".
h, w = 112, 92
num_to_show = min(16, pca.components_.shape[0])
cols = 4
rows = int(np.ceil(num_to_show / cols))
plt.figure(figsize=(2.2*cols, 2.6*rows))
for i in range(num_to_show):
    eigenface = pca.components_[i].reshape(h, w)
    ax = plt.subplot(rows, cols, i + 1)
    ax.imshow(eigenface, cmap='gray')
    ax.set_title(f"PC {i+1}")
    ax.axis('off')
plt.suptitle("First eigenfaces (principal components)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(pca_dir, "primeras_eigenfaces_grid.png"))
plt.show()
