import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

# Global random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory for figures
OUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 0) LOAD DATA =====
mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'umist_cropped.mat'))
facedat = mat['facedat']

for i in range(5):
    print(facedat[0,0][:,:,i].shape)

# Sample faces grid
plt.figure(figsize=(10,3))
for i in range(5):
    img = facedat[0,0][:,:,i]
    plt.subplot(1,5,i+1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title(f"Img {i+1}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sample_faces_grid.png"), dpi=150)
plt.close()

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
plt.savefig(os.path.join(OUT_DIR, "class_balance_per_person.png"), dpi=150)
plt.close()

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

# ===== 1) DATAFRAME WITH LABELS =====
# Build a single Pandas DataFrame with flattened pixels + labels
h, w = 112, 92  # UMIST cropped dimensions
X_flat = np.array([img.flatten() for img in X])  # (n_samples, n_pixels)
feature_cols = [f"px_{i}" for i in range(X_flat.shape[1])]
df = pd.DataFrame(X_flat, columns=feature_cols)
df["label"] = y

print("\nDataFrame shape:", df.shape)
print("DataFrame columns (first 5 + label):", list(df.columns[:5]) + ["...", "label"])

# ===== 2) STRATIFIED SPLIT & SCALING =====
# Convert to numpy arrays
y = np.array(y)

# stratified split: 70% train, 15% val, 15% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X_flat, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=y_temp
)

print("\nSplit sizes:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Visualize split distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, (labels, title) in enumerate([(y_train, "Train"), (y_val, "Validation"), (y_test, "Test")]):
    unique, counts = np.unique(labels, return_counts=True)
    axes[idx].bar(unique, counts)
    axes[idx].set_title(f"Distribution in {title.upper()}")
    axes[idx].set_xlabel("Person")
    axes[idx].set_ylabel("Count in subset")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "split_distributions.png"), dpi=150)
plt.close()

# ===== 3) PCA (95% VAR) =====
# Fit PCA on the scaled training set.
# Use 95% retained variance to choose number of components automatically.
pca = PCA(n_components=0.95, svd_solver='full', random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("\n[PCA] Components selected for 95% variance:", pca.n_components_)
print("[PCA] Cumulative explained variance: {:.2f}%".format(100 * np.sum(pca.explained_variance_ratio_)))
print("[PCA] Shapes -> Train:", X_train_pca.shape, "Val:", X_val_pca.shape, "Test:", X_test_pca.shape)

# (1) Scree plot / Cumulative explained variance
cum_var = np.cumsum(pca.explained_variance_ratio_)
plt.figure(figsize=(8,3))
plt.plot(np.arange(1, len(cum_var) + 1), cum_var, marker='o')
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("PCA - Cumulative explained variance")
plt.grid(True, linewidth=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_cumulative_explained_variance.png"), dpi=150)
plt.close()

# (2) Visualization of the first eigenfaces (principal components as images)
# Each PCA component has dimension equal to the number of pixels (112*92).
# Reshape to 112x92 to visualize as an "eigenface".
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
plt.savefig(os.path.join(OUT_DIR, "pca_eigenfaces_grid.png"), dpi=150)
plt.close()

# ===== 4) PCA (k = 10,20,50,100) =====
# Compare cumulative variance for fixed component counts
fixed_ks = [10, 20, 50, 100]
cum_vars = []
for k in fixed_ks:
    p = PCA(n_components=k, svd_solver="full", random_state=RANDOM_STATE)
    p.fit(X_train_scaled)  # train-only
    cum_vars.append(float(p.explained_variance_ratio_.sum()))

# Plot a simple bar chart comparing cumulative variance for each k
plt.figure(figsize=(6,4))
plt.bar([str(k) for k in fixed_ks], cum_vars)
plt.xlabel("PCA components (k)")
plt.ylabel("Cumulative explained variance")
plt.title("PCA cumulative variance by component count")
for i, v in enumerate(cum_vars):
    plt.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pca_k_comparison.png"), dpi=150)
plt.close()

print("\n[PCA] Cumulative variance for fixed component counts:")
for k, v in zip(fixed_ks, cum_vars):
    print(f"  k={k}: {v:.4f}")

# ===== 5) t-SNE (2D) =====
# t-SNE on PCA-50 features (train set)
print("\n[t-SNE] Computing t-SNE on PCA-50 features (train set)...")
pca_50 = PCA(n_components=50, svd_solver="full", random_state=RANDOM_STATE).fit(X_train_scaled)
Z_train = pca_50.transform(X_train_scaled)

tsne = TSNE(n_components=2, init="pca", perplexity=30, learning_rate="auto",
            n_iter=1000, random_state=RANDOM_STATE, verbose=0)
Z2 = tsne.fit_transform(Z_train)

plt.figure(figsize=(6,5))
scatter = plt.scatter(Z2[:, 0], Z2[:, 1], c=y_train, s=10, cmap='tab20')
plt.title("t-SNE (on PCA-50) – Train set")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tsne_pca50_train.png"), dpi=150)
plt.close()

print("[t-SNE] t-SNE visualization saved to figs/tsne_pca50_train.png")
print("\n✓ All figures saved to:", OUT_DIR)
