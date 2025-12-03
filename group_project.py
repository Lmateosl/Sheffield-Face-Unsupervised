import os
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Global random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output directory for figures
OUT_DIR = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 0) LOAD DATA =====
mat = scipy.io.loadmat(os.path.join(os.path.dirname(__file__), 'umist_cropped.mat'))
facedat = mat['facedat']

outputs_dir = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(outputs_dir, exist_ok=True)
data_outputs_dir = os.path.join(outputs_dir, "data_outputs")
os.makedirs(data_outputs_dir, exist_ok=True)
metrics_path = os.path.join(data_outputs_dir, "metrics.txt")

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

# Save class balance stats as tabular data
class_balance_df = pd.DataFrame({
    "person_index": np.arange(1, facedat.shape[1] + 1),
    "num_images": counts,
})
class_balance_df["percentage"] = (class_balance_df["num_images"] / counts.sum() * 100).round(2)
class_balance_df.to_csv(os.path.join(data_outputs_dir, "class_balance.csv"), index=False)

# Append summary stats to metrics.txt
with open(metrics_path, "a") as f:
    f.write("=== Class balance statistics ===\n")
    f.write(f"Total persons: {facedat.shape[1]}\n")
    f.write(f"Total images: {counts.sum()}\n")
    f.write(f"Mean images per person: {counts.mean():.2f}\n")
    f.write(f"Std deviation: {counts.std():.4f}\n")
    f.write(f"Min: {counts.min()} | Max: {counts.max()}\n\n")

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

# Save PCA numeric outputs
pca_summary_path = os.path.join(data_outputs_dir, "pca_summary.txt")
with open(pca_summary_path, "w") as f:
    f.write("PCA with 95% retained variance\n")
    f.write(f"Components selected: {pca.n_components_}\n")
    f.write("Cumulative explained variance: {:.2f}%\n".format(100 * np.sum(pca.explained_variance_ratio_)))
    f.write(f"Train shape after PCA: {X_train_pca.shape}\n")
    f.write(f"Val shape after PCA: {X_val_pca.shape}\n")
    f.write(f"Test shape after PCA: {X_test_pca.shape}\n")

# Save explained variance ratios as table
evr_df = pd.DataFrame({
    "component_index": np.arange(1, len(pca.explained_variance_ratio_) + 1),
    "explained_variance_ratio": pca.explained_variance_ratio_,
    "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
})
evr_df.to_csv(os.path.join(data_outputs_dir, "pca_explained_variance.csv"), index=False)

# Append PCA summary to global metrics file
with open(metrics_path, "a") as f:
    f.write("=== PCA summary (95% variance) ===\n")
    f.write(f"Components selected: {pca.n_components_}\n")
    f.write("Cumulative explained variance: {:.2f}%\n".format(100 * np.sum(pca.explained_variance_ratio_)))
    f.write("\n")

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
plt.savefig(os.path.join(OUT_DIR, "pca_cumulative_explained_variance.png"), dpi=150)
plt.show()
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
plt.savefig(os.path.join(pca_dir, "first_eigenfaces_grid.png"))
plt.show()

# === Autoencoder: nonlinear dimensionality reduction and reconstruction (TensorFlow/Keras) ===
# The autoencoder learns a compressed latent representation and tries to reconstruct the input.

input_dim = X_train_scaled.shape[1]
latent_dim = 64  # size of the bottleneck (latent space)

# Define encoder and decoder with Keras
inputs = keras.Input(shape=(input_dim,), name="ae_input")
x = layers.Dense(512, activation="relu")(inputs)
latent = layers.Dense(latent_dim, name="latent")(x)

x_dec = layers.Dense(512, activation="relu")(latent)
outputs = layers.Dense(input_dim, name="reconstruction")(x_dec)

autoencoder = keras.Model(inputs, outputs, name="autoencoder")
autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

print("\n[Autoencoder] Model summary:")
autoencoder.summary()

num_epochs = 20
history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,
    validation_data=(X_val_scaled, X_val_scaled),
    epochs=num_epochs,
    batch_size=64,
    shuffle=True,
    verbose=1,
)

# Extract final training and validation loss
final_train_loss = float(history.history["loss"][-1])
final_val_loss = float(history.history["val_loss"][-1])

# Save autoencoder loss history
ae_history_df = pd.DataFrame({
    "epoch": np.arange(1, len(history.history["loss"]) + 1),
    "train_loss": history.history["loss"],
    "val_loss": history.history["val_loss"],
})
ae_history_df.to_csv(os.path.join(data_outputs_dir, "autoencoder_loss_history.csv"), index=False)

# Append autoencoder summary to metrics.txt
with open(metrics_path, "a") as f:
    f.write("=== Autoencoder reconstruction loss ===\n")
    f.write(f"Final train MSE: {final_train_loss:.6f}\n")
    f.write(f"Final val MSE: {final_val_loss:.6f}\n\n")

# === Autoencoder reconstructions (original vs reconstructed faces) ===
# Use the trained autoencoder to reconstruct some test images.
recon_test_scaled = autoencoder.predict(X_test_scaled)
# Inverse-transform from scaled space back to original pixel space
recon_test = scaler.inverse_transform(recon_test_scaled)

# Create an output folder for autoencoder figures
ae_dir = os.path.join(os.path.dirname(__file__), "autoencoder_outputs")
os.makedirs(ae_dir, exist_ok=True)

# Plot a comparison between original and reconstructed faces
num_examples = min(8, X_test.shape[0])
plt.figure(figsize=(2.5 * num_examples, 5))
for i in range(num_examples):
    # Original image (unscaled)
    orig = X_test[i].reshape(h, w)
    # Reconstructed image (from autoencoder)
    rec = recon_test[i].reshape(h, w)

    ax_orig = plt.subplot(2, num_examples, i + 1)
    ax_orig.imshow(orig, cmap="gray")
    ax_orig.set_title(f"Orig {i+1}")
    ax_orig.axis("off")

    ax_rec = plt.subplot(2, num_examples, num_examples + i + 1)
    ax_rec.imshow(rec, cmap="gray")
    ax_rec.set_title(f"Recon {i+1}")
    ax_rec.axis("off")

plt.suptitle("Autoencoder reconstructions: original (top) vs reconstructed (bottom)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(ae_dir, "autoencoder_reconstructions.png"))
plt.show()


# === Clustering on PCA embeddings: K-Means and Hierarchical ===
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

clustering_dir = os.path.join(os.path.dirname(__file__), "clustering_outputs")
os.makedirs(clustering_dir, exist_ok=True)


# K-MEANS CLUSTERING

k = 20  # since dataset has 20 persons, sensible choice
kmeans = KMeans(n_clusters=k, random_state=42)
train_clusters_km = kmeans.fit_predict(X_train_pca)
val_clusters_km = kmeans.predict(X_val_pca)
test_clusters_km = kmeans.predict(X_test_pca)

sil_km = silhouette_score(X_train_pca, train_clusters_km)
print("\n[K-Means] Silhouette Score:", sil_km)

with open(metrics_path, "a") as f:
    f.write("=== K-Means Clustering ===\n")
    f.write(f"Clusters: {k}\n")
    f.write(f"Silhouette Score: {sil_km:.4f}\n\n")

# Plot K-Means result in PCA first 2 dims
plt.figure(figsize=(7,5))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=train_clusters_km, s=10)
plt.title("K-Means Clusters (PCA space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(clustering_dir, "kmeans_pca_clusters.png"))
plt.show()


# HIERARCHICAL CLUSTERING

hier = AgglomerativeClustering(n_clusters=k, linkage='ward')
train_clusters_hier = hier.fit_predict(X_train_pca)

sil_hier = silhouette_score(X_train_pca, train_clusters_hier)
print("[Hierarchical] Silhouette Score:", sil_hier)

with open(metrics_path, "a") as f:
    f.write("=== Hierarchical Clustering ===\n")
    f.write(f"Clusters: {k}\n")
    f.write(f"Silhouette Score: {sil_hier:.4f}\n\n")

# Plot Hierarchical clusters in PCA 2D
plt.figure(figsize=(7,5))
plt.scatter(X_train_pca[:,0], X_train_pca[:,1], c=train_clusters_hier, s=10)
plt.title("Hierarchical Clusters (PCA space)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(os.path.join(clustering_dir, "hierarchical_pca_clusters.png"))
plt.show()


# Dendrogram (optional but useful)

# Use a small subset to avoid huge dendrograms
subset_size = min(150, X_train_pca.shape[0])
Z = linkage(X_train_pca[:subset_size], method='ward')

plt.figure(figsize=(12,4))
dendrogram(Z, truncate_mode='level', p=6, leaf_rotation=90)
plt.title("Hierarchical Dendrogram (subset of training set)")
plt.tight_layout()
plt.savefig(os.path.join(clustering_dir, "hierarchical_dendrogram.png"))
plt.show()
