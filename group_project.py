"""
Face Recognition Project - Data Loading and Preprocessing
This script loads the UMIST face dataset, performs exploratory data analysis,
and prepares the data for machine learning models.
"""

# ===== IMPORTS =====
# Standard library imports
import os  # For file path operations and directory creation

# Scientific computing and data manipulation
import scipy.io  # For loading MATLAB .mat files (UMIST dataset format)
import numpy as np  # For numerical operations and array manipulation
import pandas as pd  # For structured data handling and analysis

# Visualization
import matplotlib.pyplot as plt  # For creating plots and figures

# Machine learning preprocessing and dimensionality reduction
from sklearn.preprocessing import StandardScaler  # For feature normalization (zero mean, unit variance)
from sklearn.decomposition import PCA  # Principal Component Analysis for dimensionality reduction
from sklearn.manifold import TSNE  # t-SNE for non-linear dimensionality reduction and visualization
from sklearn.model_selection import train_test_split  # For splitting data into train/val/test sets

# Deep learning framework
import tensorflow as tf  # TensorFlow for building neural networks
from tensorflow import keras  # Keras API for high-level neural network construction
from tensorflow.keras import layers  # Pre-built neural network layers

# ===== REPRODUCIBILITY SETUP =====
# Set global random seed for reproducibility across all libraries
# This ensures that random operations (data shuffling, weight initialization, etc.)
# produce the same results every time the script runs
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)  # NumPy random operations
tf.random.set_seed(RANDOM_STATE)  # TensorFlow random operations

# ===== SCRIPT DIRECTORY DETECTION =====
# Get the directory where this script is located
# Works in both regular script execution and interactive environments (Spyder, Jupyter)
try:
    # When running as a script file, __file__ is defined
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # When running interactively (Spyder, Jupyter), __file__ is not defined
    # Use current working directory instead
    SCRIPT_DIR = os.getcwd()

# ===== OUTPUT DIRECTORY SETUP =====
# Create directory for saving visualization figures
# The 'figs' folder will contain all generated plots and charts
OUT_DIR = os.path.join(SCRIPT_DIR, "figs")
os.makedirs(OUT_DIR, exist_ok=True)  # Create directory if it doesn't exist, don't error if it does

# ===== 0) LOAD DATA =====
# Load the UMIST cropped face dataset from MATLAB file format
# The dataset contains face images of multiple individuals with varying poses/expressions
mat = scipy.io.loadmat(os.path.join(SCRIPT_DIR, 'umist_cropped.mat'))
facedat = mat['facedat']  # Extract the face data array from the MATLAB structure
# facedat structure: facedat[0, person_index] contains all images for one person
# Each person's data is a 3D array: height x width x num_images

# ===== OUTPUT DIRECTORIES FOR RESULTS =====
# Create nested directory structure for organizing different types of outputs:
# - outputs/data_outputs/ : CSV files, metrics, and numerical results
# - figs/ : All visualization figures (already created above)
outputs_dir = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(outputs_dir, exist_ok=True)
data_outputs_dir = os.path.join(outputs_dir, "data_outputs")
os.makedirs(data_outputs_dir, exist_ok=True)
metrics_path = os.path.join(data_outputs_dir, "metrics.txt")  # Path for aggregated metrics file

# ===== EXPLORATORY DATA ANALYSIS: SAMPLE IMAGES =====
# First, inspect the shape/dimensions of sample images to understand the data structure
# This helps verify the data loaded correctly and understand the image dimensions
print("\n=== Sample Image Shapes ===")
for i in range(5):
    print(f"Image {i+1} shape: {facedat[0,0][:,:,i].shape}")
    # Each image is accessed as facedat[0, person_index][:, :, image_index]
    # The shape shows: (height, width) - typically 112x92 for UMIST cropped

# Visualize a sample of face images to get a visual understanding of the dataset
# This helps identify data quality, image characteristics, and potential preprocessing needs
print("\n=== Displaying Sample Faces Grid ===")
plt.figure(figsize=(10,3))  # Create figure with width=10, height=3 inches
for i in range(5):
    img = facedat[0,i][:,:,0]   # first image of 5 different people
    plt.subplot(1,5,i+1)  # Create subplot in 1 row, 5 columns, position i+1
    plt.imshow(img, cmap='gray')  # Display as grayscale image
    plt.axis('off')  # Hide axes for cleaner visualization
    plt.title(f"Img {i+1}")
plt.tight_layout()  # Automatically adjust spacing between subplots
plt.savefig(os.path.join(OUT_DIR, "sample_faces_grid.png"), dpi=150)  # Save high-resolution figure
plt.show()  # Display figure in interactive window
plt.close()  # Close figure to free memory

# ===== CLASS BALANCE ANALYSIS =====
# Analyze the distribution of images across different people (classes)
# Class imbalance can affect model performance, so it's important to understand the data distribution
# This helps inform decisions about sampling strategies, loss functions, and evaluation metrics
print("\n=== Class Balance Analysis ===")
counts = []  # List to store number of images per person
for person in range(facedat.shape[1]):  # Iterate through all people in the dataset
    num_images = facedat[0, person].shape[2]  # Get number of images (3rd dimension)
    counts.append(num_images)
    print(f"Person {person+1}: {num_images} images")

# Convert to numpy array for easier statistical computations
counts = np.array(counts)

# Calculate and display summary statistics about class balance
# These metrics help identify if the dataset is balanced or imbalanced
print("\n=== Class Balance Summary Statistics ===")
print(f"Total persons: {facedat.shape[1]}")  # Number of unique classes (people)
print(f"Total images: {counts.sum()}")  # Total number of images across all people
print(f"Mean images per person: {counts.mean():.2f}")  # Average number of images per person
print(f"Std deviation: {counts.std():.4f}")  # Standard deviation - measures variability
print(f"Min: {counts.min()} | Max: {counts.max()}")  # Range of image counts
# Low std deviation and similar min/max indicate balanced dataset
# High std deviation indicates imbalanced dataset (some people have many more images)

# Save class balance statistics as a CSV file for later analysis or reporting
# This creates a structured table with person index, number of images, and percentage
class_balance_df = pd.DataFrame({
    "person_index": np.arange(1, facedat.shape[1] + 1),  # Person IDs (1-indexed for readability)
    "num_images": counts,  # Number of images per person
})
# Calculate percentage of total dataset each person represents
class_balance_df["percentage"] = (class_balance_df["num_images"] / counts.sum() * 100).round(2)
class_balance_df.to_csv(os.path.join(data_outputs_dir, "class_balance.csv"), index=False)
# index=False prevents saving row indices, keeping CSV clean

# Append summary statistics to a centralized metrics file
# This aggregates all important metrics in one place for easy reference
with open(metrics_path, "a") as f:  # "a" mode appends to file (creates if doesn't exist)
    f.write("=== Class balance statistics ===\n")
    f.write(f"Total persons: {facedat.shape[1]}\n")
    f.write(f"Total images: {counts.sum()}\n")
    f.write(f"Mean images per person: {counts.mean():.2f}\n")
    f.write(f"Std deviation: {counts.std():.4f}\n")
    f.write(f"Min: {counts.min()} | Max: {counts.max()}\n\n")

# Visualize class balance with a bar chart
# This graphical representation makes it easy to spot imbalances at a glance
# Bars of similar height indicate balanced classes, varying heights indicate imbalance
print("\n=== Displaying Class Balance Chart ===")
plt.figure(figsize=(10,4))  # Width=10, height=4 inches for good visibility
plt.bar(range(1, facedat.shape[1] + 1), counts)  # X-axis: person IDs, Y-axis: image counts
plt.xlabel("Person")  # Label for x-axis
plt.ylabel("Number of images")  # Label for y-axis
plt.title("Number of images per person (class balance)")  # Chart title
plt.tight_layout()  # Adjust spacing to prevent label cutoff
plt.savefig(os.path.join(OUT_DIR, "class_balance_per_person.png"), dpi=150)  # Save high-res version
plt.show()  # Display interactive chart
plt.close()  # Free memory

# ===== BUILD FEATURE MATRIX (X) AND TARGET VECTOR (y) =====
# Transform the nested MATLAB structure into flat lists suitable for machine learning
# X: List of images (each image is a 2D numpy array: height x width)
# y: List of labels (person IDs: 0, 1, 2, ..., num_persons-1)
# This format is required by scikit-learn and TensorFlow/Keras
X = []  # Will contain all images as 2D arrays
y = []  # Will contain corresponding person labels (class labels)
for person in range(facedat.shape[1]):  # Iterate through each person
    imgs = facedat[0, person]  # Get all images for this person: shape (H x W x N_imgs)
    for i in range(imgs.shape[2]):  # Iterate through each image for this person
        X.append(imgs[:, :, i])  # Extract single image (2D slice: height x width)
        y.append(person)  # Assign label: person index (0-indexed: 0, 1, 2, ..., 19)
# After this loop, X contains all images and y contains corresponding labels
# The order is preserved: X[i] corresponds to label y[i]

# Analyze the target variable (y) distribution
# This provides another view of class balance using the flattened label list
# Useful for verifying the data extraction was correct and understanding label distribution
df_y = pd.DataFrame({"persona": y})  # Create DataFrame with labels for easy analysis
print("\n=== Target Composition (Count per Person) ===")
# value_counts() counts occurrences of each unique label, sort_index() orders by label
print(df_y["persona"].value_counts().sort_index())
# This shows: how many images belong to each person (class)

# Calculate percentage distribution of classes
# normalize=True converts counts to proportions (0-1), multiply by 100 for percentages
target_pct = (df_y["persona"].value_counts(normalize=True).sort_index() * 100).round(2)

print("\n=== Percentage per Person (%) ===")
print(target_pct)
# Shows what percentage of the total dataset each person represents
# Balanced dataset: percentages should be similar (~5% each for 20 people)
# Imbalanced dataset: percentages vary significantly

# ===== 1) CREATE PANDAS DATAFRAME WITH FLATTENED FEATURES =====
# Convert 2D images into 1D feature vectors for machine learning
# Most ML algorithms expect 2D input: (n_samples, n_features)
# Each pixel becomes a feature, so a 112x92 image becomes a 10,304-dimensional vector
h, w = 112, 92  # UMIST cropped face dimensions: height=112 pixels, width=92 pixels
# Flatten each 2D image into a 1D array: [pixel1, pixel2, ..., pixel10304]
# Result: (n_samples, n_pixels) where n_pixels = h * w = 112 * 92 = 10,304
X_flat = np.array([img.flatten() for img in X])  # List comprehension: flatten each image
# Create descriptive column names for each pixel feature
feature_cols = [f"px_{i}" for i in range(X_flat.shape[1])]  # ["px_0", "px_1", ..., "px_10303"]
# Build DataFrame: rows = images, columns = pixel features + label
df = pd.DataFrame(X_flat, columns=feature_cols)
df["label"] = y  # Add label column for supervised learning

# Display DataFrame structure information
print("\n=== DataFrame Information ===")
print(f"DataFrame shape: {df.shape}")  # (n_samples, n_features + 1 label column)
print(f"DataFrame columns (first 5 + label): {list(df.columns[:5]) + ['...', 'label']}")
# Shows: DataFrame has ~10,304 pixel columns + 1 label column

# ===== 2) STRATIFIED DATA SPLITTING & FEATURE SCALING =====
# Convert label list to numpy array for scikit-learn compatibility
y = np.array(y)

# STRATIFIED SPLIT: Divide data into train/validation/test sets
# Stratified splitting ensures each split maintains the same class distribution as the original
# This prevents one split from having all images of a rare person, which would bias evaluation
# Split strategy: 70% train, 15% validation, 15% test
# First split: separate 70% train from 30% temp (val+test combined)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_flat, y, 
    test_size=0.30,  # 30% goes to temp (will be split into val and test)
    random_state=RANDOM_STATE,  # For reproducibility
    stratify=y  # Maintain class proportions in both train and temp
)
# Second split: divide temp (30%) into val (15%) and test (15%)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.5,  # Split temp 50/50: 15% val, 15% test
    random_state=RANDOM_STATE,  # For reproducibility
    stratify=y_temp  # Maintain class proportions in val and test
)
# Result: Train (70%), Validation (15%), Test (15%)
# Validation set: used for hyperparameter tuning and early stopping
# Test set: used only for final evaluation (never used during training)

# Display split sizes to verify the data was divided correctly
print("\n=== Train/Validation/Test Split Sizes ===")
print(f"Train: {X_train.shape}")  # Should be ~70% of total samples
print(f"Validation: {X_val.shape}")  # Should be ~15% of total samples
print(f"Test: {X_test.shape}")  # Should be ~15% of total samples
# Shape format: (n_samples, n_features) where n_features = 10,304 pixels

# FEATURE NORMALIZATION (StandardScaler)
# Normalize pixel values to have zero mean and unit variance: z = (x - mean) / std
# Why normalize?
#   1. Prevents features with large scales from dominating the model
#   2. Helps neural networks converge faster (gradients are more stable)
#   3. Required for PCA and many ML algorithms to work effectively
#   4. Pixel values originally range 0-255 (or similar), normalization centers them around 0
print("\n=== Applying StandardScaler Normalization ===")
scaler = StandardScaler()  # Create scaler object
# Fit scaler on training data ONLY (compute mean and std from training set)
# Then transform all splits using the training statistics
# CRITICAL: Never fit on validation/test - this would cause data leakage!
X_train_scaled = scaler.fit_transform(X_train)  # Fit (compute stats) + Transform (apply scaling)
X_val_scaled = scaler.transform(X_val)  # Transform only (use training stats)
X_test_scaled = scaler.transform(X_test)  # Transform only (use training stats)
# After scaling: each feature has mean ≈ 0 and std ≈ 1
print("Normalization complete.")

# Visualize class distributions across train/validation/test splits
# This verifies that stratified splitting worked correctly
# Each split should have similar class proportions (bars should look similar across subplots)
# If distributions differ significantly, stratified splitting may have failed
print("\n=== Displaying Split Distributions Chart ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns for train/val/test
for idx, (labels, title) in enumerate([(y_train, "Train"), (y_val, "Validation"), (y_test, "Test")]):
    # Count how many images each person has in this split
    unique, counts = np.unique(labels, return_counts=True)  # unique: person IDs, counts: frequencies
    axes[idx].bar(unique, counts)  # Create bar chart: x=person ID, y=count
    axes[idx].set_title(f"Distribution in {title.upper()}")  # Subplot title
    axes[idx].set_xlabel("Person")  # X-axis label
    axes[idx].set_ylabel("Count in subset")  # Y-axis label
plt.tight_layout()  # Adjust spacing between subplots
plt.savefig(os.path.join(OUT_DIR, "split_distributions.png"), dpi=150)  # Save figure
plt.show()  # Display interactive figure
plt.close()  # Free memory
# Expected result: All three bar charts should show similar patterns
# (proportional heights), confirming stratified split preserved class balance

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
pca_dir = os.path.join(SCRIPT_DIR, "pca_outputs")
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
            max_iter=1000, random_state=RANDOM_STATE, verbose=0)
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

# Autoencoder: nonlinear dimensionality reduction and reconstruction (TensorFlow/Keras)
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

# Autoencoder reconstructions (original vs reconstructed faces)
# Use the trained autoencoder to reconstruct some test images.
recon_test_scaled = autoencoder.predict(X_test_scaled)
# Inverse-transform from scaled space back to original pixel space
recon_test = scaler.inverse_transform(recon_test_scaled)

# Create an output folder for autoencoder figures
ae_dir = os.path.join(SCRIPT_DIR, "autoencoder_outputs")
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


# Clustering on PCA embeddings: K-Means and Hierarchical
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

clustering_dir = os.path.join(SCRIPT_DIR, "clustering_outputs")
os.makedirs(clustering_dir, exist_ok=True)

# === Helper: cluster composition & purity analysis (Task 4) ===
def compute_and_save_cluster_composition(cluster_labels, true_labels, algo_name):
    """
    Compute cluster composition (cluster x true label),
    per-cluster purity, and overall weighted purity.

    Saves:
      - {algo_name}_cluster_composition.csv
      - {algo_name}_cluster_purity.csv

    Returns:
      overall_purity (float)
    """
    # Crosstab: rows = clusters, columns = true labels
    comp_df = pd.crosstab(cluster_labels, true_labels)
    comp_path = os.path.join(clustering_dir, f"{algo_name}_cluster_composition.csv")
    comp_df.to_csv(comp_path)

    # Per-cluster sizes and majority-class counts
    cluster_sizes = comp_df.sum(axis=1).values
    max_per_cluster = comp_df.max(axis=1).values

    # Overall weighted purity across all clusters
    overall_purity = float(max_per_cluster.sum() / cluster_sizes.sum())

    # Per-cluster purity table
    per_cluster_purity = max_per_cluster / cluster_sizes
    per_cluster_df = pd.DataFrame({
        "cluster": comp_df.index,
        "size": cluster_sizes,
        "dominant_class": comp_df.idxmax(axis=1).values,
        "purity": per_cluster_purity,
    })
    purity_path = os.path.join(clustering_dir, f"{algo_name}_cluster_purity.csv")
    per_cluster_df.to_csv(purity_path, index=False)

    print(f"[{algo_name}] Overall weighted purity: {overall_purity:.4f}")
    return overall_purity


# K-MEANS CLUSTERING

k = 20  # since dataset has 20 persons, sensible choice
kmeans = KMeans(n_clusters=k, random_state=42)
train_clusters_km = kmeans.fit_predict(X_train_pca)
val_clusters_km = kmeans.predict(X_val_pca)
test_clusters_km = kmeans.predict(X_test_pca)

sil_km = silhouette_score(X_train_pca, train_clusters_km)
purity_km = compute_and_save_cluster_composition(train_clusters_km, y_train, "kmeans")

print("\n[K-Means] Silhouette Score:", sil_km)

with open(metrics_path, "a") as f:
    f.write("=== K-Means Clustering ===\n")
    f.write(f"Clusters: {k}\n")
    f.write(f"Silhouette Score: {sil_km:.4f}\n")
    f.write(f"Overall weighted cluster purity: {purity_km:.4f}\n\n")

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
purity_hier = compute_and_save_cluster_composition(train_clusters_hier, y_train, "hierarchical")

print("[Hierarchical] Silhouette Score:", sil_hier)

with open(metrics_path, "a") as f:
    f.write("=== Hierarchical Clustering ===\n")
    f.write(f"Clusters: {k}\n")
    f.write(f"Silhouette Score: {sil_hier:.4f}\n")
    f.write(f"Overall weighted cluster purity: {purity_hier:.4f}\n\n")

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


# GMM + t-SNE on Autoencoder Latent Space 
print("\n=== Extracting Autoencoder Latent Features ===")

# Build encoder model to get the 64-D latent representation
encoder = keras.Model(inputs=autoencoder.input,
                      outputs=autoencoder.get_layer("latent").output)

# Extract latent features
X_train_latent = encoder.predict(X_train_scaled)
X_val_latent = encoder.predict(X_val_scaled)
X_test_latent = encoder.predict(X_test_scaled)


latent_dir = os.path.join(SCRIPT_DIR, "latent_clustering_outputs")
os.makedirs(latent_dir, exist_ok=True)

print("Latent shape:", X_train_latent.shape)


# Gaussian Mixture Model (GMM) CLUSTERING

from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score as sk_silhouette_score  # alias to avoid confusion

print("\n=== Running GMM on Autoencoder Latent Space ===")

gmm = GaussianMixture(
    n_components=20,
    covariance_type='full',
    reg_covar=1e-3,
    random_state=42
)


train_clusters_gmm = gmm.fit_predict(X_train_latent)
val_clusters_gmm = gmm.predict(X_val_latent)
test_clusters_gmm = gmm.predict(X_test_latent)

sil_gmm = sk_silhouette_score(X_train_latent, train_clusters_gmm)
purity_gmm = compute_and_save_cluster_composition(train_clusters_gmm, y_train, "gmm_latent")

print("[GMM-Latent] Silhouette Score:", sil_gmm)

with open(metrics_path, "a") as f:
    f.write("=== GMM on Autoencoder Latent Space ===\n")
    f.write("Clusters (components): 20\n")
    f.write(f"Silhouette Score: {sil_gmm:.4f}\n")
    f.write(f"Overall weighted cluster purity: {purity_gmm:.4f}\n\n")


# t-SNE VISUALIZATION OF LATENT SPACE

from sklearn.manifold import TSNE

print("\n=== Running t-SNE on Latent Space ===")

tsne = TSNE(
    n_components=2,
    init="pca",
    perplexity=30,
    learning_rate="auto",
    max_iter=1000
)

X_train_tsne = tsne.fit_transform(X_train_latent)

# Plot: GMM clusters colored on t-SNE
plt.figure(figsize=(7,5))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
            c=train_clusters_gmm, s=10, cmap="viridis")
plt.title("GMM Clusters (Autoencoder Latent Space, t-SNE Projection)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(latent_dir, "gmm_tsne_latent_clusters.png"))
plt.show()

# Plot: True labels on t-SNE
plt.figure(figsize=(7,5))
plt.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1],
            c=y_train, s=10, cmap="tab20")
plt.title("True Labels (Autoencoder Latent Space, t-SNE Projection)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.tight_layout()
plt.savefig(os.path.join(latent_dir, "tsne_latent_true_labels.png"))
plt.show()

print("\nGMM + t-SNE on Autoencoder Latent Space completed.")


# Supervised Learning – Neural Network Classifier (Task 5)
print("\n=== Preparing Neural Network Classifier ===")
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
)

nn_dir = os.path.join(SCRIPT_DIR, "nn_outputs")
os.makedirs(nn_dir, exist_ok=True)

print("\n=== Building Supervised Neural Network Classifier ===")

num_classes = len(np.unique(y_train))

# Use PCA features + K-Means cluster label (one-hot) as input to classifier
def build_classifier_features(X_pca, cluster_labels, num_clusters):
    cluster_onehot = keras.utils.to_categorical(cluster_labels, num_classes=num_clusters)
    return np.concatenate([X_pca, cluster_onehot], axis=1)

X_train_clf = build_classifier_features(X_train_pca, train_clusters_km, k)
X_val_clf = build_classifier_features(X_val_pca, val_clusters_km, k)
X_test_clf = build_classifier_features(X_test_pca, test_clusters_km, k)

# Baseline features: PCA only (no cluster labels)
X_train_pca_only = X_train_pca
X_val_pca_only = X_val_pca
X_test_pca_only = X_test_pca

print("Classifier feature shapes:")
print("Train:", X_train_clf.shape, "Val:", X_val_clf.shape, "Test:", X_test_clf.shape)

# Evaluation helpers
def evaluate_split(name, X_split, y_split, model, out_dir):
    y_prob = model.predict(X_split)
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_split, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_split, y_pred, average="macro", zero_division=0
    )

    print(f"\n[NN Classifier] {name} metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    # Save classification report
    report_txt = classification_report(y_split, y_pred, digits=4, zero_division=0)
    with open(os.path.join(out_dir, f"classification_report_{name.lower()}.txt"), "w") as f:
        f.write(report_txt)

    return y_pred, acc, precision, recall, f1

def run_classifier_hparam_search(
    X_train,
    y_train,
    X_val,
    y_val,
    num_classes,
    metrics_path,
    nn_dir,
):
    print("\n=== Hyperparameter evaluation (classifier on PCA+KMeans features) ===")

    search_space = [
        {"name": "hp1_lr1e-3_dr0.3_u256_128", "lr": 1e-3, "dropout": 0.3, "units": [256, 128]},
        {"name": "hp2_lr5e-4_dr0.4_u512_256", "lr": 5e-4, "dropout": 0.4, "units": [512, 256]},
        {"name": "hp3_lr1e-3_dr0.5_u256_64", "lr": 1e-3, "dropout": 0.5, "units": [256, 64]},
    ]

    results = []
    input_dim = X_train.shape[1]

    for cfg in search_space:
        print(f"Training {cfg['name']} ...")
        inputs = keras.Input(shape=(input_dim,), name=f"clf_hp_input_{cfg['name']}")
        z = layers.Dense(cfg["units"][0], activation="relu")(inputs)
        z = layers.Dropout(cfg["dropout"])(z)
        z = layers.Dense(cfg["units"][1], activation="relu")(z)
        z = layers.Dropout(cfg["dropout"])(z)
        outputs = layers.Dense(num_classes, activation="softmax")(z)

        model = keras.Model(inputs, outputs, name=f"clf_hp_{cfg['name']}")
        model.compile(
            optimizer=keras.optimizers.Adam(cfg["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        early_stop = keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        )

        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=25,
            batch_size=64,
            shuffle=True,
            callbacks=[early_stop],
            verbose=0,
        )

        # Validation evaluation only; reports saved for traceability
        _, acc, prec, rec, f1 = evaluate_split(
            f"VAL_HP_{cfg['name']}", X_val, y_val, model, nn_dir
        )

        results.append({
            "name": cfg["name"],
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "cfg": cfg,
        })

    if results:
        best = max(results, key=lambda r: r["acc"])
        with open(metrics_path, "a") as f:
            f.write("=== Classifier Hyperparameter Evaluation (PCA + KMeans features) ===\n")
            for r in results:
                cfg = r["cfg"]
                f.write(
                    f"{r['name']}: lr={cfg['lr']}, dropout={cfg['dropout']}, units={cfg['units']} | "
                    f"Val Acc: {r['acc']:.4f}, Prec: {r['prec']:.4f}, Rec: {r['rec']:.4f}, F1: {r['f1']:.4f}\n"
                )
            f.write(f"Best (by val accuracy): {best['name']}\n\n")

def build_and_train_classifier(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    num_classes,
    nn_dir,
    metrics_path,
    label_suffix,
    write_history=True,
):
    """
    label_suffix is a short string used to distinguish outputs, for example:
      "_pca_kmeans" or "_pca_only"
    write_history controls whether to save training history and curves (we only need it for the main model).
    """

    input_dim_clf = X_train.shape[1]

    clf_inputs = keras.Input(shape=(input_dim_clf,), name="clf_input" + label_suffix)
    z = layers.Dense(256, activation="relu")(clf_inputs)
    z = layers.Dropout(0.3)(z)
    z = layers.Dense(128, activation="relu")(z)
    z = layers.Dropout(0.3)(z)
    clf_outputs = layers.Dense(num_classes, activation="softmax", name="clf_output" + label_suffix)(z)

    clf_model = keras.Model(clf_inputs, clf_outputs, name="face_classifier" + label_suffix)

    clf_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(f"\n[Classifier{label_suffix}] Model summary:")
    clf_model.summary()

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    history_clf = clf_model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        shuffle=True,
        callbacks=[early_stop],
        verbose=1,
    )

    # Save training history and curves only for the main model (PCA + KMeans)
    if write_history:
        clf_history_df = pd.DataFrame({
            "epoch": np.arange(1, len(history_clf.history["loss"]) + 1),
            "train_loss": history_clf.history["loss"],
            "val_loss": history_clf.history["val_loss"],
            "train_accuracy": history_clf.history["accuracy"],
            "val_accuracy": history_clf.history["val_accuracy"],
        })
        clf_history_df.to_csv(os.path.join(data_outputs_dir, f"nn_classifier_history{label_suffix}.csv"), index=False)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(clf_history_df["epoch"], clf_history_df["train_loss"], label="Train")
        plt.plot(clf_history_df["epoch"], clf_history_df["val_loss"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"NN Classifier{label_suffix} - Loss")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(clf_history_df["epoch"], clf_history_df["train_accuracy"], label="Train")
        plt.plot(clf_history_df["epoch"], clf_history_df["val_accuracy"], label="Val")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"NN Classifier{label_suffix} - Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(nn_dir, f"nn_training_curves{label_suffix}.png"))
        plt.show()

    # Evaluate on validation and test sets using existing helper
    y_val_pred, acc_val, prec_val, rec_val, f1_val = evaluate_split(
        "VAL" + label_suffix.upper(), X_val, y_val, clf_model, nn_dir
    )
    y_test_pred, acc_test, prec_test, rec_test, f1_test = evaluate_split(
        "TEST" + label_suffix.upper(), X_test, y_test, clf_model, nn_dir
    )

    # Save confusion matrix for test set
    cm_test = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm_test, interpolation="nearest")
    plt.title(f"NN Classifier{label_suffix} - Confusion Matrix (Test)")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(os.path.join(nn_dir, f"nn_confusion_matrix_test{label_suffix}.png"))
    plt.show()

    # Append metrics to global metrics file
    with open(metrics_path, "a") as f:
        f.write(f"=== Neural Network Classifier {label_suffix} ===\n")
        f.write(f"Validation - Accuracy: {acc_val:.4f}, Precision (macro): {prec_val:.4f}, "
                f"Recall (macro): {rec_val:.4f}, F1 (macro): {f1_val:.4f}\n")
        f.write(f"Test       - Accuracy: {acc_test:.4f}, Precision (macro): {prec_test:.4f}, "
                f"Recall (macro): {rec_test:.4f}, F1 (macro): {f1_test:.4f}\n\n")

    return clf_model

# Hyperparameter evaluation (lightweight search on classifier settings)
run_classifier_hparam_search(
    X_train_clf,
    y_train,
    X_val_clf,
    y_val,
    num_classes,
    metrics_path,
    nn_dir,
)

# Train classifiers
clf_model_pca_kmeans = build_and_train_classifier(
    X_train_clf,
    y_train,
    X_val_clf,
    y_val,
    X_test_clf,
    y_test,
    num_classes=num_classes,
    nn_dir=nn_dir,
    metrics_path=metrics_path,
    label_suffix="_pca_kmeans",
    write_history=True,  # save curves and history for this main model
)

clf_model_pca_only = build_and_train_classifier(
    X_train_pca_only,
    y_train,
    X_val_pca_only,
    y_val,
    X_test_pca_only,
    y_test,
    num_classes=num_classes,
    nn_dir=nn_dir,
    metrics_path=metrics_path,
    label_suffix="_pca_only",
    write_history=False,  # optional, we do not need extra curves for the baseline
)

# Sample test predictions: display images with true & predicted labels 
num_show = min(8, X_test.shape[0])
indices = np.random.choice(X_test.shape[0], size=num_show, replace=False)

y_test_pred = np.argmax(clf_model_pca_kmeans.predict(X_test_clf), axis=1)

plt.figure(figsize=(2.5 * num_show, 3))
for i, idx in enumerate(indices):
    img = X_test[idx].reshape(h, w)
    true_label = y_test[idx]
    pred_label = y_test_pred[idx]

    ax = plt.subplot(1, num_show, i + 1)
    ax.imshow(img, cmap="gray")
    ax.axis("off")
    ax.set_title(f"T:{true_label}\nP:{pred_label}", fontsize=8)

plt.suptitle("NN Classifier - Sample Test Predictions\nT=true, P=predicted", fontsize=10)
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig(os.path.join(nn_dir, "nn_sample_predictions.png"))
plt.show()


print("\nNeural Network Classifier training and evaluation completed.")

# ==== SAVE MODELS FOR API USE ====
from joblib import dump

models_dir = os.path.join(SCRIPT_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Save sklearn objects
dump(scaler, os.path.join(models_dir, "scaler.joblib"))
dump(pca, os.path.join(models_dir, "pca.joblib"))
dump(kmeans, os.path.join(models_dir, "kmeans.joblib"))

# Save Keras classifier model
clf_model_pca_kmeans.save(os.path.join(models_dir, "face_classifier_pca_kmeans.h5"))

print("All models saved to:", models_dir)
