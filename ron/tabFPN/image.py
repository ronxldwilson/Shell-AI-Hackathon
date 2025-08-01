import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class TabularToImageTransformer:
    def __init__(self, method='grid', image_size=(8, 7), scaling='standard'):
        """
        Transform tabular data to images for CNN processing
        
        Parameters:
        - method: 'grid', 'correlation', 'tsne', 'pca'
        - image_size: tuple for output image dimensions
        - scaling: 'standard', 'minmax', 'none'
        """
        self.method = method
        self.image_size = image_size
        self.scaling = scaling
        self.scaler = None
        self.feature_positions = None
        self.fitted = False
        
    def fit(self, X):
        """Fit the transformer on training data"""
        X = np.array(X)
        n_features = X.shape[1]
        
        # Fit scaler
        if self.scaling == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling == 'minmax':
            self.scaler = MinMaxScaler()
        
        if self.scaler:
            self.scaler.fit(X)
        
        # Generate feature positions based on method
        if self.method == 'grid':
            self.feature_positions = self._create_grid_positions(n_features)
        elif self.method == 'correlation':
            self.feature_positions = self._create_correlation_positions(X)
        elif self.method == 'tsne':
            self.feature_positions = self._create_tsne_positions(X)
        elif self.method == 'pca':
            self.feature_positions = self._create_pca_positions(X)
        
        self.fitted = True
        return self
    
    def transform(self, X):
        """Transform tabular data to images"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted first")
        
        X = np.array(X)
        
        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Create images
        n_samples = X_scaled.shape[0]
        images = np.zeros((n_samples, self.image_size[0], self.image_size[1], 1))
        
        for i, (row, col) in enumerate(self.feature_positions):
            if row < self.image_size[0] and col < self.image_size[1]:
                images[:, row, col, 0] = X_scaled[:, i]
        
        return images
    
    def fit_transform(self, X):
        """Fit and transform in one step"""
        return self.fit(X).transform(X)
    
    def _create_grid_positions(self, n_features):
        """Simple grid arrangement"""
        positions = []
        for i in range(n_features):
            row = i // self.image_size[1]
            col = i % self.image_size[1]
            positions.append((row, col))
        return positions
    
    def _create_correlation_positions(self, X):
        """Arrange features based on correlation similarity"""
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X.T)
        
        # Handle NaN values that might occur with constant features
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        # Create distance matrix (1 - absolute correlation)
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Ensure diagonal is exactly 0 and matrix is symmetric
        np.fill_diagonal(distance_matrix, 0)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Ensure all values are non-negative and bounded
        distance_matrix = np.clip(distance_matrix, 0, 2)
        
        # Convert to condensed distance matrix for linkage
        condensed_dist = squareform(distance_matrix, checks=False)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Get feature order from clustering
        from scipy.cluster.hierarchy import leaves_list
        feature_order = leaves_list(linkage_matrix)
        
        # Arrange in grid based on clustering order
        positions = []
        for i, feature_idx in enumerate(feature_order):
            row = i // self.image_size[1]
            col = i % self.image_size[1]
            positions.append((row, col))
        
        # Reorder positions to match original feature order
        ordered_positions = [None] * len(feature_order)
        for i, pos in enumerate(positions):
            ordered_positions[feature_order[i]] = pos
        
        return ordered_positions
    
    def _create_tsne_positions(self, X):
        """Use t-SNE to arrange features in 2D space"""
        # Use feature vectors (transpose to treat features as samples)
        X_features = X.T
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_features)-1))
        positions_2d = tsne.fit_transform(X_features)
        
        # Scale positions to fit in image grid
        positions_2d_scaled = self._scale_positions_to_grid(positions_2d)
        
        return positions_2d_scaled
    
    def _create_pca_positions(self, X):
        """Use PCA to arrange features in 2D space"""
        # Use feature vectors (transpose to treat features as samples)
        X_features = X.T
        
        # Apply PCA
        pca = PCA(n_components=2, random_state=42)
        positions_2d = pca.fit_transform(X_features)
        
        # Scale positions to fit in image grid
        positions_2d_scaled = self._scale_positions_to_grid(positions_2d)
        
        return positions_2d_scaled
    
    def _scale_positions_to_grid(self, positions_2d):
        """Scale 2D positions to fit in the image grid"""
        # Normalize positions to [0, 1]
        positions_norm = (positions_2d - positions_2d.min(axis=0)) / (positions_2d.max(axis=0) - positions_2d.min(axis=0))
        
        # Scale to grid size
        positions_scaled = []
        for i in range(len(positions_norm)):
            row = int(positions_norm[i, 0] * (self.image_size[0] - 1))
            col = int(positions_norm[i, 1] * (self.image_size[1] - 1))
            positions_scaled.append((row, col))
        
        return positions_scaled
    
    def visualize_feature_arrangement(self, feature_names=None):
        """Visualize how features are arranged in the image"""
        if not self.fitted:
            raise ValueError("Transformer must be fitted first")
        
        # Create visualization grid
        grid = np.full(self.image_size, -1, dtype=int)
        
        for feature_idx, (row, col) in enumerate(self.feature_positions):
            if row < self.image_size[0] and col < self.image_size[1]:
                grid[row, col] = feature_idx
        
        plt.figure(figsize=(10, 8))
        plt.imshow(grid, cmap='tab20', aspect='auto')
        plt.colorbar(label='Feature Index')
        plt.title(f'Feature Arrangement ({self.method} method)')
        
        # Add feature labels if provided
        if feature_names:
            for feature_idx, (row, col) in enumerate(self.feature_positions):
                if row < self.image_size[0] and col < self.image_size[1] and feature_idx < len(feature_names):
                    plt.text(col, row, f'{feature_idx}', ha='center', va='center', fontsize=8, color='white')
        
        plt.xlabel('Column')
        plt.ylabel('Row')
        plt.show()

def create_cnn_model(input_shape, n_outputs=10):
    """Create CNN model for tabular image data"""
    model = keras.Sequential([
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        
        # Global pooling and dense layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        # Output layer (no activation for regression)
        layers.Dense(n_outputs, activation='linear')
    ])
    
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae', 'mape']  # Added MAPE as a metric
    )
    
    return model

def evaluate_model_performance(model, X_test_images, y_test):
    """Evaluate model performance with MAPE as primary metric"""
    y_pred = model.predict(X_test_images)
    
    # Calculate metrics for each output
    mse_scores = []
    r2_scores = []
    mape_scores = []
    
    for i in range(y_test.shape[1]):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        # Calculate MAPE with handling for zero values
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        y_true_safe = np.where(np.abs(y_test[:, i]) < epsilon, epsilon, y_test[:, i])
        mape = np.mean(np.abs((y_test[:, i] - y_pred[:, i]) / y_true_safe)) * 100
        
        mse_scores.append(mse)
        r2_scores.append(r2)
        mape_scores.append(mape)
    
    print("Performance by Output:")
    for i in range(len(mape_scores)):
        print(f"Output {i+1}: MAPE = {mape_scores[i]:.2f}%, MSE = {mse_scores[i]:.4f}, R² = {r2_scores[i]:.4f}")
    
    print(f"\nOverall Performance:")
    print(f"Average MAPE: {np.mean(mape_scores):.2f}%")
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    print(f"Average R²: {np.mean(r2_scores):.4f}")
    
    return mape_scores, mse_scores, r2_scores, y_pred

# Example usage function
def run_complete_pipeline(X, y, test_size=0.2, method='correlation', image_size=(8, 7)):
    """
    Complete pipeline for tabular to image CNN
    
    Parameters:
    - X: Feature matrix (samples x features)
    - y: Target matrix (samples x targets)
    - test_size: Train/test split ratio
    - method: Transformation method ('grid', 'correlation', 'tsne', 'pca')
    - image_size: Dimensions for the feature image
    """
    
    print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features, {y.shape[1]} targets")
    print(f"Using {method} method with {image_size[0]}x{image_size[1]} images\n")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Transform to images
    transformer = TabularToImageTransformer(method=method, image_size=image_size, scaling='standard')
    X_train_images = transformer.fit_transform(X_train)
    X_test_images = transformer.transform(X_test)
    
    print(f"Image shape: {X_train_images.shape}")
    
    # Visualize feature arrangement
    print("\nFeature arrangement in image:")
    transformer.visualize_feature_arrangement()
    
    # Create and train model
    model = create_cnn_model(X_train_images.shape[1:], n_outputs=y.shape[1])
    print(f"\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train_images, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=7, factor=0.5)
        ]
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    mape_scores, mse_scores, r2_scores, y_pred = evaluate_model_performance(model, X_test_images, y_test)
    
    # Plot training history
    plt.figure(figsize=(18, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history.history['mape'], label='Training MAPE')
    plt.plot(history.history['val_mape'], label='Validation MAPE')
    plt.title('Model MAPE (%)')
    plt.xlabel('Epoch')
    plt.ylabel('MAPE (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    return model, transformer, history, (X_test_images, y_test, y_pred)

# Example with synthetic data (replace with your actual data)
def create_example_data():
    """Create synthetic data for demonstration"""
    np.random.seed(42)
    n_samples = 1000
    n_features = 55
    n_targets = 10
    
    # Create correlated features
    X = np.random.randn(n_samples, n_features)
    
    # Create some feature interactions
    X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)
    X[:, 2] = X[:, 0] * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    # Create targets based on complex feature interactions
    y = np.zeros((n_samples, n_targets))
    for i in range(n_targets):
        # Each target depends on different feature combinations
        start_idx = (i * 5) % n_features
        end_idx = min(start_idx + 10, n_features)
        y[:, i] = np.sum(X[:, start_idx:end_idx], axis=1) + 0.1 * np.random.randn(n_samples)
    
    return X, y

# Run example focused on correlation method
if __name__ == "__main__":
    # Create example data (replace this with loading your actual data)
    X_example, y_example = create_example_data()
    
    print("Running pipeline with CORRELATION method (recommended for better performance)")
    print("Replace 'create_example_data()' with your actual data loading\n")
    
    # Focus on correlation method - the best approach
    print(f"{'='*60}")
    print("USING CORRELATION METHOD - Groups Similar Features Together")
    print('='*60)
    
    model, transformer, history, results = run_complete_pipeline(
        X_example, y_example, 
        method='correlation',  # Using correlation method exclusively
        image_size=(8, 7)
    )
    
    print(f"\n{'='*60}")
    print("CORRELATION METHOD COMPLETE")
    print("This method places correlated features near each other,")
    print("giving the CNN more meaningful spatial patterns to learn from!")
    print('='*60)