import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from sklearn.metrics import mean_squared_error
from skimage import data, color, transform, feature
from skimage.io import imshow


def recover_sparse_vector(N, s, p):
    
    alpha = np.zeros(N) 
    non_zero_indices = np.random.choice(alpha.shape[0], s, replace = False)
    alpha[non_zero_indices] = np.abs(np.random.randn(s))

    X = np.random.randn(N,N)        # Random square gausian matrix of size(N,N)
    y = X @ alpha                   # Noiseless Transformation y = Xα

    plt.stem(alpha)
    plt.title('Ground truth Alpha Values')
    plt.show()

    # **Now take a subsample Xp of X corresponding to n randomly chosen rows:**

    total_rows = X.shape[0]
    random_indices = np.random.choice(total_rows, size=p, replace=False)  # Indices
    X_p = X[random_indices, :]      # Selecting random rows from X_n
    y_p = y[random_indices]         # Selecting random rows from y_n


    # Solving the minimization problem
    alpha_hat = cp.Variable(N)

    objective = cp.Minimize(cp.norm1(alpha_hat))

    constraints = [y_p == X_p @ alpha_hat ]

    # Solve the problem
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    alpha_hat = alpha_hat.value
    return(alpha, alpha_hat)

N = 1000    # Size of the sparse vector, Alpha
s = 10      # Non-zero entries in Alpha
p = 200     # For subsampling: total number of rows for indexing
alpha, alpha_hat = recover_sparse_vector(N,s,p)
plt.title('Calculated Alpha Vs Ground truth')
plt.stem(alpha, markerfmt='bo', label = 'Ground Truth')
plt.stem(alpha_hat, markerfmt='co', label = 'Alpha_hat')
plt.legend()
plt.show()

mse_value = mean_squared_error(alpha,alpha_hat)

def recover_prob_function(N, s, p, n):
    
    recover_prob = []
    for i in range(100):
        alpha, alpha_hat = recover_sparse_vector(N,s,p)
        mse_value = mean_squared_error(alpha,alpha_hat)
        result = np.isclose(mse_value, 0, atol=1e-5 )
        output = 1 if result else 0
        recover_prob.append(output)
    prob = (sum(recover_prob)/len(recover_prob))*100
    print('Probability of recovering the correct support when experiment was ran {0} times: {1} %'.format(100, prob))    
    return prob

# Parameters
N = 100  # Size of alpha
s_values = [5, 10, 15, 20, 25, 30]  # Sparsity levels
p_values = list(range(10, N, 10))  # Number of measurements
n=10
# Phase Transition Data Storage
phase_transition_data = np.zeros((len(s_values), len(p_values)))

# 7. Phase Transition
for i, s in enumerate(s_values):
    for j, p in enumerate(p_values):
        phase_transition_data[i, j] = recover_prob_function(N, s,p, n)

phase_transition_data

#  Load sample image
image = data.camera()
image = transform.resize(image, (128, 128))  # Resizing to manageable size

# Convert the image to grayscale
gray_image = color.rgb2gray(image) if len(image.shape) == 3 else image

# Edge detection to get a sparse representation
edges = feature.canny(gray_image)

# Visualize original and sparse representation
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(edges, cmap='gray')
ax[1].set_title('Sparse Representation (Edges)')
ax[1].axis('off')

plt.tight_layout()
plt.show()    

from skimage.transform import radon, iradon

# Set of angles for the Radon transform
theta = np.linspace(0., 180., max(gray_image.shape), endpoint=False)

# Apply Radon transform to get projections
sinogram = radon(edges, theta=theta, circle=True)

# Subsample: Taking every 4th projection angle as an example
subsampled_theta = theta[::4]
subsampled_sinogram = sinogram[:, ::4]

# Visualize original and subsampled sinograms (projections)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(sinogram, cmap='gray', aspect='auto')
ax[0].set_title('Full Projections')
ax[0].set_xlabel('Projection angle (deg)')
ax[0].set_ylabel('Projection position (pixels)')

ax[1].imshow(subsampled_sinogram, cmap='gray', aspect='auto')
ax[1].set_title('Subsampled Projections')
ax[1].set_xlabel('Projection angle (deg)')
ax[1].set_ylabel('Projection position (pixels)')

plt.tight_layout()
plt.show()

# Use Inverse Radon transform for reconstruction
reconstruction = iradon(subsampled_sinogram, theta=subsampled_theta, circle=True)

# Visualize the original sparse image and the reconstruction
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(edges, cmap='gray')
ax[0].set_title('Original Sparse Image')
ax[0].axis('off')

ax[1].imshow(reconstruction, cmap='gray')
ax[1].set_title('Reconstruction from Subsampled Projections')
ax[1].axis('off')

plt.tight_layout()
plt.show()



# Apply Radon transform to the original image to get projections
original_sinogram = radon(gray_image, theta=theta, circle=True)

# Subsample: Taking every 4th projection angle as an example
subsampled_original_sinogram = original_sinogram[:, ::4]

# Reconstruction from Subsampled Projections of the original image
reconstruction_from_original = iradon(subsampled_original_sinogram, theta=subsampled_theta, circle=True)

# Visualize the original image and the reconstruction
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(gray_image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(reconstruction_from_original, cmap='gray')
ax[1].set_title('Reconstruction from Subsampled Projections')
ax[1].axis('off')

plt.tight_layout()
plt.show()
