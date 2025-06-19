
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Example data
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# pt_path = "/home/zzq/Documents/apps/optimization/pruning/PaS_Timm/save_dir/resnet50_A3_search_lr_0.0005_eps_50_bs_128_p0.55_scale_1e3_fp32/20250617-060241-resnet50-160/convergence_0.55_pmacs_1.8366825580596924_tmacs_1.839211344718933.pt"
pt_path = "/home/zzq/Documents/apps/optimization/pruning/PaS_Timm/save_dir/resnet50_A3_search_lr_0.0005_eps_50_bs_128_p0.55_scale_1e2_fp32/20250619-013227-resnet50-160/ep_23_convergence_0.55_pmacs_1.8321411609649658_tmacs_1.839211344718933.pt"
data = torch.load(pt_path)

# X: iterations, Y: layers
iterations = np.arange(data.shape[0])
layers = np.arange(data.shape[1])
X, Y = np.meshgrid(iterations, layers)

# Z: channel values — transpose so shape matches meshgrid
Z = data.T  # shape (50, 100) to match meshgrid
print("Z:", Z.shape)

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_xlabel("Iteration")
ax.set_ylabel("Layer")
ax.set_zlabel("Channel Magnitude")
plt.title("Channel Activations Across Iterations and Layers")
plt.colorbar(surf, ax=ax, shrink=0.5)
plt.show()
plt.savefig("prune.png")
