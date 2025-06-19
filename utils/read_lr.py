import torch
import torch.nn as nn
import torch.nn.functional as F

def adjust_learning_rate(init_lr, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1
        print("epoch: {}, factor: {}".format(epoch, factor))

    lr = init_lr * (0.1 ** factor) # 20 --> 0.02

    """Warmup"""
    if epoch < 5:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)
    return lr

init_lr = 20
epoch = 81
len_epoch = 10010

import matplotlib.pyplot as plt
# Sample learning rate data (replace with your own during training)
lrs = []
steps = []
epochs = []

# fig_name = "lr_vs_epochs_search.png"
# for e in range(81, 90):
#     lr = adjust_learning_rate(init_lr, e, 0, len_epoch)
#     lrs.append(lr)
#     epochs.append(e)

# fig_name = "lr_vs_epochs_finetune.png"
# init_lr = 0.128*2
# for e in range(52, 90):
#     lr = adjust_learning_rate(init_lr, e, 0, len_epoch)
#     lrs.append(lr)
#     epochs.append(e)


fig_name = "lr_vs_epochs_finetune_cosine.png"
class simplenet(nn.Module):
    def __init__(self):
        super(simplenet, self).__init__()
        self.fc1 = nn.Linear(3, 16)
        self.fc2 = nn.Linear(16, 8)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return x

model = simplenet()
init_lr = 0.1
train_epochs = 100
optimizer = torch.optim.SGD(model.parameters(), init_lr,
                            momentum=0.9,
                            weight_decay=0.)

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_epochs - 0, eta_min=1e-3
    )

for e in range(train_epochs):
    lr_scheduler.step()
    lr = optimizer.param_groups[0]['lr'] 
    lrs.append(lr)
    epochs.append(e)

print(lrs)

# Plotting
plt.figure(figsize=(8, 5))
plt.plot(epochs, lrs, label="Learning Rate")
plt.xlabel("Epochs")
plt.ylabel("Learning Rate")
plt.title("Learning Rate vs Epochs")
plt.grid(True)
plt.legend()

# Save figure
plt.savefig(fig_name, dpi=300)
plt.close()

print("Saved plot to lr_vs_epochs.png")