import torch

pt_path = "/home/zzq/Documents/apps/optimization/pruning/PaS_Timm/save_dir/resnet50_A3_finetune_lr_0.0005_eps_50_bs_256_p0.55_scale_1e3_fp32/20250618-015135-resnet50-160/convergence_0.55_pmacs_1.8366825580596924_tmacs_1.839211344718933.pt"
pt_path = "/home/zzq/Documents/apps/optimization/pruning/PaS_Timm/save_dir/resnet50_A3_search_lr_0.0005_eps_50_bs_128_p0.55_scale_1e3_fp32/20250617-060241-resnet50-160/convergence_0.55_pmacs_1.8366825580596924_tmacs_1.839211344718933.pt"

data = torch.load(pt_path)


print(data)
print(data.shape)
for step in range(data.shape[0]):
    print(step, data[step])

