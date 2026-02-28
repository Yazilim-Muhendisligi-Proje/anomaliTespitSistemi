import torch

try:
    path = r"c:\Users\isley\Desktop\Yazılım Mühendisliği\anomali\anomaliTespitSistemi\anomaliProje\aiops_gnn_model.pth"
    model = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(model, dict):
        print("Model keys:", list(model.keys()))
        for k, v in model.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: Tensor of shape {v.shape}")
            else:
                print(f"{k}: {type(v)}")
    else:
        print(type(model))
        print(model)
except Exception as e:
    print(f"Error: {e}")
