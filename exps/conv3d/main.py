import mlx.core as mx
import mlx.nn
import torch


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        in_ch, out_ch = 3, 5
        kernel_size = [2, 4, 4]
        self.conv3d = torch.nn.Conv3d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def forward(self, x):
        return self.conv3d(x)


class MLXModel(mlx.nn.Module):
    def __init__(self):
        super().__init__()
        in_ch, out_ch = 3, 5
        kernel_size = [2, 4, 4]
        self.conv3d = mlx.nn.Conv3d(
            in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size, stride=kernel_size, bias=False
        )

    def __call__(self, x):
        x = x.transpose(0, 2, 3, 4, 1)
        d = self.conv3d(x)
        return d


def tensor_to_mlx(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        new_state_dict[k] = mx.array(v.numpy()).transpose(0, 2, 3, 4, 1)
    return new_state_dict


if __name__ == "__main__":
    state_dict = torch.load("torch_model.pt")
    torch_model = TorchModel()
    # torch.save(torch_model.state_dict(), "torch_model.pt")
    torch_model.load_state_dict(state_dict)
    mlx_model = MLXModel()
    state_dict = tensor_to_mlx(state_dict)
    mlx_model.load_weights(list(state_dict.items()))

    # print("torch weights:", torch_model.conv3d.weight, torch_model.conv3d.weight.shape)
    # print("mlx weights:", mlx_model.conv3d.weight, mlx_model.conv3d.weight.shape)

    in_ch, out_ch = 3, 5
    patch_size = 4
    x = torch.randn(1, in_ch, out_ch, patch_size, patch_size)
    torch_y = torch_model(x)
    mlx_y = mlx_model(mx.array(x.numpy()))

    print("torch_y:", torch_y, torch_y.shape, torch_y.dtype)
    print("mlx_y:", mlx_y, mlx_y.shape)
