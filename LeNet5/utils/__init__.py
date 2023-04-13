import torch

def getTorchDevice():
    if not torch.backends.mps.is_available():
        device = torch.device("cpu")
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")

        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")

    else:
        mps_device = torch.device("mps")
    if (mps_device is not None):
        device = mps_device

    return device