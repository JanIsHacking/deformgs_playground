from argparse import ArgumentParser
import torch
import numpy as np
from typing import NamedTuple
from arguments import ModelHiddenParams
from scene.gaussian_model import GaussianModel

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

# 1. Instantiate the model with the same `sh_degree` and `args` as when the checkpoint was created
sh_degree = 3 # Example value, should match the original
args = ModelHiddenParams(ArgumentParser())
model = GaussianModel(sh_degree, args)

# 2. Load the checkpoint (assuming the path to the checkpoint is known)
checkpoint_path = 'output/synthetic/scene_1/chkpnt_20000_fine.pth'

# 3. Load the checkpoint data (model_args, training_args)
checkpoint_data = torch.load(checkpoint_path)

# 4. Restore the model from the checkpoint
model.restore(checkpoint_data[0], checkpoint_data[1])  # checkpoint_data is a tuple (model_args, training_args)

# 5. Inspect the model
print(f"Active SH Degree: {model.active_sh_degree}")
print(f"XYZ: {model.get_xyz}")
print(f"Scaling: {model.get_scaling}")
print(f"Rotation: {model.get_rotation}")
print(f"Opacity: {model.get_opacity}")

# Example: Get and print the covariance
scaling_modifier = 1
covariance = model.get_covariance(scaling_modifier)
print(f"Covariance: {covariance}")
