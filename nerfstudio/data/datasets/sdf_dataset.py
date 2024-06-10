# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
SDFStudio dataset.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import Tensor
import cv2

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.model_components.ray_generators import RayGenerator


class SDFDataset(InputDataset):
    """Dataset that returns images and depths.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth", "normal"]

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)

        # can be none if monoprior not included
        self.depth_filenames = self.metadata["depth_filenames"]
        self.normal_filenames = self.metadata["normal_filenames"]
        self.camera_to_worlds = self.metadata["camera_to_worlds"]
        # can be none if auto orient not enabled in dataparser
        self.transform = self.metadata["transform"]
        self.include_mono_prior = self.metadata["include_mono_prior"]
        self.train_ray_generator = RayGenerator(self.cameras)

    def get_metadata(self, data: Dict) -> Dict:
        # TODO supports foreground_masks
        metadata = {}
        if self.include_mono_prior:
            depth_filepath = self.depth_filenames[data["image_idx"]]
            normal_filepath = self.normal_filenames[data["image_idx"]]
            camtoworld = self.camera_to_worlds[data["image_idx"]]

            # Scale depth images to meter units and also by scaling applied to cameras
            depth_image, normal_image = self.get_depths_and_normals(
                depth_filepath=depth_filepath, normal_filename=normal_filepath, camtoworld=camtoworld
            )
            metadata["depth"] = depth_image
            metadata["normal"] = normal_image

        if "source_collider" in self.metadata:
            X_steps = torch.linspace(start=0, end=data["image"].shape[0]-1, steps=data["image"].shape[0])
            Y_steps = torch.linspace(start=0, end=data["image"].shape[1]-1, steps=data["image"].shape[1])
            pixel_indices_X, pixel_indices_Y = torch.meshgrid(X_steps, Y_steps)
            camera_indices = torch.full(data["image"].shape[0:2], data["image_idx"])
            ray_indices = torch.dstack([camera_indices, pixel_indices_X, pixel_indices_Y])
            ray_indices = torch.flatten(ray_indices, end_dim=1).to(dtype=torch.int32)
            ray_bundle = self.train_ray_generator(ray_indices)
            source_collider = self.metadata["source_collider"]
            ray_bundle = source_collider(ray_bundle)
            ray_bundle = self.metadata["collider"](ray_bundle, filter_out_nan=False)
            nears_mask = ~torch.isnan(ray_bundle.nears)
            fars_mask = ~torch.isnan(ray_bundle.fars)
            bbox_mask = nears_mask & fars_mask
            bbox_mask = bbox_mask.expand(-1, 3).reshape(data["image"].shape)

            source_intersections = ray_bundle.source_intersections
            source_shape_mask = ~torch.isnan(source_intersections)
            source_shape_mask = source_shape_mask.reshape(data["image"].shape)

            source_distances = ray_bundle.source_distances            
            source_position_mask = torch.full(source_distances.shape, True)
            source_position_mask[source_distances < ray_bundle.nears + 1e-6] = False
            source_position_mask = source_position_mask.expand(-1, 3).reshape(data["image"].shape)

            rays_init_power = source_collider.get_rays_init_power(ray_bundle, self.metadata["def_absorption"], self.metadata["pixel_size"])
            image = torch.mean(torch.tensor(data["image"]), dim=-1)
            pixel_resolution = 1/(2**torch.min(torch.tensor(data["bit_depth"]))-1).reshape(-1,1)
            rays_init_power = rays_init_power.reshape(data["image"].shape[:-1])
            rays_init_intensity = rays_init_power * 0.444444 / self.metadata["pixel_size"]**2

            # check that theoretical initial power computed and image power measured are compatible
            assert torch.all(torch.nan_to_num(rays_init_intensity, nan=1) >= image - pixel_resolution/2), str(self.image_filenames[data["image_idx"]])+" has at least a pixel incompatible with source theroretical power."
            object_mask = rays_init_intensity > image + pixel_resolution/2
            object_mask = cv2.dilate(np.array(object_mask*255, np.uint8),np.ones((3,3), np.uint8),iterations=self.metadata["boundary_thickness"])
            object_mask = torch.tensor(object_mask/255, dtype=torch.bool).unsqueeze(-1).expand(-1,-1, 3)


            source_mask = source_shape_mask & source_position_mask

            final_mask = bbox_mask & source_mask & object_mask

            # check that at least one ray of the camera is useful
            assert torch.any(final_mask), str(self.image_filenames[data["image_idx"]])+" has no useful pixels."

            if "mask" in data:
                data["mask"] &= final_mask
            else:
                data["mask"] = final_mask

        return metadata

    def get_depths_and_normals(self, depth_filepath: Path, normal_filename: Path, camtoworld: Tensor):
        """function to process additional depths and normal information
        Args:
            depth_filepath: path to depth file
            normal_filename: path to normal file
            camtoworld: camera to world transformation matrix
        """

        # load mono depth
        depth = np.load(depth_filepath)
        depth = torch.from_numpy(depth).float()

        # load mono normal
        normal = np.load(normal_filename)

        # transform normal to world coordinate system
        normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here
        normal = torch.from_numpy(normal).float()

        rot = camtoworld[:3, :3]

        normal_map = normal.reshape(3, -1)
        normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

        normal_map = rot @ normal_map
        normal = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)

        if self.transform is not None:
            h, w, _ = normal.shape
            normal = self.transform[:3, :3] @ normal.reshape(-1, 3).permute(1, 0)
            normal = normal.permute(1, 0).reshape(h, w, 3)

        return depth, normal
