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
Implementation of Base volume model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type, cast

import torch
from torch.nn import Parameter

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.field_components.field_heads import FieldHeadNames

from nerfstudio.model_components.renderers import AbsorptionRenderer
from nerfstudio.model_components.scene_colliders import NearFarCollider, AABBBoxCollider, SphereCollider
from nerfstudio.models.base_surface_model import SurfaceModel, SurfaceModelConfig



@dataclass
class VolumeModelConfig(SurfaceModelConfig):
    """Volume Model Config"""

    _target: Type = field(default_factory=lambda: VolumeModel)


class VolumeModel(SurfaceModel):
    """Base volume model

    Args:
        config: Base volume model configuration to instantiate model
    """

    config: VolumeModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()
        metadata = self.kwargs["metadata"]
        collider_type = metadata["collider_type"]
        if collider_type == 'near_far':
            self.collider = NearFarCollider(near_plane=metadata["near"], far_plane=metadata["far"])
        elif collider_type == 'box':
            self.collider = AABBBoxCollider(self.scene_box)
        elif collider_type == 'sphere':
            self.collider = SphereCollider(center=metadata["center"], radius=metadata["radius"])
        else:
            raise NotImplementedError("collider type not implemented")
        self.renderer_absorption = AbsorptionRenderer()

    def get_outputs(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        assert (
            ray_bundle.metadata is not None and "directions_norm" in ray_bundle.metadata
        ), "directions_norm is required in ray_bundle.metadata"

        samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)

        # shortcuts
        field_outputs: Dict[FieldHeadNames, torch.Tensor] = cast(
            Dict[FieldHeadNames, torch.Tensor], samples_and_field_outputs["field_outputs"]
        )
        ray_samples = samples_and_field_outputs["ray_samples"]
        weights = samples_and_field_outputs["weights"]

        power = self.renderer_absorption(
            initial_power=field_outputs[FieldHeadNames.POWER],
            absorption=field_outputs[FieldHeadNames.ABSORPTION],
            samples_width=ray_samples.deltas
        )

        # print(field_outputs[FieldHeadNames.SDF][0])
        # print(field_outputs[FieldHeadNames.ABSORPTION][0])
        # print(ray_samples.deltas[0])

        # convert power to intensity
        pixel_size = samples_and_field_outputs["pixel_size"]
        intensity = power/pixel_size**2

        # convert intensity to Blender pixel value
        value = torch.clip(intensity * 0.25, min=0, max=1).expand(-1, 3)
        # print("min value", torch.min(value), "max value", torch.max(value))

        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        # the rendered depth is point-to-point distance and we should convert to depth
        depth = depth / ray_bundle.metadata["directions_norm"]

        normal = self.renderer_normal(semantics=field_outputs[FieldHeadNames.NORMALS], weights=weights)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": value,
            "accumulation": accumulation,
            "depth": depth,
            "normal": normal,
            "weights": weights,
            # used to scale z_vals for free space and sdf loss
            "directions_norm": ray_bundle.metadata["directions_norm"],
        }

        if self.training:
            grad_points = field_outputs[FieldHeadNames.GRADIENT]
            outputs.update({"eik_grad": grad_points})
            outputs.update(samples_and_field_outputs)

        if "weights_list" in samples_and_field_outputs:
            weights_list = cast(List[torch.Tensor], samples_and_field_outputs["weights_list"])
            ray_samples_list = cast(List[torch.Tensor], samples_and_field_outputs["ray_samples_list"])

            for i in range(len(weights_list) - 1):
                outputs[f"prop_depth_{i}"] = self.renderer_depth(
                    weights=weights_list[i], ray_samples=ray_samples_list[i]
                )
        # this is used only in viewer
        outputs["normal_vis"] = (outputs["normal"] + 1.0) / 2.0
        return outputs
