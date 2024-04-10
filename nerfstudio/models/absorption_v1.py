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
Implementation of Absorption model v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Type

import torch

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.ray_samplers import AbsorptionSampler
from nerfstudio.models.base_volume_model import VolumeModel, VolumeModelConfig


@dataclass
class AbsorptionModelConfig(VolumeModelConfig):
    """Absorption Model Config"""

    _target: Type = field(default_factory=lambda: AbsorptionModel)
    num_samples: int = 32
    """Number of uniform samples"""
    num_samples_importance_per_step: int = 8
    """Number of importance samples per step"""
    num_up_sample_steps: int = 8
    """Number of up sample step, 1 for simple coarse-to-fine sampling"""
    base_variance: float = 256
    """Fixed base variance in NeuS sampler, the inv_s will be base * 2 ** iter during upsample"""
    perturb: bool = True
    """Use to use perturb for the sampled points"""
    init_mat_absorption: float = 1
    """Absorption constant of the object's material"""
    def_absorption: float = 0
    """Absorption constant outside of the object"""
    source_power: float = 1
    """Total power of the ray source"""
    source_diameter: float = 1
    """Diameter of the ray source"""
    source_position: tuple = (0,0,0)
    """3D position of the ray source"""
    pixel_size: float = 1
    """Pixel size of the sensor"""


class AbsorptionModel(VolumeModel):
    """Absorption model

    Args:
        config: Absorption model configuration to instantiate model
    """

    config: AbsorptionModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        self.sampler = AbsorptionSampler(
            num_samples=self.config.num_samples,
            num_samples_importance_per_step=self.config.num_samples_importance_per_step,
            num_upsample_steps=self.config.num_up_sample_steps,
            base_variance=self.config.base_variance,
        )

        self.def_absorption = self.config.def_absorption
        self.source_power = self.config.source_power
        self.source_diameter = self.config.source_diameter
        self.source_position = self.config.source_position
        self.pixel_size = self.config.pixel_size
        self.anneal_end = 50000
        self.field_scaling = self.kwargs["metadata"]["sdf_field_scaling"]

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        return callbacks

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        ray_samples = self.sampler(ray_bundle, sdf_fn=self.field.get_sdf)
        field_outputs = self.field(
            ray_samples,
            mid_points=True,
            return_absorption=True,
            def_absorption=self.def_absorption,
            return_initial_power=True,
            source_power=self.source_power,
            source_diameter=self.source_diameter,
            source_position=self.source_position,
            pixel_size=self.pixel_size,
            field_scaling=self.field_scaling
        )

        dummy_weights = torch.ones_like(field_outputs[FieldHeadNames.ABSORPTION])

        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": dummy_weights,
            "pixel_size": self.pixel_size,
        }
        return samples_and_field_outputs

    def get_metrics_dict(self, outputs, batch) -> Dict:
        metrics_dict = super().get_metrics_dict(outputs, batch)
        if self.training:
            # training statics
            metrics_dict["s_val"] = self.field.deviation_network.get_variance().item()
            metrics_dict["inv_s"] = 1.0 / self.field.deviation_network.get_variance().item()

        return metrics_dict
