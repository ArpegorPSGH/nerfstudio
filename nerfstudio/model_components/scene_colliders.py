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
Scene Colliders
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox


class SceneCollider(nn.Module):
    """Module for setting near and far values for rays."""

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        super().__init__()

    def intersect_with_source(self, ray_bundle: RayBundle) -> RayBundle:
        if ray_bundle.source_intersections is not None:
            origin_to_source = ray_bundle.source_intersections - ray_bundle.origins
            source_distances = torch.nan_to_num(origin_to_source / ray_bundle.directions, nan=torch.inf, neginf=torch.inf)
            source_distances = source_distances.min(dim=-1, keepdim=True).values
            fars = torch.nan_to_num(ray_bundle.fars)
            ray_bundle.fars = ray_bundle.fars.where(source_distances > fars, source_distances)
            ray_bundle.source_distances = source_distances
        return ray_bundle

    def set_nears_and_fars(self, ray_bundle: RayBundle, filter_out_nan: bool) -> RayBundle:
        """To be implemented."""
        raise NotImplementedError

    def forward(self, ray_bundle: RayBundle, filter_out_nan: bool = True) -> RayBundle:
        """Sets the nears and fars if they are not set already."""
        if ray_bundle.nears is not None and ray_bundle.fars is not None:
            return ray_bundle
        return self.intersect_with_source(self.set_nears_and_fars(ray_bundle, filter_out_nan=filter_out_nan))


class AABBBoxCollider(SceneCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        scene_box: scene box to apply to dataset
    """

    def __init__(self, scene_box: SceneBox, near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scene_box = scene_box
        self.near_plane = near_plane

    def _intersect_with_aabb(
        self, rays_o: Float[Tensor, "num_rays 3"], rays_d: Float[Tensor, "num_rays 3"], aabb: Float[Tensor, "2 3"]
    ):
        """Returns collection of valid rays within a specified near/far bounding box along with a mask
        specifying which rays are valid

        Args:
            rays_o: (num_rays, 3) ray origins
            rays_d: (num_rays, 3) ray directions
            aabb: (2, 3) This is [min point (x,y,z), max point (x,y,z)]
        """
       
        aabb = aabb.to(device=rays_o.device)
        
        # x
        t1 = (aabb[0, 0] - rays_o[:, 0:1]) / rays_d[:, 0:1]
        t2 = (aabb[1, 0] - rays_o[:, 0:1]) / rays_d[:, 0:1]
        # y
        t3 = (aabb[0, 1] - rays_o[:, 1:2]) / rays_d[:, 1:2]
        t4 = (aabb[1, 1] - rays_o[:, 1:2]) / rays_d[:, 1:2]
        # z
        t5 = (aabb[0, 2] - rays_o[:, 2:3]) / rays_d[:, 2:3]
        t6 = (aabb[1, 2] - rays_o[:, 2:3]) / rays_d[:, 2:3]

        nears = torch.max(
            torch.nan_to_num(torch.cat([torch.minimum(t1, t2), torch.minimum(t3, t4), torch.minimum(t5, t6)], dim=1), posinf=0), dim=1, keepdim=True
        ).values
        fars = torch.min(
            torch.nan_to_num(torch.cat([torch.maximum(t1, t2), torch.maximum(t3, t4), torch.maximum(t5, t6)], dim=1), neginf=torch.inf), dim=1, keepdim=True
        ).values

        intersections_nears = rays_o + nears*rays_d
        intersections_fars = rays_o + fars*rays_d

        nears[torch.any((intersections_nears < (aabb[0] - 1e-6)) | (intersections_nears > (aabb[1] + 1e-6)), dim=1)] = torch.nan
        fars[torch.any((intersections_fars < (aabb[0] - 1e-6)) | (intersections_fars > (aabb[1] + 1e-6)), dim=1)] = torch.nan

        # clamp to near plane
        near_plane = self.near_plane if self.training else 0
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)

        return nears, fars

    def set_nears_and_fars(self, ray_bundle: RayBundle, filter_out_nan: bool) -> RayBundle:
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        aabb = self.scene_box.aabb
        nears, fars = self._intersect_with_aabb(ray_bundle.origins, ray_bundle.directions, aabb)
        ray_bundle.nears = nears
        ray_bundle.fars = fars
        return ray_bundle


class SphereCollider(SceneCollider):
    """Module for colliding rays with the scene box to compute near and far values.

    Args:
        center: center of sphere to intersect [3]
        radius: radius of sphere to intersect
        near_plane: near plane to clamp to
    """

    def __init__(self, center: torch.Tensor, radius: float, near_plane: float = 0.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.center = center
        self.radius = radius
        self.near_plane = near_plane

    def _intersect_with_sphere(self,
        rays_o: torch.Tensor, rays_d: torch.Tensor, center: torch.Tensor, filter_out_nan: bool, radius: float = 1.0, near_plane: float = 0.0
    ):
        a = (rays_d * rays_d).sum(dim=-1, keepdim=True)
        b = 2 * (rays_o - center) * rays_d
        b = b.sum(dim=-1, keepdim=True)
        c = (rays_o - center) * (rays_o - center)
        c = c.sum(dim=-1, keepdim=True) - radius**2

        delta = torch.square(b) - 4 * a * c
        if filter_out_nan:
            # replace negative values with 0s
            delta = torch.maximum(torch.zeros_like(delta), delta)

        nears = (-b - torch.sqrt(delta)) / (2 * a)
        fars = (-b + torch.sqrt(delta)) / (2 * a)
        # clamp to near plane
        nears = torch.clamp(nears, min=near_plane)
        fars = torch.maximum(fars, nears + 1e-6)  

        return nears, fars

    def set_nears_and_fars(self, ray_bundle: RayBundle, filter_out_nan: bool) -> RayBundle:
        """Intersects the rays with the scene box and updates the near and far values.
        Populates nears and fars fields and returns the ray_bundle.

        Args:
            ray_bundle: specified ray bundle to operate on
        """
        self.center = self.center.to(ray_bundle.origins.device)
        near_plane = self.near_plane if self.training else 0
        nears, fars = self._intersect_with_sphere(
            rays_o=ray_bundle.origins,
            rays_d=ray_bundle.directions,
            center=self.center,
            radius=self.radius,
            near_plane=near_plane,
            filter_out_nan=filter_out_nan
        )
        ray_bundle.nears = nears
        ray_bundle.fars = fars
        return ray_bundle


class NearFarCollider(SceneCollider):
    """Sets the nears and fars with fixed values.

    Args:
        near_plane: distance to near plane
        far_plane: distance to far plane
        reset_near_plane: whether to reset the near plane to 0.0 during inference. The near plane can be
            helpful for reducing floaters during training, but it can cause clipping artifacts during
            inference when an evaluation or viewer camera moves closer to the object.
    """

    def __init__(self, near_plane: float, far_plane: float, reset_near_plane: bool = True, **kwargs) -> None:
        self.near_plane = near_plane
        self.far_plane = far_plane
        self.reset_near_plane = reset_near_plane
        super().__init__(**kwargs)

    def set_nears_and_fars(self, ray_bundle: RayBundle, filter_out_nan: bool) -> RayBundle:
        ones = torch.ones_like(ray_bundle.origins[..., 0:1])
        near_plane = self.near_plane if (self.training or not self.reset_near_plane) else 0
        ray_bundle.nears = ones * near_plane
        ray_bundle.fars = ones * self.far_plane
        return ray_bundle
        