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
Source Colliders
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from nerfstudio.cameras.rays import RayBundle


class SourceCollider(nn.Module):
    """Module for colliding rays with a planar source.

    Args:
        X_size: size along X
        Y_size: size along Y
        transformations: transformations of source
    """

    def __init__(self, X_size: float, Y_size: float, transformations: torch.Tensor, **kwargs) -> None:
        self.X_size = X_size
        self.Y_size = Y_size
        self.normal = torch.Tensor([0, 0, 1])
        self.transformations = transformations
        self.inv_transformations = torch.linalg.inv(transformations)
        super().__init__(**kwargs)

    def _transform_rays(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        self.transformations = self.transformations.to(device=rays_o.device)
        self.inv_transformations = self.inv_transformations.to(device=rays_o.device)
        new_rays_o = torch.matmul(self.inv_transformations, torch.cat((rays_o.unsqueeze(-1), torch.ones((rays_o.shape[0],1,1), device=rays_o.device)),1))
        new_rays_d = torch.matmul(self.inv_transformations[:-1,:-1], rays_d.unsqueeze(-1))
        return new_rays_o, new_rays_d
    
    def _intersect_with_plane(self, rays_o: torch.Tensor, rays_d: torch.Tensor):
        x_intersect = rays_o[:,0] - rays_o[:,2]*rays_d[:,0]/rays_d[:,2]
        y_intersect = rays_o[:,1] - rays_o[:,2]*rays_d[:,1]/rays_d[:,2]
        return x_intersect, y_intersect

    def _is_touching_source(self, plane_intersection: torch.Tensor, rays_d: torch.Tensor):
        """To be implemented."""
        raise NotImplementedError
    
    def set_source_intersection(self, ray_bundle: RayBundle) -> RayBundle:
        new_rays_o, new_rays_d = self._transform_rays(rays_o=ray_bundle.origins, rays_d=ray_bundle.directions)
        plane_x_intersection, plane_y_intersection = self._intersect_with_plane(rays_o=new_rays_o, rays_d=new_rays_d)
        plane_intersection = torch.cat((plane_x_intersection, plane_y_intersection, torch.zeros_like(plane_x_intersection), torch.ones_like(plane_x_intersection)), 1)
        source_intersection = self._is_touching_source(plane_intersection, new_rays_d.squeeze(-1))
        ray_bundle.source_intersection = torch.matmul(self.transformations, source_intersection.unsqueeze(-1))[:,:-1].squeeze(-1)
        return ray_bundle

    def forward(self, ray_bundle: RayBundle) -> RayBundle:
        """Set the source intersection coordinates in the source coordinate system (XoY plane)"""
        if ray_bundle.source_intersection is not None:
            return ray_bundle
        return self.set_source_intersection(ray_bundle)

class RectangleSourceCollider(SourceCollider):
    """Module for colliding rays with a rectangle planar source.
    """

    def _is_touching_source(self, plane_intersection: torch.Tensor, rays_d: torch.Tensor):
         plane_intersection[(torch.matmul(rays_d, self.normal) > 0) | (torch.abs(plane_intersection[:,0]) > self.X_size/2) | (torch.abs(plane_intersection[:,1]) > self.Y_size/2)] = torch.nan
         return plane_intersection

class EllipseSourceCollider(SourceCollider):
    """Module for colliding rays with an ellipse planar source.
    """

    def _is_touching_source(self, plane_intersection: torch.Tensor, rays_d: torch.Tensor):
         self.normal = self.normal.to(device=rays_d.device)
         plane_intersection[(torch.matmul(rays_d, self.normal) > 0) | (plane_intersection[:,0]**2 / (self.X_size/2)**2 + plane_intersection[:,1]**2 / (self.Y_size/2)**2 > 1)] = torch.nan
         return plane_intersection





    
