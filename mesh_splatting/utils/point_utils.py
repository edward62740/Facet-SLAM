import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

import collections
_PIX_GRID = {}                                   # (W, H) -> (ndc2pix, homogeneous pixel grid)
_RAYS_LRU = collections.OrderedDict()            # id(world_view_transform) -> (that tensor, rays_d, rays_o); bounded
_RAYS_LRU_MAX = 16


def _rays(view):
    """World-frame ray directions (H*W, 3) and origin for `view` (same expressions as before). The pixel grid is built
    once per resolution; rays are kept for the 16 most recent poses in a module-level cache keyed on the identity of
    the view's cached world_view_transform tensor, which is replaced whenever the pose changes (the entry holds a
    reference, so the id cannot be recycled while cached). Nothing is attached to the Camera objects."""
    wvt = view.world_view_transform
    W, H = view.image_width, view.image_height
    hit = _RAYS_LRU.get(id(wvt))
    if hit is not None and hit[0] is wvt and hit[1].shape[0] == W * H:
        _RAYS_LRU.move_to_end(id(wvt))
        return hit[1], hit[2]
    c2w = torch.linalg.inv_ex(wvt.T)[0]                  # no host-side singularity check (sync)
    g = _PIX_GRID.get((W, H))
    if g is None:
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W) / 2],
            [0, H / 2, 0, (H) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        g = (ndc2pix, points)
        _PIX_GRID[(W, H)] = g
    ndc2pix, points = g
    projection_matrix = c2w.T @ view.full_proj_transform
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    rays_d = points @ torch.linalg.inv_ex(intrins)[0].T @ c2w[:3,:3].T
    rays_o = c2w[:3,3]
    _RAYS_LRU[id(wvt)] = (wvt, rays_d, rays_o)
    while len(_RAYS_LRU) > _RAYS_LRU_MAX:
        _RAYS_LRU.popitem(last=False)
    return rays_d, rays_o

def depths_to_points(view, depthmap):
    rays_d, rays_o = _rays(view)
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    output = torch.zeros_like(points)
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    output[1:-1, 1:-1, :] = normal_map
    return output