from pathlib import Path
import json
from functools import reduce

import torch
import numpy as np
import cv2 as cv
from typing import NamedTuple

MIN_DEPTH=1e-1
MAX_DEPTH=1e3

def reformat_input(x):
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x)
    # convert to float unless it is a bool
    if not x.dtype == torch.bool:
        x = x.to(torch.float32)
    return x

def absrel(pred, target, mask):
    def _absrel(t_m, p_m):
        err_heatmap = torch.abs(t_m - p_m) / (t_m + 1e-10)  # (H, W)
        err = err_heatmap.sum() / t_m.shape[0]
        assert not (torch.isnan(err) | torch.isinf(err))
        return err.item()

    if mask.sum() == 0:
        return 0.
    return _absrel(target[mask], pred[mask])


def rmse(pred, target, mask):
    def _rmse(t_m, p_m):
        err_heatmap = (t_m - p_m) ** 2  # (H, W)
        err = torch.sqrt(err_heatmap.sum() / t_m.shape[0])
        assert not (torch.isnan(err) | torch.isinf(err))
        return err.item()

    if mask.sum() == 0:
        return 0.
    return _rmse(target[mask], pred[mask])



def delta(pred, target, mask):
    def _delta(t_m, p_m):
        gt_pred = t_m / (p_m + 1e-10)  # (H, W)
        pred_gt = p_m / (t_m + 1e-10)  # (H, W)
        gt_pred_gt = torch.stack([gt_pred, pred_gt], dim=-1)  # (H, W, 2)
        ratio_max = torch.amax(gt_pred_gt, dim=-1)  # (H, W)

        delta_0125_sum = torch.sum(ratio_max < 1.25 ** 0.125)
        delta_025_sum = torch.sum(ratio_max < 1.25 ** 0.25)
        delta_05_sum = torch.sum(ratio_max < 1.25 ** 0.5)
        delta_1_sum = torch.sum(ratio_max < 1.25)
        delta_2_sum = torch.sum(ratio_max < 1.25 ** 2)
        delta_3_sum = torch.sum(ratio_max < 1.25 ** 3)
        delta_0125, delta_025, delta_05 = (delta_0125_sum / t_m.shape[0]), (delta_025_sum / t_m.shape[0]), (delta_05_sum / t_m.shape[0])
        delta_1, delta_2, delta_3 = (delta_1_sum / t_m.shape[0]), (delta_2_sum / t_m.shape[0]), (delta_3_sum / t_m.shape[0])
        assert not (torch.isnan(delta_0125) | torch.isinf(delta_0125))
        assert not (torch.isnan(delta_025) | torch.isinf(delta_025))
        assert not (torch.isnan(delta_05) | torch.isinf(delta_05))
        assert not (torch.isnan(delta_1) | torch.isinf(delta_1))
        assert not (torch.isnan(delta_2) | torch.isinf(delta_2))
        assert not (torch.isnan(delta_3) | torch.isinf(delta_3))
        return delta_0125.item(), delta_025.item(), delta_05.item(), delta_1.item(), delta_2.item(), delta_3.item()

    if mask.sum() == 0:
        return 0., 0., 0., 0., 0., 0.
    return _delta(target[mask], pred[mask])


def align_disparity(true, pred, mask, scale_only=False):
    assert not scale_only # not needed for models selected
    true = 1 / true
    if mask is None:
        b = true.reshape(-1)
        A = np.stack([pred.reshape(-1), np.ones_like(b)], axis=1)
    else:
        b = true[mask]
        A = np.stack([pred[mask], np.ones_like(b)], axis=1)

    assert np.isfinite(A).all() and np.isfinite(b).all(), ((~np.isfinite(A)).sum(), (~np.isfinite(b)).sum())
    a,b = np.linalg.lstsq(A,b,rcond=None)[0]
    return 1 / (pred * a + b)

def align_depth(true, pred, mask=None, scale_only=False):
    if scale_only:
        return align_depth_scale(true, pred, mask)
    if mask is None:
        b = true.reshape(-1)
        A = np.stack([pred.reshape(-1), np.ones_like(b)], axis=1)
    else:
        b = true[mask]
        A = np.stack([pred[mask], np.ones_like(b)], axis=1)

    assert np.isfinite(A).all() and np.isfinite(b).all(), ((~np.isfinite(A)).sum(), (~np.isfinite(b)).sum())
    a,b = np.linalg.lstsq(A,b,rcond=None)[0]
    return pred * a + b

def align_depth_scale(true, pred, mask=None):
    if mask is None:
        b = true.reshape(-1)
        A = pred.reshape(-1,1)
    else:
        b = true[mask]
        A = pred[mask].reshape(-1,1)

    assert np.isfinite(A).all() and np.isfinite(b).all(), ((~np.isfinite(A)).sum(), (~np.isfinite(b)).sum())
    a = np.linalg.lstsq(A,b,rcond=None)[0]
    return pred * a

def rotate_image_f(img, theta):
    H, W = img.shape[:2]
    center = (W / 2, H / 2)
    rot_mat = cv.getRotationMatrix2D(center, theta * 180 / np.pi, 1.0)
    rot_img = cv.warpAffine(img, rot_mat, (W, H), flags=cv.INTER_NEAREST)
    return rot_img

class Instance(NamedTuple):
    identifiers: list[str] # object, variation, scene, id
    depth: np.array
    obj_mask: np.array
    background_mask: np.array
    theta: float | None
    base_identifiers: list[str] # object, variation, scene, id
    img_path: str = None

    def from_folder(folder: str):
        folder = Path(folder)

        with open(folder / "variation.json") as f:
            variation_data = json.load(f)
        
        maps = np.load(folder / "maps.npz")

        obj_mask = maps['object_segmentation'] == variation_data['obj_index']
        img_path = folder / "image.png"
        base_identifiers = list(variation_data['base'].split('/'))
        return Instance(
            depth=maps['depth'], obj_mask=obj_mask, background_mask=maps['background'], theta=variation_data.get('theta'), 
            img_path=img_path, identifiers=folder.parts[-4:], base_identifiers=base_identifiers
        )
    
    def from_identifiers(root: str, identifiers: list[str]):
        obj, variation, scene, id = identifiers
        return Instance.from_folder(Path(root) / obj / variation / scene / id)
    
    def get_base(self, root: str):
        return Instance.from_identifiers(root, self.base_identifiers)
    
    @property
    def is_base(self):
        res = tuple(self.base_identifiers) == tuple(self.identifiers)
        return res


def analyze_instance(model, root, instance_identifiers, align_method_str, full_scene=False, scale_only=False, compare_with_base=False, erosion=2):
    obj, variation, scene, id = instance_identifiers
    match align_method_str:
        case "depth":
            align_method = align_depth
        case "disparity":
            align_method = align_disparity
        case _:
            raise ValueError()

    instance = Instance.from_identifiers(root, instance_identifiers)

    rotate = compare_with_base and variation == 'rotate_camera'
    # get object masks
    if compare_with_base:
        assert not instance.is_base, "Comparing identical items"
        base_instance = instance.get_base(root)
        obj_mask = instance.obj_mask
        base_obj_mask = base_instance.obj_mask
        if rotate:
            obj_mask = rotate_image_f(obj_mask.astype(np.uint8), instance.theta).astype(bool)
            base_obj_mask = rotate_image_f(base_obj_mask.astype(np.uint8), base_instance.theta).astype(bool)

        obj_mask = obj_mask & base_obj_mask
    else:
        obj_mask = instance.obj_mask

    # use erosion kernel size 2 to account for border ambiguity
    kernel = np.ones((erosion, erosion), np.uint8)
    obj_mask = cv.erode(obj_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    if full_scene:
        obj_mask = np.ones_like(obj_mask)

    # get background mask
    background_mask = instance.background_mask
    # dilate background mask to likewise avoid boundary ambiguity
    background_mask = cv.dilate(background_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    if rotate:
        background_mask = rotate_image_f(background_mask.astype(np.uint8), instance.theta).astype(bool)

    # get depths
    pred_depth = model.predict(instance)
    if rotate:
        pred_depth = rotate_image_f(pred_depth, instance.theta)

    if compare_with_base:
        base_pred = model.predict(base_instance)
        true_depth = base_pred / np.median(base_pred)
        if rotate:
            true_depth = rotate_image_f(true_depth, base_instance.theta)
    else:
        true_depth = np.clip(instance.depth, MIN_DEPTH, MAX_DEPTH)

    # get evaluation mask
    valid_mask = np.isfinite(true_depth) & np.isfinite(pred_depth) & (true_depth > 0)
    if compare_with_base:
        eval_mask = obj_mask & valid_mask
    else:
        eval_mask = obj_mask & ~background_mask & valid_mask

    # align depths
    pred_depth = align_method(true_depth, pred_depth, eval_mask, scale_only)
    if not compare_with_base:
        pred_depth = np.clip(pred_depth, MIN_DEPTH, MAX_DEPTH)

    deltas = dict(zip(
        ['delta_0125', 'delta_025', 'delta_05', 'delta_1', 'delta_2', 'delta_3'],
        delta(reformat_input(pred_depth), reformat_input(true_depth), reformat_input(eval_mask))
    ))

    metrics = {
        'absrel': absrel(reformat_input(pred_depth), reformat_input(true_depth), reformat_input(eval_mask)),
        'rmse': rmse(reformat_input(pred_depth), reformat_input(true_depth), reformat_input(eval_mask)),
        'delta_0125': deltas['delta_0125'],
    }

    return metrics