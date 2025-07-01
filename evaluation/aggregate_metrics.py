from pathlib import Path
import json
import sys
from itertools import chain
import json
import os
from collections import defaultdict

import numpy as np

from compute_metrics import analyze_instance, Instance

def calc_avg_error(model, align_method, full_scene, scale_only, path, erosion=2):
    """Calculates the average error for a scene and its variations"""
    root = str(path.parent.parent.parent.resolve())
    all_errors = defaultdict(list)
    for instance_folder in Path(path).iterdir():
        instance = Instance.from_folder(instance_folder)
        if instance.is_base:
            print("skipping base instance", instance.identifiers)
            continue
        if (res := analyze_instance(model, root, instance.identifiers, align_method, full_scene=full_scene, scale_only=scale_only, compare_with_base=False, erosion=erosion)) is not None:
            for metric in ['absrel', 'rmse', 'delta_0125']:
                all_errors[metric].append(res[metric])

    base_instance = instance.get_base(root)
    res = analyze_instance(model, root, base_instance.identifiers, align_method, full_scene=full_scene, scale_only=scale_only, compare_with_base=False, erosion=erosion)
    for metric in ['absrel', 'rmse', 'delta_0125']:
        all_errors[metric].append(res[metric])

    agg = lambda x: np.mean(x)
    return {err: agg(np.array(vals, dtype=np.float32)) for err, vals in all_errors.items()}

def calc_accuracy_stability(model, align_method, full_scene, scale_only, path, erosion=2):
    """Calculates the accuracy stability for a scene and its variations"""
    root = str(path.parent.parent.parent.resolve())
    all_errors = defaultdict(list)
    for instance_folder in Path(path).iterdir():
        instance = Instance.from_folder(instance_folder)
        if instance.is_base:
            continue
        res = analyze_instance(model, root, instance.identifiers, align_method, full_scene=full_scene, scale_only=scale_only, compare_with_base=False, erosion=erosion)
        for metric in ['absrel', 'rmse']:
            all_errors[metric].append(res[metric])
        all_errors['delta_0125_inv'].append(1 - res['delta_0125'])

    base_instance = instance.get_base(root)
    res = analyze_instance(model, root, base_instance.identifiers, align_method, full_scene=full_scene, scale_only=scale_only, compare_with_base=False, erosion=erosion)
    for metric in ['absrel', 'rmse']:
        all_errors[metric].append(res[metric])
    all_errors['delta_0125_inv'].append(1 - res['delta_0125'])

    agg = lambda x: ((x - x.mean())**2).sum() / (len(x) - 1)
    return {err: agg(np.array(vals, dtype=np.float32)) for err, vals in all_errors.items()}

def calc_self_inconsistency(model, align_method, full_scene, scale_only, path, erosion=2):
    """Calculates the self-inconsistency for a scene and its variations"""
    root = str(path.parent.parent.parent.resolve())
    all_errors = defaultdict(list)
    for instance_folder in Path(path).iterdir():
        instance = Instance.from_folder(instance_folder)
        if instance.is_base:
            continue

        res = analyze_instance(model, root, instance.identifiers, align_method, full_scene=full_scene, scale_only=scale_only, compare_with_base=True, erosion=erosion)
        for metric in ['absrel', 'rmse']:
            all_errors[metric].append(res[metric])
        all_errors['delta_0125_inv'].append(1 - res['delta_0125'])

    agg = lambda x: (x**2).sum() / len(x)
    return {err: agg(np.array(vals, dtype=np.float32)) for err, vals in all_errors.items()}

def agg_list_dict(dicts, agg):
    all_metrics = defaultdict(list)
    for d in dicts:
        for metric, val in d.items():
            all_metrics[metric].append(val)
    return {err: agg(np.array(vals, dtype=np.float32)) for err, vals in all_metrics.items()}


def compute_object_variation(dataset_root, obj, variation, model, align_method, full_scene, scale_only, erosion, calculate_self_consistency):
    """Aggregates the average error, accuracy stability, and self-inconsistency for a single object and variation type"""
    scenes = list((Path(dataset_root) / obj / variation).iterdir())

    errors = [calc_avg_error(model, align_method, full_scene, scale_only, scene, erosion) for scene in scenes]
    stabilities = [calc_accuracy_stability(model, align_method, full_scene, scale_only, scene, erosion) for scene in scenes]
    if calculate_self_consistency:
        self_inconsistencies = [calc_self_inconsistency(model, align_method, full_scene, scale_only, scene, erosion) for scene in scenes]
    
    err = agg_list_dict(errors, lambda x: x.mean())
    stability = agg_list_dict(stabilities, lambda x: np.sqrt(x).mean())
    if calculate_self_consistency:
        consistency = agg_list_dict(self_inconsistencies, lambda x: np.sqrt(x).mean())
        return err, stability, consistency
    return err, stability

def compute_variation(dataset_root, variation, model, align_method, full_scene, scale_only, erosion, calculate_self_consistency):
    """Aggregates the average error, accuracy stability, and self-inconsistency for all objects and a single variation type"""
    results = [compute_object_variation(dataset_root, obj, variation, model, align_method, full_scene, scale_only, erosion, calculate_self_consistency) for obj in ['chairs', 'cactus', 'cabinets', 'desks', 'fishes']]
    err = np.mean([r[0]['absrel'] for r in results])
    stability = np.mean([r[1]['absrel'] for r in results])
    if calculate_self_consistency:
        consistency = np.mean([r[2]['absrel'] for r in results])
        return err, stability, consistency
    return err, stability


class FileLoaderModel:
    def __init__(self, predictions_root):
        """This model model loads precomputed depth predictions"""
        self.predictions_root = predictions_root
    
    def predict(self, instance: Instance):
        obj, var, scene, id = instance.identifiers
        pred_depth_file = Path(self.predictions_root) / obj / var / scene / f"{id}.npy"
        return np.load(pred_depth_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument('--full_scene', action='store_true')
    parser.add_argument('--scale_only', action='store_true')
    parser.add_argument('--erosion_kernel_size', type=int, default=2)

    args = parser.parse_args()

    model = FileLoaderModel("<predictions folder>")
    variations_list = [p.name for p in (Path(args.dataset_root) / "chairs").iterdir()]
    variations_list.remove('ood_background_swap')
    variations_list.remove('base')

    for var in variations_list:
        self_con = var in ['camera_roll', 'lighting', 'object_material_swap', 'scene_material_swap', 'object_resizing']
        res = compute_variation(args.dataset_root, var, model, 'depth', args.full_scene, args.scale_only, args.erosion_kernel_size, calculate_self_consistency=self_con)
        print("Results:", var, res)
    



