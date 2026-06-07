import os
import pydicom
import pandas as pd
from pathlib import Path
from django.apps import apps

import cv2
import numpy as np
import logging
logger = logging.getLogger(__name__)

def process_dicom_folder(folder_path, patient_id="0000"):
    rsna_app = apps.get_app_config('rsna')
    models = rsna_app.ml_models
    device = rsna_app.device
    temp_scales = getattr(rsna_app, 'temp_scales', {})
    nfn_offsets = getattr(rsna_app, 'nfn_offsets', {})
    ss_offsets = getattr(rsna_app, 'ss_offsets', {})
    
    folder_path = Path(folder_path)
    study_id = int(folder_path.name)
    
    import sys
    import inspect
    rsna_app_dir = os.path.dirname(os.path.abspath(__file__))
    if rsna_app_dir not in sys.path:
        sys.path.insert(0, rsna_app_dir)
        
    from config.settings import CFG
    from src.module5_inference import (
        run_study, predict_keypoints, extract_patch_from_series,
        find_series, load_sorted_series, build_instance_map,
        find_axial_slice_for_level, get_axial_lr_coords,
        scale_keypoints_to_t1,
        LEVELS, MODEL3_CONDITIONS, MODEL4_CONDITIONS,
    )
    
    # 1. Build series_desc
    data = []
    kaggle_series = pd.DataFrame()
    if CFG.SERIES_CSV.exists():
        kaggle_series = pd.read_csv(CFG.SERIES_CSV)
        
    for series_dir in [d for d in folder_path.iterdir() if d.is_dir()]:
        dcm_files = list(series_dir.glob('*.dcm'))
        if not dcm_files:
            continue
        try:
            ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True)
            desc = ds.SeriesDescription if hasattr(ds, 'SeriesDescription') else ""
            series_id = series_dir.name
            
            if not desc or desc in ["T2", "T1", "T2W_FSE"]: 
                if not kaggle_series.empty:
                    match = kaggle_series[(kaggle_series['study_id'].astype(str) == str(study_id)) & 
                                          (kaggle_series['series_id'].astype(str) == str(series_id))]
                    if not match.empty:
                        desc = match.iloc[0]['series_description']
                        
            data.append({"study_id": study_id, "series_id": int(series_id), "series_description": desc})
        except Exception as e:
            logger.error(f"Failed to read DICOM in {series_dir}: {e}")
            
    if not data:
        raise ValueError("No DICOM series found.")
    series_desc = pd.DataFrame(data)
    
    original_dicom_root = CFG.DICOM_ROOT
    CFG.DICOM_ROOT = folder_path.parent
    
    try:
        sig = inspect.signature(run_study)
        kwargs = {
            'study_id': study_id,
            'series_desc': series_desc,
            'model1': models[1],
            'model2': models[2],
            'model4': models[4],
            'temp_scales': temp_scales,
            'nfn_offsets': nfn_offsets,
            'device': device,
            'logger': logger,
            'run_id': "django_app",
            'ss_offsets': ss_offsets
        }
        if 'model3' in sig.parameters:
            kwargs['model3'] = models.get(3)
        else:
            kwargs['model3l'] = models.get("3L")
            kwargs['model3r'] = models.get("3R")
            
        preds = run_study(**kwargs)
        
        # ── Post-processing: generate visualisation images ──────────────────
        from django.conf import settings
        VIS_DIR = Path(settings.MEDIA_ROOT) / 'rsna_uploads' / 'traces' / str(study_id)
        VIS_DIR.mkdir(parents=True, exist_ok=True)
        
        full_scan_url = None
        patch_urls = {}
        
        # Re-run keypoint prediction to capture the full scan image
        sag_t2_id = find_series(series_desc, study_id, "Sagittal T2", logger, "django_vis")
        sag_t1_id = find_series(series_desc, study_id, "Sagittal T1", logger, "django_vis")
        axl_t2_id = find_series(series_desc, study_id, "Axial T2",    logger, "django_vis")
        
        if sag_t2_id is not None:
            kps, sag_t2_windowed, sag_t2_H, sag_t2_W = predict_keypoints(
                models[1], study_id, sag_t2_id, device
            )
            
            # Save full scan with keypoint overlay
            if sag_t2_windowed is not None and kps:
                full_scan_url = _save_full_scan(
                    VIS_DIR, study_id, sag_t2_windowed, kps
                )
                logger.info(f"  [vis] Full scan saved: {full_scan_url}")
            else:
                logger.warning(f"  [vis] No scan image or keypoints from predict_keypoints")
            
            sag_t2_dir = CFG.DICOM_ROOT / str(study_id) / str(sag_t2_id)
            sag_t2_datasets = load_sorted_series(sag_t2_dir)
            sag_t2_mid_ds = sag_t2_datasets[len(sag_t2_datasets) // 2] if sag_t2_datasets else None
            sag_t2_mid_inst = int(getattr(sag_t2_mid_ds, "InstanceNumber",
                                          len(sag_t2_datasets) // 2)) if sag_t2_mid_ds else 0
            
            # SCS patches (Sagittal T2)
            if kps:
                for lvl in LEVELS:
                    cx, cy = kps[lvl]
                    patch = extract_patch_from_series(sag_t2_dir, sag_t2_mid_inst, cx, cy)
                    url = _save_patch(VIS_DIR, study_id, f"spinal_canal_stenosis_{lvl}", patch, CFG)
                    if url:
                        patch_urls[f"spinal_canal_stenosis_{lvl}"] = url
            
            # NFN patches (Sagittal T1)
            if sag_t1_id is not None and kps:
                sag_t1_dir = CFG.DICOM_ROOT / str(study_id) / str(sag_t1_id)
                t1_kps, t1_H, t1_W = scale_keypoints_to_t1(
                    kps, sag_t2_H, sag_t2_W, sag_t1_dir, logger
                )
                t1_datasets = load_sorted_series(sag_t1_dir)
                t1_mid_inst = int(getattr(
                    t1_datasets[len(t1_datasets) // 2], "InstanceNumber",
                    len(t1_datasets) // 2
                )) if t1_datasets else 0
                
                for cond in MODEL3_CONDITIONS:
                    for lvl in LEVELS:
                        cx_base, cy_base = t1_kps[lvl]
                        off = nfn_offsets.get(cond, {}).get(lvl, {})
                        if "dx_frac" in off:
                            dx_abs = float(off["dx_frac"]) * t1_W
                            dy_abs = float(off["dy_frac"]) * t1_H
                        else:
                            dx_abs = float(off.get("dx", 0.0))
                            dy_abs = float(off.get("dy", 0.0))
                        cx = float(cx_base + dx_abs)
                        cy = float(cy_base + dy_abs)
                        
                        patch = extract_patch_from_series(sag_t1_dir, t1_mid_inst, cx, cy)
                        url = _save_patch(VIS_DIR, study_id, f"{cond}_{lvl}", patch, CFG)
                        if url:
                            patch_urls[f"{cond}_{lvl}"] = url
            
            # SS patches (Axial T2)
            if axl_t2_id is not None and sag_t2_mid_ds is not None and kps:
                axl_t2_dir = CFG.DICOM_ROOT / str(study_id) / str(axl_t2_id)
                axial_datasets = load_sorted_series(axl_t2_dir)
                
                if axial_datasets:
                    for lvl in LEVELS:
                        cx_sag, cy_sag = kps[lvl]
                        try:
                            axl_idx = find_axial_slice_for_level(
                                axial_datasets, sag_t2_mid_ds,
                                cx_sag, cy_sag, logger, study_id, lvl
                            )
                            axl_ds = axial_datasets[axl_idx]
                            axl_inst = int(getattr(axl_ds, "InstanceNumber", axl_idx))
                            
                            (left_cx, left_cy), (right_cx, right_cy) = get_axial_lr_coords(
                                axl_ds, sag_t2_mid_ds, cx_sag, cy_sag,
                                ss_offsets=ss_offsets, level=lvl,
                            )
                            
                            for cond, cx, cy in [
                                ("left_subarticular_stenosis",  left_cx,  left_cy),
                                ("right_subarticular_stenosis", right_cx, right_cy),
                            ]:
                                patch = extract_patch_from_series(axl_t2_dir, axl_inst, cx, cy)
                                url = _save_patch(VIS_DIR, study_id, f"{cond}_{lvl}", patch, CFG)
                                if url:
                                    patch_urls[f"{cond}_{lvl}"] = url
                        except Exception as e:
                            logger.warning(f"  [vis] SS patch error for {lvl}: {e}")
        
        logger.info(f"  [vis] Generated {len(patch_urls)} patch images, full_scan={'yes' if full_scan_url else 'no'}")
            
    finally:
        CFG.DICOM_ROOT = original_dicom_root
        
    return preds, series_desc, {"full_scan": full_scan_url, "patches": patch_urls}


def _save_full_scan(vis_dir, study_id, windowed_img, kps):
    """Save the Sagittal T2 full scan with keypoint overlay. Returns URL or None."""
    try:
        img_color = cv2.cvtColor(windowed_img, cv2.COLOR_GRAY2BGR)
        
        level_colors = {
            "l1_l2": (60, 76, 231),
            "l2_l3": (18, 156, 243),
            "l3_l4": (113, 204, 46),
            "l4_l5": (219, 152, 52),
            "l5_s1": (182, 89, 155),
        }
        
        for lvl, (px, py) in kps.items():
            col = level_colors.get(lvl, (0, 255, 0))
            cv2.circle(img_color, (int(px), int(py)), radius=4, color=(255, 255, 255), thickness=-1)
            cv2.circle(img_color, (int(px), int(py)), radius=2, color=col, thickness=-1)
            label = lvl.upper().replace('_', '/')
            cv2.putText(img_color, label, (int(px) + 8, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(img_color, label, (int(px) + 8, int(py) + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)
        
        out_path = vis_dir / f"full_scan_{study_id}.jpg"
        ok = cv2.imwrite(str(out_path), img_color)
        if ok:
            return f"/media/rsna_uploads/traces/{study_id}/full_scan_{study_id}.jpg"
        else:
            logger.error(f"  [vis] cv2.imwrite failed: {out_path}")
    except Exception as e:
        logger.error(f"  [vis] Full scan save error: {e}")
    return None


def _save_patch(vis_dir, study_id, key, patch, CFG):
    """De-normalise a (3,128,128) patch and save as JPEG. Returns URL or None."""
    if patch is None:
        return None
    try:
        _MEAN = np.array(CFG.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
        _STD  = np.array(CFG.IMAGENET_STD,  dtype=np.float32)[:, None, None]
        p01 = np.clip(patch * _STD + _MEAN, 0.0, 1.0)
        img = (p01.transpose(1, 2, 0) * 255).astype(np.uint8)
        
        out_path = vis_dir / f"patch_{key}.jpg"
        ok = cv2.imwrite(str(out_path), img[:, :, ::-1])  # RGB -> BGR
        if ok:
            return f"/media/rsna_uploads/traces/{study_id}/{out_path.name}"
        else:
            logger.error(f"  [vis] cv2.imwrite failed for patch: {out_path}")
    except Exception as e:
        logger.error(f"  [vis] Patch save error for {key}: {e}")
    return None
