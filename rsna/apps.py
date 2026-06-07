from django.apps import AppConfig


class RsnaConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'rsna'

    def ready(self):
        import sys
        import os
        
        # Prevent loading models during manage.py migrate, makemigrations, etc. unless runserver
        if 'runserver' not in sys.argv and 'wsgi' not in sys.argv and 'gunicorn' not in sys.argv:
            return
            
        # Use local src/ and config/ copied into this Django app
        rsna_app_dir = os.path.dirname(__file__)
        if rsna_app_dir not in sys.path:
            sys.path.insert(0, rsna_app_dir)
            
        print("Loading RSNA models... This might take a while.")
        import torch
        from src.module5_inference import load_model1, load_classifier
        from config.settings import CFG
        from pathlib import Path
        
        # Check for local checkpoints
        local_ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
        os.makedirs(local_ckpt_dir, exist_ok=True)
        if os.path.exists(os.path.join(local_ckpt_dir, 'model1_best.pt')):
            CFG.CKPT_DIR = Path(local_ckpt_dir)
            CFG.OUTPUT_DIR = Path(local_ckpt_dir)
            if hasattr(CFG, 'NFN_OFFSETS_PATH'): CFG.NFN_OFFSETS_PATH = CFG.OUTPUT_DIR / "nfn_keypoint_offsets.json"
            if hasattr(CFG, 'SS_OFFSETS_PATH'): CFG.SS_OFFSETS_PATH = CFG.OUTPUT_DIR / "ss_axial_offsets.json"
            if hasattr(CFG, 'TEMP_SCALES_PATH'): CFG.TEMP_SCALES_PATH = CFG.OUTPUT_DIR / "temperature_scales.json"
            print(f"Using local Django checkpoints and config: {CFG.CKPT_DIR}")
        
        has_merged_m3 = (CFG.CKPT_DIR / "model3_best.pt").exists()
        
        # Need device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.ml_models = {
            1: load_model1(self.device),
            2: load_classifier(2, self.device),
            4: load_classifier(4, self.device),
        }
        
        if has_merged_m3:
            self.ml_models[3] = load_classifier(3, self.device)
            print("Detected merged Model 3 checkpoint.")
        else:
            if (CFG.CKPT_DIR / "model3_left_best.pt").exists():
                self.ml_models["3L"] = load_classifier("3_left", self.device)
                self.ml_models["3R"] = load_classifier("3_right", self.device)
                print("Detected split Model 3L and 3R checkpoints.")
        
        import logging
        logger = logging.getLogger(__name__)
        
        # Load pre-computed JSON offsets and temperature scales from local checkpoints.
        # These JSONs already exist in healthcare/rsna/checkpoints so master/val CSVs
        # are not needed — load_nfn_offsets / load_ss_offsets only use master when
        # the JSON is absent (recompute path, which we never hit here).
        import json
        nfn_path  = Path(local_ckpt_dir) / "nfn_keypoint_offsets.json"
        ss_path   = Path(local_ckpt_dir) / "ss_axial_offsets.json"
        temp_path = Path(local_ckpt_dir) / "temperature_scales.json"

        print("Checking for pre-computed offsets and temperatures...")
        if nfn_path.exists():
            with open(nfn_path) as f:
                self.nfn_offsets = json.load(f)
            print(f"NFN offsets loaded from {nfn_path}")
        else:
            print("NFN offsets not found. Using defaults.")
            self.nfn_offsets = {}

        if ss_path.exists():
            with open(ss_path) as f:
                self.ss_offsets = json.load(f)
            print(f"SS offsets loaded from {ss_path}")
        else:
            print("SS offsets not found. Using defaults.")
            self.ss_offsets = {}

        if temp_path.exists():
            with open(temp_path) as f:
                scales = json.load(f)
            self.temp_scales = {int(k): float(v) for k, v in scales.items()}
            print(f"Temperature scales loaded: {self.temp_scales}")
        else:
            print("Temperature scales missing. Models will use T=1.0")
            self.temp_scales = {}
        
        print("RSNA models loaded successfully!")

