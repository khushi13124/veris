import os
import shutil
import time
import zipfile
import pydicom
from django.shortcuts import render, redirect
from django.conf import settings
from .services import process_dicom_folder

# Setup media root
UPLOAD_DIR = getattr(settings, 'MEDIA_ROOT') / 'rsna_uploads'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def upload_dicom(request):
    if request.method == "POST":
        patient_id = request.GET.get('patient_id', 'unknown')
        visit_id = request.GET.get('visit_id', '')
        
        # We can either accept a ZIP or multiple files from webkitdirectory
        files = request.FILES.getlist('dicom_files')
        
        if not files:
            return render(request, 'rsna/upload.html', {"error": "No files uploaded."})
            
        temp_dir = UPLOAD_DIR / str(int(time.time() * 1000))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        if len(files) == 1 and files[0].name.endswith('.zip'):
            zip_path = temp_dir / 'upload.zip'
            with open(zip_path, 'wb+') as dest:
                for chunk in files[0].chunks():
                    dest.write(chunk)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir / "raw")
            os.remove(zip_path)
            
            def get_all_files(path):
                f_paths = []
                for root, _, filenames in os.walk(path):
                    for name in filenames:
                        f_paths.append(os.path.join(root, name))
                return f_paths
                
            raw_files = get_all_files(temp_dir / "raw")
        else:
            raw_dir = temp_dir / "raw"
            raw_dir.mkdir(exist_ok=True)
            raw_files = []
            for i, f in enumerate(files):
                f_path = raw_dir / f"{i}_{f.name}"
                with open(f_path, 'wb+') as dest:
                    for chunk in f.chunks():
                        dest.write(chunk)
                raw_files.append(str(f_path))

        study_id = None
        for f_path in raw_files:
            try:
                ds = pydicom.dcmread(f_path, stop_before_pixels=True)
                uid_study = str(ds.StudyInstanceUID)
                uid_series = str(ds.SeriesInstanceUID).split('.')[-1]
                
                if study_id is None:
                    study_id = uid_study
                    
                s_dir = UPLOAD_DIR / uid_study / uid_series
                s_dir.mkdir(parents=True, exist_ok=True)
                
                target_path = s_dir / os.path.basename(f_path)
                shutil.move(f_path, target_path)
            except Exception:
                pass 
                
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if study_id is None:
            return render(request, 'rsna/upload.html', {"error": "No valid DICOM files found."})
            
        study_dir = UPLOAD_DIR / str(study_id)
        
        try:
            preds, series_desc, visuals = process_dicom_folder(study_dir, patient_id=patient_id)
            
            serializable_preds = {k: v.tolist() for k, v in preds.items()}
            
            request.session[f'rsna_preds_{study_id}'] = serializable_preds
            request.session[f'rsna_path_{study_id}'] = str(study_dir)
            request.session[f'rsna_visuals_{study_id}'] = visuals
            request.session[f'rsna_patient_{study_id}'] = patient_id
            request.session[f'rsna_visit_{study_id}'] = visit_id
            
            return redirect('rsna:results', prediction_id=study_id)
            
        except Exception as e:
            return render(request, 'rsna/upload.html', {"error": str(e), "patient_id": request.GET.get('patient_id', '')})

    patient_id = request.GET.get('patient_id', '')
    visit_id = request.GET.get('visit_id', '')
    return render(request, 'rsna/upload.html', {"patient_id": patient_id, "visit_id": visit_id})

def results(request, prediction_id):
    preds = request.session.get(f'rsna_preds_{prediction_id}')
    path_val = request.session.get(f'rsna_path_{prediction_id}')
    visuals = request.session.get(f'rsna_visuals_{prediction_id}', {})
    patient_id = request.session.get(f'rsna_patient_{prediction_id}', '')
    visit_id = request.session.get(f'rsna_visit_{prediction_id}', '')
    
    if not preds:
        return redirect('rsna:upload')
        
    return render(request, 'rsna/results.html', {
        'prediction_id': prediction_id,
        'preds': preds,
        'path': path_val,
        'full_scan': visuals.get('full_scan'),
        'patches': visuals.get('patches', {}),
        'patient_id': patient_id,
        'visit_id': visit_id
    })
