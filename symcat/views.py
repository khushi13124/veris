# symcat/views.py
# Updated to pass visit context to the symptom checker template

from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn



class ImprovedDiseaseModel(nn.Module):
    def __init__(self, num_symptoms, num_diseases):
        super().__init__()
        layers = []
        in_dim = num_symptoms
        for hidden_dim, dropout_rate in zip([1024, 512, 256], [0.4, 0.3, 0.2]):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, num_diseases))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


MODEL = None
SYMPTOM_TO_IDX = None
IDX_TO_DISEASE = None
DISEASE_SYMPTOMS = None
AVAILABLE_SYMPTOMS = None

def load_model_and_data():
    global MODEL, SYMPTOM_TO_IDX, IDX_TO_DISEASE, DISEASE_SYMPTOMS, AVAILABLE_SYMPTOMS
    if MODEL is not None:
        return

    try:
        BASE_DIR = Path(__file__).resolve().parent

        MODEL_PATH = BASE_DIR / "models" / "symptom_disease_best.pt"
        SYMPTOM_ENCODER_PATH = BASE_DIR / "data" / "symptom_encoder.pkl"
        IDX_TO_DISEASE_PATH = BASE_DIR / "data" / "idx_to_disease.pkl"
        MERGED_DATA_PATH = BASE_DIR / "data" / "merged_symptom_disease_data.csv"

        with open(SYMPTOM_ENCODER_PATH, 'rb') as f:
            SYMPTOM_TO_IDX = pickle.load(f)
        with open(IDX_TO_DISEASE_PATH, 'rb') as f:
            IDX_TO_DISEASE = pickle.load(f)

        checkpoint = torch.load(MODEL_PATH, map_location='cpu')
        MODEL = ImprovedDiseaseModel(
            num_symptoms=len(SYMPTOM_TO_IDX),
            num_diseases=len(IDX_TO_DISEASE)
        )
        MODEL.load_state_dict(checkpoint['model_state_dict'])
        MODEL.eval()

        df = pd.read_csv(MERGED_DATA_PATH)
        df['disease'] = df['disease'].str.lower().str.strip()
        df['symptom'] = df['symptom'].str.lower().str.strip()

        DISEASE_SYMPTOMS = (
            df.groupby('disease')[['symptom', 'weight']]
            .apply(lambda x: dict(zip(x['symptom'], x['weight'])))
            .to_dict()
        )
        AVAILABLE_SYMPTOMS = sorted(SYMPTOM_TO_IDX.keys())
       
        print(f"Model loaded: {len(SYMPTOM_TO_IDX)} symptoms, {len(IDX_TO_DISEASE)} diseases")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@login_required(login_url='/login/')
def symptom_checker_page(request):
    """
    Renders the symptom checker page.
    If ?visit_id=N is passed, the page will show a Save button
    and redirect back to patient_detail after saving.
    """
    context = {}

    visit_id = request.GET.get('visit_id')
    if visit_id:
        try:
            # Lazy import to avoid circular imports
            from visits.models import Visit
            from doctors.models import Doctor

            doctor  = Doctor.objects.get(user=request.user)
            visit   = Visit.objects.select_related('patient').get(id=visit_id, doctor=doctor)
            context['visit']    = visit
            context['visit_id'] = visit_id
        except Exception:
            # If visit not found or doctor mismatch, just show without save context
            pass

    return render(request, 'symcat/symptom_disease_page.html', context)



@require_http_methods(["GET"])
def get_symptoms(request):
    load_model_and_data()
    return JsonResponse({
        'symptoms': AVAILABLE_SYMPTOMS,
        'count': len(AVAILABLE_SYMPTOMS)
    })


@csrf_exempt
@require_http_methods(["POST"])
def predict_disease(request):
    load_model_and_data()

    try:
        data = json.loads(request.body)
        input_symptoms = data.get('symptoms', [])

        if not input_symptoms:
            return JsonResponse({'success': False, 'error': 'No symptoms provided'}, status=400)

        input_symptoms = [s.lower().strip() for s in input_symptoms]
        x = np.zeros(len(SYMPTOM_TO_IDX), dtype=np.float32)
        matched_symptoms = []
        unknown_symptoms = []

        for symptom in input_symptoms:
            if symptom in SYMPTOM_TO_IDX:
                x[SYMPTOM_TO_IDX[symptom]] = 1.0
                matched_symptoms.append(symptom)
            else:
                unknown_symptoms.append(symptom)
        
        if not matched_symptoms:
            return JsonResponse({
                'success': False,
                'error': 'No recognized symptoms. Please check spelling.',
                'unknown_symptoms': unknown_symptoms
            }, status=400)

        with torch.no_grad():
            x_tensor = torch.tensor(x).unsqueeze(0)
            logits = MODEL(x_tensor)
            probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        all_indices = np.argsort(probs)[::-1]
        filtered_indices = []
        symptom_set = set(matched_symptoms)
        for idx in all_indices:
            prob = probs[idx]
            disease = IDX_TO_DISEASE[int(idx)]
            known_symptoms = DISEASE_SYMPTOMS.get(disease, {})

            if not known_symptoms:
                continue

            matched = sum(1 for s in symptom_set if s in known_symptoms)
            completeness = matched / len(known_symptoms) if known_symptoms else 0

            include = (
                len(filtered_indices) < 10 or
                prob > 0.01 or
                completeness > 0.5
            )

            if include:
                filtered_indices.append(idx)

            if len(filtered_indices) >= 25:
                break

        predictions = []
        for rank, idx in enumerate(filtered_indices, 1):
            disease = IDX_TO_DISEASE[int(idx)]
            probability = float(probs[idx])
            known_symptoms = DISEASE_SYMPTOMS.get(disease, {})

            matched = {s: {'weight': known_symptoms[s], 'importance': _get_importance(known_symptoms[s])} 
                      for s in symptom_set if s in known_symptoms}
            missing = {s: {'weight': known_symptoms[s], 'importance': _get_importance(known_symptoms[s])} 
                      for s in known_symptoms if s not in symptom_set}
            completeness = len(matched) / len(known_symptoms) if known_symptoms else 0
            interpretation= _generate_interpretation(matched, missing, completeness, probability)

            predictions.append({
                'rank': rank,
                'disease': disease,
                'probability': round(probability, 4),
                'matched_symptoms': matched,
                'missing_symptoms': dict(list(missing.items())[:10]),
                'completeness': round(completeness, 2),
                'num_matched': len(matched),
                'num_missing': len(missing),
                'interpretation': interpretation
            })

        return JsonResponse({
            'success': True,
            'predictions': predictions,
            'input_info': {
                'total_entered': len(input_symptoms),
                'recognized': len(matched_symptoms),
                'unknown': len(unknown_symptoms),
                'unknown_list': unknown_symptoms,
                'matched_symptoms_list': matched_symptoms
            }
        })

    except json.JSONDecodeError:
        return JsonResponse({'success': False, 'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)}, status=500)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_importance(weight):
    if weight > 0.7:   return "high"
    elif weight > 0.4: return "medium"
    return "low"


def _generate_interpretation(matched, missing, completeness, probability):
    if probability > 0.7 and completeness > 0.7:
        confidence = "Strong match"
    elif probability > 0.4 or completeness > 0.5:
        confidence = "Moderate match"
    else:
        confidence = "Weak match"

    interpretation = f"{confidence}. "
    total_symptoms = len(matched) + len(missing)
    if total_symptoms > 0:
        if completeness > 0.7:
            interpretation += f"Patient exhibits {len(matched)}/{total_symptoms} known symptoms. "
        elif completeness > 0.5:
            interpretation += f"Patient has {len(matched)}/{total_symptoms} symptoms. "
        else:
            interpretation += f"Only {len(matched)}/{total_symptoms} symptoms present. "

    critical_missing = sum(1 for info in missing.values() if info['weight'] > 0.7)
    if critical_missing > 0:
        interpretation += f"{critical_missing} critical symptom(s) missing. "

    if completeness > 0.7 and critical_missing == 0 and probability > 0.5:
        interpretation += "Strong candidate for diagnosis."
    elif completeness > 0.5 or probability > 0.4:
        interpretation += "Consider for differential diagnosis."
    else:
        interpretation += "Low likelihood - explore other diagnoses."

    return interpretation