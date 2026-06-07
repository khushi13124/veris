from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.db.models import Q
from django.utils import timezone

from .forms import PatientForm, MedicalHistoryForm
from .models import Patient, PatientMedicalHistory


def register_patient(request):
    if request.method == 'POST':
        form = PatientForm(request.POST, request.FILES)
        if form.is_valid():
            patient = form.save()
            return redirect('medical_history', patient_id=patient.id)
    else:
        form = PatientForm()

    return render(request, 'patients/register.html', {
        'form': form,
        'now_year': timezone.now().year,
    })


def medical_history(request, patient_id):
    patient     = get_object_or_404(Patient, id=patient_id)
    existing_mh = PatientMedicalHistory.objects.filter(patient=patient).first()

    if request.method == 'POST':
        conditions  = request.POST.getlist('medical_conditions')
        med_names   = request.POST.getlist('current_medications_name[]')
        med_dosages = request.POST.getlist('current_medications_dosage[]')
        med_freqs   = request.POST.getlist('current_medications_frequency[]')
        med_since   = request.POST.getlist('current_medications_since[]')

        medications = []
        for i in range(len(med_names)):
            if med_names[i].strip():
                medications.append({
                    'name':      med_names[i].strip(),
                    'dosage':    med_dosages[i].strip() if i < len(med_dosages) else '',
                    'frequency': med_freqs[i].strip()   if i < len(med_freqs)   else '',
                    'since':     med_since[i].strip()   if i < len(med_since)   else '',
                })

        form = MedicalHistoryForm(request.POST, instance=existing_mh)

        if form.is_valid():
            mh                     = form.save(commit=False)
            mh.patient             = patient
            mh.visit_date          = timezone.now()
            mh.medical_conditions  = conditions
            mh.current_medications = medications

            try:
                h      = float(request.POST.get('height_cm', 0))
                w      = float(request.POST.get('weight_kg', 0))
                mh.bmi = round(w / ((h / 100) ** 2), 2) if h > 0 and w > 0 else None
            except (ValueError, ZeroDivisionError):
                mh.bmi = None

            mh.save()
            return redirect('success', patient_id=patient.id)
    else:
        form = MedicalHistoryForm(instance=existing_mh)

    return render(request, 'patients/medicalhistory.html', {
        'form':    form,
        'patient': patient,
    })


def success(request, patient_id):
    patient         = get_object_or_404(Patient, id=patient_id)
    medical_history = PatientMedicalHistory.objects.filter(patient=patient).first()

    return render(request, 'patients/success.html', {
        'patient':         patient,
        'medical_history': medical_history,
    })


# ✅ AJAX search — used by doctor dashboard "Find Patient" tab
@login_required(login_url='doctor_login')
def search_json(request):
    q = request.GET.get('q', '').strip()

    if len(q) < 2:
        return JsonResponse({'patients': []})

    # ✅ Patient has NO user FK — search full_name and patient_id directly
    qs = Patient.objects.filter(
        Q(full_name__icontains=q) |
        Q(patient_id__icontains=q) |
        Q(mobile_number__icontains=q)
    ).order_by('full_name')[:20]

    from django.urls import reverse
    results = []
    for p in qs:
        results.append({
            'id':        p.id,
            'name':      p.full_name,
            'age':       p.age,
            'gender':    p.gender,
            'visit_url': reverse('visits:patient_detail', args=[p.id]),
        })

    return JsonResponse({'patients': results})