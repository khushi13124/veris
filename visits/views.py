from django.shortcuts import render, get_object_or_404, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from django.contrib import messages
from django.utils import timezone
from datetime import date
from collections import defaultdict

from patients.models import Patient
from doctors.models import Doctor
from .models import Visit
from django.views.decorators.http import require_http_methods

@login_required(login_url='/login/')
@require_http_methods(["POST"])
def save_ai_result(request, visit_id):
    """
    Saves AI differential diagnosis text to Visit.ai_differential.

    Called by symptom_disease_page.html via:
        fetch(`/visits/save-ai/${visitId}/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded', 'X-CSRFToken': csrfToken },
            body: new URLSearchParams({ field: 'ai_differential', result: resultText })
        })
    """
    try:
        from visits.models import Visit
        from doctors.models import Doctor

        doctor = Doctor.objects.get(user=request.user)
        visit  = Visit.objects.get(id=visit_id, doctor=doctor)

        field  = request.POST.get('field', '')
        result = request.POST.get('result', '')

        if field == 'ai_differential' and result:
            visit.ai_differential = result
            visit.save(update_fields=['ai_differential'])
            return JsonResponse({'status': 'ok'})
        else:
            return JsonResponse({'status': 'error', 'message': 'Missing field or result'}, status=400)

    except Doctor.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Doctor not found'}, status=403)
    except Visit.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Visit not found or access denied'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

def _get_doctor(request):
    return get_object_or_404(Doctor, user=request.user)


# ─────────────────────────────────────────────
#  TODAY'S QUEUE — admin/reception view (read-only)
# ─────────────────────────────────────────────
@login_required
def today_queue(request):
    today = timezone.localdate()   # ← uses TIME_ZONE from settings, not UTC

    all_today = (
        Visit.objects
        .filter(visit_date=today)
        .select_related('patient', 'doctor')
        .order_by('checked_in_at')
    )

    active_visits = []
    done_today    = []
    for idx, v in enumerate(all_today, start=1):
        v.token = idx
        if v.status in ('waiting', 'with_doctor'):
            active_visits.append(v)
        else:
            done_today.append(v)

    past_done = (
        Visit.objects
        .filter(status='done')
        .exclude(visit_date=today)
        .select_related('patient', 'doctor')
        .order_by('-visit_date', 'checked_in_at')
    )

    past_by_date = defaultdict(list)
    for v in past_done:
        past_by_date[v.visit_date].append(v)

    past_by_date_sorted = sorted(
        past_by_date.items(), key=lambda x: x[0], reverse=True
    )

    return render(request, 'visits/today_queue.html', {
        'active_visits':    active_visits,
        'done_today':       done_today,
        'past_by_date':     past_by_date_sorted,
        'today':            today,
        'active_count':     len(active_visits),
        'done_today_count': len(done_today),
    })


# ─────────────────────────────────────────────
#  TODAY'S QUEUE — JSON for dashboard panel
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
def today_queue_json(request):
    doctor = _get_doctor(request)
    today  = timezone.localdate()

    visits = (
        Visit.objects
        .filter(doctor=doctor, visit_date=today)
        .select_related('patient')
        .order_by('checked_in_at')
    )

    result = []
    for idx, v in enumerate(visits, start=1):
        prev_count = Visit.objects.filter(
            doctor=doctor, patient=v.patient,
        ).exclude(id=v.id).count()

        photo_url = v.patient.profile_photo.url if v.patient.profile_photo else None

        result.append({
            'visit_id':      v.id,
            'token':         idx,
            'patient_name':  v.patient.full_name,
            'patient_id':    v.patient.patient_id,
            'patient_photo': photo_url,
            'status':        v.status,
            'is_new':        prev_count == 0,
            'checked_in_at': v.checked_in_at.isoformat() if v.checked_in_at else None,
            'detail_url':    f'/visits/patient/{v.patient.id}/',
            'start_url':     f'/visits/status/{v.id}/with_doctor/',
            'complete_url':  f'/visits/status/{v.id}/done/',
        })

    return JsonResponse({'visits': result})


# ─────────────────────────────────────────────
#  ASSIGN VISIT (check-in)
# ─────────────────────────────────────────────
@login_required
def assign_visit(request):
    if request.method == 'POST':
        patient_id = request.POST.get('patient_id')
        doctor_id  = request.POST.get('doctor_id')
        remarks    = request.POST.get('remarks', '')

        try:
            patient = Patient.objects.get(id=patient_id)
            doctor  = Doctor.objects.get(id=doctor_id)
        except (Patient.DoesNotExist, Doctor.DoesNotExist):
            messages.error(request, "Invalid patient or doctor.")
            return redirect('visits:assign_visit')

        Visit.objects.create(
            patient=patient,
            doctor=doctor,
            visit_date=timezone.localdate(),
            remarks=remarks,
            status='waiting',
            checked_in_at=timezone.now(),
        )
        messages.success(request, f"{patient.full_name} checked in successfully ✅")
        return redirect('visits:assign_visit')

    patients = Patient.objects.all().order_by('full_name')
    doctors  = Doctor.objects.all().order_by('full_name')
    return render(request, 'visits/assign_visit.html', {
        'patients': patients, 'doctors': doctors,
    })


# ─────────────────────────────────────────────
#  PATIENT DETAIL — doctor view
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
def patient_detail(request, patient_id):
    doctor  = _get_doctor(request)
    patient = get_object_or_404(Patient, id=patient_id)
    today   = timezone.localdate()

    active_visit = (
        Visit.objects
        .filter(patient=patient, doctor=doctor, visit_date=today)
        .exclude(status='done')
        .order_by('-checked_in_at')
        .first()
    )

    # Auto-advance waiting → with_doctor when doctor opens patient detail
    if active_visit and active_visit.status == 'waiting':
        active_visit.status = 'with_doctor'
        active_visit.save(update_fields=['status'])

    my_visits = (
        Visit.objects
        .filter(patient=patient, doctor=doctor)
        .order_by('-visit_date', '-checked_in_at')
    )

    other_visits = (
        Visit.objects
        .filter(patient=patient)
        .exclude(doctor=doctor)
        .select_related('doctor')
        .order_by('-visit_date', '-checked_in_at')
    )

    prior_count = (
        my_visits.exclude(id=active_visit.id).count()
        if active_visit else my_visits.count()
    )

    doctor.assigned_patients.add(patient)

    return render(request, 'visits/patient_detail.html', {
        'patient':      patient,
        'doctor':       doctor,
        'active_visit': active_visit,
        'my_visits':    my_visits,
        'other_visits': other_visits,
        'is_returning': prior_count > 0,
        'today':        today,
    })


# ─────────────────────────────────────────────
#  ADD VISIT
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
@require_POST
def add_visit(request, patient_id):
    doctor  = _get_doctor(request)
    patient = get_object_or_404(Patient, id=patient_id)
    doctor.assigned_patients.add(patient)

    next_appointment = None
    next_appt_str = request.POST.get('next_appointment', '').strip()
    if next_appt_str:
        try:
            next_appointment = date.fromisoformat(next_appt_str)
        except ValueError:
            pass

    visit = Visit.objects.create(
        patient=patient, doctor=doctor,
        remarks=request.POST.get('remarks', '').strip(),
        medication=request.POST.get('medication', '').strip(),
        diagnosis_notes=request.POST.get('diagnosis_notes', '').strip(),
        next_appointment=next_appointment,
        visit_date=timezone.localdate(),
        status='with_doctor',
        checked_in_at=timezone.now(),
    )
    messages.success(request, f"Visit recorded on {visit.visit_date.strftime('%d %b %Y')} ✅")
    return redirect('visits:patient_detail', patient_id=patient_id)


# ─────────────────────────────────────────────
#  UPDATE VISIT
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
@require_POST
def update_visit(request, visit_id):
    doctor = _get_doctor(request)
    visit  = get_object_or_404(Visit, id=visit_id, doctor=doctor)

    visit.remarks         = request.POST.get('remarks',         visit.remarks         or '').strip()
    visit.diagnosis_notes = request.POST.get('diagnosis_notes', visit.diagnosis_notes or '').strip()
    visit.medication      = request.POST.get('medication',      visit.medication      or '').strip()

    next_appt_str = request.POST.get('next_appointment', '').strip()
    if next_appt_str:
        try:
            visit.next_appointment = date.fromisoformat(next_appt_str)
        except ValueError:
            pass
    elif 'next_appointment' in request.POST:
        visit.next_appointment = None

    keep_status = request.POST.get('keep_status', '0')
    if keep_status != '1' and visit.status != 'done':
        visit.status = 'done'

    visit.save()
    messages.success(request, "Visit saved ✅")
    return redirect('visits:patient_detail', patient_id=visit.patient.id)


# ─────────────────────────────────────────────
#  MARK DONE (kept for queue buttons)
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
@require_POST
def mark_done(request, visit_id):
    doctor = _get_doctor(request)
    visit  = get_object_or_404(Visit, id=visit_id, doctor=doctor)
    visit.status = 'done'
    visit.save(update_fields=['status'])
    messages.success(request, f"Visit for {visit.patient.full_name} marked as done ✅")
    return redirect('doctor_dashboard')


# ─────────────────────────────────────────────
#  UPDATE VISIT STATUS
#  Restricted to the doctor assigned to the visit.
#  Admin/reception hitting this URL gets a 403.
# ─────────────────────────────────────────────
@login_required
def update_visit_status(request, visit_id, status):
    visit = get_object_or_404(Visit, id=visit_id)

    # Only the assigned doctor may change status
    try:
        requesting_doctor = Doctor.objects.get(user=request.user)
    except Doctor.DoesNotExist:
        messages.error(request, "Only the assigned doctor can update visit status.")
        return redirect('visits:today_queue')

    if visit.doctor != requesting_doctor:
        messages.error(request, "You are not authorised to update this visit.")
        return redirect('visits:today_queue')

    if status in ('waiting', 'with_doctor', 'done'):
        visit.status = status
        visit.save(update_fields=['status'])

    next_url = request.POST.get('next') or request.GET.get('next') or 'visits:today_queue'
    if next_url == 'visits:today_queue':
        return redirect('visits:today_queue')
    return redirect(next_url)


# ─────────────────────────────────────────────
#  SAVE AI / SYMCAT RESULT to a visit
# ─────────────────────────────────────────────
@login_required(login_url='doctor_login')
@require_POST
def save_ai_result(request, visit_id):
    doctor = _get_doctor(request)
    visit  = get_object_or_404(Visit, id=visit_id, doctor=doctor)

    field  = request.POST.get('field', '')
    result = request.POST.get('result', '').strip()

    if field not in ('ai_differential', 'ai_medical'):
        return JsonResponse({'status': 'error', 'message': 'Invalid field.'}, status=400)

    setattr(visit, field, result)
    visit.save(update_fields=[field])
    return JsonResponse({'status': 'ok', 'field': field, 'visit_id': visit.id})