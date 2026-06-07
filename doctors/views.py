from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils import timezone
from django.db.models import Q
from datetime import timedelta, date
import secrets
import string
import json

from .models import Doctor
from .forms import DoctorRegistrationForm, DoctorLoginForm, ChangePasswordForm, OTPVerificationForm
from patients.models import Patient
from visits.models import Visit


def _get_doctor(request):
    return get_object_or_404(Doctor, user=request.user)


@login_required(login_url='doctor_login')
def dashboard(request):
    doctor = _get_doctor(request)
    today  = timezone.now().date()

    todays_visits = list(
        Visit.objects
        .filter(doctor=doctor, visit_date=today)
        .select_related('patient')
        .order_by('checked_in_at')
    )
    for idx, v in enumerate(todays_visits, start=1):
        v.token = idx

    pending_reports = Visit.objects.filter(
        doctor=doctor,
    ).filter(
        Q(diagnosis_notes='') | Q(diagnosis_notes__isnull=True)
    ).count()

    year_start = date(today.year, 1, 1)
    all_visits_this_year = (
        Visit.objects
        .filter(doctor=doctor, visit_date__gte=year_start)
        .select_related('patient')
        .order_by('visit_date', 'checked_in_at')
    )

    visits = Visit.objects.filter(doctor=doctor).select_related('patient')

    calendar_events = [
        {
            "title": v.patient.full_name,
            "date": str(v.visit_date),   # ✅ use visit_date (your field)
            "patient_id": v.patient.id,
            "color": "#2563eb" if v.status != 'done' else "#16a34a"
        }
        for v in visits
    ]

    return render(request, 'doctors/dashboard.html', {
        'doctor': doctor,
        'todays_visits': todays_visits,
        'today_appointments': len(todays_visits),
        'pending_reports': pending_reports,
        'all_visits_this_year': all_visits_this_year,
        'today': today,
        'calendar_events': json.dumps(calendar_events),  # ✅ ADD THIS
    })


def register_doctor(request):
    if request.method == 'POST':
        form = DoctorRegistrationForm(request.POST, request.FILES)
        if form.is_valid():
            from django.contrib.auth import get_user_model
            User = get_user_model()
            alphabet = string.ascii_letters + string.digits
            temp_pw  = ''.join(secrets.choice(alphabet) for _ in range(10))
            email    = form.cleaned_data['email']
            user = User.objects.create_user(email=email, password=temp_pw)
            doctor = form.save(commit=False)
            doctor.user           = user
            doctor.temp_password  = temp_pw
            doctor.is_first_login = True
            doctor.save()
            return redirect('doctor_credentials', doctor_id=doctor.doctor_id)
    else:
        form = DoctorRegistrationForm()
    return render(request, 'doctors/register_doctor.html', {'form': form})


def doctor_credentials(request, doctor_id):
    doctor = get_object_or_404(Doctor, doctor_id=doctor_id)
    return render(request, 'doctors/doctor_credentials.html', {'doctor': doctor})


def doctor_login(request):
    if request.method == 'POST':
        form = DoctorLoginForm(request.POST)
        if form.is_valid():
            email    = form.cleaned_data['username']
            password = form.cleaned_data['password']
            from django.contrib.auth import get_user_model
            User = get_user_model()
            try:
                User.objects.get(email=email)
                user = authenticate(request, email=email, password=password)
            except User.DoesNotExist:
                user = None
            if user:
                login(request, user)
                try:
                    doctor = user.doctor_profile
                    if doctor.is_first_login:
                        return redirect('change_password')
                except Doctor.DoesNotExist:
                    pass
                return redirect('doctor_dashboard')
            else:
                form.add_error(None, 'Invalid email or password.')
    else:
        form = DoctorLoginForm()
    return render(request, 'doctors/doctor_login.html', {'form': form})


@login_required(login_url='doctor_login')
def doctor_logout(request):
    logout(request)
    return redirect('/login/')


@login_required(login_url='doctor_login')
def change_password(request):
    doctor = _get_doctor(request)
    if request.method == 'POST':
        form = ChangePasswordForm(request.POST)
        if form.is_valid():
            new_pw = form.cleaned_data['new_password']
            request.user.set_password(new_pw)
            request.user.save()
            doctor.is_first_login = False
            doctor.temp_password  = ''
            doctor.save()
            login(request, request.user)
            return redirect('doctor_dashboard')
    else:
        form = ChangePasswordForm()
    return render(request, 'doctors/change_password.html', {'form': form})


@login_required(login_url='doctor_login')
def verify_otp_view(request):
    if request.method == 'POST':
        form = OTPVerificationForm(request.POST)
        if form.is_valid():
            return redirect('doctor_dashboard')
    else:
        form = OTPVerificationForm()
    return render(request, 'doctors/verify_otp.html', {'form': form})


@login_required(login_url='doctor_login')
def edit_profile(request):
    doctor = _get_doctor(request)
    if request.method == 'POST':
        form = DoctorRegistrationForm(request.POST, request.FILES, instance=doctor)
        if form.is_valid():
            form.save()
            return redirect('doctor_dashboard')
    else:
        form = DoctorRegistrationForm(instance=doctor)
    return render(request, 'doctors/edit_profile.html', {'form': form, 'doctor': doctor})


@login_required(login_url='doctor_login')
def patients_search_json(request):
    q = request.GET.get('q', '').strip()

    if not q:
        return JsonResponse({"patients": []})

    patients = Patient.objects.filter(
        Q(full_name__icontains=q) | Q(patient_id__icontains=q)
    ).order_by('full_name')[:10]

    data = [{
        "name": p.full_name,
        "id": p.patient_id,
        "age": p.age,
        "gender": p.gender,
        "visit_url": f"/visits/patient/{p.id}/",
    } for p in patients]

    return JsonResponse({"patients": data})

@login_required(login_url='doctor_login')
def patients_list_json(request):
    patients = Patient.objects.all().order_by('full_name')

    data = [{
        "name": p.full_name,
        "id": p.patient_id,
        "age": p.age,
        "gender": p.gender,
        "photo": p.profile_photo.url if p.profile_photo else None,
        "visit_url": f"/visits/patient/{p.id}/",
    } for p in patients]

    return JsonResponse({"patients": data})

@login_required(login_url='doctor_login')
def today_queue_json(request):
    doctor = _get_doctor(request)
    today = timezone.localdate()   

    visits = (
        Visit.objects
        .filter(doctor=doctor, visit_date=today)
        .select_related('patient')
        .order_by('checked_in_at')
    )

    data = []
    for idx, v in enumerate(visits, start=1):
        data.append({
            "token": idx,
            "patient_id": v.patient.patient_id,
            "patient_name": v.patient.full_name,
            "patient_photo": v.patient.profile_photo.url if v.patient.profile_photo else None,
            "status": v.status,
            "checked_in_at": v.checked_in_at.isoformat() if v.checked_in_at else None,
            "visit_id": v.id,
            "detail_url": f"/visits/patient/{v.patient.id}/",
            "start_url": f"/visits/status/{v.id}/with_doctor/",
            "complete_url": f"/visits/status/{v.id}/done/",
        })

    return JsonResponse({"visits": data})