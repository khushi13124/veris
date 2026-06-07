from django.views.generic import FormView, TemplateView
from django.contrib.auth import authenticate, login
from django.contrib.auth.mixins import LoginRequiredMixin
from django.shortcuts import redirect
from django import forms
from django.shortcuts import render
from django.contrib.auth.decorators import login_required

from django.utils import timezone
from patients.models import Patient
from doctors.models import Doctor
from visits.models import Visit
from datetime import timedelta
from django.contrib.auth import authenticate, login, logout  # add logout here

def doctor_logout(request):
    logout(request)
    return redirect('/login/')
@login_required(login_url='/login/')
def home_view(request):
    if request.user.role != 'ADMIN':
        from django.contrib.auth import logout
        logout(request)
        return redirect('/login/')

    today = timezone.now().date()

    # Last 7 days visit counts
    last_7_days = []
    for i in range(6, -1, -1):
        day = today - timedelta(days=i)
        count = Visit.objects.filter(visit_date=day).count()
        last_7_days.append({
            'label': day.strftime('%a'),   # Mon, Tue ...
            'count': count,
            
        })

    context = {
        'total_patients':    Patient.objects.count(),
        'total_doctors':     Doctor.objects.count(),
        'today_queue_count': Visit.objects.filter(visit_date=today, status='waiting').count(),
        'last_7_days':       last_7_days,
    }
    return render(request, 'accounts/home.html', context)

class LoginForm(forms.Form):
    email = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-500',
            'placeholder': 'Enter your email'
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-slate-500',
            'placeholder': 'Enter your password'
        })
    )


class LoginView(FormView):
    template_name = 'accounts/login.html'
    form_class = LoginForm

    def dispatch(self, request, *args, **kwargs):
        if request.user.is_authenticated:

            # 🔵 DOCTOR FLOW (use your existing logic)
            if request.user.role == 'DOCTOR':
                try:
                    doctor = request.user.doctor_profile

                    if doctor.is_first_login:
                        return redirect('/doctors/change-password/')

                    return redirect('/doctors/dashboard/')
                except Exception:
                    logout(request)
                    return redirect('/login/')

            # 🟣 ADMIN FLOW
            if request.user.role == 'ADMIN':
                return redirect('/home/')

            logout(request)

        return super().dispatch(request, *args, **kwargs)

    def form_valid(self, form):
        email = form.cleaned_data.get('email')
        password = form.cleaned_data.get('password')

        user = authenticate(self.request, email=email, password=password)

        if user is not None:
            login(self.request, user)

            # 🔵 DOCTOR LOGIN (reuse your flow)
            if user.role == 'DOCTOR':
                try:
                    doctor = user.doctor_profile

                    if doctor.is_first_login:
                        return redirect('/doctors/change-password/')

                    return redirect('/doctors/dashboard/')

                except Exception:
                    logout(self.request)
                    form.add_error(None, "Doctor profile not found.")
                    return self.form_invalid(form)

            # 🟣 ADMIN LOGIN
            if user.role == 'ADMIN':
                return redirect('/home/')

            logout(self.request)
            form.add_error(None, "Invalid role.")
            return self.form_invalid(form)

        else:
            form.add_error(None, "Invalid email or password.")
            return self.form_invalid(form)


