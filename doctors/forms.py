from django import forms
from django.contrib.auth import get_user_model
from .models import Doctor

User = get_user_model()


class DoctorRegistrationForm(forms.ModelForm):
    class Meta:
        model = Doctor
        fields = [
            'full_name',
            'profile_photo',
            'specialization',
            'department',
            'qualification',
            'years_of_experience',
            'contact_number',
            'email',
            'chamber_number',
            'days_available',
            'shift_timings',
            'bio',
        ]
        widgets = {
            'full_name': forms.TextInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': 'Dr. Full Name',
            }),
            'specialization': forms.TextInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': 'e.g. Cardiologist',
            }),
            'department': forms.Select(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            }),
            'qualification': forms.TextInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': 'e.g. MBBS, MD, DM',
            }),
            'years_of_experience': forms.NumberInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'min': 0,
                'max': 60,
            }),
            'contact_number': forms.TextInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': '+91XXXXXXXXXX',
            }),
            'email': forms.EmailInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': 'doctor@hospital.com',
            }),
            'chamber_number': forms.TextInput(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'placeholder': 'e.g. Room 204, Block B',
            }),
            'days_available': forms.Select(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            }),
            'shift_timings': forms.Select(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            }),
            'bio': forms.Textarea(attrs={
                'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
                'rows': 3,
                'placeholder': 'Brief professional bio...',
            }),
            'profile_photo': forms.FileInput(attrs={
                'class': 'w-full text-sm text-slate-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-primary hover:file:bg-blue-100',
                'accept': 'image/*',
            }),
        }


class DoctorLoginForm(forms.Form):
    username = forms.EmailField(
        widget=forms.EmailInput(attrs={
            'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            'placeholder': 'doctor@hospital.com',
            'autofocus': True,
        })
    )
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={
            'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            'placeholder': 'Password',
        })
    )


class ChangePasswordForm(forms.Form):
    new_password = forms.CharField(
        label='New Password',
        widget=forms.PasswordInput(attrs={
            'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            'placeholder': 'New password',
            'id': 'new_password',
        })
    )
    confirm_password = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(attrs={
            'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary',
            'placeholder': 'Confirm new password',
            'id': 'confirm_password',
        })
    )

    def clean(self):
        cleaned_data = super().clean()
        new_password = cleaned_data.get('new_password')
        confirm_password = cleaned_data.get('confirm_password')

        if new_password and confirm_password:
            if new_password != confirm_password:
                raise forms.ValidationError("Passwords do not match.")
            if len(new_password) < 8:
                raise forms.ValidationError("Password must be at least 8 characters.")
        return cleaned_data


class OTPVerificationForm(forms.Form):
    otp = forms.CharField(
        max_length=6,
        min_length=6,
        widget=forms.TextInput(attrs={
            'class': 'w-full border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-primary tracking-widest text-center text-lg',
            'placeholder': '000000',
            'autocomplete': 'off',
            'maxlength': '6',
            'inputmode': 'numeric',
        })
    )