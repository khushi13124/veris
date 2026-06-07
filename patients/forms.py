import re
from django import forms
from django.core.exceptions import ValidationError
from django.utils import timezone

from .models import Patient, PatientMedicalHistory


# ── ID number format rules ────────────────────────────────────────────────────
ID_NUMBER_PATTERNS = {
    'Aadhar':   (r'^\d{12}$',                 'Aadhaar number must be exactly 12 digits.'),
    'PAN':      (r'^[A-Z]{5}[0-9]{4}[A-Z]$', 'PAN must be in AAAAA9999A format (5 letters, 4 digits, 1 letter).'),
    'Passport': (r'^[A-Z][0-9]{7}$',          'Passport number must be 1 uppercase letter followed by 7 digits (e.g. A1234567).'),
}


class PatientForm(forms.ModelForm):

    date_of_birth = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-input'}),
        label='Date of Birth'
    )

    profile_photo = forms.ImageField(
        required=False,
        widget=forms.ClearableFileInput(attrs={
            'class': 'form-input'
        })
    )

    class Meta:
        model   = Patient
        exclude = ['patient_id', 'created_at']
        widgets = {
            'full_name': forms.TextInput(attrs={
                'placeholder': 'Enter full name', 'class': 'form-input'
            }),
            'age': forms.NumberInput(attrs={
                'placeholder': 'Auto-calculated',
                'class': 'form-input',
                'readonly': 'readonly'
            }),
            'gender':         forms.Select(attrs={'class': 'form-select'}),
            'blood_group':    forms.Select(attrs={'class': 'form-select'}),
            'marital_status': forms.Select(attrs={'class': 'form-select'}),
            'mobile_number': forms.TextInput(attrs={
                'placeholder': '10-digit mobile number',
                'class': 'form-input', 'maxlength': '10'
            }),
            'email': forms.EmailInput(attrs={
                'placeholder': 'example@email.com', 'class': 'form-input'
            }),
            'alternate_contact': forms.TextInput(attrs={
                'placeholder': '10-digit number (optional)',
                'class': 'form-input', 'maxlength': '10'
            }),
            'address': forms.Textarea(attrs={
                'rows': 3, 'placeholder': 'Enter full address', 'class': 'form-input'
            }),
            'pin_code': forms.TextInput(attrs={
                'placeholder': '6-digit PIN code',
                'class': 'form-input', 'maxlength': '6', 'id': 'id_pin_code'
            }),
            'city':  forms.TextInput(attrs={'placeholder': 'City',  'class': 'form-input'}),
            'state': forms.TextInput(attrs={'placeholder': 'State', 'class': 'form-input'}),
            'id_type': forms.Select(attrs={'class': 'form-select'}),
            'id_number': forms.TextInput(attrs={
                'placeholder': 'Enter ID number', 'class': 'form-input'
            }),
            'insurance_provider': forms.Select(attrs={'class': 'form-select'}),
            'insurance_policy_number': forms.TextInput(attrs={
                'placeholder': 'Policy number (if applicable)', 'class': 'form-input'
            }),
        }

    def clean_date_of_birth(self):
        dob = self.cleaned_data['date_of_birth']
        if dob >= timezone.now().date():
            raise ValidationError("Date of Birth cannot be today or in the future.")
        return dob

    def clean_age(self):
        dob = self.cleaned_data.get('date_of_birth')
        if dob:
            today = timezone.now().date()
            age   = today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))
            return max(age, 0)
        return self.cleaned_data.get('age', 0)

    def clean_email(self):
        email = self.cleaned_data['email']
        qs = Patient.objects.filter(email=email)
        if self.instance and self.instance.pk:
            qs = qs.exclude(pk=self.instance.pk)
        if qs.exists():
            raise ValidationError("A patient with this email already exists.")
        return email

    def clean_id_number(self):
        id_type   = self.cleaned_data.get('id_type', '')
        id_number = self.cleaned_data.get('id_number', '').strip().replace(' ', '')

        if not id_number:
            raise ValidationError("ID number is required.")

        rule = ID_NUMBER_PATTERNS.get(id_type)
        if rule:
            pattern, message = rule
            if not re.match(pattern, id_number):
                raise ValidationError(message)

        return id_number


class MedicalHistoryForm(forms.ModelForm):

    smoking = forms.ChoiceField(
        choices=PatientMedicalHistory.SMOKING_CHOICES,
        widget=forms.RadioSelect(),
        required=True,
        label='Smoking Habit',
    )
    alcohol = forms.ChoiceField(
        choices=PatientMedicalHistory.ALCOHOL_CHOICES,
        widget=forms.RadioSelect(),
        required=True,
        label='Alcohol Consumption',
    )
    exercise = forms.ChoiceField(
        choices=PatientMedicalHistory.EXERCISE_CHOICES,
        widget=forms.RadioSelect(),
        required=True,
        label='Exercise Frequency',
    )

    class Meta:
        model   = PatientMedicalHistory
        # ✅ visit_date and doctor_notes added — set automatically in the view
        exclude = [
            'patient',
            'medical_conditions',
            'current_medications',
            'bmi',
            'created_at',
            'visit_date',      # ← set to timezone.now() in the view
            'doctor_notes',    # ← not part of the patient-facing registration form
        ]
        widgets = {
            'condition_details': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Describe condition details (e.g. diagnosed year, severity)',
                'class': 'form-input'
            }),
            'past_medications': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'List past medications (name, dosage, duration)',
                'class': 'form-input'
            }),
            'surgeries': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'List past surgeries or hospitalizations with year',
                'class': 'form-input'
            }),
            'allergies': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'List known allergies (food, medicine, environmental)',
                'class': 'form-input'
            }),
            'family_history': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Family history of diseases (e.g. Father: Diabetes)',
                'class': 'form-input'
            }),
            'height_cm': forms.NumberInput(attrs={
                'placeholder': 'Height in cm', 'class': 'form-input', 'step': '0.1'
            }),
            'weight_kg': forms.NumberInput(attrs={
                'placeholder': 'Weight in kg', 'class': 'form-input', 'step': '0.1'
            }),
            'bmi': forms.NumberInput(attrs={
                'placeholder': 'Auto-calculated', 'class': 'form-input', 'readonly': 'readonly'
            }),
        }