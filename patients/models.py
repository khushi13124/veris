from django.db import models
from django.core.validators import RegexValidator
from django.utils import timezone


class Patient(models.Model):
    """
    Stores all personal, contact, and identification details for a patient.

    Patient ID format: PT{DDMMYYYY}{NNNNN}
    e.g. PT2303202600001 — sequential per date, never reused.

    All medical data is stored in PatientMedicalHistory linked via ForeignKey,
    allowing multiple visit records per patient.
    """

    # ── Auto-generated unique patient identifier ──────────────────────────────
    patient_id = models.CharField(
        max_length=20,
        unique=True,
        blank=True,
        editable=False,
    )

    # ── Section A: Personal Info ───────────────────────────────────────────────
    full_name = models.CharField(
        max_length=100,
        validators=[RegexValidator(r'^[a-zA-Z ]+$', 'Only alphabets and spaces are allowed.')]
    )
    date_of_birth = models.DateField()
    age = models.PositiveIntegerField()

    GENDER_CHOICES = [
        ('Male', 'Male'),
        ('Female', 'Female'),
        ('Other', 'Other'),
        ('Prefer not to say', 'Prefer not to say'),
    ]
    gender = models.CharField(max_length=25, choices=GENDER_CHOICES)

    BLOOD_GROUP_CHOICES = [
        ('A+', 'A+'), ('A-', 'A-'),
        ('B+', 'B+'), ('B-', 'B-'),
        ('O+', 'O+'), ('O-', 'O-'),
        ('AB+', 'AB+'), ('AB-', 'AB-'),
    ]
    blood_group = models.CharField(max_length=5, choices=BLOOD_GROUP_CHOICES)

    MARITAL_STATUS = [
        ('Single', 'Single'),
        ('Married', 'Married'),
        ('Divorced', 'Divorced'),
        ('Widowed', 'Widowed'),
    ]
    marital_status = models.CharField(max_length=10, choices=MARITAL_STATUS)

    # 🆕 Profile Photo
    profile_photo = models.ImageField(
        upload_to='patient_photos/',
        null=True,
        blank=True
    )

    # ── Section B: Contact Info ────────────────────────────────────────────────
    mobile_number = models.CharField(
        max_length=10,
        validators=[RegexValidator(r'^\d{10}$', 'Enter a valid 10-digit mobile number.')]
    )
    email = models.EmailField(unique=True)
    alternate_contact = models.CharField(
        max_length=10, blank=True, null=True,
        validators=[RegexValidator(r'^\d{10}$', 'Enter a valid 10-digit mobile number.')]
    )
    address = models.TextField()
    pin_code = models.CharField(
        max_length=6,
        validators=[RegexValidator(r'^\d{6}$', 'Enter a valid 6-digit PIN code.')]
    )
    city = models.CharField(max_length=50)
    state = models.CharField(max_length=50)

    # ── Section C: Identification ──────────────────────────────────────────────
    ID_TYPES = [
        ('Aadhar',   'Aadhar'),
        ('PAN',      'PAN'),
        ('Passport', 'Passport'),
    ]
    id_type   = models.CharField(max_length=20, choices=ID_TYPES)
    id_number = models.CharField(max_length=20, default='', blank=False)

    INSURANCE_PROVIDERS = [
        ('Star Health', 'Star Health'),
        ('LIC',         'LIC'),
        ('HDFC Ergo',   'HDFC Ergo'),
        ('Bajaj Allianz', 'Bajaj Allianz'),
        ('New India Assurance', 'New India Assurance'),
        ('None',        'None'),
    ]
    insurance_provider      = models.CharField(max_length=50, choices=INSURANCE_PROVIDERS, default='None')
    insurance_policy_number = models.CharField(max_length=50, blank=True, null=True)

    # ── Metadata ───────────────────────────────────────────────────────────────
    created_at = models.DateTimeField(default=timezone.now)

    # ── Auto-assign patient_id on first save ───────────────────────────────────
    def save(self, *args, **kwargs):
        if not self.patient_id:
            now = timezone.now()
            date_str = now.strftime('%d%m%Y')
            prefix   = f'PT{date_str}'

            last = (
                Patient.objects.filter(patient_id__startswith=prefix)
                .order_by('patient_id')
                .last()
            )
            if last:
                try:
                    last_seq = int(last.patient_id[len(prefix):])
                except (ValueError, IndexError):
                    last_seq = 0
            else:
                last_seq = 0

            self.patient_id = f'{prefix}{last_seq + 1:05d}'

        super().save(*args, **kwargs)

    def __str__(self):
        return f"{self.patient_id} — {self.full_name}"

    class Meta:
        verbose_name        = 'Patient'
        verbose_name_plural = 'Patients'
        ordering            = ['-created_at']


class PatientMedicalHistory(models.Model):
    """
    Stores multiple medical visit records for a patient.

    Each row = ONE VISIT.
    """

    patient = models.ForeignKey(
        'Patient',
        on_delete=models.CASCADE,
        related_name='medical_histories'
    )

    visit_date = models.DateTimeField(default=timezone.now)
    doctor_notes = models.TextField(blank=True, null=True)

    CONDITIONS_CHOICES = [
        ('Diabetes', 'Diabetes'),
        ('Hypertension', 'Hypertension'),
        ('Heart Disease', 'Heart Disease'),
        ('Asthma', 'Asthma'),
        ('Thyroid Disorder', 'Thyroid Disorder'),
        ('Kidney Disease', 'Kidney Disease'),
        ('Liver Disease', 'Liver Disease'),
        ('Cancer', 'Cancer'),
        ('Stroke', 'Stroke'),
        ('Arthritis', 'Arthritis'),
        ('Mental Health Disorders', 'Mental Health Disorders'),
        ('Other', 'Other'),
        ('None', 'None'),
    ]

    medical_conditions = models.JSONField(default=list, blank=True)
    condition_details  = models.TextField(blank=True, null=True)

    current_medications = models.JSONField(default=list, blank=True)
    past_medications    = models.TextField(blank=True, null=True)

    surgeries      = models.TextField(blank=True, null=True)
    allergies      = models.TextField(blank=True, null=True)
    family_history = models.TextField(blank=True, null=True)

    SMOKING_CHOICES  = [('Never', 'Never'), ('Occasionally', 'Occasionally'), ('Regular', 'Regular')]
    ALCOHOL_CHOICES  = [('Never', 'Never'), ('Social', 'Social'), ('Regular', 'Regular')]
    EXERCISE_CHOICES = [('Daily', 'Daily'), ('Weekly', 'Weekly'), ('Rare', 'Rare')]

    smoking   = models.CharField(max_length=20, choices=SMOKING_CHOICES)
    alcohol   = models.CharField(max_length=20, choices=ALCOHOL_CHOICES)
    exercise  = models.CharField(max_length=20, choices=EXERCISE_CHOICES)
    height_cm = models.FloatField()
    weight_kg = models.FloatField()
    bmi       = models.FloatField(blank=True, null=True)

    confirm_information = models.BooleanField(default=False)
    agree_data_policy   = models.BooleanField(default=False)
    agree_research      = models.BooleanField(default=False)

    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.patient.patient_id} — Visit on {self.visit_date.strftime('%d-%m-%Y')}"

    class Meta:
        verbose_name        = 'Patient Medical History'
        verbose_name_plural = 'Patient Medical Histories'