from django.db import models
from django.conf import settings


class Doctor(models.Model):
    DEPARTMENT_CHOICES = [
        ('Cardiology', 'Cardiology'),
        ('Neurology', 'Neurology'),
        ('Orthopedics', 'Orthopedics'),
        ('Pediatrics', 'Pediatrics'),
        ('Dermatology', 'Dermatology'),
        ('Radiology', 'Radiology'),
        ('General Medicine', 'General Medicine'),
        ('Surgery', 'Surgery'),
        ('Gynecology', 'Gynecology'),
        ('Psychiatry', 'Psychiatry'),
        ('ENT', 'ENT'),
        ('Ophthalmology', 'Ophthalmology'),
        ('Oncology', 'Oncology'),
        ('Urology', 'Urology'),
        ('Nephrology', 'Nephrology'),
        ('Other', 'Other'),
    ]

    DAYS_CHOICES = [
        ('Mon-Fri', 'Monday to Friday'),
        ('Mon-Sat', 'Monday to Saturday'),
        ('Tue-Sat', 'Tuesday to Saturday'),
        ('Mon-Wed-Fri', 'Monday, Wednesday, Friday'),
        ('Tue-Thu-Sat', 'Tuesday, Thursday, Saturday'),
        ('Weekends', 'Weekends Only'),
        ('All Days', 'All Days'),
    ]

    SHIFT_CHOICES = [
        ('Morning (8AM - 2PM)', 'Morning (8AM - 2PM)'),
        ('Afternoon (2PM - 8PM)', 'Afternoon (2PM - 8PM)'),
        ('Evening (4PM - 10PM)', 'Evening (4PM - 10PM)'),
        ('Night (10PM - 6AM)', 'Night (10PM - 6AM)'),
        ('Full Day (8AM - 8PM)', 'Full Day (8AM - 8PM)'),
    ]

    # Identity
    doctor_id       = models.CharField(max_length=10, unique=True, editable=False)
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='doctor_profile')
    full_name       = models.CharField(max_length=150)
    profile_photo   = models.ImageField(upload_to='doctors/photos/', blank=True, null=True)

    # Professional
    specialization  = models.CharField(max_length=100)
    department      = models.CharField(max_length=100, choices=DEPARTMENT_CHOICES)
    qualification   = models.CharField(max_length=200)
    years_of_experience = models.PositiveIntegerField(default=0)

    # Contact
    contact_number  = models.CharField(max_length=15)
    email           = models.EmailField()

    # Schedule
    chamber_number  = models.CharField(max_length=20, blank=True)
    days_available  = models.CharField(max_length=50, choices=DAYS_CHOICES)
    shift_timings   = models.CharField(max_length=50, choices=SHIFT_CHOICES)

    # Patients
    assigned_patients = models.ManyToManyField(
        'patients.Patient',
        blank=True,
        related_name='assigned_doctors'
    )

    # Bio
    bio             = models.TextField(blank=True)

    # Auth flow
    is_first_login  = models.BooleanField(default=True)
    temp_password   = models.CharField(max_length=20, blank=True)
    otp_secret      = models.CharField(max_length=32, blank=True)

    # Timestamp
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.doctor_id} — {self.full_name}"

    def save(self, *args, **kwargs):
        # Auto-generate doctor_id like D-0001
        if not self.doctor_id:
            last = Doctor.objects.order_by('-created_at').first()
            if last and last.doctor_id:
                try:
                    num = int(last.doctor_id.split('-')[1]) + 1
                except (IndexError, ValueError):
                    num = 1
            else:
                num = 1
            self.doctor_id = f"D-{num:04d}"
        super().save(*args, **kwargs)