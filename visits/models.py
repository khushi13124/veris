from django.db import models
from datetime import date


STATUS_CHOICES = [
    ('waiting',    'Waiting'),
    ('with_doctor','With Doctor'),
    ('done',       'Done'),
]


class Visit(models.Model):
    patient = models.ForeignKey(
        'patients.Patient',
        on_delete=models.CASCADE,
        related_name='visits'
    )
    doctor = models.ForeignKey(
        'doctors.Doctor',
        on_delete=models.CASCADE,
        related_name='visits'
    )

    # ✅ DateField (not DateTimeField) so visit_date=today works perfectly
    visit_date       = models.DateField(default=date.today)
    created_at       = models.DateTimeField(auto_now_add=True)

    remarks          = models.TextField(blank=True)
    medication       = models.TextField(blank=True)
    diagnosis_notes  = models.TextField(blank=True)
    ai_differential  = models.TextField(blank=True)
    
    entered_symptoms = models.TextField(blank=True, default='')
    selected_diagnosis = models.TextField(blank=True, default='')
    ai_medical       = models.TextField(blank=True)
    next_appointment = models.DateField(null=True, blank=True)

    # ── Step 1 additions ──────────────────────────────────────────────────────
    status       = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='waiting',
    )
    checked_in_at = models.DateTimeField(
        null=True, blank=True,
        help_text="Set automatically when the visit is created via check-in."
    )
    # ─────────────────────────────────────────────────────────────────────────

    class Meta:
        ordering = ['-visit_date', '-created_at']

    def __str__(self):
        return f"Visit – {self.patient} by Dr. {self.doctor} on {self.visit_date}"

    # ── Convenience helpers ───────────────────────────────────────────────────
    @property
    def is_today(self):
        return self.visit_date == date.today()

    @property
    def status_badge_class(self):
        """Returns a Bootstrap / Tailwind-friendly class string per status."""
        return {
            'waiting':     'badge-waiting',
            'with_doctor': 'badge-with-doctor',
            'done':        'badge-done',
        }.get(self.status, '')
        