import random
import string
import pyotp
from django.conf import settings


def generate_doctor_id():
    from doctors.models import Doctor
    last = Doctor.objects.order_by('-created_at').first()
    if last and last.doctor_id:
        try:
            num = int(last.doctor_id.split('-')[1]) + 1
        except (IndexError, ValueError):
            num = 1
    else:
        num = 1
    return f"D-{num:04d}"


def generate_temp_password(length=8):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length))


def generate_otp_secret():
    return pyotp.random_base32()


def generate_otp(otp_secret):
    totp = pyotp.TOTP(otp_secret, interval=600)
    return totp.now()


def verify_otp(otp_secret, otp_entered):
    totp = pyotp.TOTP(otp_secret, interval=600)
    return totp.verify(otp_entered, valid_window=1)


def generate_username(full_name):
    # "Dr. John Smith" → "john.smith" + random 2 digits if taken
    base = full_name.lower().strip()
    base = base.replace('dr.', '').replace('dr ', '').strip()
    parts = base.split()
    if len(parts) >= 2:
        username = f"{parts[0]}.{parts[-1]}"
    else:
        username = parts[0]
    # Remove any non-alphanumeric except dot
    username = ''.join(c for c in username if c.isalnum() or c == '.')

    # Check uniqueness
    from django.contrib.auth import get_user_model
    User = get_user_model()
    original = username
    counter = 1
    while User.objects.filter(username=username).exists():
        username = f"{original}{counter}"
        counter += 1
    return username


def send_credentials_sms(phone_number, full_name, username, temp_password, doctor_id):
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        message = (
            f"Hello Dr. {full_name},\n"
            f"Your Healthcare Portal credentials:\n"
            f"Doctor ID: {doctor_id}\n"
            f"Username: {username}\n"
            f"Password: {temp_password}\n"
            f"Please login and change your password immediately."
        )
        client.messages.create(
            body=message,
            from_=settings.TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return True
    except Exception as e:
        # Don't crash if Twilio not configured yet
        print(f"SMS not sent: {e}")
        return False


def send_otp_sms(phone_number, otp):
    try:
        from twilio.rest import Client
        client = Client(settings.TWILIO_ACCOUNT_SID, settings.TWILIO_AUTH_TOKEN)
        client.messages.create(
            body=f"Your Healthcare Portal OTP is: {otp}. Valid for 10 minutes.",
            from_=settings.TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return True
    except Exception as e:
        print(f"OTP SMS not sent: {e}")
        return False