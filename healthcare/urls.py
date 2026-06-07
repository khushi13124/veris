from django.contrib import admin
from django.urls import path, include
from django.shortcuts import redirect
from django.conf import settings
from django.conf.urls.static import static


def root_redirect(request):
    return redirect('/login/')


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', root_redirect),
    path('', include('accounts.urls')),
    path('', include('symcat.urls')),
    path('', include('patients.urls')),
    path('doctors/', include('doctors.urls')),
    path('visits/', include('visits.urls')),  
    path('symcat/', include('symcat.urls')),  
    path('rsna/', include('rsna.urls')),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)