from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.views.generic import CreateView
from django.views.generic.base import TemplateView


class IndexView(LoginRequiredMixin, TemplateView):
    template_name = 'user/index.html'
    login_url = reverse_lazy('login-view')


class SignupView(CreateView):
    form_class = UserCreationForm
    success_url = reverse_lazy('login-view')
    template_name = 'user/signup.html'
