from django.contrib import messages
from django.db.models import Count
from django.urls import reverse
from django.views.generic import DetailView, ListView, CreateView, UpdateView, DeleteView

from django.contrib.auth.mixins import PermissionRequiredMixin, LoginRequiredMixin, AccessMixin

from .models import Requirement


class RequirementListView(LoginRequiredMixin, ListView):
    # login_url = '/user/login/'
    # redirect_field_name = 'list.html'

    model = Requirement
    context_object_name = 'requirements'
    template_name = 'requirements/list.html'

    def get_queryset(self):
        queryset = Requirement.objects.annotate(smell_count=Count('smells'))
        params = {}
        project_id = self.request.GET.get('project_id')
        if project_id:
            params['project_id'] = int(project_id)
        type_ = self.request.GET.get('type')
        if type_:
            params['requirement_type'] = type_
        have_smell = self.request.GET.get('have_smell')
        if have_smell == '1':
            params['smell_count__gte'] = 1
        return queryset.filter(**params)


class RequirementDetailView(DetailView):
    model = Requirement
    context_object_name = 'requirement'
    template_name = 'requirements/detail.html'

    def get_context_data(self, **kwargs):
        context = super(RequirementDetailView, self).get_context_data(**kwargs)
        context['word_size'] = len(self.get_object().description.split())
        context['smells'] = self.get_object().smells \
            .annotate(smell_count=Count('smell__title')).values('smell__title', 'smell__description')
        return context


class RequirementCreateView(CreateView):

    model = Requirement
    template_name = 'requirements/create.html'
    fields = ('title', 'number', 'description', 'requirement_type', 'project')

    def get_success_url(self):
        return reverse('requirement_detail_view', kwargs={'pk': self.object.id})


class RequirementUpdateView(PermissionRequiredMixin, UpdateView):
    permission_required = 'requirements.change_requirement'
    model = Requirement
    template_name = 'requirements/update.html'
    fields = ('title', 'number', 'description', 'requirement_type', 'project')

    def get_success_url(self):
        return reverse('requirement_detail_view', kwargs={'pk': self.object.id})


class RequirementDeleteView(PermissionRequiredMixin, DeleteView):
    """
    https://docs.djangoproject.com/en/3.1/topics/auth/default/#the-permissionrequiredmixin-mixin
    """
    permission_required = 'requirements.delete_requirement'
    model = Requirement
    template_name = 'requirements/delete.html'

    def get_success_url(self):
        messages.success(self.request, f"requirement {self.object.title} deleted!")
        return reverse('requirement_list_view')
