import xlrd
from django.contrib import messages
from django.db.models import Count
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils.datastructures import MultiValueDictKeyError
from django.views.generic import ListView, CreateView, DeleteView, UpdateView

from django.contrib.auth.mixins import PermissionRequiredMixin

from requirements.models import Requirement
from .models import Project


def import_view(request):
    try:
        xlsx_file = request.FILES.get('xlsx_file', None)
    except MultiValueDictKeyError:
        messages.success(request, f'please specify a xlsx file')
        return redirect('projects_list_view')
    if request.method == 'POST' and request.FILES['xlsx_file']:
        del request.FILES['xlsx_file']
        wb = xlrd.open_workbook(file_contents=xlsx_file.read())
        sheet = wb.sheet_by_index(0)
        for i in range(1, sheet.nrows):
            project_name = sheet.cell_value(i, 0)
            requirement = sheet.cell_value(i, 1)
            project_obj, is_project_created = Project.objects.get_or_create(name=project_name)
            requirement_obj = Requirement.objects.create(
                title=f'REQ {i} {project_obj.name}',
                number=f'REQ {i}',
                description=requirement,
                requirement_type=Requirement.TYPE_FUNCTIONAL,
                project=project_obj
            )
        messages.success(request, f'project successfully inserted')
        return redirect('projects_list_view')
    return render(request, 'projects/import.html')


class ProjectsList(ListView):
    model = Project
    context_object_name = 'projects'
    template_name = 'projects/list.html'

    def get_queryset(self):
        query_set = super(ProjectsList, self).get_queryset()
        query_set = query_set.annotate(smells_count=Count('requirements__smells'))
        return query_set


class ProjectCreate(CreateView):
    model = Project
    template_name = 'projects/create.html'
    fields = ('name', 'description')

    def get_success_url(self):
        messages.success(self.request, f"project {self.object.name} created!")
        return reverse('projects_list_view')


class ProjectDelete(PermissionRequiredMixin, DeleteView):
    permission_required = 'requirements.delete_project'
    model = Project
    template_name = 'projects/delete.html'

    def get_success_url(self):
        messages.success(self.request, f"project {self.object.name} deleted!")
        return reverse('projects_list_view')


class ProjectUpdate(PermissionRequiredMixin, UpdateView):
    permission_required = 'requirements.change_project'
    model = Project
    fields = ('name', 'description')
    template_name = 'projects/update.html'

    def get_success_url(self):
        messages.success(self.request, f"project {self.object.name} edited!")
        return reverse('projects_list_view')
