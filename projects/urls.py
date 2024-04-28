from django.urls import path

from . import views

urlpatterns = [
    path('list', views.ProjectsList.as_view(), name='projects_list_view'),
    path('import', views.import_view, name='projects_import_view'),
    path('create', views.ProjectCreate.as_view(), name='projects_create_view'),
    path('delete/<int:pk>', views.ProjectDelete.as_view(), name='projects_delete_view'),
    path('update/<int:pk>', views.ProjectUpdate.as_view(), name='projects_update_view'),
]
