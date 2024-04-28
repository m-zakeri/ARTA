from django.urls import path

from . import views

urlpatterns = [
    path('detail/<int:pk>', views.RequirementDetailView.as_view(), name='requirement_detail_view'),
    path('update/<int:pk>', views.RequirementUpdateView.as_view(), name='requirement_update_view'),
    path('delete/<int:pk>', views.RequirementDeleteView.as_view(), name='requirement_delete_view'),
    path('list/', views.RequirementListView.as_view(), name='requirement_list_view'),
    path('create/', views.RequirementCreateView.as_view(), name='requirement_create_view'),
]
