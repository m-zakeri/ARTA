from django.urls import path

from . import views

urlpatterns = [
    path('metrics/', views.MetricView.as_view(), name='metrics_view'),
    path('requirement_size_fig', views.requirement_size_view, name='size_fig_view'),
    path('smell_rel_fig', views.size_smell_rel_view, name='smell_rel_fig_view'),
    path('tree_map_view/<str:project_name>', views.tree_map_view, name='tree_map_view'),
    path('get_excel/<str:by_expert>', views.get_excel, name='excel_view'),
]
