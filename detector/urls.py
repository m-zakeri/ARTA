from django.urls import path

from . import views

urlpatterns = [
    path('comments/list/', views.CommentListView.as_view(), name='comments_list_view'),
    path('comments/list/<int:finding_pk>', views.CommentListView.as_view(), name='comments_list_view'),
    path('comments/create', views.CommentCreateView.as_view(), name='comments_create_view'),
    path('comments/create/<int:finding_pk>', views.CommentCreateView.as_view(), name='comments_create_view'),
]
