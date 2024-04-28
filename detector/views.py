from django.views.generic import ListView
from rest_framework.generics import CreateAPIView

from .models import Comment
from .serializers import CommentSerializer


class CommentListView(ListView):
    model = Comment
    template_name = 'detector/comments_list.html'
    context_object_name = 'comments'

    def get_queryset(self):
        return Comment.objects.filter(finding_id=self.kwargs.get('finding_pk')).order_by('created_date')


class CommentCreateView(CreateAPIView):
    serializer_class = CommentSerializer
