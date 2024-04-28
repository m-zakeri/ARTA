from rest_framework import serializers

from .models import Comment


class CommentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Comment
        fields = ('content', 'finding')

    def create(self, validated_data):
        return Comment.objects.create(
            content=validated_data['content'],
            finding=validated_data['finding'],
            user=self.context['request'].user
        )
