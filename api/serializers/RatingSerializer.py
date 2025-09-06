from rest_framework import serializers
from ..models import Rating, Movie
from ..serializers import MovieSerializer
from ..serializers import UserSerializer


class RatingSerializer(serializers.ModelSerializer):
    movie = MovieSerializer(read_only=True)
    movie_id = serializers.PrimaryKeyRelatedField(
        queryset=Movie.objects.all(), source="movie", write_only=True
    )
    user = UserSerializer(read_only=True)

    class Meta:
        model = Rating
        fields = (
            "id",
            "movie",
            "movie_id",
            "user",
            "rating",
            "review",
            "created_at",
            "updated_at",
        )
        read_only_fields = ("created_at", "updated_at")
