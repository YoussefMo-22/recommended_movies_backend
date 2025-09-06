from rest_framework import serializers
from ..models import Movie
from ..serializers.GenreSerializer import GenreSerializer

class MovieSerializer(serializers.ModelSerializer):
    avg_rating = serializers.FloatField(read_only=True)
    ratings_count = serializers.IntegerField(read_only=True)
    genres = GenreSerializer(many=True, read_only=True)  # Show genre names instead of IDs

    class Meta:
        model = Movie
        fields = (
            "id",
            "movielens_id",
            "title",
            "year",
            "genres",         
            "imdb_id",
            "tmdb_id",
            "description",
            "poster_url",
            "avg_rating",
            "ratings_count",
        )
