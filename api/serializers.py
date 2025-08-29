from rest_framework import serializers
from django.contrib.auth.models import User
from .models import Movie, Rating, UserProfile, Genre


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, min_length=8)
    password2 = serializers.CharField(write_only=True, min_length=8)

    class Meta:
        model = User
        fields = ("username", "email", "password", "password2")

    def validate(self, data):
        if data["password"] != data["password2"]:
            raise serializers.ValidationError("Passwords do not match.")
        return data

    def create(self, validated_data):
        validated_data.pop("password2")
        return User.objects.create_user(**validated_data)


class UserProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UserProfile
        fields = ("favorite_genres", "bio", "avatar_url")


class UserSerializer(serializers.ModelSerializer):
    profile = UserProfileSerializer(read_only=True)

    class Meta:
        model = User
        fields = ("id", "username", "email", "profile")


# ✅ Genre serializer
class GenreSerializer(serializers.ModelSerializer):
    class Meta:
        model = Genre
        fields = ("id", "name")


class MovieSerializer(serializers.ModelSerializer):
    avg_rating = serializers.FloatField(read_only=True)
    ratings_count = serializers.IntegerField(read_only=True)
    genres = GenreSerializer(many=True, read_only=True)  # ✅ Show genre names instead of IDs

    class Meta:
        model = Movie
        fields = (
            "id",
            "movielens_id",
            "title",
            "year",
            "genres",         # will now return [{"id": 1, "name": "Action"}, ...]
            "imdb_id",
            "tmdb_id",
            "description",
            "poster_url",
            "avg_rating",
            "ratings_count",
        )




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
