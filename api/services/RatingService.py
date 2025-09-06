from django.shortcuts import get_object_or_404
from ..models import Rating, Movie


class RatingService:
    @staticmethod
    def get_user_ratings(user):
        return Rating.objects.filter(user=user).select_related("movie")

    @staticmethod
    def create_or_update_rating(user, movie, rating_value, review=""):
        rating_obj, created = Rating.objects.update_or_create(
            user=user,
            movie=movie,
            defaults={"rating": rating_value, "review": review},
        )
        return rating_obj, created

    @staticmethod
    def get_movie(movie_id):
        return get_object_or_404(Movie, id=movie_id)

    @staticmethod
    def get_user_movie_rating(user, movie):
        return Rating.objects.filter(user=user, movie=movie).first()
