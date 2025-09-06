# movies/services/movie_service.py
from django.db.models import Avg, Count
from ..models import Movie

class MovieService:
    def get_movies(self, filters=None, ordering=None):
        """
        Return queryset of movies with applied filters and ordering.
        """
        qs = Movie.objects.annotate(
            avg_rating=Avg("ratings__rating"),
            ratings_count=Count("ratings"),
        )

        if filters:
            title = filters.get("title")
            if title:
                qs = qs.filter(title__icontains=title)

            genre = filters.get("genre")
            if genre:
                qs = qs.filter(genres__name__icontains=genre)

            year = filters.get("year")
            if year and str(year).isdigit():
                qs = qs.filter(year=int(year))

        if ordering:
            qs = qs.order_by(*ordering)

        return qs

    def get_movies_by_genre(self, genre):
        return self.get_movies(filters={"genre": genre})
