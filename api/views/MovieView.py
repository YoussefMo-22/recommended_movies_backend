# movies/views/movie_views.py
from rest_framework import viewsets, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.pagination import PageNumberPagination

from ..serializers.MovieSerializer import MovieSerializer
from ..services.MovieService import MovieService

class MoviePagination(PageNumberPagination):
    page_size = 24
    page_size_query_param = "page_size"
    max_page_size = 100


class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    serializer_class = MovieSerializer
    permission_classes = [AllowAny]
    pagination_class = MoviePagination
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["year", "avg_rating", "ratings_count"]
    ordering = ["id"]

    movie_service = MovieService()

    def get_queryset(self):
        filters_dict = {
            "title": self.request.query_params.get("title"),
            "genre": self.request.query_params.get("genre"),
            "year": self.request.query_params.get("year"),
        }
        return self.movie_service.get_movies(filters=filters_dict, ordering=self.ordering)

    @action(detail=False, methods=["get"], url_path="by-genre/(?P<genre>[^/.]+)")
    def by_genre(self, request, genre=None):
        qs = self.movie_service.get_movies_by_genre(genre)
        page = self.paginate_queryset(qs)
        if page:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
