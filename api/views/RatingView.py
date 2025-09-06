from rest_framework import viewsets, mixins, status
from rest_framework.permissions import IsAuthenticated
from rest_framework.decorators import action
from rest_framework.response import Response

from ..serializers.RatingSerializer import RatingSerializer
from ..services.RatingService import RatingService


class RatingViewSet(
    mixins.CreateModelMixin,
    mixins.UpdateModelMixin,
    mixins.DestroyModelMixin,
    mixins.ListModelMixin,
    viewsets.GenericViewSet,
):
    serializer_class = RatingSerializer
    permission_classes = [IsAuthenticated]

    def get_queryset(self):
        return RatingService.get_user_ratings(self.request.user)

    def perform_create(self, serializer):
        movie = serializer.validated_data["movie"]
        rating = serializer.validated_data["rating"]
        review = serializer.validated_data.get("review", "")
        obj, _ = RatingService.create_or_update_rating(
            user=self.request.user, movie=movie, rating_value=rating, review=review
        )
        serializer.instance = obj

    @action(detail=False, methods=["get", "post", "patch"], url_path="movie/(?P<movie_id>[^/.]+)")
    def user_movie_rating(self, request, movie_id=None):
        movie = RatingService.get_movie(movie_id)
        rating_obj = RatingService.get_user_movie_rating(request.user, movie)

        if request.method == "GET":
            if rating_obj:
                return Response(self.get_serializer(rating_obj).data)
            return Response({"rating": None})

        rating_value = request.data.get("rating")
        review = request.data.get("review", "")
        if rating_value is None:
            return Response(
                {"error": "Rating value is required"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        rating_obj, created = RatingService.create_or_update_rating(
            user=request.user, movie=movie, rating_value=rating_value, review=review
        )

        return Response(
            self.get_serializer(rating_obj).data,
            status=status.HTTP_201_CREATED if created else status.HTTP_200_OK,
        )
