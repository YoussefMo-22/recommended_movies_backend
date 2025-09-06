from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

from api.views.UserView import RegisterView, MeView

from api.views.MovieView import MovieViewSet
from api.views.RatingView import RatingViewSet
from api.views.RecommenationView import recommend_movies, recommend_similar_movies
from api.views.HybridView import HybridRecommendationView
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

# Router for viewsets
router = DefaultRouter()
router.register(r"movies", MovieViewSet, basename="movie")
router.register(r"ratings", RatingViewSet, basename="rating")

urlpatterns = [
    # Auth
    path("auth/register/", RegisterView.as_view(), name="register"),
    path("auth/login/", TokenObtainPairView.as_view(), name="login"),
    path("auth/refresh/", TokenRefreshView.as_view(), name="token_refresh"),
    path("auth/me/", MeView.as_view(), name="me"),

    # Recommendations
    path("recommend/<int:user_id>/", recommend_movies, name="recommend_movies"), 
    path("movies/<int:movie_id>/recommendations/", recommend_similar_movies, name="recommend_similar_movies"),
    path("movies/<int:movie_id>/hybrid/", HybridRecommendationView.as_view(), name="hybrid-recommendations"),

    # API schema & docs
    path("schema/", SpectacularAPIView.as_view(), name="schema"),
    path("schema/swagger-ui/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("schema/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),

    # Router endpoints (movies, ratings, etc.)
    path("", include(router.urls)),
]
