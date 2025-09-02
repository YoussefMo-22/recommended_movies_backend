from django.urls import path, include
from rest_framework.routers import DefaultRouter
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)

from .views import (
    RegisterView,
    UserProfileView,
    MovieViewSet,
    RatingViewSet,
    RecommendationView,
    MeView
)
from api import views
from api.views import recommend_similar_movies
from .views import HybridRecommendationView
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView

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
    path("recommendations/", RecommendationView.as_view(), name="recommendations"),
    path("recommend/<int:user_id>/", views.recommend_movies, name="recommend_movies"), 
    path("movies/<int:movie_id>/recommendations/", recommend_similar_movies, name="recommend_similar_movies"),
    path("movies/<int:movie_id>/hybrid/", HybridRecommendationView.as_view(), name="hybrid-recommendations"),
]
