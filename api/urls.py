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

    # User profile
    path("user/profile/", UserProfileView.as_view(), name="user-profile"),

    # Recommendations
    path("recommendations/", RecommendationView.as_view(), name="recommendations"),
    path("recommend/<int:user_id>/", views.recommend_movies, name="recommend_movies"), 
    path("movies/<int:movie_id>/recommendations/", recommend_similar_movies, name="recommend_similar_movies"),
    path("movies/<int:movie_id>/hybrid/", HybridRecommendationView.as_view(), name="hybrid-recommendations"),
    
    # API schema
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),

    # Optional UI:
    path("api/schema/swagger-ui/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/schema/redoc/", SpectacularRedocView.as_view(url_name="schema"), name="redoc"),
    
    # Router (movies & ratings)
    path("", include(router.urls)),
]
