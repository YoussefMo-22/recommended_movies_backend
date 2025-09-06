from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.response import Response

from ..serializers import MovieSerializer
from ..recommender.collaborative import recommend_for_user
from ..recommender.content_based import get_content_based_recommendations

@api_view(["GET"]) 
@permission_classes([IsAuthenticated]) 
def recommend_movies(request, user_id):
    movies = recommend_for_user(user_id)
    serializer = MovieSerializer(movies, many=True)
    return Response(serializer.data)

@api_view(["GET"])
@permission_classes([AllowAny])
def recommend_similar_movies(request, movie_id):
    recommendations = get_content_based_recommendations(movie_id)
    return Response(MovieSerializer(recommendations, many=True).data)