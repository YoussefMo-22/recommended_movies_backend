from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import permissions
from django.shortcuts import get_object_or_404

from ..models import Movie
from ..services.HybridService import HybridRecommendationService


class HybridRecommendationView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request, movie_id):
        movie_id = int(movie_id)
        target_movie = get_object_or_404(Movie, id=movie_id)

        weight_cf = float(request.query_params.get("w_cf", 0.6))
        weight_cb = float(request.query_params.get("w_cb", 0.4))

        recommended = HybridRecommendationService.get_hybrid_recommendations(
            movie_id=movie_id,
            weight_cf=weight_cf,
            weight_cb=weight_cb,
        )

        DEFAULT_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"

        result = []
        for m in recommended:
            try:
                gm = getattr(m, "genres", None)
                if hasattr(gm, "all"):
                    genres_list = [{"id": g.id, "name": g.name} for g in gm.all()]
                else:
                    genres_str = str(gm) if gm else ""
                    genres_list = [{"name": g} for g in genres_str.split("|") if g]
            except Exception:
                genres_list = []

            result.append({
                "id": m.id,
                "title": m.title,
                "year": getattr(m, "year", None),
                "genres": genres_list,
                "imdb_id": getattr(m, "imdb_id", "") or "",
                "tmdb_id": getattr(m, "tmdb_id", "") or "",
                "description": getattr(m, "description", "") or "",
                "poster_url": getattr(m, "poster_url", "") or DEFAULT_POSTER,
            })

        return Response({"movie": target_movie.title, "recommendations": result})
