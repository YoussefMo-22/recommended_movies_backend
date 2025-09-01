from django.db.models import Avg, Count, Q
from django.contrib.auth.models import User
from rest_framework import viewsets, mixins, permissions, status, filters
from rest_framework.decorators import action
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import Movie, Rating, UserProfile
from .serializers import (
    MovieSerializer,
    RatingSerializer,
    RegisterSerializer,
    UserSerializer,
    UserProfileSerializer,
)
from .recommender.collaborative import recommend_for_user
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse

class RegisterView(APIView):
    """Registers a new user with password confirmation"""
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {"message": "User created", "user": UserSerializer(user).data},
            status=status.HTTP_201_CREATED,
        )


class UserProfileView(APIView):
    """View & update user profile"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        return Response(UserProfileSerializer(profile).data)

    def patch(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


from rest_framework.pagination import PageNumberPagination
from rest_framework.decorators import action

class MoviePagination(PageNumberPagination):
    page_size = 10  # default items per page
    page_size_query_param = "page_size"  # allow client to override
    max_page_size = 100


class MovieViewSet(viewsets.ReadOnlyModelViewSet):
    """Browse/search movies with filters, pagination & genre filtering"""
    serializer_class = MovieSerializer
    permission_classes = [permissions.AllowAny]
    pagination_class = MoviePagination
    filter_backends = [filters.OrderingFilter]
    ordering_fields = ["year", "avg_rating", "ratings_count"]
    ordering = ["id"]

    def get_queryset(self):
        qs = Movie.objects.annotate(
            avg_rating=Avg("ratings__rating"),
            ratings_count=Count("ratings"),
        )
        q = self.request.query_params.get("q")
        if q:
            qs = qs.filter(Q(title__icontains=q) | Q(genres__icontains=q))
        genre = self.request.query_params.get("genre")
        if genre:
            qs = qs.filter(genres__icontains=genre)
        year = self.request.query_params.get("year")
        if year and year.isdigit():
            qs = qs.filter(year=int(year))
        return qs

    @action(detail=False, methods=["get"], url_path="by-genre/(?P<genre>[^/.]+)")
    def by_genre(self, request, genre=None):
        """Get all movies in a specific genre with pagination"""
        qs = self.get_queryset().filter(genres__name__icontains=genre)
        page = self.paginate_queryset(qs)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)


class RatingViewSet(mixins.CreateModelMixin,
                    mixins.UpdateModelMixin,
                    mixins.DestroyModelMixin,
                    mixins.ListModelMixin,
                    viewsets.GenericViewSet):
    """Create/update/delete/list user ratings"""
    serializer_class = RatingSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Rating.objects.filter(user=self.request.user).select_related("movie")

    def perform_create(self, serializer):
        # Upsert behavior: if rating exists, update
        movie = serializer.validated_data["movie"]
        rating = serializer.validated_data["rating"]
        review = serializer.validated_data.get("review", "")
        obj, _ = Rating.objects.update_or_create(
            user=self.request.user,
            movie=movie,
            defaults={"rating": rating, "review": review},
        )
        serializer.instance = obj


class RecommendationView(APIView):
    """Return personalized recommendations (collaborative → content → popular)"""
    permission_classes = [permissions.IsAuthenticated]

    def get(self, request):
        user = request.user

        # Try collaborative filtering
        collab = recommend_for_user(user.id, top_k=20)
        if collab:
            strategy = "collaborative"
            movie_ids = [m.id for m in collab]
        else:
            # Fallback: content-based
            if collab:
                strategy = "content"
                movie_ids = [m.id for m in collab]
            else:
                # Final fallback: most popular movies
                popular = Movie.objects.annotate(
                    avg_rating=Avg("ratings__rating"),
                    ratings_count=Count("ratings"),
                ).filter(ratings_count__gte=10).order_by(
                    "-avg_rating", "-ratings_count"
                )[:20]
                data = MovieSerializer(popular, many=True).data
                return Response({"strategy": "popular", "results": data})

        # Fetch movie details while preserving order
        movies = list(
            Movie.objects.filter(id__in=movie_ids).annotate(
                avg_rating=Avg("ratings__rating"),
                ratings_count=Count("ratings"),
            )
        )
        movies_sorted = sorted(movies, key=lambda m: movie_ids.index(m.id))
        data = MovieSerializer(movies_sorted, many=True).data

        return Response({"strategy": strategy, "results": data})

from rest_framework.decorators import api_view
from rest_framework.response import Response
from api.serializers import MovieSerializer
from .recommender.collaborative import recommend_for_user


@extend_schema(
    summary="User-based Movie Recommendation",
    description="Recommend movies for a given user based on their ratings and preferences.",
    parameters=[
        OpenApiParameter(name="user_id", description="ID of the user", required=True, type=int),
    ],
    responses={
        200: OpenApiResponse(response=dict, description="List of recommended movies for the user"),
        404: OpenApiResponse(description="User not found"),
    },
)
@api_view(["GET"])
def recommend_movies(request, user_id):
    movies = recommend_for_user(user_id)
    serializer = MovieSerializer(movies, many=True)
    return Response(serializer.data)


from .recommender.content_based import get_content_based_recommendations
from api.serializers import MovieSerializer
@extend_schema(
    summary="Movie-based Similar Recommendations",
    description="Get a list of movies similar to the given movie based on content similarity.",
    parameters=[
        OpenApiParameter(name="movie_id", description="ID of the reference movie", required=True, type=int),
    ],
    responses={
        200: OpenApiResponse(response=dict, description="List of similar movies"),
        404: OpenApiResponse(description="Movie not found"),
    },
)
@api_view(["GET"])
def recommend_similar_movies(request, movie_id):
    recommendations = get_content_based_recommendations(movie_id)
    serializer = MovieSerializer(recommendations, many=True)
    return Response(serializer.data)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from api.models import Movie, Rating

@extend_schema(
    summary="Hybrid Recommendation (Content + Collaborative)",
    description="Generate hybrid recommendations by combining collaborative filtering and content-based methods.",
    parameters=[
        OpenApiParameter(name="movie_id", description="ID of the movie for hybrid recommendations", required=True, type=int),
    ],
    responses={
        200: OpenApiResponse(response=dict, description="List of hybrid recommended movies"),
        404: OpenApiResponse(description="Movie not found"),
    },
)
class HybridRecommendationView(APIView):
    """
    Hybrid recommender: combines collaborative filtering (ratings-based)
    and content-based (tf-idf on genres+description) into a weighted score.
    """

    def get(self, request, movie_id):
        # ensure movie_id is an int
        movie_id = int(movie_id)

        # 1) target movie
        target_movie = get_object_or_404(Movie, id=movie_id)

        # ------------------------------
        # COLLABORATIVE FILTERING PART
        # ------------------------------
        ratings_qs = Rating.objects.all().values("user_id", "movie_id", "rating")
        ratings_df = pd.DataFrame(list(ratings_qs))

        collaborative_scores = pd.Series(dtype=float)

        if not ratings_df.empty:
            # pivot table: users x movies (rows: users, cols: movie_ids)
            pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)

            # Only compute similarity if our movie appears in the pivot columns
            if movie_id in pivot.columns:
                # item-based approach: similarity between movies (cosine of movie rating vectors)
                # pivot.T -> rows = movies, cols = users
                movie_item_matrix = pivot.T.values  # shape (n_movies, n_users)
                # compute cosine similarity between movies
                sim_matrix = cosine_similarity(movie_item_matrix)
                sim_df = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)

                # get similarity series for the target movie
                # (this is a pandas Series indexed by movie_id)
                collaborative_scores = sim_df[movie_id].sort_values(ascending=False)

        # ------------------------------
        # CONTENT-BASED FILTERING PART
        # ------------------------------
        movies_qs = Movie.objects.all().values("id", "title", "genres", "description")
        movies_df = pd.DataFrame(list(movies_qs))

        content_scores = pd.Series(dtype=float)

        if not movies_df.empty and movie_id in movies_df["id"].values:
            # Ensure genres & description are strings (handle M2M vs string)
            # If genres is a pipe-separated string field in DB, keep it;
            # If genres were normalized to M2M earlier, the `.values()` will contain whatever you stored.
            movies_df["genres"] = movies_df["genres"].fillna("").astype(str)
            movies_df["description"] = movies_df["description"].fillna("").astype(str)

            # Combine genres + description for TF-IDF
            movies_df["content"] = (movies_df["genres"].str.replace("|", " ") + " " + movies_df["description"]).str.strip()

            # Build TF-IDF over the content
            tfidf = TfidfVectorizer(stop_words="english")
            tfidf_matrix = tfidf.fit_transform(movies_df["content"])  # shape (n_movies, n_features)

            # cosine similarity between movies based on content
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)  # shape (n_movies, n_movies)

            # map movie_id -> row index in movies_df
            id_to_index = pd.Series(movies_df.index, index=movies_df["id"])

            # Handle duplicates safely
            idx_val = id_to_index.loc[movie_id]
            if isinstance(idx_val, pd.Series):
                idx_val = idx_val.iloc[0]
            idx = int(idx_val)

            row = cosine_sim[idx]

            # Build list of (row_index, score) and ensure scores are Python floats
            sim_scores = [(i, float(np.squeeze(score))) for i, score in enumerate(row)]

            # sort by score desc, skip self (highest will be self = 1.0)
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

            # take top N (skip first entry which is itself)
            top_sim = sim_scores[1:11]  # top 10 similar movies

            # map back to movie IDs
            content_scores = pd.Series({int(movies_df.iloc[i].id): score for i, score in top_sim})

        # ------------------------------
        # HYBRID COMBINATION
        # ------------------------------
        # Normalize safely (avoid divide by zero and ambiguous numpy truth checks)
        if not collaborative_scores.empty:
            max_val = float(np.max(collaborative_scores.values)) if getattr(collaborative_scores, "values", None) is not None else float(collaborative_scores.max())
            if max_val == 0:
                max_val = 1.0
            collaborative_scores = collaborative_scores.astype(float) / max_val

        if not content_scores.empty:
            max_val = float(np.max(content_scores.values)) if getattr(content_scores, "values", None) is not None else float(content_scores.max())
            if max_val == 0:
                max_val = 1.0
            content_scores = content_scores.astype(float) / max_val

        # Ensure both are float Series and have proper index types
        collaborative_scores = collaborative_scores.astype(float)
        content_scores = content_scores.astype(float)

        # Weighted combination (configurable via query params if you want)
        weight_cf = float(request.query_params.get("w_cf", 0.6))
        weight_cb = float(request.query_params.get("w_cb", 0.4))

        hybrid_scores = (weight_cf * collaborative_scores).add(weight_cb * content_scores, fill_value=0.0)

        # Remove target movie if present
        if movie_id in hybrid_scores.index:
            hybrid_scores = hybrid_scores.drop(movie_id)

        if hybrid_scores.empty:
            # fallback: popular movies by average rating
            popular = Movie.objects.annotate(avg_rating=pd.NamedAgg("ratings__rating", "mean"),
                                             ratings_count=pd.NamedAgg("ratings__rating", "count")) \
                                   .filter(ratings_count__gte=5) \
                                   .order_by("-avg_rating")[:10]
            recommended = list(popular)
        else:
            # Top-N by hybrid score
            top_ids = hybrid_scores.sort_values(ascending=False).head(10).index.tolist()

            # Fetch movies and preserve the order from top_ids
            recommended_qs = Movie.objects.filter(id__in=top_ids)
            recommended = list(recommended_qs)
            recommended.sort(key=lambda m: top_ids.index(m.id))

        # ------------------------------
        # Build JSON-serializable result (serialize genres properly)
        # ------------------------------
        result = []
        for m in recommended:
            # If genres is ManyToMany on the model, fetch names; otherwise handle string field
            genres_list = []
            try:
                # If Movie.genres is ManyToMany, getattr returns a manager
                gm = getattr(m, "genres")
                if hasattr(gm, "all"):
                    genres_list = [g.name for g in gm.all()]
                else:
                    # If it's a pipe-separated string
                    genres_str = str(gm) if gm is not None else ""
                    genres_list = [g for g in genres_str.split("|") if g]
            except Exception:
                # fallback: empty
                genres_list = []

            result.append({
                "id": m.id,
                "movielens_id": getattr(m, "movielens_id", None),
                "title": m.title,
                "year": getattr(m, "year", None),
                "genres": genres_list,
                "imdb_id": getattr(m, "imdb_id", ""),
                "tmdb_id": getattr(m, "tmdb_id", ""),
                "description": getattr(m, "description", "") or "",
                "poster_url": getattr(m, "poster_url", "") or "",
            })

        return Response({"movie": target_movie.title, "recommendations": result})
