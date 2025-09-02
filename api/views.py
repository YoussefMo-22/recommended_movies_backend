# Standard library
import logging

# Django
from django.db.models import Avg, Count
from django.shortcuts import get_object_or_404
from django.contrib.auth import get_user_model
from django.core.cache import cache

# DRF
from rest_framework import viewsets, mixins, permissions, status, filters
from rest_framework.views import APIView
from rest_framework.decorators import action, api_view, permission_classes
from rest_framework.response import Response
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.pagination import PageNumberPagination

# Third-party
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from drf_spectacular.utils import extend_schema, OpenApiParameter, OpenApiResponse

# Local apps
from .models import Movie, Rating, UserProfile
from .serializers import (
    MovieSerializer,
    RatingSerializer,
    RegisterSerializer,
    UserSerializer,
    UserProfileSerializer,
)
from .recommender.collaborative import recommend_for_user
from .recommender.content_based import get_content_based_recommendations


logger = logging.getLogger(__name__)

# TF-IDF cache keys
TFIDF_CACHE_KEY = "movies:tfidf_matrix"
TFIDF_INDEX_KEY = "movies:tfidf_index"
TFIDF_IDS_KEY = "movies:tfidf_ids"
TFIDF_VECTORIZER_KEY = "movies:tfidf_vectorizer"
TFIDF_CACHE_TTL = 60 * 60 * 6  # 6 hours

COLLAB_TOP_MOVIES = 3000
HYBRID_TOP_K = 10


User = get_user_model()

class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        serializer = RegisterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return Response(
            {"message": "User created", "user": UserSerializer(user).data},
            status=status.HTTP_201_CREATED,
        )


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(UserSerializer(request.user).data)


class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        return Response(UserProfileSerializer(profile).data)

    def patch(self, request):
        profile, _ = UserProfile.objects.get_or_create(user=request.user)
        serializer = UserProfileSerializer(profile, data=request.data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data)


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

    def get_queryset(self):
        qs = Movie.objects.annotate(
            avg_rating=Avg("ratings__rating"),
            ratings_count=Count("ratings"),
        )

        title = self.request.query_params.get("title")
        if title:
            qs = qs.filter(title__icontains=title)

        genre = self.request.query_params.get("genre")
        if genre:
            qs = qs.filter(genres__icontains=genre)

        year = self.request.query_params.get("year")
        if year and year.isdigit():
            qs = qs.filter(year=int(year))

        return qs

    @action(detail=False, methods=["get"], url_path="by-genre/(?P<genre>[^/.]+)")
    def by_genre(self, request, genre=None):
        qs = self.get_queryset().filter(genres__name__icontains=genre)
        page = self.paginate_queryset(qs)
        if page:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)
        serializer = self.get_serializer(qs, many=True)
        return Response(serializer.data)


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
        return Rating.objects.filter(user=self.request.user).select_related("movie")

    def perform_create(self, serializer):
        movie = serializer.validated_data["movie"]
        rating = serializer.validated_data["rating"]
        review = serializer.validated_data.get("review", "")
        obj, _ = Rating.objects.update_or_create(
            user=self.request.user, movie=movie, defaults={"rating": rating, "review": review}
        )
        serializer.instance = obj

    @action(detail=False, methods=["get", "post", "patch"], url_path="movie/(?P<movie_id>[^/.]+)")
    def user_movie_rating(self, request, movie_id=None):
        movie = get_object_or_404(Movie, id=movie_id)
        rating_obj = Rating.objects.filter(user=request.user, movie=movie).first()

        if request.method == "GET":
            if rating_obj:
                return Response(self.get_serializer(rating_obj).data)
            return Response({"rating": None})

        rating_value = request.data.get("rating")
        review = request.data.get("review", "")
        if rating_value is None:
            return Response({"error": "Rating value is required"}, status=status.HTTP_400_BAD_REQUEST)

        rating_obj, created = Rating.objects.update_or_create(
            user=request.user, movie=movie, defaults={"rating": rating_value, "review": review}
        )
        return Response(self.get_serializer(rating_obj).data,
                        status=status.HTTP_201_CREATED if created else status.HTTP_200_OK)


class RecommendationView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        user = request.user
        collab = recommend_for_user(user.id, top_k=20)

        if collab:
            strategy = "collaborative"
            movie_ids = [m.id for m in collab]
        else:
            popular = Movie.objects.annotate(
                avg_rating=Avg("ratings__rating"),
                ratings_count=Count("ratings")
            ).filter(ratings_count__gte=10).order_by("-avg_rating", "-ratings_count")[:20]
            return Response({"strategy": "popular", "results": MovieSerializer(popular, many=True).data})

        movies = list(Movie.objects.filter(id__in=movie_ids).annotate(
            avg_rating=Avg("ratings__rating"), ratings_count=Count("ratings")))
        movies_sorted = sorted(movies, key=lambda m: movie_ids.index(m.id))
        return Response({"strategy": strategy, "results": MovieSerializer(movies_sorted, many=True).data})

@api_view(["GET"]) 
@permission_classes([IsAuthenticated]) 
def recommend_movies(request, user_id):
    movies = recommend_for_user(user_id)
    serializer = MovieSerializer(movies, many=True)
    return Response(serializer.data)

@api_view(["GET"])
def recommend_similar_movies(request, movie_id):
    recommendations = get_content_based_recommendations(movie_id)
    return Response(MovieSerializer(recommendations, many=True).data)


def build_or_get_tfidf():
    """
    Build (once) a TF-IDF matrix over movies' content (genres + description),
    cache it, and return (movies_ids, tfidf_matrix).
    The tfidf_matrix is a scipy.sparse matrix of shape (n_movies, n_features).
    """
    tfidf_matrix = cache.get(TFIDF_CACHE_KEY)
    ids = cache.get(TFIDF_IDS_KEY)
    index_map = cache.get(TFIDF_INDEX_KEY)

    if tfidf_matrix is not None and ids is not None and index_map is not None:
        return ids, index_map, tfidf_matrix

    # Load only necessary fields to keep memory lower
    qs = Movie.objects.all().values("id", "genres", "description")
    df = pd.DataFrame(list(qs))

    if df.empty:
        # nothing to build
        cache.set(TFIDF_CACHE_KEY, None, TFIDF_CACHE_TTL)
        cache.set(TFIDF_IDS_KEY, [], TFIDF_CACHE_TTL)
        cache.set(TFIDF_INDEX_KEY, {}, TFIDF_CACHE_TTL)
        return [], {}, None

    # convert genres representation to plain string
    # If genres in DB are pipe-separated strings: keep them. If M2M, values() returns whatever stored.
    df["genres"] = df["genres"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)

    # combine to single document per movie
    df["content"] = (df["genres"].str.replace("|", " ") + " " + df["description"]).str.strip()

    # Build TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)  # limit features
    tfidf_matrix = vectorizer.fit_transform(df["content"])

    ids = df["id"].astype(int).tolist()
    index_map = {mid: idx for idx, mid in enumerate(ids)}

    # Cache (store sparse matrix, index map and ids)
    cache.set(TFIDF_CACHE_KEY, tfidf_matrix, TFIDF_CACHE_TTL)
    cache.set(TFIDF_VECTORIZER_KEY, vectorizer, TFIDF_CACHE_TTL)
    cache.set(TFIDF_IDS_KEY, ids, TFIDF_CACHE_TTL)
    cache.set(TFIDF_INDEX_KEY, index_map, TFIDF_CACHE_TTL)

    return ids, index_map, tfidf_matrix


# ------------------------------
# Helper: compute collaborative similarity row for one movie
# ------------------------------
def compute_collaborative_scores_for_movie(target_movie_id):
    """
    Efficiently compute item-item collaborative similarity *scores* for the target movie only.
    Strategy:
      - Load ratings, restrict to top-N most-rated movies to keep pivot sparse (COLLAB_TOP_MOVIES).
      - Build user x movie pivot (sparse)
      - Compute cosine similarity between the target movie column and all movie columns using sparse operations.
    Returns pandas.Series indexed by movie_id -> similarity score.
    """
    try:
        ratings_qs = Rating.objects.all().values("user_id", "movie_id", "rating")
        ratings_df = pd.DataFrame(list(ratings_qs))
        if ratings_df.empty:
            return pd.Series(dtype=float)

        # Restrict to top most-rated movie ids to keep pivot width manageable
        top_movie_ids = ratings_df["movie_id"].value_counts().head(COLLAB_TOP_MOVIES).index.tolist()
        ratings_df = ratings_df[ratings_df["movie_id"].isin(top_movie_ids)]

        # If target movie isn't in the top movies, skip CF part
        if target_movie_id not in ratings_df["movie_id"].values:
            return pd.Series(dtype=float)

        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
        # Use sparse representation of pivot (users x movies)
        pivot_sparse = csr_matrix(pivot.values)  # shape (n_users, n_movies)

        # find column index for target_movie_id
        try:
            col_idx = list(pivot.columns).index(target_movie_id)
        except ValueError:
            return pd.Series(dtype=float)

        # compute similarity between target column and all columns: use transpose to get movies x users
        movie_item_sparse = pivot_sparse.T  # shape (n_movies, n_users)
        # compute cosine similarity between row col_idx and all rows
        # movie_item_sparse[col_idx] yields a 1 x n_users sparse matrix
        sim_row = cosine_similarity(movie_item_sparse[col_idx], movie_item_sparse).flatten()
        scores = pd.Series(sim_row, index=pivot.columns)
        # sort desc
        scores = scores.sort_values(ascending=False)
        return scores
    except Exception as e:
        logger.exception("Error computing collaborative scores: %s", e)
        return pd.Series(dtype=float)


# ------------------------------
# HybridRecommendationView (optimized)
# ------------------------------
class HybridRecommendationView(APIView):
    permission_classes = [permissions.AllowAny]  # allow or require auth as you want

    def get(self, request, movie_id):
        movie_id = int(movie_id)
        target_movie = get_object_or_404(Movie, id=movie_id)

        # 1) Content-based (use cached TF-IDF matrix)
        ids, index_map, tfidf_matrix = build_or_get_tfidf()
        content_scores = pd.Series(dtype=float)
        try:
            if tfidf_matrix is not None and movie_id in index_map:
                idx = index_map[movie_id]  # row index in tfidf_matrix
                # compute cosine similarity between the target row and all rows
                sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
                # map similarity vector to movie ids
                content_scores = pd.Series(sim_row, index=ids)
                # drop self
                content_scores = content_scores.drop(movie_id, errors="ignore")
                # keep top candidates (keep more to allow CF to combine)
                content_scores = content_scores.sort_values(ascending=False).head(100)
        except Exception as e:
            logger.exception("Content-based part failed: %s", e)
            content_scores = pd.Series(dtype=float)

        # 2) Collaborative filtering (compute only for clicked movie)
        collaborative_scores = pd.Series(dtype=float)
        try:
            collaborative_scores = compute_collaborative_scores_for_movie(movie_id)
            # drop self
            collaborative_scores = collaborative_scores.drop(movie_id, errors="ignore")
            # keep top candidates
            collaborative_scores = collaborative_scores.head(100)
        except Exception as e:
            logger.exception("Collaborative part failed: %s", e)
            collaborative_scores = pd.Series(dtype=float)

        # 3) Normalize both series (safe)
        def safe_normalize(s: pd.Series):
            if s.empty:
                return s
            arr = s.values.astype(float)
            maxv = arr.max() if arr.max() != 0 else 1.0
            return s.astype(float) / float(maxv)

        collaborative_scores = safe_normalize(collaborative_scores)
        content_scores = safe_normalize(content_scores)

        # 4) Weighted hybrid combination
        try:
            weight_cf = float(request.query_params.get("w_cf", 0.6))
            weight_cb = float(request.query_params.get("w_cb", 0.4))
        except Exception:
            weight_cf, weight_cb = 0.6, 0.4

        # Align indices before combining; use pandas add with fill_value=0
        hybrid_scores = (weight_cf * collaborative_scores).add(weight_cb * content_scores, fill_value=0.0)
        # Remove target id if present
        hybrid_scores = hybrid_scores.drop(movie_id, errors="ignore")

        # 5) If hybrid is empty, fallback to popular
        if hybrid_scores.empty:
            recommended_qs = (
                Movie.objects.annotate(avg_rating=Avg("ratings__rating"), ratings_count=Count("ratings"))
                .filter(ratings_count__gte=5)
                .order_by("-avg_rating")[:HYBRID_TOP_K]
                .prefetch_related("genres")
            )
            recommended = list(recommended_qs)
        else:
            top_ids = hybrid_scores.sort_values(ascending=False).head(HYBRID_TOP_K).index.tolist()
            # fetch movies preserving order
            recommended_qs = Movie.objects.filter(id__in=top_ids).prefetch_related("genres")
            recommended = list(recommended_qs)
            recommended.sort(key=lambda m: top_ids.index(m.id))

        # 6) Build JSON response with genre objects and safe image defaults
        DEFAULT_POSTER = "https://via.placeholder.com/300x450?text=No+Poster"
        DEFAULT_BACKDROP = "https://via.placeholder.com/1280x720?text=No+Backdrop"

        result = []
        for m in recommended:
            # genres: return list of {id, name} when possible
            genres_list = []
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
                "movielens_id": getattr(m, "movielens_id", None),
                "title": m.title,
                "year": getattr(m, "year", None),
                "genres": genres_list,
                "imdb_id": getattr(m, "imdb_id", "") or "",
                "tmdb_id": getattr(m, "tmdb_id", "") or "",
                "description": getattr(m, "description", "") or "",
                "poster_url": getattr(m, "poster_url", "") or DEFAULT_POSTER,
            })

        return Response({"movie": target_movie.title, "recommendations": result})