import pandas as pd
from django.db.models import Count, Avg
from django.core.cache import cache
from django.shortcuts import get_object_or_404
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

from ..models import Movie, Rating
from ..constants.constants import (
    TFIDF_CACHE_KEY,
    TFIDF_IDS_KEY,
    TFIDF_INDEX_KEY,
    TFIDF_VECTORIZER_KEY,
    TFIDF_CACHE_TTL,
    COLLAB_TOP_MOVIES,
    HYBRID_TOP_K,
)
import logging

logger = logging.getLogger(__name__)


class HybridRecommendationService:
    @staticmethod
    def build_or_get_tfidf():
        tfidf_matrix = cache.get(TFIDF_CACHE_KEY)
        ids = cache.get(TFIDF_IDS_KEY)
        index_map = cache.get(TFIDF_INDEX_KEY)

        if tfidf_matrix is not None and ids is not None and index_map is not None:
            return ids, index_map, tfidf_matrix

        qs = Movie.objects.all().values("id", "genres", "description")
        df = pd.DataFrame(list(qs))

        if df.empty:
            cache.set(TFIDF_CACHE_KEY, None, TFIDF_CACHE_TTL)
            cache.set(TFIDF_IDS_KEY, [], TFIDF_CACHE_TTL)
            cache.set(TFIDF_INDEX_KEY, {}, TFIDF_CACHE_TTL)
            return [], {}, None

        df["genres"] = df["genres"].fillna("").astype(str)
        df["description"] = df["description"].fillna("").astype(str)
        df["content"] = (df["genres"].str.replace("|", " ") + " " + df["description"]).str.strip()

        vectorizer = TfidfVectorizer(stop_words="english", max_features=50_000)
        tfidf_matrix = vectorizer.fit_transform(df["content"])

        ids = df["id"].astype(int).tolist()
        index_map = {mid: idx for idx, mid in enumerate(ids)}

        cache.set(TFIDF_CACHE_KEY, tfidf_matrix, TFIDF_CACHE_TTL)
        cache.set(TFIDF_VECTORIZER_KEY, vectorizer, TFIDF_CACHE_TTL)
        cache.set(TFIDF_IDS_KEY, ids, TFIDF_CACHE_TTL)
        cache.set(TFIDF_INDEX_KEY, index_map, TFIDF_CACHE_TTL)

        return ids, index_map, tfidf_matrix

    @staticmethod
    def compute_collaborative_scores(target_movie_id):
        ratings_qs = Rating.objects.all().values("user_id", "movie_id", "rating")
        ratings_df = pd.DataFrame(list(ratings_qs))
        if ratings_df.empty:
            return pd.Series(dtype=float)

        top_movie_ids = ratings_df["movie_id"].value_counts().head(COLLAB_TOP_MOVIES).index.tolist()
        ratings_df = ratings_df[ratings_df["movie_id"].isin(top_movie_ids)]

        if target_movie_id not in ratings_df["movie_id"].values:
            return pd.Series(dtype=float)

        pivot = ratings_df.pivot_table(index="user_id", columns="movie_id", values="rating").fillna(0)
        pivot_sparse = csr_matrix(pivot.values)

        try:
            col_idx = list(pivot.columns).index(target_movie_id)
        except ValueError:
            return pd.Series(dtype=float)

        sim_row = cosine_similarity(pivot_sparse.T[col_idx], pivot_sparse.T).flatten()
        scores = pd.Series(sim_row, index=pivot.columns)
        scores = scores.sort_values(ascending=False)
        return scores

    @staticmethod
    def normalize_scores(s: pd.Series):
        if s.empty:
            return s
        arr = s.values.astype(float)
        maxv = arr.max() if arr.max() != 0 else 1.0
        return s.astype(float) / float(maxv)

    @staticmethod
    def get_hybrid_recommendations(movie_id, weight_cf=0.6, weight_cb=0.4):
        ids, index_map, tfidf_matrix = HybridRecommendationService.build_or_get_tfidf()

        # Content-based
        content_scores = pd.Series(dtype=float)
        if tfidf_matrix is not None and movie_id in index_map:
            idx = index_map[movie_id]
            sim_row = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            content_scores = pd.Series(sim_row, index=ids).drop(movie_id, errors="ignore")
            content_scores = content_scores.sort_values(ascending=False).head(100)

        # Collaborative
        collaborative_scores = HybridRecommendationService.compute_collaborative_scores(movie_id)
        collaborative_scores = collaborative_scores.drop(movie_id, errors="ignore").head(100)

        # Normalize
        collaborative_scores = HybridRecommendationService.normalize_scores(collaborative_scores)
        content_scores = HybridRecommendationService.normalize_scores(content_scores)

        # Combine
        hybrid_scores = (weight_cf * collaborative_scores).add(weight_cb * content_scores, fill_value=0.0)
        hybrid_scores = hybrid_scores.drop(movie_id, errors="ignore")

        if hybrid_scores.empty:
            recommended_qs = (
                Movie.objects.annotate(avg_rating=Avg("ratings__rating"), ratings_count=Count("ratings"))
                .filter(ratings_count__gte=5)
                .order_by("-avg_rating")[:HYBRID_TOP_K]
                .prefetch_related("genres")
            )
            return list(recommended_qs)

        top_ids = hybrid_scores.sort_values(ascending=False).head(HYBRID_TOP_K).index.tolist()
        recommended_qs = Movie.objects.filter(id__in=top_ids).prefetch_related("genres")
        recommended = list(recommended_qs)
        recommended.sort(key=lambda m: top_ids.index(m.id))
        return recommended
