import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from django.core.cache import cache
from ..models import Rating, Movie


def recommend_for_user(user_id: int, top_k: int = 20, neighbor_k: int = 50):
    """
    Optimized user-based collaborative filtering.
    - Uses sparse matrices for speed
    - Caches similarity matrix to avoid recomputation
    - Limits to top similar users for scoring
    """
    # Load ratings
    ratings_qs = Rating.objects.all().values_list("user_id", "movie_id", "rating")
    if not ratings_qs.exists():
        return []

    df = pd.DataFrame(list(ratings_qs), columns=["user_id", "movie_id", "rating"])

    # Pivot into sparse user-movie matrix
    user_movie_matrix = df.pivot_table(
        index="user_id", columns="movie_id", values="rating", aggfunc="mean"
    ).fillna(0)

    if user_id not in user_movie_matrix.index:
        return []

    sparse_matrix = csr_matrix(user_movie_matrix.values)

    # Try cached similarity matrix (reuse across requests)
    similarity_df = cache.get("user_similarity")
    if similarity_df is None:
        similarity_matrix = cosine_similarity(sparse_matrix, dense_output=False)
        similarity_df = pd.DataFrame(
            similarity_matrix.toarray(),
            index=user_movie_matrix.index,
            columns=user_movie_matrix.index,
        )
        cache.set("user_similarity", similarity_df, timeout=3600)  # 1 hour

    # Get top similar users (exclude self)
    similar_users = (
        similarity_df[user_id]
        .drop(user_id, errors="ignore")
        .sort_values(ascending=False)
        .head(neighbor_k)
    )
    if similar_users.empty:
        return []

    # Weighted recommendation scores
    target_ratings = user_movie_matrix.loc[user_id]
    scores = {}

    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = user_movie_matrix.loc[sim_user]
        unrated_movies = target_ratings[target_ratings == 0].index
        for movie_id in unrated_movies:
            rating = sim_user_ratings[movie_id]
            if rating > 0:
                scores[movie_id] = scores.get(movie_id, 0.0) + sim_score * rating

    if not scores:
        return []

    # Pick top movies
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    movie_ids = [m[0] for m in top_movies]

    # Fetch all movies in one query
    movies_map = Movie.objects.in_bulk(movie_ids)
    return [movies_map[mid] for mid in movie_ids if mid in movies_map]
