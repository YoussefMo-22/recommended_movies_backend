import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from ..models import Rating, Movie


def recommend_for_user(user_id: int, top_k: int = 20):
    """
    User-based collaborative filtering recommendations.
    1. Build user-item rating matrix
    2. Find similar users
    3. Recommend unseen movies, weighted by similarity
    """

    # Load ratings
    ratings_qs = Rating.objects.all().values("user_id", "movie_id", "rating")
    df = pd.DataFrame(ratings_qs)

    if df.empty:
        return []

    # Pivot: users x movies
    user_movie_matrix = df.pivot_table(
        index="user_id", columns="movie_id", values="rating"
    ).fillna(0)

    if user_id not in user_movie_matrix.index:
        return []

    # Compute cosine similarity between users
    similarity_matrix = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=user_movie_matrix.index,
        columns=user_movie_matrix.index,
    )

    # Get similar users (exclude self)
    similar_users = similarity_df[user_id].sort_values(ascending=False).drop(user_id)

    if similar_users.empty:
        return []

    # Weighted recommendation scores
    target_ratings = user_movie_matrix.loc[user_id]
    scores = {}

    for sim_user, sim_score in similar_users.items():
        sim_user_ratings = user_movie_matrix.loc[sim_user]
        for movie_id, rating in sim_user_ratings.items():
            if target_ratings[movie_id] == 0 and rating > 0:
                scores[movie_id] = scores.get(movie_id, 0) + sim_score * rating

    # Pick top movies
    top_movies = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    movie_ids = [m[0] for m in top_movies]

    return Movie.objects.filter(id__in=movie_ids)
