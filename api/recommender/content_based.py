from api.models import Movie

def get_content_based_recommendations(movie_id, top_n=5):
    try:
        # Fetch the target movie with its genres in one query
        movie = Movie.objects.prefetch_related("genres").get(id=movie_id)
    except Movie.DoesNotExist:
        return []

    movie_genres = set(movie.genres.values_list("name", flat=True))
    if not movie_genres:
        return []

    # Fetch all other movies + their genres in one query
    similar_movies = (
        Movie.objects.exclude(id=movie.id)
        .filter(genres__isnull=False)
        .prefetch_related("genres")
        .distinct()
    )

    scored_movies = []
    for m in similar_movies:
        # Use cached prefetched genres
        genres = {g.name for g in m.genres.all()}
        if not genres:
            continue

        # Jaccard similarity
        intersection = len(movie_genres & genres)
        union = len(movie_genres | genres)
        similarity = intersection / union if union else 0

        if similarity > 0:
            scored_movies.append((m, similarity))

    # Sort and return only top_n movies
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored_movies[:top_n]]
