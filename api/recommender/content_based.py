from api.models import Movie

def get_content_based_recommendations(movie_id, top_n=5):
    try:
        movie = Movie.objects.get(id=movie_id)
    except Movie.DoesNotExist:
        return []

    # ✅ Fetch genres as a set of strings (genre names)
    movie_genres = set(movie.genres.values_list("name", flat=True))

    if not movie_genres:
        return []

    # Find other movies with genres
    similar_movies = Movie.objects.exclude(id=movie.id).filter(genres__isnull=False).distinct()

    scored_movies = []
    for m in similar_movies:
        genres = set(m.genres.values_list("name", flat=True))  # ✅ Extract genre names
        intersection = movie_genres & genres
        union = movie_genres | genres
        similarity = len(intersection) / len(union) if union else 0

        if similarity > 0:
            scored_movies.append((m, similarity))

    # Sort by similarity (highest first)
    scored_movies.sort(key=lambda x: x[1], reverse=True)

    return [m for m, score in scored_movies[:top_n]]

