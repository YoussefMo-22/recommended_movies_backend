from django.db import models
from django.contrib.auth import get_user_model

User = get_user_model()


class Movie(models.Model):
    # External IDs
    movielens_id = models.PositiveIntegerField(unique=True, db_index=True)
    imdb_id = models.CharField(max_length=16, blank=True, default="", db_index=True)
    tmdb_id = models.CharField(max_length=16, blank=True, default="", db_index=True)

    # Core metadata
    title = models.CharField(max_length=512, db_index=True)
    year = models.PositiveIntegerField(null=True, blank=True, db_index=True)
    description = models.TextField(blank=True, default="")
    poster_url = models.URLField(blank=True, default="")

    # Relationships
    genres = models.ManyToManyField("Genre", related_name="movies", blank=True)

    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["title"]
        indexes = [
            models.Index(fields=["title"]),
            models.Index(fields=["year"]),
        ]

    def __str__(self):
        return f"{self.title} ({self.year or 'n/a'})"


class Genre(models.Model):
    """Movie genres (e.g., Action, Comedy)."""
    name = models.CharField(max_length=64, unique=True, db_index=True)

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name


class Rating(models.Model):
    """Represents a user's rating for a movie (ratings.csv)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="ratings")
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="ratings")
    rating = models.DecimalField(max_digits=3, decimal_places=1)  # 0.5 to 5.0
    review = models.TextField(blank=True, default="")
    timestamp = models.BigIntegerField(null=True, blank=True, db_index=True)  # from dataset

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("user", "movie")
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["movie"]),
            models.Index(fields=["rating"]),
        ]
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.user_id}-{self.movie_id}:{self.rating}"


class Tag(models.Model):
    """Free-text tags users assign to movies (tags.csv)."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="tags")
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="tags")
    tag = models.CharField(max_length=255, db_index=True)
    timestamp = models.BigIntegerField(null=True, blank=True, db_index=True)  # from dataset

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        indexes = [
            models.Index(fields=["user"]),
            models.Index(fields=["movie"]),
            models.Index(fields=["tag"]),
        ]
        ordering = ["-updated_at"]

    def __str__(self):
        return f"{self.user_id}-{self.movie_id}:{self.tag}"


class UserProfile(models.Model):
    """Optional extension for user preferences and analytics."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="profile")
    favorite_genres = models.ManyToManyField(Genre, related_name="fans", blank=True)  # pipe-separated
    bio = models.TextField(blank=True, default="")
    avatar_url = models.URLField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Profile: {self.user.username}"
