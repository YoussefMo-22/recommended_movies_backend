from django.contrib import admin
from django.db.models import Avg, Count
from .models import Movie, Rating, Tag, UserProfile, Genre


class RatingInline(admin.TabularInline):
    """Inline ratings for a movie."""
    model = Rating
    extra = 0
    fields = ("user", "rating", "created_at")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    show_change_link = True
    autocomplete_fields = ("user",)


class TagInline(admin.TabularInline):
    """Inline tags for a movie."""
    model = Tag
    extra = 0
    fields = ("user", "tag", "created_at")
    readonly_fields = ("created_at",)
    ordering = ("-created_at",)
    autocomplete_fields = ("user",)


@admin.register(Genre)
class GenreAdmin(admin.ModelAdmin):
    """Admin for Genres (linked with movies)."""
    list_display = ("id", "name")
    search_fields = ("name",)
    ordering = ("name",)


@admin.register(Movie)
class MovieAdmin(admin.ModelAdmin):
    list_display = (
        "id", "title", "year", "get_genres", "get_avg_rating", "get_ratings_count"
    )
    list_display_links = ("id", "title")
    search_fields = ("title", "imdb_id", "tmdb_id", "genres__name")
    list_filter = ("year", "genres")
    ordering = ("title",)
    inlines = [RatingInline, TagInline]
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("genres",)
    filter_horizontal = ("genres",)  # ✅ Better UI for many-to-many genres
    date_hierarchy = "created_at"

    def get_queryset(self, request):
        # Annotate queryset for performance (instead of per-row queries)
        qs = super().get_queryset(request)
        return qs.annotate(
            avg_rating=Avg("ratings__rating"),
            ratings_count=Count("ratings")
        ).prefetch_related("genres")

    def get_genres(self, obj):
        """Display genres as a comma-separated list."""
        return ", ".join(g.name for g in obj.genres.all())
    get_genres.short_description = "Genres"

    def get_avg_rating(self, obj):
        return round(obj.avg_rating or 0, 2)
    get_avg_rating.admin_order_field = "avg_rating"
    get_avg_rating.short_description = "Avg Rating"

    def get_ratings_count(self, obj):
        return obj.ratings_count
    get_ratings_count.admin_order_field = "ratings_count"
    get_ratings_count.short_description = "Ratings Count"


@admin.register(Rating)
class RatingAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "movie", "rating", "created_at", "updated_at")
    list_filter = ("rating", "created_at")
    search_fields = ("movie__title", "user__username")
    ordering = ("-updated_at",)
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("movie", "user")


@admin.register(Tag)
class TagAdmin(admin.ModelAdmin):
    list_display = ("id", "tag", "user", "movie", "created_at")
    list_filter = ("created_at",)
    search_fields = ("tag", "movie__title", "user__username")
    ordering = ("-created_at",)
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("movie", "user")


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "get_favorite_genres", "created_at")
    search_fields = ("user__username", "favorite_genres__name")
    list_filter = ("created_at",)
    readonly_fields = ("created_at", "updated_at")
    autocomplete_fields = ("user", "favorite_genres")
    filter_horizontal = ("favorite_genres",)

    def get_favorite_genres(self, obj):
        """Display favorite genres for a user."""
        return ", ".join(g.name for g in obj.favorite_genres.all())
    get_favorite_genres.short_description = "Favorite Genres"
