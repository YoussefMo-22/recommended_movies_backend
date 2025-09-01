import requests
from django.core.management.base import BaseCommand
from django.conf import settings
from api.models import Movie

TMDB_API_URL = "https://api.themoviedb.org/3/movie/{}?api_key={}&language=en-US"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

class Command(BaseCommand):
    help = "Fetch poster and description for ALL movies using TMDb API"

    def handle(self, *args, **kwargs):
        movies = Movie.objects.all()  # ✅ update ALL movies
        updated = 0
        skipped = 0
        failed = 0

        for movie in movies:
            if not movie.tmdb_id:
                skipped += 1
                self.stdout.write(self.style.WARNING(f"Skipping {movie.title} (no TMDb ID)"))
                continue

            url = TMDB_API_URL.format(movie.tmdb_id, settings.TMDB_API_KEY)
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                poster_path = data.get("poster_path")
                overview = data.get("overview")

                # ✅ always refresh poster_url and description if available
                if poster_path:
                    movie.poster_url = f"{TMDB_IMAGE_BASE}{poster_path}"
                if overview:
                    movie.description = overview.strip()

                movie.save(update_fields=["poster_url", "description"])
                updated += 1
                self.stdout.write(self.style.SUCCESS(f"Updated {movie.title}"))
            else:
                failed += 1
                self.stdout.write(
                    self.style.ERROR(f"Failed to fetch {movie.title} (status {response.status_code})")
                )

        self.stdout.write(
            self.style.SUCCESS(
                f"\nDone! ✅ Updated: {updated}, Skipped: {skipped}, Failed: {failed}, Total: {movies.count()}"
            )
        )
