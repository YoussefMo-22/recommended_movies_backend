import csv
import os
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from api.models import Movie, Rating, Tag, Genre  # 👈 Genre added

User = get_user_model()


class Command(BaseCommand):
    help = "Import MovieLens (ml-latest-small) dataset into the database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--path",
            type=str,
            required=True,
            help="Path to the extracted ml-latest-small dataset folder"
        )

    def handle(self, *args, **options):
        base_path = options["path"]

        movies_csv = os.path.join(base_path, "movies.csv")
        links_csv = os.path.join(base_path, "links.csv")
        ratings_csv = os.path.join(base_path, "ratings.csv")
        tags_csv = os.path.join(base_path, "tags.csv")

        self.stdout.write(self.style.WARNING("Starting MovieLens import..."))

        self.import_movies_and_links(movies_csv, links_csv)
        self.import_ratings(ratings_csv)
        self.import_tags(tags_csv)

        self.stdout.write(self.style.SUCCESS("✅ Import completed successfully."))

    def import_movies_and_links(self, movies_csv, links_csv):
        self.stdout.write("Importing movies and links...")

        # Load links mapping: movieId -> (imdbId, tmdbId)
        links_map = {}
        with open(links_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                links_map[int(row["movieId"])] = {
                    "imdb_id": row["imdbId"] or "",
                    "tmdb_id": row["tmdbId"] or "",
                }

        # Import movies
        created_count = 0
        with open(movies_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                movie_id = int(row["movieId"])
                title = row["title"]
                year = None
                # Extract year from title if available (e.g., "Toy Story (1995)")
                if title.strip().endswith(")"):
                    try:
                        year = int(title.strip()[-5:-1])
                    except ValueError:
                        pass

                link_data = links_map.get(movie_id, {})

                movie, created = Movie.objects.get_or_create(
                    movielens_id=movie_id,
                    defaults={
                        "imdb_id": link_data.get("imdb_id", ""),
                        "tmdb_id": link_data.get("tmdb_id", ""),
                        "title": title,
                        "year": year,
                    },
                )

                if created:
                    created_count += 1

                # 👇 Handle genres as many-to-many
                genres_str = row["genres"].strip()
                if genres_str and genres_str != "(no genres listed)":
                    genre_names = [g.strip() for g in genres_str.split("|")]
                    for gname in genre_names:
                        genre, _ = Genre.objects.get_or_create(name=gname)
                        movie.genres.add(genre)

        self.stdout.write(self.style.SUCCESS(f"✔ Imported/linked {created_count} movies."))

    def import_ratings(self, ratings_csv):
        self.stdout.write("Importing ratings...")

        with open(ratings_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ratings = []
            users_created = set()

            for row in reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])

                # Ensure user exists
                if user_id not in users_created:
                    User.objects.get_or_create(
                        id=user_id,
                        defaults={"username": f"user_{user_id}"}
                    )
                    users_created.add(user_id)

                try:
                    movie = Movie.objects.get(movielens_id=movie_id)
                except Movie.DoesNotExist:
                    continue

                ratings.append(
                    Rating(
                        user_id=user_id,
                        movie=movie,
                        rating=float(row["rating"]),
                        timestamp=int(row["timestamp"]),
                    )
                )

            Rating.objects.bulk_create(ratings, ignore_conflicts=True)

        self.stdout.write(self.style.SUCCESS(f"✔ Imported {len(ratings)} ratings."))

    def import_tags(self, tags_csv):
        self.stdout.write("Importing tags...")

        with open(tags_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            tags = []
            users_created = set(User.objects.values_list("id", flat=True))

            for row in reader:
                user_id = int(row["userId"])
                movie_id = int(row["movieId"])

                # Ensure user exists
                if user_id not in users_created:
                    User.objects.get_or_create(
                        id=user_id,
                        defaults={"username": f"user_{user_id}"}
                    )
                    users_created.add(user_id)

                try:
                    movie = Movie.objects.get(movielens_id=movie_id)
                except Movie.DoesNotExist:
                    continue

                tags.append(
                    Tag(
                        user_id=user_id,
                        movie=movie,
                        tag=row["tag"].strip(),
                        timestamp=int(row["timestamp"]),
                    )
                )

            Tag.objects.bulk_create(tags, ignore_conflicts=True)

        self.stdout.write(self.style.SUCCESS(f"✔ Imported {len(tags)} tags."))
