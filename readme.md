# Movie Recommendation Backend

This is the backend of a **Movie Recommendation System** built with **Django** and **Django REST Framework (DRF)**.  
It supports user registration, authentication, movie browsing, ratings, and both collaborative/content-based recommendations.

---

## Features

- **User Authentication & Profiles**
  - Register, login, JWT authentication.
  - User profile view & update.

- **Movies**
  - Browse and search movies by title, genre, or year.
  - Pagination and ordering by year, average rating, or rating count.

- **Ratings**
  - Users can rate and review movies.
  - Update or remove ratings.
  - Retrieve user's rating for a specific movie.

- **Recommendations**
  - **Collaborative Filtering**: Personalized recommendations based on user ratings.
  - **Content-Based Filtering**: Recommendations based on movie content (genres & description).
  - **Hybrid Recommendations**: Combines collaborative and content-based methods.
  - Fallback to popular movies if no personalized data available.

- **API Documentation**
  - OpenAPI schema generated with **drf-spectacular**.
  - Swagger UI via **drf-yasg** or spectacular sidecar.

- **Optimizations**
  - TF-IDF matrix caching for content-based recommendations.
  - Sparse matrices for efficient collaborative filtering.
  - Prefetching related objects to reduce queries.

---

## Tech Stack

- **Backend:** Django, Django REST Framework
- **Authentication:** SimpleJWT
- **Database:** PostgreSQL
- **Caching:** Django Cache Framework (for TF-IDF matrix)
- **Machine Learning:** scikit-learn (TF-IDF, cosine similarity)
- **Others:** pandas, numpy, requests, django-cors-headers

---

## Installation

1. **Clone the repo**
```bash
git clone <repository-url>
cd <repo-directory>

2. **Create a virtual environment**
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows

3. **Install dependencies**
pip install -r requirements.txt

4. **Configure environment variables**
Create a .env file in the project root:
DEBUG=True
SECRET_KEY=your-secret-key
DATABASE_URL=postgres://USER:PASSWORD@HOST:PORT/DB_NAME

5. **Apply migrations**
python manage.py migrate

6. **Run the server**
python manage.py runserver

The API will be available at http://127.0.0.1:8000/.

## API Endpoints

### Auth
- `POST /auth/register/` → Register a new user
- `POST /auth/login/` → Obtain JWT token
- `GET /auth/me/` → Get current user

### Movies
- `GET /movies/` → List movies (supports filtering & ordering)
- `GET /movies/:id/` → Movie details
- `GET /movies/by-genre/:genre/` → Movies by genre

### Ratings
- `GET /ratings/movie/:movie_id/` → Get user's rating for a movie
- `POST/PATCH /ratings/movie/:movie_id/` → Create or update rating
- `DELETE /ratings/:id/` → Remove rating

### Recommendations
- `GET /recommendations/` → Personalized recommendations
- `GET /recommendations/movie/:movie_id/` → Hybrid recommendations for a movie
- `GET /recommend_movies/:user_id/` → User-based collaborative recommendations
- `GET /recommend_similar_movies/:movie_id/` → Content-based similar movies

### Notes
- Use JWT token in the `Authorization` header for protected routes:  

- Caching is applied to the TF-IDF matrix to speed up content-based recommendations.
- Recommended movies are limited to top 10-20 results to reduce payload size.
