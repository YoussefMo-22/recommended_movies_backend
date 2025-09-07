from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny, IsAuthenticated
from django.contrib.auth import get_user_model
from ..services.UserService import UserService

User = get_user_model()
user_service = UserService()

class RegisterView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        try:
            user = user_service.register_user(request.data)
            profile = user_service.get_or_create_profile(user)
            return Response(
                {"message": "User created", "user": user_service.get_me(user), "profile": profile},
                status=status.HTTP_201_CREATED,
            )
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        username = request.data.get("username")
        password = request.data.get("password")

        if not username or not password:
            return Response(
                {"error": "Username and password are required"},
                status=status.HTTP_400_BAD_REQUEST
            )
        try:
            data = user_service.login_user(username, password)
            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_401_UNAUTHORIZED)


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(user_service.get_me(request.user))


class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(user_service.get_or_create_profile(request.user))

    def patch(self, request):
        try:
            updated_profile = user_service.update_profile(request.user, request.data)
            return Response(updated_profile, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
