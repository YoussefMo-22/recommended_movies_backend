# views.py
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
        user = user_service.register_user(request.data)
        return Response(
            {"message": "User created", "user": user_service.get_me(user)},
            status=status.HTTP_201_CREATED,
        )


class MeView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(user_service.get_me(request.user))


class UserProfileView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        return Response(user_service.get_or_create_profile(request.user))

    def patch(self, request):
        return Response(user_service.update_profile(request.user, request.data))
