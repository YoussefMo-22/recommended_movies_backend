# services/UserService.py
from django.contrib.auth import get_user_model
from rest_framework.exceptions import ValidationError
from ..serializers import RegisterSerializer, UserSerializer, UserProfileSerializer
from ..models import UserProfile

User = get_user_model()

class UserService:
    def register_user(self, data):
        serializer = RegisterSerializer(data=data)
        serializer.is_valid(raise_exception=True)
        user = serializer.save()
        return user

    def get_me(self, user):
        return UserSerializer(user).data

    def get_or_create_profile(self, user):
        profile, _ = UserProfile.objects.get_or_create(user=user)
        return UserProfileSerializer(profile).data

    def update_profile(self, user, data):
        profile, _ = UserProfile.objects.get_or_create(user=user)
        serializer = UserProfileSerializer(profile, data=data, partial=True)
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return serializer.data
