import os
from repositories.UserRepository import UserRepository
from services.user_service import UserService

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///backend.db'
    JWT_SECRET_KEY = 'your_jwt_secret_key'
    SECRET_KEY = os.urandom(24).hex()
    CORS_SUPPORTS_CREDENTIALS = True

    # Inicializace UserRepository a UserService
    user_repository = UserRepository()
    user_service = UserService(user_repository)