import os
from repositories.user_repository import UserRepository
from repositories.notification_repository import NotificationRepository
from services.user_service import UserService
from services.notification_service import NotificationService

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///backend.db'
    JWT_SECRET_KEY = 'your_jwt_secret_key'
    SECRET_KEY = os.urandom(24).hex()
    CORS_SUPPORTS_CREDENTIALS = True

    DATASET_FOLDER = 'datasets'
    ALLOWED_EXTENSIONS = {'csv'}

    # Inicializace UserRepository a UserService
    user_repository = UserRepository()
    user_service = UserService(user_repository)

    notification_repository =  NotificationRepository()
    notification_service = NotificationService(notification_repository)