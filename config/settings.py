import os
from repositories.user_repository import UserRepository
from repositories.notification_repository import NotificationRepository
from services.user_service import UserService
from services.notification_service import NotificationService

class Config:
    SQLALCHEMY_DATABASE_URI = 'sqlite:///backend.db'
    API_KEY = "ASF4561.AF545wagdA56fds89911a2vaASF32!"
    JWT_SECRET_KEY = 'your_jwt_secret_key'
    SECRET_KEY = os.urandom(24).hex()
    CORS_SUPPORTS_CREDENTIALS = True

    DATASET_FOLDER = 'datasets'
    #extensions that are acceptable from frontend file upload
    ALLOWED_EXTENSIONS = {'csv', 'jpg', 'jpeg', 'png', 'tfrecord', 'h5', 'json'}
    #parameters, that require conversion to integer becouse microsoft NNI returns all the values as float
    KERAS_INT_PARAMS = ['units', 'filters', 'kernel_size', 'strides', 'pool_size'] 

    # Inicializace UserRepository a UserService
    user_repository = UserRepository()
    user_service = UserService(user_repository)

    notification_repository =  NotificationRepository()
    notification_service = NotificationService(notification_repository)