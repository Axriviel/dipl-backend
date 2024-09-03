from flask import Blueprint, jsonify, request
from models import db
from models.notification import Notification
from models.user import User

notification_bp = Blueprint('notification_bp', __name__)

@notification_bp.route("/notifications", methods=["GET"])
def get_notifications():
    user_name = request.args.get('user')
    #print(user_name)
    
    if not user_name:
        return jsonify({"error": "User name is required"}), 400
    
    try:
        # Najít uživatele podle jména
        user = User.query.filter_by(username=user_name).first()
        #print(user.id)
        if not user:
            return jsonify({"error": "User not found"}), 404
        
        # Dotaz na notifikace tohoto uživatele
        notifications = Notification.query.filter_by(user_id=user.id).all()
        #print(notifications)
        
        # Serializace dat do seznamu slovníků
        notifications_list = [
            {
                'id': notification.id,
                'message': notification.message,
                'timestamp': notification.timestamp,
                'was_read': notification.was_read,
                'user': user.username  # nebo jiné relevantní údaje o uživateli
            }
            for notification in notifications
        ]
        #print(notifications_list)
        
        # Vrácení dat jako JSON odpověď
        return jsonify(notifications_list)
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500
