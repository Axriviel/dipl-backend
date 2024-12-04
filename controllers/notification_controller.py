from flask import Blueprint, jsonify, request
from models import db
from models.notification import Notification
from config.settings import Config
from controllers.user_controller import session
from DTO.mapper import map_notification_to_dto


notification_bp = Blueprint('notification_bp', __name__)


def create_notification(for_user_id, message):
    return Config.notification_service.create_notification(for_user_id = for_user_id, message = message)

@notification_bp.route("/notifications", methods=["GET"])
def get_notifications():
    # user_id = request.args.get('user')
    user_id = session.get("user_id")
    #print(user_name)
    page = request.args.get('page', 1, type=int)  # Předvolba na stránku 1
    limit = request.args.get('limit', 10, type=int)  # Předvolba na 10 notifikací na stránku



    if not user_id:
        return jsonify({"error": "User is required"}), 400
    
    try:
        # Dotaz na notifikace uživatele
        notifications_query = Notification.query.filter_by(user_id=user_id)
        total_notifications = notifications_query.count()  # Celkový počet notifikací
        notifications = notifications_query.order_by(Notification.timestamp.desc()) \
            .offset((page - 1) * limit) \
            .limit(limit) \
            .all()
        #print(notifications)
        
        # Serializace dat do seznamu slovníků
        notifications_list = [
            map_notification_to_dto(notification, user_id).to_dict() 
            for notification in notifications
        ]
        #print(notifications_list)
        
        # Vrácení dat jako JSON odpověď
        return jsonify({
            'notifications': notifications_list,
            'totalPages': (total_notifications + limit - 1) // limit,  # Výpočet celkového počtu stránek
            'currentPage': page
        })
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500

@notification_bp.route("/notifications/<int:notification_id>/mark-as-read", methods=["PUT"])
def mark_as_read(notification_id):
    user_id = session.get("user_id")

    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        # Najdeme notifikaci podle ID a ověříme, že patří aktuálnímu uživateli
        notification = Notification.query.filter_by(id=notification_id, user_id=user_id).first()

        if not notification:
            return jsonify({"error": "Notification not found"}), 404

        # Aktualizujeme stav was_read
        notification.was_read = True
        db.session.commit()

        return jsonify({"message": "Notification marked as read successfully."}), 200
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500



