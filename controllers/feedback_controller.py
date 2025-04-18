# blueprints/user_routes.py
from flask import Blueprint, jsonify, request
from models.feedback import Feedback
from models.user import User
from models import db
from DTO.mapper import map_feedback_to_dto

feedback_bp = Blueprint('feedback_bp', __name__)

@feedback_bp.route("/getfeedback", methods=["GET"])
def return_feedback():
    try:
        data = Feedback.query.all()
        feedback_list = [
            map_feedback_to_dto(feedback).to_dict()
            for feedback in data
        ]
        
        return jsonify(feedback_list)
    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


@feedback_bp.route("/feedback", methods=["POST"])
def receive_feedback():
    data = request.get_json()
    try:
        feedback_text = data["feedback"]
        user_name = data["user"]

        user = User.query.filter_by(username=user_name).first()

        feedback = Feedback(feedback=feedback_text, user_id=user.id)
        # print("Feedback: "+ feedback_text +" od uzivatele: " + user_name)

        db.session.add(feedback)
        db.session.commit()

        return jsonify({"status": "success", "message": "Feedback received successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500