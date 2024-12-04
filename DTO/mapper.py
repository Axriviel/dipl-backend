def map_notification_to_dto(notification, user_id):
    from DTO.notification_DTO import NotificationDTO
    return NotificationDTO(
        id=notification.id,
        message=notification.message,
        timestamp=notification.timestamp,
        was_read=notification.was_read,
        user=user_id
    )

def map_feedback_to_dto(feedback):
    from DTO.feedback_DTO import FeedbackDTO
    return FeedbackDTO(
        id=feedback.id,
        feedback=feedback.feedback,
        timestamp=feedback.timestamp,
        user=feedback.user.username 
    )