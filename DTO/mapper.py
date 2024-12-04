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

def map_model_to_dto(model):
    from DTO.model_DTO import ModelDTO

    return ModelDTO(
        id=model.id,
        name=model.model_name,
        accuracy=model.accuracy,
        metric_value=model.metric_value,
        watched_metric=model.watched_metric,
        metric_values_history=model.metric_values_history,
        used_opt_method=model.used_opt_method,
        error=model.error,
        dataset=model.dataset,
    )