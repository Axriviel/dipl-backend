from models.notification import db, Notification

def create_notification(user, message):
    try:
        new_notification = Notification(message=message)
        db.session.add(new_notification)
        db.session.commit()
    
    except Exception as e:
        print(e)