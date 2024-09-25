from models.notification import db, Notification
class NotificationRepository:
        def get_all(self):
            pass

        def create_notification(self, for_user_id, message):
            
            try:
                #print("xD")
                #print("Notifikace: " +message+ " pro uživatele " + str(for_user_id))
                 new_notification = Notification(message = message, user_id = for_user_id)
                 db.session.add(new_notification)
                 db.session.commit()

            except Exception as e:
                print(e)