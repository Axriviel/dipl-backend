from models import db

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String(2000), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())  # Nastavení aktuálního času při vytvoření záznamu
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    was_read = db.Column(db.Boolean, default=False)
    user = db.relationship('User', backref=db.backref('notifications', lazy=True))

    def __repr__(self):
        return f'<Notification {self.message}>'
