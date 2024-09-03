from models import db

class Feedback(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feedback = db.Column(db.String(2000), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())  # Nastavení aktuálního času při vytvoření záznamu
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('feedbacks', lazy=True))

    def __repr__(self):
        return f'<Feedback {self.feedback}>'
