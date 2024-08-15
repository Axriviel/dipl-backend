from models import db
from models.user import User

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    error = db.Column(db.Float, nullable=False)
    dataset = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('models', lazy=True))

    def __repr__(self):
        return f'<Model {self.model_name}>'
