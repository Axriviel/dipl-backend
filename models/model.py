from models import db
from models.user import User
from sqlalchemy import JSON

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    watched_metric = db.Column(db.String(100), nullable=False)
    metric_values_history = db.Column(JSON, nullable=True)  # JSON sloupec pro uložení epoch a hodnot
    error = db.Column(db.Float, nullable=False)
    dataset = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('models', lazy=True))

    def __repr__(self):
        return f'<Model {self.model_name}>'
