from models import db
from models.user import User
from sqlalchemy import JSON

class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    metric_value = db.Column(db.Float, nullable=False)
    watched_metric = db.Column(db.String(100), nullable=False)
    metric_values_history = db.Column(JSON, nullable=True)  # JSON for epochs and values
    creation_config = db.Column(JSON, nullable=True) #config used to create the model
    used_params = db.Column(JSON, nullable=True)  # used params
    used_opt_method = db.Column(db.String(100), nullable=False)
    used_task = db.Column(db.String(100), nullable=True)
    used_tags = db.Column(JSON, nullable=True)
    
    dataset = db.Column(db.String(100), nullable=False)
    used_designer = db.Column(db.String(30), nullable=False) #auto or semi
    task_protocol = db.Column(JSON, nullable=True)  

    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    user = db.relationship('User', backref=db.backref('models', lazy=True))

    def __repr__(self):
        return f'<Model {self.model_name}>'
