from flask import Flask
from config import Config, basedir
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import pandas as pd
import os

from .Thoracic import ThoracicPredictor 


app = Flask (__name__)
app.config.from_object(Config)
db = SQLAlchemy(app)
migrate = Migrate(app, db)
predictor = ThoracicPredictor(pd.read_csv(os.path.join(basedir , 'thoracic-data.csv')))
from app import routes

