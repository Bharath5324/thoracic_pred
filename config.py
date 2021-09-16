import os
basedir = os.path.abspath(os.path.dirname(__file__))

class Config():
    SECRET_KEY = 'SAMPLE KEY'
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(basedir, 'app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TEMPLATES_AUTO_RELOAD = True
    DEBUG = True
    