import os
import re
from flask import Flask
from backend.config import (SECRET_KEY, JWT_SECRET_KEY, DATABASE_URL,
                            CHECKPOINT_PATH, FRONTEND_URL)
from backend.extensions import db, jwt, cors
from backend.routes.auth import auth_bp
from backend.routes.predict import predict_bp
from backend.routes.scans import scans_bp

# Allow any localhost port (covers 5173, 5174, etc.) plus whatever FRONTEND_URL is set to
_ALLOWED_ORIGINS = [re.compile(r'http://localhost:\d+'), FRONTEND_URL]


def create_app():
    app = Flask(__name__)

    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['JWT_SECRET_KEY'] = JWT_SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    app.config['JWT_ALGORITHM'] = 'HS256'
    app.config['JWT_HEADER_NAME'] = 'Authorization'
    app.config['JWT_HEADER_TYPE'] = 'Bearer'

    db.init_app(app)
    jwt.init_app(app)
    cors.init_app(
        app,
        resources={r'/api/*': {
            'origins': _ALLOWED_ORIGINS,
            'methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            'allow_headers': ['Content-Type', 'Authorization'],
        }},
    )

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(scans_bp, url_prefix='/api')

    with app.app_context():
        from backend.models import user, scan  # noqa: F401 — registers models
        db.create_all()

        if os.path.isabs(CHECKPOINT_PATH):
            checkpoint = CHECKPOINT_PATH
        else:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            checkpoint = os.path.normpath(os.path.join(project_root, CHECKPOINT_PATH))

        try:
            from backend.ml.inference import init_model
            init_model(checkpoint)
        except Exception as exc:
            print(f'[WARNING] Model failed to load: {exc}')
            print('[WARNING] /api/predict will return 500 until the model is available.')

    return app


if __name__ == '__main__':
    application = create_app()
    application.run(host='0.0.0.0', port=5000, debug=False)
