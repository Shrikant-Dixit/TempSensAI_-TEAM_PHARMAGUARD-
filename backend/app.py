from flask import Flask
from routes.predict import predict_bp
from routes.upload import upload_bp

def create_app():
    app = Flask(__name__)

    # Register blueprints
    app.register_blueprint(predict_bp)
    app.register_blueprint(upload_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=True)
