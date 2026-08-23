from flask import Flask
from routes.predict import predict_bp
from routes.upload import upload_bp

def create_app():
    # Initialize Flask web server
    app = Flask(__name__)

    # Register your route blueprints
    # predict_bp handles AI evaluation + graph generation
    # upload_bp handles manual CSV uploads
    app.register_blueprint(predict_bp)
    app.register_blueprint(upload_bp)

    return app

if __name__ == "__main__":
    # Create the app
    app = create_app()

    # Run the server locally
    # host="0.0.0.0" makes it accessible on your LAN (for QR testing)
    # port=5000 is the default Flask port
    app.run(host="0.0.0.0", port=5000, debug=True)
