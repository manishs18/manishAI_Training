from flask import Flask, jsonify
from pydantic import ValidationError
from src.routes import bp
from src.logger import configure_logger

def create_app():
    app = Flask(__name__)
    
    # Initialize structured logging configurations
    configure_logger()

    # Register blueprints
    app.register_blueprint(bp)

    # --- Step 4: Global Error Handlers ---
    
    @app.errorhandler(ValidationError)
    def handle_pydantic_validation_error(e):
        """Returns 422 with structural error details when Pydantic parsing fails."""
        return jsonify({
            "error": "Unprocessable Entity",
            "details": e.errors(include_url=False, include_context=False)
        }), 422

    @app.errorhandler(404)
    def handle_not_found(e):
        return jsonify({"error": "Resource not found"}), 404

    @app.errorhandler(500)
    def handle_internal_server_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app