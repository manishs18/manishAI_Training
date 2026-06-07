import time
from flask import Blueprint, request, jsonify, g
import structlog
from pydantic import ValidationError

from src.models import PredictionRequest, PredictionResponse
from src.logger import add_correlation_id

bp = Blueprint("api", __name__)

@bp.before_request
def start_timer():
    g.start_time = time.time()

@bp.after_request
def log_request_info(response):
    # Calculate duration
    duration = time.time() - g.start_time
    
    # Create a contextual logger containing the correlation_id
    log = structlog.get_logger().bind(**add_correlation_id())
    log.info(
        "request_processed",
        method=request.method,
        path=request.path,
        status=response.status_code,
        duration=f"{duration:.4f}s"
    )
    return response

@bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200

@bp.route("/predictions", methods=["POST"])
def create_prediction():
    log = structlog.get_logger().bind(**add_correlation_id())
    
    try:
        # Validate incoming JSON against Pydantic schema
        data = request.get_json(force=True)
        req_model = PredictionRequest(**data)
    except (ValidationError, Exception) as e:
        # Reraise ValidationError to be handled by the app-level error handler
        if isinstance(e, ValidationError):
            raise e
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    # Mock ML Inference logic
    log.info("running_inference", feature_count=len(req_model.features))
    response_data = PredictionResponse(prediction=1, probability=0.94)

    return jsonify(response_data.model_dump()), 200

@bp.route("/predictions/<int:pred_id>", methods=["GET"])
def get_prediction(pred_id):
    # Mocking a simple fetch
    if pred_id == 999:  # Mocking a "not found" scenario
        return jsonify({"error": f"Prediction {pred_id} not found"}), 404
        
    response_data = PredictionResponse(prediction=0, probability=0.12)
    return jsonify(response_data.model_dump()), 200