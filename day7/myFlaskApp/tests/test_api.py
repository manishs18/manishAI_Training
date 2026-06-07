import json

def test_health_endpoint(client):
    """Test the basic GET /health route."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "healthy"}

def test_create_prediction_success(client):
    """Test valid POST payload validation and response structure."""
    payload = {"features": [1.2, 3.4, -0.5]}
    response = client.post("/predictions", json=payload)
    
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert "probability" in data
    assert data["status"] == "success"

def test_create_prediction_invalid_payload(client):
    """Test that bad input formats successfully route through the 422 error handler."""
    # Features must be a list, passing a string instead
    payload = {"features": "not-a-list"} 
    response = client.post("/predictions", json=payload)
    
    assert response.status_code == 422
    data = response.get_json()
    assert data["error"] == "Unprocessable Entity"
    assert "details" in data

def test_endpoint_not_found(client):
    """Test 404 custom global error handler."""
    response = client.get("/invalid-route-abc")
    assert response.status_code == 404
    assert response.get_json() == {"error": "Resource not found"}