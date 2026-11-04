"""
Unit and integration tests for FastAPI backend.
"""

import os
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

from api.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data


def test_index_html_serving():
    response = client.get("/")
    assert response.status_code == 200
    assert "PINN Relative Depth" in response.text


def test_samples_endpoint():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0


def test_predict_with_sample():
    response = client.post("/api/predict", data={"sample_id": "urban_street", "colormap": "turbo"})
    if response.status_code != 200:
        print("ERROR DETAIL:", response.json())
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "depth_map_png" in data
    assert "original_image" in data
    assert "probe_grid" in data
    assert "physics" in data


def test_compare_points():
    # 3x3 dummy grid where row 0 is far (1.0) and row 2 is near (0.2)
    dummy_grid = [
        [0.9, 0.9, 0.9],
        [0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2],
    ]
    payload = {
        "point_a": [1, 2],  # Bottom (near, 0.2)
        "point_b": [1, 0],  # Top (far, 0.9)
        "depth_grid": dummy_grid,
    }
    response = client.post("/api/compare", json=payload)
    assert response.status_code == 200
    res = response.json()
    assert "Point A" in res["closer"]
