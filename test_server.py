"""Test suite verifying IrisAI server.py static serving and REST API endpoints."""

import json
import sys
import threading
import time
import urllib.request
import urllib.error

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from http.server import HTTPServer
from server import IrisDashboardHandler, load_ml_model, load_dataset, get_metrics_payload

TEST_PORT = 8011
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"


def start_test_server():
    load_ml_model()
    load_dataset()
    get_metrics_payload()
    httpd = HTTPServer(("127.0.0.1", TEST_PORT), IrisDashboardHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.5)
    return httpd


def test_endpoint(name, url, method="GET", data=None):
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
        body = json.dumps(data).encode("utf-8")
    else:
        body = None

    try:
        with urllib.request.urlopen(req, data=body, timeout=5) as response:
            status = response.status
            content = response.read()
            return status, content
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def run_tests():
    print("=" * 60)
    print("Testing IrisAI Server & REST API")
    print("=" * 60)

    server = start_test_server()
    passed = 0
    total = 0

    try:
        # 1. Health API
        total += 1
        status, content = test_endpoint("Health Check", f"{BASE_URL}/api/health")
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("status") == "ok"
        print("  ✓ [PASS] GET /api/health returns status ok")
        passed += 1

        # 2. Predict API - Setosa
        total += 1
        status, content = test_endpoint(
            "Predict Setosa",
            f"{BASE_URL}/api/predict",
            method="POST",
            data={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
        )
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("predicted_species") == "setosa"
        assert data.get("confidence") > 0.90
        print(f"  ✓ [PASS] POST /api/predict correctly identifies Setosa ({data['confidence']:.1%})")
        passed += 1

        # 3. Predict API - Versicolor
        total += 1
        status, content = test_endpoint(
            "Predict Versicolor",
            f"{BASE_URL}/api/predict",
            method="POST",
            data={"sepal_length": 6.0, "sepal_width": 2.8, "petal_length": 4.5, "petal_width": 1.4}
        )
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("predicted_species") == "versicolor"
        print(f"  ✓ [PASS] POST /api/predict correctly identifies Versicolor ({data['confidence']:.1%})")
        passed += 1

        # 4. Predict API - Virginica
        total += 1
        status, content = test_endpoint(
            "Predict Virginica",
            f"{BASE_URL}/api/predict",
            method="POST",
            data={"sepal_length": 6.7, "sepal_width": 3.1, "petal_length": 5.6, "petal_width": 2.4}
        )
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("predicted_species") == "virginica"
        print(f"  ✓ [PASS] POST /api/predict correctly identifies Virginica ({data['confidence']:.1%})")
        passed += 1

        # 5. Metrics API
        total += 1
        status, content = test_endpoint("Metrics Check", f"{BASE_URL}/api/metrics")
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert "best_model" in data
        assert "confusion_matrix" in data
        assert "dataset_summary" in data
        print("  ✓ [PASS] GET /api/metrics returns full evaluation payloads")
        passed += 1

        # 6. Dataset API
        total += 1
        status, content = test_endpoint("Dataset Check", f"{BASE_URL}/api/dataset")
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("total") == 150
        assert len(data.get("records", [])) == 150
        print(f"  ✓ [PASS] GET /api/dataset returns {data['total']} specimens")
        passed += 1

        # 7. Dataset API Search
        total += 1
        status, content = test_endpoint("Dataset Filter", f"{BASE_URL}/api/dataset?species=setosa")
        data = json.loads(content.decode("utf-8"))
        assert status == 200
        assert data.get("filtered_count") == 50
        print(f"  ✓ [PASS] GET /api/dataset?species=setosa filters 50 Setosa specimens")
        passed += 1

        # 8. Static HTML
        total += 1
        status, content = test_endpoint("Static HTML", f"{BASE_URL}/index.html")
        assert status == 200
        assert b"IrisAI" in content
        print("  ✓ [PASS] GET /index.html serves Stitch UI dashboard HTML")
        passed += 1

        # 9. Static JS
        total += 1
        status, content = test_endpoint("Static JS", f"{BASE_URL}/app.js")
        assert status == 200
        assert b"switchTab" in content
        print("  ✓ [PASS] GET /app.js serves frontend controller script")
        passed += 1

    finally:
        server.shutdown()
        server.server_close()

    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed successfully!")
    print("=" * 60)
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    if not success:
        sys.exit(1)
