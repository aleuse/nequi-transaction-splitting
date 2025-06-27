"""
Ejemplos de uso de la API de detección de fraccionamiento de transacciones.
"""

import requests
import json
from datetime import datetime
import pandas as pd

# Configuración
API_BASE_URL = "http://localhost:8000"


def test_health_check():
    """Verificar estado de la API."""
    response = requests.get(f"{API_BASE_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_single_transaction():
    """Ejemplo de predicción para una transacción individual."""
    transaction = {
        "user_id": 123456,
        "merchant_id": 789012,
        "transaction_date": "2023-06-15 14:30:00",
        "transaction_amount": 150000.0,
        "transaction_type": "debit",
    }

    response = requests.post(f"{API_BASE_URL}/predict/single", json=transaction)
    print("Single Transaction Prediction:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_batch_transactions():
    """Ejemplo de predicción para lote de transacciones."""
    batch = {
        "transactions": [
            {
                "user_id": 123456,
                "merchant_id": 789012,
                "transaction_date": "2023-06-15 14:30:00",
                "transaction_amount": 50000.0,
                "transaction_type": "debit",
            },
            {
                "user_id": 123456,
                "merchant_id": 789012,
                "transaction_date": "2023-06-15 14:35:00",
                "transaction_amount": 50000.0,
                "transaction_type": "debit",
            },
            {
                "user_id": 123456,
                "merchant_id": 789012,
                "transaction_date": "2023-06-15 14:40:00",
                "transaction_amount": 50000.0,
                "transaction_type": "debit",
            },
        ]
    }

    response = requests.post(f"{API_BASE_URL}/predict/batch", json=batch)
    print("Batch Transactions Prediction:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_file_upload():
    """Ejemplo de predicción desde archivo CSV."""
    # Crear archivo CSV de ejemplo
    data = {
        "user_id": [123456, 123456, 123456, 789012, 789012],
        "merchant_id": [111, 111, 111, 222, 222],
        "transaction_date": [
            "2023-06-15 14:30:00",
            "2023-06-15 14:35:00",
            "2023-06-15 14:40:00",
            "2023-06-15 15:00:00",
            "2023-06-15 15:05:00",
        ],
        "transaction_amount": [50000.0, 50000.0, 50000.0, 100000.0, 100000.0],
        "transaction_type": ["debit", "debit", "debit", "credit", "credit"],
    }

    df = pd.DataFrame(data)
    df.to_csv("test_transactions.csv", index=False)

    # Subir archivo
    with open("test_transactions.csv", "rb") as f:
        files = {"file": ("test_transactions.csv", f, "text/csv")}
        params = {
            "threshold_percentile": 90,
            "include_scores": True,
            "alerts_only": False,
        }

        response = requests.post(
            f"{API_BASE_URL}/predict/upload", files=files, params=params
        )

    print("File Upload Prediction:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_with_config():
    """Ejemplo con configuración personalizada."""
    transaction = {
        "user_id": 123456,
        "merchant_id": 789012,
        "transaction_date": "2023-06-15 14:30:00",
        "transaction_amount": 150000.0,
        "transaction_type": "debit",
    }

    config = {"threshold_percentile": 90, "include_scores": True, "alerts_only": True}

    payload = {"transaction": transaction, "config": config}

    response = requests.post(f"{API_BASE_URL}/predict/single", json=payload)
    print("Prediction with Custom Config:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def test_model_status():
    """Verificar estado de los modelos."""
    response = requests.get(f"{API_BASE_URL}/models/status")
    print("Model Status:")
    print(json.dumps(response.json(), indent=2))
    print("-" * 50)


def main():
    """Ejecutar todos los ejemplos."""
    print("Ejemplos de uso de la API de detección de fraccionamiento")
    print("=" * 60)

    try:
        # Verificar que la API esté funcionando
        test_health_check()
        test_model_status()

        # Ejemplos de predicción
        test_single_transaction()
        test_batch_transactions()
        test_file_upload()

        print("✅ Todos los ejemplos ejecutados correctamente")

    except requests.exceptions.ConnectionError:
        print(
            "❌ Error: No se puede conectar a la API. Asegúrate de que esté ejecutándose en http://localhost:8000"
        )
    except Exception as e:
        print(f"❌ Error inesperado: {str(e)}")


if __name__ == "__main__":
    main()
