# Transaction Splitting Detection API

API REST para la detección de fraccionamiento de transacciones usando modelos de machine learning ensemble (Isolation Forest + DBSCAN).

## 🚀 Características

- **Múltiples formatos de entrada**: CSV, Excel (.xlsx, .xls), Parquet
- **Tres tipos de inferencia**: Transacción individual, lote de transacciones, archivo completo
- **Configuración flexible**: Umbrales de alerta personalizables
- **Respuestas detalladas**: Scores de riesgo, métricas de confianza y metadatos
- **Documentación automática**: Swagger UI y ReDoc integrados
- **Manejo robusto de errores**: Validación de entrada y respuestas estructuradas

## 📋 Requisitos

```bash
# Instalar dependencias específicas de la API
pip install -r requirements_api.txt
```

## 🔧 Instalación y Configuración

1. **Asegurar que los modelos estén entrenados**:
```bash
# Si no tienes modelos entrenados
python -m src.transaction_splitting.scripts.train
```

2. **Instalar dependencias de la API**:
```bash
pip install -r requirements_api.txt
```

3. **Verificar estructura de archivos**:
```
models/
├── if_model.pkl          # Modelo Isolation Forest
├── dbscan_model.pkl      # Modelo DBSCAN
└── scaler.pkl           # Escalador StandardScaler
```

## 🚀 Ejecución

### Opción 1: Script dedicado (recomendado)
```bash
python src/transaction_splitting/api/server.py
```

### Opción 2: Uvicorn directo
```bash
uvicorn src.transaction_splitting.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Opción 3: Desde el módulo
```bash
python -m src.transaction_splitting.api.main
```

La API estará disponible en: http://localhost:8000

## 📚 Documentación

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## 🎯 Endpoints

### 1. Health Check
```
GET /health
```
Verifica el estado de la API y los modelos.

**Respuesta**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "version": "1.0.0",
  "timestamp": "2023-06-15T14:30:00"
}
```

### 2. Predicción Individual
```
POST /predict/single
```

Predice fraccionamiento para una transacción individual.

**Request Body**:
```json
{
  "user_id": 123456,
  "merchant_id": 789012,
  "transaction_date": "2023-06-15 14:30:00",
  "transaction_amount": 150000.0,
  "transaction_type": "debit"
}
```

**Respuesta**:
```json
{
  "message": "Prediction completed successfully",
  "candidate_groups_found": 1,
  "predictions": [
    {
      "user_id": 123456,
      "merchant_id": 789012,
      "transaction_type": "debit",
      "transaction_count": 3,
      "total_group_amount": 450000.0,
      "combined_risk_score": 0.85,
      "is_fractionment_alert": true,
      "alert_threshold": 0.80,
      "isolation_forest_score": -0.2,
      "dbscan_noise_flag": true
    }
  ]
}
```

### 3. Predicción por Lotes
```
POST /predict/batch
```

Procesa múltiples transacciones en una sola llamada.

**Request Body**:
```json
{
  "transactions": [
    {
      "user_id": 123456,
      "merchant_id": 789012,
      "transaction_date": "2023-06-15 14:30:00",
      "transaction_amount": 50000.0,
      "transaction_type": "debit"
    },
    {
      "user_id": 123456,
      "merchant_id": 789012,
      "transaction_date": "2023-06-15 14:35:00",
      "transaction_amount": 50000.0,
      "transaction_type": "debit"
    }
  ]
}
```

### 4. Carga de Archivos
```
POST /predict/upload
```

Carga y procesa archivos (CSV, Excel, Parquet).

**Parámetros**:
- `file`: Archivo a procesar
- `threshold_percentile`: Percentil para alertas (50-99.9, default: 95)
- `include_scores`: Incluir scores individuales (default: true)
- `alerts_only`: Solo retornar alertas (default: false)

**Ejemplo con curl**:
```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@transactions.csv" \
  -F "threshold_percentile=90" \
  -F "include_scores=true" \
  -F "alerts_only=false"
```

## 📊 Formato de Datos

### Columnas Requeridas

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `user_id` | int | ID del usuario | 123456 |
| `merchant_id` | int | ID del comercio | 789012 |
| `transaction_date` | string/datetime | Fecha y hora | "2023-06-15 14:30:00" |
| `transaction_amount` | float | Monto (>0) | 150000.0 |
| `transaction_type` | string | Tipo ("credit" o "debit") | "debit" |

### Formatos de Fecha Soportados
- ISO 8601: `2023-06-15T14:30:00`
- Estándar: `2023-06-15 14:30:00`
- Con timezone: `2023-06-15T14:30:00Z`

## 🔧 Configuración Avanzada

### PredictionConfig

```json
{
  "threshold_percentile": 95,
  "include_scores": true,
  "alerts_only": false
}
```

- `threshold_percentile`: Percentil para determinar alertas (50-99.9)
- `include_scores`: Incluir scores detallados de Isolation Forest y DBSCAN
- `alerts_only`: Filtrar solo alertas en las respuestas

## 🧪 Ejemplos de Uso

### Python con requests

```python
import requests
import json

# Predicción individual
transaction = {
    "user_id": 123456,
    "merchant_id": 789012,
    "transaction_date": "2023-06-15 14:30:00",
    "transaction_amount": 150000.0,
    "transaction_type": "debit"
}

response = requests.post(
    "http://localhost:8000/predict/single",
    json=transaction
)

print(json.dumps(response.json(), indent=2))
```

### cURL

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Predicción individual
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123456,
    "merchant_id": 789012,
    "transaction_date": "2023-06-15 14:30:00",
    "transaction_amount": 150000.0,
    "transaction_type": "debit"
  }'

# Carga de archivo
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@transactions.csv" \
  -F "threshold_percentile=90"
```

## 📈 Interpretación de Resultados

### Risk Scores
- `combined_risk_score`: Score combinado normalizado (0-1)
  - 0-0.5: Riesgo bajo
  - 0.5-0.8: Riesgo medio
  - 0.8-1.0: Riesgo alto

- `isolation_forest_score`: Score de anomalía IF (valores negativos = más anómalos)
- `dbscan_noise_flag`: Si el grupo fue clasificado como ruido por DBSCAN

### Alertas
- `is_fractionment_alert`: Booleano indicando si se detectó fraccionamiento
- `alert_threshold`: Umbral usado para generar la alerta

## 🚨 Manejo de Errores

### Códigos de Error

- `400 Bad Request`: Datos de entrada inválidos
- `422 Unprocessable Entity`: Errores de validación Pydantic
- `503 Service Unavailable`: Modelos no disponibles
- `500 Internal Server Error`: Errores internos

### Ejemplo de Respuesta de Error

```json
{
  "error": "Missing required columns: ['user_id']",
  "detail": "The uploaded file must contain all required columns",
  "timestamp": "2023-06-15T14:30:00"
}
```

## 🔄 Gestión de Modelos

### Verificar Estado
```
GET /models/status
```

### Recargar Modelos
```
POST /models/load
```

Opcional: especificar directorio personalizado
```json
{
  "model_dir": "/path/to/custom/models"
}
```

## 📊 Monitoreo

### Analytics
```
GET /analytics/stats
```

Endpoint para estadísticas de uso (implementar con base de datos en producción).

## 🧪 Testing

### Ejecutar ejemplos
```bash
python api_examples.py
```

### Tests unitarios
```bash
# Instalar pytest
pip install pytest pytest-asyncio httpx

# Ejecutar tests (cuando se implementen)
pytest tests/api/
```

## 🚀 Deployment

### Docker (recomendado para producción)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.transaction_splitting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Consideraciones de Producción

1. **Configurar CORS** apropiadamente en `main.py`
2. **Variables de entorno** para configuración
3. **Logging** estructurado (ELK stack)
4. **Métricas** con Prometheus
5. **Rate limiting** con slowapi
6. **Autenticación** (JWT, API keys)
7. **Load balancer** para múltiples instancias

## 🔧 Troubleshooting

### Problemas Comunes

1. **"Models not available"**
   - Verificar que existen los archivos en `models/`
   - Ejecutar entrenamiento: `python -m src.transaction_splitting.scripts.train`

2. **"Missing required columns"**
   - Verificar formato de datos de entrada
   - Consultar sección "Formato de Datos"

3. **"Connection refused"**
   - Verificar que la API esté ejecutándose
   - Comprobar puerto (default: 8000)

### Logs

La API genera logs detallados:
```bash
# Ver logs en tiempo real
python src/transaction_splitting/api/server.py
```

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama para feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit cambios: `git commit -m 'Agregar nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

## 📄 Licencia

[Especificar licencia del proyecto]

---

Para más información técnica sobre los modelos, consultar `README.md` principal del proyecto. 