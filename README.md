# Transaction Splitting Detection System

Sistema de detección de fraccionamiento de transacciones utilizando modelos de machine learning ensemble (Isolation Forest + DBSCAN) con API REST y scripts modulares.

##  Descripción

Este proyecto implementa un sistema completo para la detección automática de fraccionamiento de transacciones financieras, una técnica comúnmente utilizada para evadir controles de detección de lavado de dinero y límites de transacciones.

###  Características Principales

- **Detección Inteligente**: Modelos ensemble (Isolation Forest + DBSCAN) para identificar patrones anómalos
- **API REST Completa**: Endpoints para inferencia individual, por lotes y carga de archivos
- **Scripts Modulares**: Herramientas de línea de comandos para entrenamiento, predicción y preprocesamiento
- **Múltiples Formatos**: Soporte para CSV, Excel (.xlsx, .xls) y Parquet
- **MLflow Integration**: Seguimiento de experimentos y gestión de modelos
- **Configuración Flexible**: Parámetros personalizables para diferentes casos de uso

## Arquitectura del Sistema

```
transaction-splitting/
├── src/transaction_splitting/          # Código fuente principal
│   ├── api/                           # API REST FastAPI
│   │   ├── main.py                    # Aplicación FastAPI
│   │   ├── models.py                  # Schemas Pydantic
│   │   ├── services.py                # Lógica de negocio
│   │   └── server.py                  # Script de inicio
│   ├── scripts/                       # Scripts CLI
│   │   ├── train.py                   # Entrenamiento de modelos
│   │   ├── predict.py                 # Inferencia
│   │   └── preprocess.py              # Preprocesamiento
│   ├── config.py                      # Configuración central
│   ├── models.py                      # Clases de modelos ML
│   ├── preprocessing.py               # Preprocesamiento de datos
│   └── utils.py                       # Utilidades
├── models/                            # Modelos entrenados
│   ├── if_model.pkl                   # Isolation Forest
│   ├── dbscan_model.pkl               # DBSCAN
│   └── scaler.pkl                     # StandardScaler
├── data/                              # Datos
│   ├── raw/                           # Datos crudos
│   └── processed/                     # Datos procesados
├── notebooks/                         # Jupyter notebooks
│   ├── 1_eda.ipynb                    # Análisis exploratorio
│   └── 2_modelado.ipynb               # Desarrollo de modelos
└── mlruns/                            # Experimentos MLflow
```

##  Instalación Rápida

### Requisitos
- Python 3.10+
- UV package manager (recomendado) o pip

### 1. Clonar y Configurar
```bash
git clone https://github.com/aleuse/nequi-transaction-splitting.git
cd transaction-splitting
```

### 2. Instalar Dependencias
```bash
# Con UV (recomendado)
uv sync

# O con pip
pip install -e .
```

### 3. Entrenar Modelos (si es necesario)
```bash
uv run python -m src.transaction_splitting.scripts.train
```

### 4. Iniciar API
```bash
uv run src/transaction_splitting/api/server.py
```

 **¡Listo!** La API estará disponible en http://localhost:8000

## Uso del Sistema

### Scripts de Línea de Comandos

#### Entrenamiento de Modelos
```bash
# Entrenamiento básico
python -m src.transaction_splitting.scripts.train

# Con parámetros personalizados
python -m src.transaction_splitting.scripts.train \
  --if-contamination 0.01 \
  --dbscan-eps 0.3 \
  --dbscan-min-samples 5 \
  --ensemble-weight 0.7 \
  --verbose
```

#### Predicción/Inferencia
```bash
# Desde archivo
python -m src.transaction_splitting.scripts.predict \
  --data-file data/transactions.csv \
  --output-file results.csv

# Solo alertas con umbral personalizado
python -m src.transaction_splitting.scripts.predict \
  --data-file data/transactions.parquet \
  --threshold-percentile 90 \
  --alerts-only \
  --verbose
```

#### Preprocesamiento
```bash
# Preprocesar datos
python -m src.transaction_splitting.scripts.preprocess \
  --input-file data/raw/transactions.xlsx \
  --output-file data/processed/clean_transactions.parquet
```

### API REST

#### Iniciar Servidor
```bash
# Modo desarrollo (con auto-reload)
uv run src/transaction_splitting/api/server.py

# Modo producción
uvicorn src.transaction_splitting.api.main:app --host 0.0.0.0 --port 8000
```

#### Endpoints Principales

##### 1. Transacción Individual
```bash
curl -X POST "http://localhost:8000/predict/single" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 123456,
    "merchant_id": 789012,
    "transaction_date": "2023-06-15 14:30:00",
    "transaction_amount": 150000.0,
    "transaction_type": "debit"
  }'
```

##### 2. Lote de Transacciones
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

##### 3. Carga de Archivos
```bash
curl -X POST "http://localhost:8000/predict/upload" \
  -F "file=@transactions.csv" \
  -F "threshold_percentile=90" \
  -F "alerts_only=false"
```

#### Documentación Interactiva
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Formato de Datos

### Esquema de Transacciones

| Campo | Tipo | Descripción | Ejemplo |
|-------|------|-------------|---------|
| `user_id` | int | Identificador único del usuario | 123456 |
| `merchant_id` | int | Identificador único del comercio | 789012 |
| `transaction_date` | datetime/string | Fecha y hora de la transacción | "2023-06-15 14:30:00" |
| `transaction_amount` | float | Monto de la transacción (>0) | 150000.0 |
| `transaction_type` | string | Tipo de transacción ("credit" o "debit") | "debit" |

### Formatos Soportados
- **CSV**: Archivos de valores separados por comas
- **Excel**: .xlsx y .xls (todas las hojas)
- **Parquet**: Formato columnar optimizado
- **JSON**: Para API REST

## Modelos de Machine Learning

### Algoritmos Utilizados

#### 1. Isolation Forest
- **Propósito**: Detección de anomalías en features numéricas
- **Configuración**: Contamination rate configurable (default: 0.05)
- **Output**: Score de anomalía (valores más negativos = más anómalos)

#### 2. DBSCAN
- **Propósito**: Clustering para identificar grupos atípicos
- **Configuración**: eps y min_samples configurables
- **Output**: Etiquetas de cluster (noise = -1)

#### 3. Ensemble
- **Combinación**: Weighted average de ambos modelos
- **Normalización**: MinMax scaling para scores comparables
- **Umbral**: Percentil configurable para alertas

### Features Utilizadas

```python
features = [
    'transaction_count',        # Número de transacciones en el grupo
    'total_group_amount',       # Monto total del grupo
    'amount_stdev',            # Desviación estándar de montos
    'amount_median',           # Mediana de montos
    'amount_min',              # Monto mínimo
    'amount_max',              # Monto máximo
    'time_span_minutes',       # Duración temporal del grupo
    'amount_coeff_of_variation', # Coeficiente de variación
    'amount_entropy',          # Entropía de distribución de montos
    'avg_time_delta_minutes',  # Tiempo promedio entre transacciones
    'has_round_amounts'        # Presencia de montos redondos
]
```

## Interpretación de Resultados

### Risk Scores
- **combined_risk_score**: Score normalizado (0-1)
  - 0.0-0.5: Riesgo bajo
  - 0.5-0.8: Riesgo medio
  - 0.8-1.0: Riesgo alto (probable fraccionamiento)

### Alertas
- **is_fractionment_alert**: Indica si se detectó fraccionamiento
- **alert_threshold**: Umbral utilizado para la decisión

### Métricas Adicionales
- **isolation_forest_score**: Score raw del Isolation Forest
- **dbscan_noise_flag**: Si el grupo fue clasificado como ruido
- **candidate_groups_found**: Número de grupos candidatos analizados

## Ejemplos Prácticos

### Escenario 1: Análisis de Archivo
```python
# Cargar y analizar archivo completo
import requests

with open('transactions.csv', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/upload',
        files={'file': f},
        params={'threshold_percentile': 95, 'alerts_only': True}
    )

alerts = response.json()
print(f"Alertas encontradas: {alerts['alerts_count']}")
```

### Escenario 2: Monitoreo en Tiempo Real
```python
# Evaluar transacción en tiempo real
transaction = {
    "user_id": 123456,
    "merchant_id": 789012,
    "transaction_date": "2023-06-15 14:30:00",
    "transaction_amount": 49999.0,  # Justo bajo un umbral común
    "transaction_type": "debit"
}

response = requests.post(
    'http://localhost:8000/predict/single',
    json=transaction
)

result = response.json()
if result['predictions'] and result['predictions'][0]['is_fractionment_alert']:
    print("⚠️ ALERTA: Posible fraccionamiento detectado")
```

### Escenario 3: Entrenamiento Personalizado
```bash
# Entrenar con datos específicos de la institución
python -m src.transaction_splitting.scripts.train \
  --data-file data/historical_transactions.parquet \
  --if-contamination 0.02 \
  --test-size 0.3 \
  --random-state 42 \
  --verbose
```

## Configuración Avanzada

### Variables de Entorno
```bash
# Configurar paths personalizados
export TRANSACTION_SPLITTING_DATA_DIR="/custom/data/path"
export TRANSACTION_SPLITTING_MODEL_DIR="/custom/models/path"
export MLFLOW_TRACKING_URI="http://mlflow-server:5000"
```

### Configuración de Modelos
```python
# config.py
MODEL_PARAMS = {
    'isolation_forest': {
        'contamination': 0.05,
        'n_estimators': 100,
        'random_state': 42
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    },
    'ensemble': {
        'if_weight': 0.7,
        'dbscan_weight': 0.3
    }
}
```

## MLflow Integration

### Tracking de Experimentos
```bash
# Iniciar MLflow UI
mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db

# Acceder en: http://localhost:5000
```

### Comparar Experimentos
```python
import mlflow

# Listar experimentos
experiments = mlflow.search_runs()
best_run = experiments.loc[experiments['metrics.combined_f1_score'].idxmax()]
print(f"Mejor modelo: {best_run['run_id']}")
```

## Desarrollo

### Linting y Formateo
```bash
# Formatear código
uv run ruff format
```

## Troubleshooting

### Problemas Comunes

#### 1. "Models not found"
```bash
# Solución: Entrenar modelos
python -m src.transaction_splitting.scripts.train
```

#### 2. "Missing columns in data"
- Verificar que el archivo tenga todas las columnas requeridas
- Consultar la sección "Formato de Datos"

#### 3. "API connection refused"
```bash
# Verificar que la API esté corriendo
curl http://localhost:8000/health
```

#### 4. "MLflow tracking issues"
```bash
# Limpiar experimentos corruptos
rm -rf mlruns/
```

### Logs y Debugging
```bash
# API con logs detallados
uvicorn src.transaction_splitting.api.main:app --log-level debug

# Scripts con verbose
python -m src.transaction_splitting.scripts.train --verbose
```

##  Deployment

### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY pyproject.toml ./
RUN pip install uv && uv sync

COPY . .
EXPOSE 8000

CMD ["uvicorn", "src.transaction_splitting.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
