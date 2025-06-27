# Transaction Splitting Detection - Modular Scripts

Este proyecto proporciona un sistema modular para detectar fraccionamiento de transacciones usando métodos de machine learning ensemble.

## Estructura del Proyecto

```
src/transaction-splitting/
├── __init__.py                 # Package principal
├── config.py                   # Configuraciones y constantes
├── utils.py                    # Funciones auxiliares
├── preprocessing.py            # Preprocesamiento de datos
├── models.py                   # Modelos de ML
└── scripts/
    ├── __init__.py
    ├── train.py               # Script de entrenamiento
    ├── predict.py             # Script de predicción
    └── preprocess.py          # Script de preprocesamiento
```

## Instalación

1. Asegúrate de tener todas las dependencias instaladas:
```bash
pip install polars pandas scikit-learn scipy numpy matplotlib seaborn mlflow
```

2. Clona o descarga el proyecto y navega al directorio raíz.

## Uso

### 1. Preprocesamiento de Datos

El primer paso es preprocesar los datos raw para generar features y grupos candidatos:

```bash
# Preprocesar datos usando el primer archivo parquet encontrado en data/raw/
python -m src.transaction_splitting.scripts.preprocess --verbose

# Especificar archivo de entrada específico
python -m src.transaction_splitting.scripts.preprocess \
    --input-file data/raw/transactions.parquet \
    --output-groups data/processed/candidate_groups.parquet \
    --save-scaler models/scaler.pkl \
    --verbose

# Ver todas las opciones
python -m src.transaction_splitting.scripts.preprocess --help
```

**Opciones principales:**
- `--input-file`: Archivo de entrada (formato parquet)
- `--output-file`: Archivo de salida para datos procesados
- `--output-groups`: Archivo de salida para grupos candidatos
- `--output-features`: Archivo de salida para matriz de features
- `--save-scaler`: Ruta para guardar el scaler entrenado
- `--verbose`: Salida detallada

### 2. Entrenamiento de Modelos

Entrena los modelos ensemble (Isolation Forest + DBSCAN):

```bash
# Entrenamiento básico con parámetros por defecto
python -m src.transaction_splitting.scripts.train --verbose

# Entrenamiento con parámetros personalizados
python -m src.transaction_splitting.scripts.train \
    --data-file data/raw/transactions.parquet \
    --output-dir models/experiment_1 \
    --if-contamination 0.01 \
    --if-n-estimators 500 \
    --dbscan-eps 0.5 \
    --dbscan-min-samples 10 \
    --save-metrics \
    --verbose

# Ver todas las opciones
python -m src.transaction_splitting.scripts.train --help
```

**Opciones principales:**
- `--data-file`: Archivo de datos de entrenamiento
- `--output-dir`: Directorio para guardar modelos entrenados
- `--if-contamination`: Parámetro de contaminación para Isolation Forest
- `--if-n-estimators`: Número de estimadores para Isolation Forest
- `--dbscan-eps`: Parámetro epsilon para DBSCAN
- `--dbscan-min-samples`: Mínimo número de muestras para DBSCAN
- `--ensemble-if-weight`: Peso de Isolation Forest en ensemble (default: 0.7)
- `--ensemble-dbscan-weight`: Peso de DBSCAN en ensemble (default: 0.3)
- `--save-metrics`: Guardar métricas de entrenamiento en JSON
- `--verbose`: Salida detallada

### 3. Predicción/Inferencia

Usa modelos entrenados para hacer predicciones sobre nuevos datos:

```bash
# Predicción básica
python -m src.transaction_splitting.scripts.predict \
    --data-file data/new_transactions.parquet \
    --verbose

# Predicción con salida personalizada
python -m src.transaction_splitting.scripts.predict \
    --data-file data/new_transactions.parquet \
    --output-file results/predictions.parquet \
    --model-dir models/experiment_1 \
    --threshold-percentile 99 \
    --include-scores \
    --verbose

# Solo mostrar alertas
python -m src.transaction_splitting.scripts.predict \
    --data-file data/new_transactions.parquet \
    --alerts-only \
    --include-scores \
    --verbose

# Ver todas las opciones
python -m src.transaction_splitting.scripts.predict --help
```

**Opciones principales:**
- `--data-file`: Archivo de datos para predicción (requerido)
- `--output-file`: Archivo de salida para predicciones
- `--model-dir`: Directorio con modelos entrenados
- `--threshold-percentile`: Percentil para umbral de alertas (default: 95)
- `--include-scores`: Incluir scores individuales de modelos
- `--alerts-only`: Mostrar solo registros con alertas
- `--verbose`: Salida detallada

## Ejemplos de Workflows Completos

### Workflow Básico

```bash
# 1. Preprocesar datos
python -m src.transaction_splitting.scripts.preprocess \
    --input-file data/raw/transactions.parquet \
    --verbose

# 2. Entrenar modelos
python -m src.transaction_splitting.scripts.train \
    --save-metrics \
    --verbose

# 3. Hacer predicciones
python -m src.transaction_splitting.scripts.predict \
    --data-file data/raw/new_transactions.parquet \
    --output-file results/predictions.parquet \
    --include-scores \
    --verbose
```

### Workflow con Experimentación

```bash
# Probar diferentes configuraciones
for contamination in 0.001 0.01 0.02; do
    for eps in 0.3 0.5 0.9; do
        echo "Training with contamination=$contamination, eps=$eps"
        python -m src.transaction_splitting.scripts.train \
            --if-contamination $contamination \
            --dbscan-eps $eps \
            --output-dir models/exp_${contamination}_${eps} \
            --save-metrics \
            --verbose
    done
done
```

## Formato de Datos

### Datos de Entrada

Los archivos de entrada deben estar en formato parquet y contener las siguientes columnas:

- `user_id`: ID del usuario
- `merchant_id`: ID del comercio
- `transaction_date`: Fecha y hora de la transacción
- `transaction_amount`: Monto de la transacción
- `transaction_type`: Tipo de transacción (ej. "credit", "debit")

### Datos de Salida

Las predicciones incluyen:

- Todas las columnas originales de los grupos candidatos
- `combined_risk_score`: Score de riesgo combinado (0-1)
- `is_fractionment_alert`: Flag booleano de alerta
- `alert_threshold`: Umbral usado para generar alertas
- `isolation_forest_score`: Score del Isolation Forest (si se incluye)
- `dbscan_noise_flag`: Flag de ruido de DBSCAN (si se incluye)

## Configuración

La configuración se encuentra en `src/transaction_splitting/config.py`:

- Rutas de archivos y directorios
- Parámetros por defecto de modelos
- Configuración de features y ventanas temporales

## Próximos Pasos

Una vez que tengas los scripts funcionando, el siguiente paso será crear una API REST con FastAPI para permitir predicciones en tiempo real y facilitar la integración con otros sistemas.

## Solución de Problemas

### Error de importación
Si obtienes errores de importación, asegúrate de ejecutar los scripts desde el directorio raíz del proyecto.

### Archivos no encontrados
Los scripts buscan automáticamente archivos en `data/raw/` si no se especifica una ruta. Asegúrate de que tus datos estén en la ubicación correcta.

### Modelos no encontrados
Para predicción, los modelos deben estar en `models/` o especificar la ruta con `--model-dir`. 