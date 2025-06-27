"""
FastAPI application for transaction splitting detection.
"""
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from ..config import MODEL_DIR
from .models import (
    TransactionRecord,
    TransactionBatch,
    PredictionConfig,
    PredictionResult,
    FileUploadResponse,
    SinglePredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from .services import (
    ModelService,
    PredictionService
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transaction Splitting Detection API",
    description="""
    API para detectar fraccionamiento de transacciones usando modelos de machine learning ensemble.
    
    ## Características
    
    * **Múltiples formatos de entrada**: Archivos (CSV, Excel, Parquet), registros individuales o lotes
    * **Detección automática**: Identifica patrones de fraccionamiento usando Isolation Forest + DBSCAN
    * **Configuración flexible**: Umbrales de alerta personalizables
    * **Respuestas detalladas**: Scores de riesgo y métricas de confianza
    
    ## Uso
    
    1. **Registro individual**: `POST /predict/single` - Para una transacción específica
    2. **Lote de registros**: `POST /predict/batch` - Para múltiples transacciones
    3. **Archivo**: `POST /predict/upload` - Para archivos CSV, Excel o Parquet
    
    ## Formato de datos requerido
    
    * `user_id`: ID del usuario
    * `merchant_id`: ID del comercio
    * `transaction_date`: Fecha y hora (YYYY-MM-DD HH:MM:SS)
    * `transaction_amount`: Monto (debe ser positivo)
    * `transaction_type`: Tipo ("credit" o "debit")
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services
model_service = ModelService()
prediction_service = PredictionService(model_service)


# Dependency for error handling
async def get_prediction_service() -> PredictionService:
    """Get prediction service dependency."""
    try:
        prediction_service.model_service.ensure_models_loaded()
        return prediction_service
    except Exception as e:
        logger.error(f"Failed to load models: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Models not available: {str(e)}"
        )


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).model_dump()
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_service.is_loaded,
        version="1.0.0",
        timestamp=datetime.now()
    )


# Prediction endpoints
@app.post("/predict/single", response_model=SinglePredictionResponse)
async def predict_single_transaction(
    transaction: TransactionRecord,
    config: Optional[PredictionConfig] = None,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predecir fraccionamiento para una transacción individual.
    
    Este endpoint toma una sola transacción y busca patrones de fraccionamiento
    considerando el historial de transacciones similares del usuario.
    """
    try:
        if config is None:
            config = PredictionConfig()
        
        predictions, metadata = service.predict_from_records([transaction], config)
        
        return SinglePredictionResponse(
            message="Prediction completed successfully",
            candidate_groups_found=metadata["total_candidate_groups"],
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Error in single prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_transactions(
    batch: TransactionBatch,
    config: Optional[PredictionConfig] = None,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predecir fraccionamiento para un lote de transacciones.
    
    Este endpoint procesa múltiples transacciones y busca patrones de
    fraccionamiento entre ellas, útil para análisis de comportamiento.
    """
    try:
        if config is None:
            config = PredictionConfig()
        
        predictions, metadata = service.predict_from_records(batch.transactions, config)
        
        return BatchPredictionResponse(
            message="Batch prediction completed successfully",
            total_transactions_processed=metadata["total_transactions_processed"],
            candidate_groups_found=metadata["total_candidate_groups"],
            alerts_count=metadata["alerts_count"],
            alert_rate=metadata["alert_rate"],
            predictions=predictions
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/upload", response_model=FileUploadResponse)
async def predict_from_file(
    file: UploadFile = File(...),
    threshold_percentile: float = 95,
    include_scores: bool = True,
    alerts_only: bool = False,
    service: PredictionService = Depends(get_prediction_service)
):
    """
    Predecir fraccionamiento desde archivo cargado.
    
    Soporta archivos en formato:
    * CSV (.csv)
    * Excel (.xlsx, .xls)
    * Parquet (.parquet)
    
    El archivo debe contener las columnas requeridas:
    user_id, merchant_id, transaction_date, transaction_amount, transaction_type
    """
    try:
        # Validate file format
        allowed_extensions = {'.csv', '.xlsx', '.xls', '.parquet'}
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_extension}. "
                       f"Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Read file content
        file_content = await file.read()
        
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Create config
        config = PredictionConfig(
            threshold_percentile=threshold_percentile,
            include_scores=include_scores,
            alerts_only=alerts_only
        )
        
        # Make predictions
        predictions, metadata = service.predict_from_file(
            file_content, file.filename, config
        )
        
        return FileUploadResponse(
            message=f"File '{file.filename}' processed successfully",
            total_candidate_groups=metadata["total_candidate_groups"],
            alerts_count=metadata["alerts_count"],
            alert_rate=metadata["alert_rate"],
            predictions=predictions
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")


# Model management endpoints
@app.post("/models/load")
async def load_models(model_dir: Optional[str] = None):
    """
    Cargar modelos desde directorio específico.
    
    Útil para cargar diferentes versiones de modelos o después de actualizaciones.
    """
    try:
        global model_service, prediction_service
        
        if model_dir:
            model_service = ModelService(model_dir)
        else:
            model_service = ModelService()
        
        model_service.load_models()
        prediction_service = PredictionService(model_service)
        
        return {"message": "Models loaded successfully", "model_dir": model_dir or "default"}
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load models: {str(e)}")


@app.get("/models/status")
async def get_model_status():
    """
    Obtener estado de los modelos cargados.
    """
    return {
        "loaded": model_service.is_loaded,
        "model_dir": str(model_service.model_dir) if model_service.model_dir else "default",
        "timestamp": datetime.now()
    }


# Analytics endpoint
@app.get("/analytics/stats")
async def get_analytics_stats():
    """
    Obtener estadísticas de uso de la API.
    
    (En una implementación real, esto vendría de una base de datos)
    """
    return {
        "message": "Analytics endpoint - implement with database for production",
        "endpoints": {
            "/predict/single": "Single transaction predictions",
            "/predict/batch": "Batch transaction predictions", 
            "/predict/upload": "File upload predictions"
        },
        "supported_formats": ["CSV", "Excel", "Parquet"],
        "timestamp": datetime.now()
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Starting Transaction Splitting Detection API...")
    try:
        # Try to load models on startup (optional)
        model_service.load_models()
        logger.info("Models loaded successfully on startup")
    except Exception as e:
        logger.warning(f"Could not load models on startup: {str(e)}")
        logger.info("Models will be loaded on first request")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Transaction Splitting Detection API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 