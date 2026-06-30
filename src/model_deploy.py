# src/model_deploy.py

# Librerías
import pandas as pd
import numpy as np
import pickle
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 1. Inicialización de la app
app = FastAPI(
    title="API de predicción de pago a tiempo",
    description="API para predecir si un cliente pagará a tiempo o no basado en su scoring crediticio",
    version="1.1.1"
)

# 2. Carga del modelo (Random Forest)
import os

# 2. Carga del modelo usando RUTAS ABSOLUTAS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"✅ Modelo cargado correctamente desde: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error al cargar el modelo en {MODEL_PATH}: {e}")
    model = None

print(f"Tipo de modelo cargado: {type(model)}")
if hasattr(model, 'named_steps'):
    print(f"Pasos del pipeline: {list(model.named_steps.keys())}")

# 3. Definición de la clase de entrada con las columnas reales de Base_de_datos.xlsx
class ClienteInput(BaseModel):
    tipo_credito: int
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int
    huella_consulta: int
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    saldo_mora_codeudor: Optional[float] = 0.0
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: str

# 4. Endpoint de saludo y verificación
@app.get("/Saludo") 
def Saludo():
    return {"message": "Hola, esta es una API para predecir si un cliente pagará a tiempo o no"}

# 5. Endpoint /predict con soporte para predicción por lotes (Batch)
@app.post("/predict", summary="Predice el pago a tiempo para una lista de clientes")
def predict(clientes: List[ClienteInput]):
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo predictivo no está disponible en el servidor.")
    
    try:
        # 1. Convertimos la lista de entrada que viene de Swagger a una lista de diccionarios
        datos_dict = [cliente.dict() for cliente in clientes]
        
        # 2. Creamos el DataFrame de Pandas exactamente con los datos crudos que ingresaron
        df_input = pd.DataFrame(datos_dict)
        
        # 3. Forzamos que las columnas de texto mantengan el tipo string
        df_input["tipo_laboral"] = df_input["tipo_laboral"].astype(str)
        df_input["tendencia_ingresos"] = df_input["tendencia_ingresos"].astype(str)
        
        # --- SOLUCIÓN: ALINEACIÓN DE COLUMNAS CON EL MODELO ---
        # Si el modelo conoce las columnas con las que fue entrenado, reordenamos df_input igual
        if hasattr(model, "feature_names_in_"):
            columnas_modelo = model.feature_names_in_
            # Reindexamos para asegurar mismo orden y evitar conflictos de posición
            df_input = df_input.reindex(columns=columnas_modelo)
        elif hasattr(model.named_steps['preprocessor'], 'feature_names_in_'):
            columnas_modelo = model.named_steps['preprocessor'].feature_names_in_
            df_input = df_input.reindex(columns=columnas_modelo)
        # ------------------------------------------------------
        
        # 4. PASAMOS EL DATAFRAME PERFECTAMENTE ALINEADO AL PIPELINE
        predicciones = model.predict(df_input)
        probabilidades = model.predict_proba(df_input)[:, 1]
        
        # 5. Construimos la respuesta estructurada que espera tu Swagger
        resultados = []
        for i, pred in enumerate(predicciones):
            resultados.append({
                "registro_nro": i + 1,
                "prediccion_pago_atiempo": int(pred),
                "estado": "Pago a tiempo (1)" if pred == 1 else "Riesgo de Morosidad (0)",
                "probabilidad_pago": float(round(probabilidades[i], 4))
            })
            
        return {
            "status": "Predicción exitosa",
            "cantidad_procesada": len(clientes),
            "resultados": resultados
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la predicción: {str(e)}")