import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Modelos recomendados para clasificar si se paga a tiempo
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Importo la función de ingeniería que acabamos de completar
from ft_engineering import ft_engineering

# 1. Definir funciones requeridas
def build_model(model_obj, X_train, y_train):
    """Entrena y devuelve un objeto de modelo ajustado."""
    model_obj.fit(X_train, y_train)
    return model_obj

def summarize_classification(model_obj, X_test, y_test):
    """Calcula las métricas esenciales de clasificación."""
    preds = model_obj.predict(X_test)
    
    # Evaluamos asumiendo que las etiquetas objetivo son numéricas (1 = Pagó a tiempo)
    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, preds, pos_label=1, zero_division=0),
        'F1-Score': f1_score(y_test, preds, pos_label=1, zero_division=0)
    }
    
    return metrics, preds

# 2. Flujo Principal de Ejecución
if __name__ == '__main__':
    print("Extrayendo datasets procesados...")
    X_train, X_test, y_train, y_test = ft_engineering()
    
    # Definir la batería de modelos a probar
    modelos_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    resultados = {}
    ultimas_predicciones = {}
    
    print("\n🚀 Iniciando entrenamiento secuencial...")
    for nombre, modelo_base in modelos_dict.items():
        # Uso obligatorio de build_model
        modelo_entrenado = build_model(modelo_base, X_train, y_train)
        
        # Uso obligatorio de summarize_classification
        metricas, preds = summarize_classification(modelo_entrenado, X_test, y_test)
        
        resultados[nombre] = metricas
        ultimas_predicciones[nombre] = preds
    
    # 3. Construcción de Tabla Resumen
    df_resumen = pd.DataFrame(resultados).T
    print("\n📊 === TABLA RESUMEN DE EVALUACIÓN ===")
    print(df_resumen.round(4))
    
    # Seleccionar el de mejor rendimiento (basado en F1-Score)
    mejor_modelo = df_resumen['F1-Score'].idxmax()
    print(f"\n🏆 El modelo ganador por mejor rendimiento es: {mejor_modelo}")
    
    # 4. Gráficos Comparativos (Seaborn)
    df_plot = df_resumen.reset_index().rename(columns={'index': 'Modelo'})
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_plot, x='Modelo', y='F1-Score', palette='magma')
    plt.title('Comparación del F1-Score entre Modelos Supervisados')
    plt.ylim(0, 1.1)
    for index, row in df_plot.iterrows():
        plt.text(index, row['F1-Score'] + 0.02, round(row['F1-Score'], 3), color='black', ha="center")
    
    plt.tight_layout()
    # Guardamos el gráfico en el directorio raíz para presentarlo en tu reporte o README
    plt.savefig('comparacion_modelos.png')
    print("\n📈 Gráfico comparativo guardado como 'comparacion_modelos.png'")
    plt.show()