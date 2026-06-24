import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from cargar_datos import cargarDatos
from ft_engineering import ft_engineering


def build_model(model_obj, X_train, y_train):
    """Entrena y devuelve un modelo ajustado."""
    model_obj.fit(X_train, y_train)
    return model_obj


def summarize_classification(model_obj, X_test, y_test):
    """Calcula metricas esenciales de clasificacion."""
    preds = model_obj.predict(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, preds),
        'Precision': precision_score(y_test, preds, pos_label=1, zero_division=0),
        'Recall': recall_score(y_test, preds, pos_label=1, zero_division=0),
        'F1-Score': f1_score(y_test, preds, pos_label=1, zero_division=0)
    }

    return metrics, preds


def detectar_multicolinealidad(df, target_col='Pago_atiempo', umbral_corr=0.80, umbral_vif=5):
    print("\n=== ANALISIS DE MULTICOLINEALIDAD ===")

    X = df.drop(columns=[target_col], errors='ignore')
    X_num = X.select_dtypes(include=['int64', 'float64']).copy()

    X_num = X_num.fillna(X_num.median())

    print(f"\nVariables numericas evaluadas: {X_num.shape[1]}")

    corr_matrix = X_num.corr().abs()

    pares_correlacion = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            col_1 = corr_matrix.columns[i]
            col_2 = corr_matrix.columns[j]
            corr_value = corr_matrix.iloc[i, j]

            if corr_value >= umbral_corr:
                pares_correlacion.append({
                    'Variable 1': col_1,
                    'Variable 2': col_2,
                    'Correlacion': round(corr_value, 4)
                })

    df_corr_alta = pd.DataFrame(pares_correlacion)

    print(f"\nPares con correlacion >= {umbral_corr}:")
    if df_corr_alta.empty:
        print("No se encontraron pares con correlacion alta.")
    else:
        print(df_corr_alta.sort_values(by='Correlacion', ascending=False))

    vif_resultados = []

    for col in X_num.columns:
        y_vif = X_num[col]
        X_vif = X_num.drop(columns=[col])

        modelo_vif = LinearRegression()
        modelo_vif.fit(X_vif, y_vif)

        r2 = modelo_vif.score(X_vif, y_vif)

        if r2 >= 0.999999:
            vif = np.inf
        else:
            vif = 1 / (1 - r2)

        vif_resultados.append({
            'Variable': col,
            'VIF': round(vif, 4) if np.isfinite(vif) else np.inf
        })

    df_vif = pd.DataFrame(vif_resultados).sort_values(by='VIF', ascending=False)

    print(f"\nVariables con VIF >= {umbral_vif}:")
    df_vif_alto = df_vif[df_vif['VIF'] >= umbral_vif]

    if df_vif_alto.empty:
        print("No se encontraron variables con VIF alto.")
    else:
        print(df_vif_alto)

    plt.figure(figsize=(14, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm')
    plt.title('Matriz de correlacion - Variables numericas')
    plt.tight_layout()
    plt.savefig('matriz_correlacion_multicolinealidad.png')
    print("\nGrafico guardado como 'matriz_correlacion_multicolinealidad.png'")

    return df_corr_alta, df_vif


if __name__ == '__main__':
    print("\nCargando dataset original para analisis de multicolinealidad...")
    df_original = cargarDatos()

    df_corr_alta, df_vif = detectar_multicolinealidad(
        df_original,
        target_col='Pago_atiempo',
        umbral_corr=0.80,
        umbral_vif=5
    )

    print("\nExtrayendo datasets procesados...")
    X_train, X_test, y_train, y_test = ft_engineering()

    modelos_dict = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    resultados = {}

    print("\nIniciando entrenamiento secuencial...")

    for nombre, modelo_base in modelos_dict.items():
        modelo_entrenado = build_model(modelo_base, X_train, y_train)
        metricas, preds = summarize_classification(modelo_entrenado, X_test, y_test)
        resultados[nombre] = metricas

    df_resumen = pd.DataFrame(resultados).T

    print("\n=== TABLA RESUMEN DE EVALUACION ===")
    print(df_resumen.round(4))

    mejor_modelo = df_resumen['F1-Score'].idxmax()
    print(f"\nEl modelo ganador por mejor rendimiento es: {mejor_modelo}")

    df_plot = df_resumen.reset_index().rename(columns={'index': 'Modelo'})

    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=df_plot,
        x='Modelo',
        y='F1-Score',
        hue='Modelo',
        palette='magma',
        legend=False
    )

    plt.title('Comparacion del F1-Score entre Modelos Supervisados')
    plt.ylim(0, 1.1)

    for index, row in df_plot.iterrows():
        plt.text(
            index,
            row['F1-Score'] + 0.02,
            round(row['F1-Score'], 3),
            color='black',
            ha="center"
        )

    plt.tight_layout()
    plt.savefig('comparacion_modelos.png')
    print("\nGrafico comparativo guardado como 'comparacion_modelos.png'")
    plt.show()