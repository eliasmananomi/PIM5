import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px


from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from cargar_datos import cargarDatos


st.set_page_config(
    page_title="Monitoreo de Modelo - Data Drift",
    layout="wide"
)


TARGET = "Pago_atiempo"
DATE_COL = "fecha_prestamo"


@st.cache_data
def load_data():
    df = cargarDatos()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def split_reference_current(df):
    df_sorted = df.sort_values(DATE_COL).copy()

    ref_df, current_df = train_test_split(
        df_sorted,
        test_size=0.2,
        random_state=42,
        stratify=df_sorted[TARGET]
    )

    return ref_df, current_df


def build_model_pipeline(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("to_string", FunctionTransformer(
        lambda x: x.fillna("missing").astype(str),
        validate=False
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    return model


def calculate_psi(expected, actual, buckets=10):
    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    if expected.nunique() <= 1 or actual.nunique() <= 1:
        return np.nan

    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return np.nan

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    expected_perc = expected_counts / max(expected_counts.sum(), 1)
    actual_perc = actual_counts / max(actual_counts.sum(), 1)

    expected_perc = np.where(expected_perc == 0, 0.0001, expected_perc)
    actual_perc = np.where(actual_perc == 0, 0.0001, actual_perc)

    psi = np.sum((actual_perc - expected_perc) * np.log(actual_perc / expected_perc))

    return psi


def calculate_js_numeric(ref_series, current_series, bins=10):
    ref_series = pd.Series(ref_series).dropna()
    current_series = pd.Series(current_series).dropna()

    if ref_series.nunique() <= 1 or current_series.nunique() <= 1:
        return np.nan

    combined = pd.concat([ref_series, current_series])
    bin_edges = np.histogram_bin_edges(combined, bins=bins)

    ref_dist = np.histogram(ref_series, bins=bin_edges)[0]
    current_dist = np.histogram(current_series, bins=bin_edges)[0]

    ref_dist = ref_dist / max(ref_dist.sum(), 1)
    current_dist = current_dist / max(current_dist.sum(), 1)

    return jensenshannon(ref_dist, current_dist)


def calculate_js_categorical(ref_series, current_series):
    ref_series = pd.Series(ref_series).fillna("missing").astype(str)
    current_series = pd.Series(current_series).fillna("missing").astype(str)

    categories = sorted(set(ref_series.unique()) | set(current_series.unique()))

    ref_dist = ref_series.value_counts(normalize=True).reindex(categories, fill_value=0)
    current_dist = current_series.value_counts(normalize=True).reindex(categories, fill_value=0)

    return jensenshannon(ref_dist, current_dist)


def chi_square_test(ref_series, current_series):
    ref_series = pd.Series(ref_series).fillna("missing").astype(str)
    current_series = pd.Series(current_series).fillna("missing").astype(str)

    categories = sorted(set(ref_series.unique()) | set(current_series.unique()))

    ref_counts = ref_series.value_counts().reindex(categories, fill_value=0)
    current_counts = current_series.value_counts().reindex(categories, fill_value=0)

    table = np.array([ref_counts.values, current_counts.values])

    if table.shape[1] <= 1:
        return np.nan

    _, p_value, _, _ = chi2_contingency(table)

    return p_value


def risk_level(row):
    if row["Tipo"] == "Numerica":
        if row["PSI"] >= 0.25 or row["KS p-value"] < 0.05:
            return "Alto"
        if row["PSI"] >= 0.10:
            return "Medio"
        return "Bajo"

    if row["Chi2 p-value"] < 0.05 or row["Jensen-Shannon"] >= 0.20:
        return "Alto"
    if row["Jensen-Shannon"] >= 0.10:
        return "Medio"
    return "Bajo"


def risk_icon(level):
    if level == "Alto":
        return "Rojo"
    if level == "Medio":
        return "Amarillo"
    return "Verde"


def drift_report(ref_df, current_df):
    features = [col for col in ref_df.columns if col != TARGET]

    numeric_cols = ref_df[features].select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = ref_df[features].select_dtypes(include=["object"]).columns

    rows = []

    for col in numeric_cols:
        ks_stat, ks_pvalue = ks_2samp(
            ref_df[col].dropna(),
            current_df[col].dropna()
        )

        psi = calculate_psi(ref_df[col], current_df[col])
        js = calculate_js_numeric(ref_df[col], current_df[col])

        rows.append({
            "Variable": col,
            "Tipo": "Numerica",
            "KS statistic": ks_stat,
            "KS p-value": ks_pvalue,
            "PSI": psi,
            "Jensen-Shannon": js,
            "Chi2 p-value": np.nan
        })

    for col in categorical_cols:
        js = calculate_js_categorical(ref_df[col], current_df[col])
        chi_pvalue = chi_square_test(ref_df[col], current_df[col])

        rows.append({
            "Variable": col,
            "Tipo": "Categorica",
            "KS statistic": np.nan,
            "KS p-value": np.nan,
            "PSI": np.nan,
            "Jensen-Shannon": js,
            "Chi2 p-value": chi_pvalue
        })

    report = pd.DataFrame(rows)
    report["Riesgo"] = report.apply(risk_level, axis=1)
    report["Semaforo"] = report["Riesgo"].apply(risk_icon)

    return report.sort_values(
        by=["Riesgo", "PSI", "Jensen-Shannon"],
        ascending=[True, False, False]
    )


def temporal_drift(df):
    df = df.copy()
    df["periodo"] = df[DATE_COL].dt.to_period("M").astype(str)

    periods = sorted(df["periodo"].dropna().unique())

    if len(periods) < 2:
        return pd.DataFrame()

    base_period = periods[0]
    ref_df = df[df["periodo"] == base_period]

    rows = []

    for period in periods[1:]:
        current_df = df[df["periodo"] == period]

        if len(current_df) < 20:
            continue

        numeric_cols = df.drop(columns=[TARGET], errors="ignore").select_dtypes(
            include=["int64", "float64"]
        ).columns

        psi_values = []

        for col in numeric_cols:
            psi = calculate_psi(ref_df[col], current_df[col])
            if not pd.isna(psi):
                psi_values.append(psi)

        rows.append({
            "Periodo": period,
            "PSI promedio": np.mean(psi_values) if psi_values else np.nan,
            "Variables evaluadas": len(psi_values)
        })

    return pd.DataFrame(rows)


def automatic_recommendations(report):
    high_risk = report[report["Riesgo"] == "Alto"]
    medium_risk = report[report["Riesgo"] == "Medio"]

    messages = []

    if high_risk.empty and medium_risk.empty:
        messages.append("No se detectan senales relevantes de data drift. El modelo puede continuar en monitoreo regular.")

    if not high_risk.empty:
        variables = ", ".join(high_risk["Variable"].head(5).tolist())
        messages.append(
            f"Se detecto drift alto en variables criticas: {variables}. "
            "Se recomienda revisar la fuente de datos, validar cambios de poblacion y evaluar retraining del modelo."
        )

    if not medium_risk.empty:
        variables = ", ".join(medium_risk["Variable"].head(5).tolist())
        messages.append(
            f"Se detecto drift medio en: {variables}. "
            "Se recomienda mantener seguimiento y comparar contra periodos futuros."
        )

    return messages


df = load_data()
ref_df, current_df = split_reference_current(df)

X_ref = ref_df.drop(columns=[TARGET, DATE_COL], errors="ignore")
y_ref = ref_df[TARGET]

X_current = current_df.drop(columns=[TARGET, DATE_COL], errors="ignore")
y_current = current_df[TARGET]

model = build_model_pipeline(X_ref)
model.fit(X_ref, y_ref)

current_df = current_df.copy()
current_df["prediccion_modelo"] = model.predict(X_current)
current_df["probabilidad_pago_a_tiempo"] = model.predict_proba(X_current)[:, 1]

preds = current_df["prediccion_modelo"]

metrics = {
    "Accuracy": accuracy_score(y_current, preds),
    "Precision": precision_score(y_current, preds, zero_division=0),
    "Recall": recall_score(y_current, preds, zero_division=0),
    "F1-Score": f1_score(y_current, preds, zero_division=0)
}

report = drift_report(ref_df, current_df.drop(columns=["prediccion_modelo", "probabilidad_pago_a_tiempo"]))
temporal_report = temporal_drift(df)


st.title("Monitoreo de Modelo y Deteccion de Data Drift")

st.markdown(
    """
    Aplicacion para monitorear cambios entre una poblacion historica de referencia
    y una poblacion actual. El objetivo es detectar data drift que pueda afectar
    el desempeno del modelo de scoring crediticio.
    """
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Registros referencia", len(ref_df))
col2.metric("Registros actuales", len(current_df))
col3.metric("Variables monitoreadas", len(report))
col4.metric("Variables con drift alto", len(report[report["Riesgo"] == "Alto"]))

st.subheader("Metricas del modelo sobre datos actuales")

metric_cols = st.columns(4)
for i, (name, value) in enumerate(metrics.items()):
    metric_cols[i].metric(name, round(value, 4))

st.subheader("Tabla de datos actuales con predicciones")

st.dataframe(
    current_df.head(100),
    width="stretch"
)

st.subheader("Metricas de data drift por variable")

st.dataframe(
    report.round(4),
    width="stretch"
)

st.subheader("Semaforo de riesgo por variable")

risk_count = report["Riesgo"].value_counts().reset_index()
risk_count.columns = ["Riesgo", "Cantidad"]

fig_risk = px.bar(
    risk_count,
    x="Riesgo",
    y="Cantidad",
    color="Riesgo",
    color_discrete_map={
        "Bajo": "green",
        "Medio": "orange",
        "Alto": "red"
    },
    title="Cantidad de variables por nivel de riesgo"
)

st.plotly_chart(fig_risk, width="stretch")

st.subheader("Comparacion de distribuciones")

selected_var = st.selectbox(
    "Selecciona una variable para comparar distribuciones",
    report["Variable"].tolist()
)

comparison_df = pd.DataFrame({
    "Referencia": ref_df[selected_var],
    "Actual": current_df[selected_var]
})

if pd.api.types.is_numeric_dtype(ref_df[selected_var]):
    fig_dist = px.histogram(
        pd.concat([
            pd.DataFrame({"Valor": ref_df[selected_var], "Grupo": "Referencia"}),
            pd.DataFrame({"Valor": current_df[selected_var], "Grupo": "Actual"})
        ]),
        x="Valor",
        color="Grupo",
        barmode="overlay",
        opacity=0.6,
        title=f"Distribucion historica vs actual - {selected_var}"
    )
else:
    dist_df = pd.concat([
        pd.DataFrame({"Valor": ref_df[selected_var].astype(str), "Grupo": "Referencia"}),
        pd.DataFrame({"Valor": current_df[selected_var].astype(str), "Grupo": "Actual"})
    ])

    fig_dist = px.histogram(
        dist_df,
        x="Valor",
        color="Grupo",
        barmode="group",
        title=f"Distribucion historica vs actual - {selected_var}"
    )

st.plotly_chart(fig_dist, width="stretch")

st.subheader("Analisis temporal del drift")

if temporal_report.empty:
    st.warning("No hay suficientes periodos temporales para calcular drift en el tiempo.")
else:
    fig_temporal = px.line(
        temporal_report,
        x="Periodo",
        y="PSI promedio",
        markers=True,
        title="Evolucion temporal del PSI promedio"
    )

    fig_temporal.add_hline(y=0.10, line_dash="dash", line_color="orange")
    fig_temporal.add_hline(y=0.25, line_dash="dash", line_color="red")

    st.plotly_chart(fig_temporal, width="stretch")
    st.dataframe(temporal_report.round(4), width="stretch")

st.subheader("Recomendaciones automaticas")

for message in automatic_recommendations(report):
    st.info(message)