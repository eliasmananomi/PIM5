# PIM5
Análisis de  scoring crediticio
## Monitoreo del modelo y detección de data drift

Se desarrolló una aplicación en Streamlit para monitorear posibles cambios en la población de datos que puedan afectar el desempeño del modelo de scoring crediticio.

El archivo principal es:

```text
src/model_monitoring.py

Desde la raíz del proyecto:
python -m streamlit run .\src\model_monitoring.py
La aplicación se abre localmente en:
http://localhost:8501
Las métricas perfectas deben interpretarse con cautela, ya que pueden indicar un dataset altamente separable o posible fuga de información.