# Dashboard Iris Streamlit

Este proyecto es un tablero interactivo desarrollado con [Streamlit](https://streamlit.io/) para explorar el clásico dataset **Iris**. Permite análisis estadísticos, visualización de datos y regresión lineal con rigurosidad estadística.

## Características

- Filtros por especie
- Estadísticas descriptivas
- Histogramas y gráficos de dispersión
- Pairplot interactivo por especie
- Análisis de regresión lineal con:
  - Gráficos
  - Pruebas estadísticas (normalidad, homocedasticidad)
  - Explicación automática de resultados

## Archivos incluidos

- `dashboard.py`: Código principal del tablero
- `iris.csv`: Dataset Iris (puedes usar el clásico o tu versión propia)
- `requirements.txt`: Paquetes necesarios para ejecutar el dashboard

## Instalación local

1. Clona el repositorio:
   ```bash
   git clone https://github.com/JuanDavidLamboglia/dashboard-iris-streamlit.git
   cd dashboard-iris-streamlit
   ```
2. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
3. Ejecuta el dashboard:
   ```bash
   streamlit run dashboard.py
   ```

## Autor

Juan Lamboglia con ayuda de GitHub Copilot

