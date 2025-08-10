import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("Análisis interactivo - Iris Dataset")

# Carga el dataset
df = pd.read_csv('iris.csv')

# Filtro de especie
species = st.multiselect("Selecciona especie(s):", options=df['species'].unique(), default=list(df['species'].unique()))
filtered_df = df[df['species'].isin(species)]

# Estadísticas básicas
st.subheader("Estadísticas básicas")
st.write(filtered_df.describe())

# Histogramas
st.subheader("Histogramas de variables")
col = st.selectbox("Selecciona variable:", options=df.columns[:-1])
fig, ax = plt.subplots()
sns.histplot(filtered_df[col], kde=True, ax=ax)
st.pyplot(fig)

# Pairplot interactivo
st.subheader("Pairplot")
fig2 = sns.pairplot(filtered_df, hue="species")
st.pyplot(fig2)

# Gráfico de dispersión personalizado
st.subheader("Gráfico de dispersión")
x_col = st.selectbox("Eje X:", options=df.columns[:-1], index=0)
y_col = st.selectbox("Eje Y:", options=df.columns[:-1], index=1)
fig3, ax3 = plt.subplots()
sns.scatterplot(data=filtered_df, x=x_col, y=y_col, hue="species", ax=ax3)
st.pyplot(fig3)

# ... (tu código actual arriba)

import statsmodels.api as sm
from scipy import stats
import numpy as np

st.header("Análisis de regresión lineal")

# Selección de variables
x_var = st.selectbox("Variable independiente (X):", options=df.columns[:-1], index=0, key="x_var")
y_var = st.selectbox("Variable dependiente (Y):", options=df.columns[:-1], index=2, key="y_var")

X = filtered_df[[x_var]]
Y = filtered_df[y_var]

# Agrega constante para statsmodels
X_const = sm.add_constant(X)

# Ajusta el modelo
model = sm.OLS(Y, X_const).fit()
filtered_df['y_pred'] = model.predict(X_const)

# 1. Scatter plot y recta de regresión
st.subheader("Gráfico de dispersión con recta de regresión")
fig, ax = plt.subplots()
ax.scatter(X, Y, label='Datos', alpha=0.7)
ax.plot(X, filtered_df['y_pred'], color='red', label='Regresión')
ax.set_xlabel(x_var)
ax.set_ylabel(y_var)
ax.legend()
st.pyplot(fig)
st.markdown(f"**Análisis:** El gráfico muestra la relación entre {x_var} y {y_var}. La línea roja representa el mejor ajuste lineal calculado por el modelo.")

# 2. Coeficientes y ecuación
st.subheader("Ecuación de regresión")
st.write(f"y = {model.params[0]:.3f} + {model.params[1]:.3f}*x")
st.write(model.summary())
st.markdown("**Análisis:** El coeficiente de la variable independiente (pendiente) indica la magnitud y dirección de la relación. El valor p y el R² ayudan a evaluar la significancia y ajuste del modelo.")

# 3. Gráfico de residuos
st.subheader("Gráfico de residuos")
residuals = model.resid
fig2, ax2 = plt.subplots()
ax2.scatter(filtered_df['y_pred'], residuals)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_xlabel("Predicción")
ax2.set_ylabel("Residuos")
st.pyplot(fig2)
st.markdown("**Análisis:** Un patrón aleatorio de residuos sugiere que el modelo es adecuado. Patrones sistemáticos pueden indicar problemas de ajuste.")

# 4. Prueba de normalidad (Shapiro-Wilk)
shapiro_stat, shapiro_p = stats.shapiro(residuals)
st.subheader("Prueba de normalidad (Shapiro-Wilk)")
st.write(f"p-valor: {shapiro_p:.4f}")
if shapiro_p > 0.05:
    st.markdown("**Análisis:** Los residuos parecen seguir una distribución normal (no se rechaza la normalidad).")
else:
    st.markdown("**Análisis:** Los residuos no siguen una distribución normal (se rechaza la normalidad).")

# 5. Prueba de homocedasticidad (Breusch-Pagan)
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(residuals, X_const)
st.subheader("Prueba de homocedasticidad (Breusch-Pagan)")
st.write(f"p-valor: {bp_test[1]:.4f}")
if bp_test[1] > 0.05:
    st.markdown("**Análisis:** No hay evidencia de heterocedasticidad. Los residuos tienen varianza constante.")
else:
    st.markdown("**Análisis:** Hay evidencia de heterocedasticidad. La varianza de los residuos no es constante.")

# 6. Conclusión
st.subheader("Conclusión")
r2 = model.rsquared
p_value = model.pvalues[1]
if p_value < 0.05 and r2 > 0.5:
    st.success("La regresión es significativa y el ajuste es bueno. Se confirma una relación lineal entre las variables seleccionadas.")
else:
    st.warning("La regresión no es significativa o el ajuste es bajo. No se puede confirmar una relación lineal fuerte entre las variables.")
