"""
Predicción del salario anual — Aplicación Streamlit
Trabajo Final (CRISP-DM) · Daniel Jaramillo Guillén y Miguel Ángel López Ríos

Ejecutar localmente:   streamlit run app.py
Estructura esperada:
    app.py
    requirements.txt
    modelos/            (archivos .joblib generados por el cuaderno del punto 4)
    test_predictions.csv
    Employee_Salaries_preparado.csv   (opcional, para la pestaña de datos)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE = Path(__file__).parent
MODELOS = BASE / "modelos"

st.set_page_config(page_title="Predicción de salario", page_icon="💼", layout="wide")


# ----------------------------------------------------------------- carga
@st.cache_resource(show_spinner="Cargando modelos…")
def cargar_modelos():
    meta = joblib.load(MODELOS / "metadata.joblib")
    modelos = {n: joblib.load(BASE / ruta) for n, ruta in meta["archivos_modelos"].items()}
    return meta, modelos


@st.cache_data
def cargar_test():
    return pd.read_csv(BASE / "test_predictions.csv")


@st.cache_data
def cargar_datos():
    ruta = BASE / "Employee_Salaries_preparado.csv"
    return pd.read_csv(ruta) if ruta.exists() else None


try:
    meta, modelos = cargar_modelos()
    test = cargar_test()
except Exception as e:  # noqa: BLE001
    st.error(
        "No se pudieron cargar los modelos. Verifica que la carpeta `modelos/` y "
        "`test_predictions.csv` estén junto a `app.py` y que la versión de "
        f"scikit-learn coincida con `requirements.txt`.\n\nDetalle: {e}"
    )
    st.stop()

FEATURES = meta["features"]
FINAL = meta["modelo_final_nombre"]
res = meta["resultados"].set_index("Modelo")
EDU = {"Bachillerato (High School)": 0, "Profesional (Bachelor)": 1, "Maestría (Master)": 2, "Doctorado (PhD)": 3}
NIVEL = {1: "1 · Junior", 2: "2 · Semi-senior", 3: "3 · Senior", 4: "4 · Líder / Manager", 5: "5 · Director"}


def usd(v):
    return f"USD {v:,.0f}"


def construir_entrada(exp, nivel, edu, remoto, tamano):
    return pd.DataFrame(
        [{
            "YearsExperience": exp,
            "JobLevel": nivel,
            "Education": edu,
            "Remote": int(remoto),
            "CompanySize_Small": int(tamano == "Pequeña"),
            "CompanySize_Medium": int(tamano == "Mediana"),
            "CompanySize_Large": int(tamano == "Grande"),
        }]
    )[FEATURES]


# ----------------------------------------------------------------- encabezado
st.title("💼 Predicción del salario anual")
st.caption(
    "Modelo de regresión entrenado con 994 empleados · Daniel Jaramillo Guillén y Miguel Ángel López Ríos"
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Modelo seleccionado", FINAL)
c2.metric("RMSE (prueba)", usd(res.loc[FINAL, "RMSE test"]))
c3.metric("MAE (prueba)", usd(res.loc[FINAL, "MAE test"]))
c4.metric("R² (prueba)", f"{res.loc[FINAL, 'R2 test']:.3f}")

tab_pred, tab_comp, tab_mod, tab_datos = st.tabs(
    ["🔮 Predicción", "📊 Comparación de modelos", "🔍 Modelo seleccionado", "🗂️ Datos"]
)

# ----------------------------------------------------------------- 1. predicción
with tab_pred:
    st.subheader("Perfil del empleado")
    col_in, col_out = st.columns([1, 1.3])

    with col_in:
        exp = st.slider("Años de experiencia", 0.0, 40.0, 10.0, 0.5)
        nivel = st.select_slider("Nivel del cargo", options=list(NIVEL), value=3, format_func=NIVEL.get)
        edu_txt = st.selectbox("Nivel educativo", list(EDU), index=1)
        tamano = st.radio("Tamaño de la empresa", ["Pequeña", "Mediana", "Grande"], index=1, horizontal=True)
        remoto = st.toggle("Trabajo remoto", value=False)
        elegido = st.selectbox(
            "Modelo a usar",
            [FINAL] + [m for m in modelos if m != FINAL],
            format_func=lambda m: f"{m} (seleccionado)" if m == FINAL else m,
        )

    entrada = construir_entrada(exp, nivel, EDU[edu_txt], remoto, tamano)
    pred = float(modelos[elegido].predict(entrada)[0])
    # Rango aproximado con los residuos reales del set de prueba (percentiles 5 y 95)
    lo, hi = pred + meta["residuo_p05"], pred + meta["residuo_p95"]

    with col_out:
        st.metric("Salario anual estimado", usd(pred), help=f"Predicción del modelo «{elegido}».")
        st.caption(
            f"Rango probable (90 %): **{usd(lo)} – {usd(hi)}** · calculado con los errores observados del modelo en el set de prueba."
        )
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"prefix": "USD ", "valueformat": ",.0f"},
            gauge={
                "axis": {"range": [test["Salary_real"].min(), test["Salary_real"].max()]},
                "bar": {"color": "#1f77b4"},
                "steps": [{"range": [lo, hi], "color": "#cfe3f5"}],
            },
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**¿Qué predicen los demás modelos para este mismo perfil?**")
    comp = pd.DataFrame({m: [float(mod.predict(entrada)[0])] for m, mod in modelos.items()}, index=["Predicción"]).T
    comp["Tipo"] = res.loc[comp.index, "Tipo"]
    comp = comp.reset_index(names="Modelo").sort_values("Predicción")
    fig = px.bar(comp, x="Predicción", y="Modelo", color="Tipo", orientation="h", text_auto=",.0f",
                 color_discrete_map={"Clásico": "#1f77b4", "Ensamble": "#ff7f0e"})
    fig.update_layout(height=380, xaxis_title="Salario anual (USD)", yaxis_title=None)
    fig.update_xaxes(range=[comp["Predicción"].min() * 0.9, comp["Predicción"].max() * 1.05])
    st.plotly_chart(fig, width="stretch")
    dispersion = comp["Predicción"].max() - comp["Predicción"].min()
    st.caption(f"Diferencia entre la predicción más baja y la más alta: {usd(dispersion)}.")

    with st.expander("Sensibilidad: ¿cómo cambia el salario con la experiencia?"):
        xs = np.arange(0, 40.5, 1.0)
        base = pd.concat([entrada] * len(xs), ignore_index=True)
        base["YearsExperience"] = xs
        curva = pd.DataFrame({"Años de experiencia": xs, "Salario estimado": modelos[elegido].predict(base)})
        fig = px.line(curva, x="Años de experiencia", y="Salario estimado", markers=True)
        fig.add_vline(x=exp, line_dash="dash", line_color="red")
        fig.update_layout(height=320, yaxis_title="USD")
        st.plotly_chart(fig, width="stretch")

# ----------------------------------------------------------------- 2. comparación
with tab_comp:
    st.subheader("Resultados de los 8 modelos")
    st.caption(
        "Hiperparámetros ajustados con el 70 % de los datos (validación cruzada de 5 pliegues, métrica RMSE) "
        "y evaluados una sola vez con el 30 % de prueba."
    )
    tabla = res[["Tipo", "RMSE train", "RMSE CV", "RMSE test", "MAE test", "R2 test", "MAPE test (%)"]].copy()
    st.dataframe(
        tabla.style.format({
            "RMSE train": "{:,.0f}", "RMSE CV": "{:,.0f}", "RMSE test": "{:,.0f}",
            "MAE test": "{:,.0f}", "R2 test": "{:.3f}", "MAPE test (%)": "{:.2f}",
        }).highlight_min(subset=["RMSE test", "MAE test"], color="#c6efce")
          .highlight_max(subset=["R2 test"], color="#c6efce"),
        width="stretch",
    )

    cA, cB = st.columns(2)
    with cA:
        long = tabla.reset_index().melt(id_vars=["Modelo", "Tipo"], value_vars=["RMSE train", "RMSE CV", "RMSE test"],
                                        var_name="Conjunto", value_name="RMSE")
        fig = px.bar(long, x="Modelo", y="RMSE", color="Conjunto", barmode="group",
                     category_orders={"Modelo": list(tabla.index)})
        fig.update_layout(title="RMSE: entrenamiento vs validación cruzada vs prueba", yaxis_title="USD", xaxis_title=None)
        st.plotly_chart(fig, width="stretch")
    with cB:
        fig = px.scatter(tabla.reset_index(), x="RMSE test", y="R2 test", color="Tipo", text="Modelo",
                         color_discrete_map={"Clásico": "#1f77b4", "Ensamble": "#ff7f0e"})
        fig.update_traces(textposition="top center", marker_size=12)
        fig.update_layout(title="Error vs R² en prueba")
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Mejores hiperparámetros encontrados**")
    st.dataframe(
        pd.DataFrame({"Hiperparámetros": {m: ", ".join(f"{k}={v}" for k, v in h.items()) for m, h in meta["hiperparametros"].items()}}),
        width="stretch",
    )

# ----------------------------------------------------------------- 3. modelo seleccionado
with tab_mod:
    st.subheader(f"Diagnóstico de «{FINAL}»")
    p = test[f"pred_{FINAL}"]
    r = test["Salary_real"] - p

    cA, cB = st.columns(2)
    with cA:
        fig = px.scatter(x=test["Salary_real"], y=p, opacity=0.6,
                         labels={"x": "Salario real (USD)", "y": "Salario predicho (USD)"}, title="Real vs predicho (set de prueba)")
        lim = [test["Salary_real"].min(), test["Salary_real"].max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color="red", dash="dash"), name="Ideal"))
        st.plotly_chart(fig, width="stretch")
    with cB:
        fig = px.histogram(r, nbins=30, title="Distribución de los errores (residuos)", labels={"value": "Error (USD)"})
        fig.update_layout(showlegend=False, yaxis_title="Frecuencia")
        st.plotly_chart(fig, width="stretch")

    imp = meta["importancia"].sort_values("Importancia (aumento RMSE USD)")
    fig = px.bar(imp, x="Importancia (aumento RMSE USD)", y="Variable", orientation="h",
                 title="Importancia de variables (aumento del RMSE al permutar cada una)")
    st.plotly_chart(fig, width="stretch")

    st.info(
        f"Error típico: el modelo se equivoca en promedio **{usd(res.loc[FINAL, 'MAE test'])}** (MAE) "
        f"y explica el **{res.loc[FINAL, 'R2 test'] * 100:.1f} %** de la variabilidad del salario en datos que nunca vio."
    )

# ----------------------------------------------------------------- 4. datos
with tab_datos:
    datos = cargar_datos()
    if datos is None:
        st.warning("No se encontró `Employee_Salaries_preparado.csv`; solo se muestra el set de prueba.")
        datos = test.rename(columns={"Salary_real": "Salary"})[FEATURES + ["Salary"]]
    st.subheader("Datos preparados")
    st.dataframe(datos, width="stretch", height=300)
    cA, cB = st.columns(2)
    with cA:
        fig = px.scatter(datos, x="YearsExperience", y="Salary", color=datos["JobLevel"].astype(str), opacity=0.6,
                         labels={"color": "Nivel"}, title="Experiencia vs salario por nivel del cargo")
        st.plotly_chart(fig, width="stretch")
    with cB:
        fig = px.box(datos, x="JobLevel", y="Salary", title="Salario por nivel del cargo")
        st.plotly_chart(fig, width="stretch")

st.divider()
st.caption(
    "Las predicciones son estimaciones estadísticas basadas en un conjunto de datos de 994 registros; "
    "no sustituyen una valoración salarial profesional."
)
