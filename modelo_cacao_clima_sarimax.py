import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Configuración para visualización más atractiva
plt.style.use('ggplot')
sns.set(style="whitegrid")

def cargar_datos_clima(ruta_archivo):
    """
    Carga los datos climáticos desde un archivo CSV.
    Asume formato con fecha en la primera columna y variables climáticas.
    """
    try:
        # Cargar datos
        df = pd.read_csv(ruta_archivo)
        
        # Convertir columna de fecha a datetime
        if 'fecha' in df.columns:
            df['fecha'] = pd.to_datetime(df['fecha'])
            df.set_index('fecha', inplace=True)
        elif 'date' in df.columns:
            df['fecha'] = pd.to_datetime(df['date'])
            df.set_index('fecha', inplace=True)
        else:
            # Intentar detectar columna de fecha
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df['fecha'] = pd.to_datetime(df[col])
                        df.set_index('fecha', inplace=True)
                        df = df.drop(col, axis=1)
                        break
                    except:
                        continue
        
        print(f"Datos cargados con éxito: {len(df)} registros")
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def analizar_estacionalidad(serie_tiempo, periodo=365):
    """
    Analiza la estacionalidad de la serie temporal.
    
    Args:
        serie_tiempo: Serie temporal a analizar
        periodo: Período estacional (365 para datos diarios anuales)
    """
    # Descomposición estacional
    descomposicion = seasonal_decompose(serie_tiempo, model='additive', period=periodo)
    
    # Visualización
    plt.figure(figsize=(12, 10))
    plt.subplot(411)
    plt.plot(serie_tiempo, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(descomposicion.trend, label='Tendencia')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(descomposicion.seasonal, label='Estacionalidad')
    plt.legend(loc='best')
    plt.subplot(414)
    plt.plot(descomposicion.resid, label='Residuos')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('descomposicion_estacional.png')
    plt.close()
    
    # ACF y PACF para identificar parámetros AR y MA
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(serie_tiempo, ax=ax1, lags=40)
    plot_pacf(serie_tiempo, ax=ax2, lags=40)
    plt.savefig('acf_pacf.png')
    plt.close()
    
    return descomposicion

def ciclo_cacao():
    """
    Genera un dataframe con las fases del ciclo vegetativo del cacao.
    
    El ciclo vegetativo del cacao incluye:
    - Floración: 2-3 veces al año (principal en meses lluviosos)
    - Desarrollo de frutos: 5-6 meses desde la polinización
    - Cosecha: 2 temporadas principales al año
    
    Returns:
        DataFrame con indicadores mensuales de las fases del cacao
    """
    # Crear índice mensual para un año
    meses = pd.date_range(start='2023-01-01', periods=12, freq='M')
    
    # Crear DataFrame
    ciclo_df = pd.DataFrame(index=meses)
    ciclo_df.index.name = 'fecha'
    
    # Floración - Suele ocurrir mayormente en temporadas de lluvia
    # (típicamente en regiones tropicales ocurre en dos temporadas)
    ciclo_df['floracion'] = [0.3, 0.2, 0.4, 0.8, 0.9, 0.6, 0.5, 0.7, 0.9, 0.8, 0.6, 0.4]
    
    # Desarrollo de frutos - comienza después de la polinización (5-6 meses de desarrollo)
    ciclo_df['desarrollo_frutos'] = [0.7, 0.6, 0.4, 0.3, 0.5, 0.7, 0.8, 0.6, 0.5, 0.7, 0.8, 0.9]
    
    # Cosecha - Mayormente 2 temporadas principales al año
    ciclo_df['cosecha'] = [0.8, 0.7, 0.4, 0.2, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.8, 0.7]
    
    # Necesidades climáticas por fase
    # Valores aproximados para temperatura óptima (°C)
    ciclo_df['temp_optima'] = [24, 24, 25, 25, 26, 26, 26, 25, 25, 24, 24, 24]
    
    # Valores aproximados para precipitación óptima (mm)
    ciclo_df['precip_optima'] = [150, 120, 100, 110, 130, 150, 160, 170, 180, 190, 180, 160]
    
    return ciclo_df

def crear_modelo_sarimax(serie_tiempo, exog=None, order=(1,1,1), seasonal_order=(1,1,1,12)):
    """
    Crea y entrena un modelo SARIMAX.
    
    Args:
        serie_tiempo: Serie temporal de la variable a predecir
        exog: Variables exógenas (opcional)
        order: Orden del componente ARIMA (p,d,q)
        seasonal_order: Orden del componente estacional (P,D,Q,s)
    
    Returns:
        Modelo SARIMAX ajustado
    """
    modelo = SARIMAX(
        serie_tiempo,
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    resultados = modelo.fit(disp=False)
    print(f"AIC: {resultados.aic}")
    
    # Visualización del diagnóstico del modelo
    fig = resultados.plot_diagnostics(figsize=(12, 8))
    plt.savefig('diagnostico_modelo.png')
    plt.close()
    
    return resultados

def evaluar_modelo(modelo, serie_tiempo, exog=None, train_size=0.8):
    """
    Evalúa el modelo SARIMAX usando datos de entrenamiento y prueba.
    
    Args:
        modelo: Modelo SARIMAX entrenado
        serie_tiempo: Serie temporal completa
        exog: Variables exógenas (opcional)
        train_size: Proporción de datos para entrenamiento
    
    Returns:
        Error cuadrático medio (RMSE)
    """
    # Dividir datos en entrenamiento y prueba
    n = len(serie_tiempo)
    train_idx = int(n * train_size)
    
    train = serie_tiempo[:train_idx]
    test = serie_tiempo[train_idx:]
    
    if exog is not None:
        exog_train = exog[:train_idx]
        exog_test = exog[train_idx:]
    else:
        exog_train = None
        exog_test = None
    
    # Entrenar con datos de entrenamiento
    modelo_eval = SARIMAX(
        train,
        exog=exog_train,
        order=modelo.model.order,
        seasonal_order=modelo.model.seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    resultados_eval = modelo_eval.fit(disp=False)
    
    # Hacer predicciones en datos de prueba
    predicciones = resultados_eval.forecast(steps=len(test), exog=exog_test)
    
    # Calcular RMSE
    rmse = sqrt(mean_squared_error(test, predicciones))
    print(f"RMSE: {rmse}")
    
    # Visualizar resultados
    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Entrenamiento')
    plt.plot(test.index, test, label='Prueba')
    plt.plot(test.index, predicciones, label='Predicciones', color='red')
    plt.title(f'Predicciones del modelo SARIMAX (RMSE: {rmse:.2f})')
    plt.legend()
    plt.savefig('evaluacion_modelo.png')
    plt.close()
    
    return rmse

def predecir_clima_cacao(modelo, pasos=24, exog_future=None):
    """
    Realiza predicciones climáticas utilizando el modelo SARIMAX
    y las relaciona con el ciclo del cacao.
    
    Args:
        modelo: Modelo SARIMAX entrenado
        pasos: Número de pasos (meses) a predecir
        exog_future: Variables exógenas futuras
    
    Returns:
        DataFrame con predicciones y recomendaciones para cultivo
    """
    # Realizar predicciones
    predicciones = modelo.forecast(steps=pasos, exog=exog_future)
    
    # Generar fechas para predicciones
    ultima_fecha = modelo.model.endog_names
    fechas_futuras = pd.date_range(start=ultima_fecha, periods=pasos, freq='M')
    
    # Crear dataframe de predicciones
    pred_df = pd.DataFrame({
        'prediccion': predicciones
    }, index=fechas_futuras)
    
    # Obtener ciclo del cacao
    ciclo = ciclo_cacao()
    
    # Expandir el ciclo para cubrir el período de predicción
    ciclos_necesarios = (pasos // 12) + 1
    ciclo_expandido = pd.concat([ciclo] * ciclos_necesarios)
    ciclo_expandido = ciclo_expandido.iloc[:pasos]
    
    # Alinear índices
    ciclo_expandido.index = fechas_futuras
    
    # Combinar predicciones con ciclo
    resultados = pd.concat([pred_df, ciclo_expandido], axis=1)
    
    # Agregar recomendaciones basadas en predicciones y ciclo
    resultados['recomendacion'] = ''
    
    # Ejemplo simple de recomendaciones (debe adaptarse según variable climática)
    for idx, row in resultados.iterrows():
        if row['floracion'] > 0.7:  # Alta floración
            if row['prediccion'] < row['precip_optima'] * 0.8:
                resultados.loc[idx, 'recomendacion'] = 'Riego suplementario recomendado para floración'
            elif row['prediccion'] > row['precip_optima'] * 1.2:
                resultados.loc[idx, 'recomendacion'] = 'Posible exceso de lluvia, mejorar drenaje'
        
        if row['cosecha'] > 0.7:  # Alta cosecha
            if row['prediccion'] > row['precip_optima'] * 1.1:
                resultados.loc[idx, 'recomendacion'] = 'Planificar cosecha anticipada por exceso de lluvia'
    
    # Visualizar predicciones y ciclo del cacao
    plt.figure(figsize=(14, 10))
    
    # Predicción climática
    ax1 = plt.subplot(311)
    plt.plot(resultados.index, resultados['prediccion'], 'b-', label='Predicción Climática')
    plt.fill_between(resultados.index, resultados['prediccion']*0.9, resultados['prediccion']*1.1, color='blue', alpha=0.2)
    plt.legend()
    plt.title('Predicción Climática')
    
    # Fases del cacao
    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(resultados.index, resultados['floracion'], 'g-', label='Floración')
    plt.plot(resultados.index, resultados['desarrollo_frutos'], 'r-', label='Desarrollo Frutos')
    plt.plot(resultados.index, resultados['cosecha'], 'y-', label='Cosecha')
    plt.legend()
    plt.title('Ciclo Vegetativo del Cacao')
    
    # Necesidades óptimas
    ax3 = plt.subplot(313, sharex=ax1)
    plt.plot(resultados.index, resultados['precip_optima'], 'b--', label='Precipitación Óptima')
    plt.plot(resultados.index, resultados['prediccion'], 'b-', label='Predicción')
    plt.legend()
    plt.title('Comparación Predicción vs. Necesidad Óptima')
    
    plt.tight_layout()
    plt.savefig('prediccion_cacao.png')
    plt.close()
    
    return resultados

def main():
    """
    Función principal que ejecuta todo el proceso.
    """
    print("Modelo SARIMAX para Predicciones Climáticas adaptado al Ciclo del Cacao")
    print("-"*80)
    
    # El usuario deberá proporcionar la ruta del archivo de datos climáticos
    # Por ejemplo: temperatura o precipitación diaria/mensual
    ruta_archivo = input("Ingrese la ruta del archivo de datos climáticos (CSV): ")
    
    # Cargar datos
    df = cargar_datos_clima(ruta_archivo)
    
    if df is None:
        print("No se pudieron cargar los datos. Finalizando programa.")
        return
    
    # Mostrar información general de los datos
    print("\nResumen de los datos:")
    print(df.describe())
    
    # Seleccionar variable climática a modelar
    print("\nColumnas disponibles:")
    for i, col in enumerate(df.columns):
        print(f"{i+1}. {col}")
    
    idx_var = int(input("\nSeleccione el número de la variable a modelar: ")) - 1
    var_modelar = df.columns[idx_var]
    
    print(f"\nModelando la variable: {var_modelar}")
    
    # Convertir a frecuencia mensual si es necesario
    if df.index.freq != 'M' and len(df) > 365:
        print("Convirtiendo datos a frecuencia mensual...")
        df_mensual = df[var_modelar].resample('M').mean()
    else:
        df_mensual = df[var_modelar]
    
    # Visualizar la serie temporal
    plt.figure(figsize=(12, 6))
    plt.plot(df_mensual)
    plt.title(f'Serie Temporal - {var_modelar}')
    plt.tight_layout()
    plt.savefig('serie_temporal.png')
    plt.close()
    
    # Analizar estacionalidad
    print("\nAnalizando estacionalidad...")
    descomposicion = analizar_estacionalidad(df_mensual, periodo=12)
    
    # Obtener ciclo del cacao
    print("\nGenerando ciclo vegetativo del cacao...")
    ciclo_df = ciclo_cacao()
    
    # Alinear ciclo con datos históricos para usarlo como variable exógena
    print("\nPreparando variables exógenas basadas en el ciclo del cacao...")
    # Expandir ciclo para cubrir período histórico
    n_ciclos = len(df_mensual) // 12 + 1
    ciclo_historico = pd.concat([ciclo_df] * n_ciclos)
    ciclo_historico = ciclo_historico.iloc[:len(df_mensual)]
    ciclo_historico.index = df_mensual.index[-len(ciclo_historico):]
    
    # Usar fases del ciclo como variables exógenas
    exog = ciclo_historico[['floracion', 'desarrollo_frutos', 'cosecha']]
    
    # Configurar parámetros SARIMAX
    print("\nConfigurando modelo SARIMAX...")
    p = int(input("Ingrese el orden autorregresivo (p) [recomendado 1]: ") or "1")
    d = int(input("Ingrese el orden de diferenciación (d) [recomendado 1]: ") or "1")
    q = int(input("Ingrese el orden de media móvil (q) [recomendado 1]: ") or "1")
    P = int(input("Ingrese el orden autorregresivo estacional (P) [recomendado 1]: ") or "1")
    D = int(input("Ingrese el orden de diferenciación estacional (D) [recomendado 1]: ") or "1")
    Q = int(input("Ingrese el orden de media móvil estacional (Q) [recomendado 1]: ") or "1")
    s = int(input("Ingrese el período estacional (s) [recomendado 12 para datos mensuales]: ") or "12")
    
    # Entrenar modelo
    print("\nEntrenando modelo SARIMAX...")
    modelo = crear_modelo_sarimax(
        df_mensual,
        exog=exog,
        order=(p, d, q),
        seasonal_order=(P, D, Q, s)
    )
    
    # Evaluar modelo
    print("\nEvaluando modelo...")
    rmse = evaluar_modelo(modelo, df_mensual, exog=exog)
    
    # Predecir para los próximos 24 meses
    print("\nGenerando predicciones para los próximos 24 meses...")
    # Crear datos exógenos futuros basados en el ciclo del cacao
    ciclo_futuro = pd.concat([ciclo_df] * 2)  # 2 años
    
    exog_futuro = ciclo_futuro[['floracion', 'desarrollo_frutos', 'cosecha']].values
    
    predicciones = predecir_clima_cacao(modelo, pasos=24, exog_future=exog_futuro)
    
    print("\nPredicciones y recomendaciones generadas con éxito.")
    print("Revise los gráficos generados para análisis visual.")
    
    # Guardar predicciones en CSV
    predicciones.to_csv('predicciones_cacao.csv')
    print("Predicciones guardadas en 'predicciones_cacao.csv'")

if __name__ == "__main__":
    main() 