# Modelo SARIMAX para Predicciones Climáticas adaptado al Ciclo del Cacao

Este modelo utiliza técnicas de series temporales con SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables) para realizar predicciones climáticas considerando el ciclo vegetativo del cacao.

## Características

- Predicción de variables climáticas (temperatura, precipitación) utilizando el modelo SARIMAX
- Incorporación del ciclo vegetativo del cacao como variables exógenas
- Análisis de estacionalidad y descomposición de series temporales
- Visualización de resultados y diagnósticos del modelo
- Generación de recomendaciones para el cultivo basadas en las predicciones

## Ciclo Vegetativo del Cacao Considerado

El modelo tiene en cuenta las siguientes características del ciclo del cacao:

- **Floración**: 2-3 períodos al año, principalmente en temporadas lluviosas
- **Desarrollo de frutos**: 5-6 meses desde la polinización hasta madurez
- **Cosecha**: 2 temporadas principales al año
- **Necesidades climáticas óptimas**: Temperatura y precipitación ideal para cada fase

## Requisitos

```
pandas
numpy
matplotlib
seaborn
statsmodels
scikit-learn
```

Puede instalar todas las dependencias con:

```
pip install -r requirements.txt
```

## Uso

1. Prepare un archivo CSV con datos climáticos históricos (temperatura, precipitación, etc.) que incluya una columna de fecha.

2. Ejecute el script principal:

```
python modelo_cacao_clima_sarimax.py
```

3. Siga las instrucciones en pantalla:
   - Ingrese la ruta de su archivo CSV con datos climáticos
   - Seleccione la variable climática a modelar
   - Configure los parámetros del modelo SARIMAX según sea necesario

4. El programa generará:
   - Gráficos de análisis de la serie temporal
   - Diagnósticos del modelo
   - Predicciones para los próximos 24 meses
   - Un archivo CSV con predicciones y recomendaciones

## Formato de Datos

El archivo CSV debe tener al menos:
- Una columna de fecha (puede llamarse 'fecha', 'date' o similar)
- Una o más columnas con variables climáticas (temperatura, precipitación, etc.)

Ejemplo:
```
fecha,temperatura,precipitacion
2020-01-01,24.5,120
2020-01-02,25.1,0
...
```

## Interpretación de Resultados

El modelo genera varios archivos de salida:
- **serie_temporal.png**: Visualización de la serie temporal original
- **descomposicion_estacional.png**: Descomposición de la serie en tendencia, estacionalidad y residuos
- **acf_pacf.png**: Gráficos de autocorrelación para identificar parámetros óptimos
- **diagnostico_modelo.png**: Diagnósticos del modelo SARIMAX
- **evaluacion_modelo.png**: Comparación de predicciones vs valores reales en datos de prueba
- **prediccion_cacao.png**: Predicciones futuras con ciclo del cacao
- **predicciones_cacao.csv**: Archivo CSV con predicciones y recomendaciones

## Personalización

Puede ajustar el ciclo vegetativo del cacao en la función `ciclo_cacao()` para adaptarlo a su región específica o variedad de cacao. 