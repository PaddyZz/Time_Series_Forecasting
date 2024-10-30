# Time Series Forecasting(LSTM)

## Introduction

Weather forecasting has always been a critical aspect of our daily lives, influencing various sectors such as agriculture, transportation, and disaster management. Accurate predictions of weather parameters, particularly temperature and humidity, play a vital role in planning and decision-making processes. As climate change continues to introduce variability and uncertainty in weather patterns, the need for reliable forecasting methods becomes increasingly important.

In recent years, advancements in machine learning and data analytics have opened new avenues for improving weather prediction models. Traditional statistical methods, while useful, often fall short in capturing complex non-linear relationships within the data. By leveraging time series analysis and modern computational techniques, we can enhance the accuracy of forecasts and provide more timely insights into atmospheric conditions.

This project focuses on developing a time series forecasting model for predicting temperature, humidity levels and other factors based on historical weather data. This project will choose the most effective approach (LSTM) for our prediction task.


## Get Started
```python
# create a new conda env and install the dependency file
conda create --name <env-name> python=3.10.3
conda activate <env-name>
pip install -r requirements.txt
```

## Execute
```python
python main.py
#or
python main.py [-c | --config] <params=value>
```

## Optional Parameters
- `BATCH_SIZE`: Set the number of samples per gradient update (e.g., `32`), type is integer, default is `32`.

- `INPUT_WIDTH`: Set the width of the input sequence (e.g., `24`), type is integer, default is `24`.

- `LABEL_WIDTH`: Set the width of the output sequence (e.g., `24`), type is integer, default is `24`.

- `MAX_EPOCHS`: Set the maximum number of epochs for training (e.g., `20`), type is integer, default is `1`.

- `MAX_SUBPLOTS`: Set the maximum number of subplots to display in visualizations (e.g., `1`), type is integer, default is `1`.

- `OUT_STEPS`: Set the number of steps to predict in the future (e.g., `24`), type is integer, default is `1`.

- `PLOT_ORIGIN`: Set whether to plot line with the original data unit (e.g., `false`), type is boolean, default is `false`.

- `SAVE_KERAS`: Set whether to save the Keras model after training (e.g., `false`), type is boolean, default is `false`.

- `SHIFT`: Set the number of days to be predicted from the input data (e.g., `24`), type is integer, default is `1`.
  
- `PLOT_COL`: Specify the column to visualize in plots (e.g., `rho (g/m³)`), type is string, default is `'rho (g/m³)'`.

## PLOT_COL column parameters
- **p (mbar)**: Measures atmospheric pressure, indicating weather systems.
- **T (degC)**: Represents the air temperature, critical for weather forecasting.
- **Tpot (K)**: Indicates potential temperature, important for understanding stability in the atmosphere.
- **Tdew (degC)**: Shows the dew point, which helps in assessing humidity and comfort levels.
- **rh (%)**: Indicates relative humidity, essential for predicting precipitation and fog.
- **VPmax (mbar)**: Maximum vapor pressure, representing the upper limit of moisture in the air.
- **VPact (mbar)**: Actual vapor pressure, showing the current moisture content.
- **VPdef (mbar)**: Vapor pressure deficit, indicating dryness in the air.
- **sh (g/kg)**: Specific humidity, quantifying the amount of water vapor present in the air.
- **H2OC (mmol/mol)**: Water vapor concentration, providing insight into atmospheric moisture.
- **rho (g/m³)**: Air density, affecting buoyancy and weather patterns.
- **wv (m/s)**: Wind velocity, indicating wind strength.
- **max. wv (m/s)**: Maximum wind velocity, highlighting peak wind conditions.
- **wd (deg)**: Wind direction, crucial for understanding weather patterns and forecasting.


## Dockerfile

```
FROM python:3.10.3
RUN pip install virtualenv
RUN virtualenv /env
ENV VIRTUAL_ENV=/env
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
WORKDIR /app
COPY . /app
RUN python -m pip install --no-cache-dir -r requirements.txt
CMD ["python", "main.py"]
```

```bash
docker build
```

## Blog
[Time Series Forecasting LSTM](https://paddyzz.github.io/projects/time_series_forecasting/)

[Deploy Model to Kubeflow and implement workflow](https://paddyzz.github.io/projects/Config_Kubeflow/)

## Reference
[Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
