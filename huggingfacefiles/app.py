import gradio as gr
import hopsworks
import joblib
import pandas as pd
import os
import openmeteo_requests
import requests_cache
import requests  # 使用原生 requests
import json
from retry_requests import retry

# ==========================================
# 🔐 安全配置
# ==========================================
if "HOPSWORKS_API_KEY" not in os.environ:
    print("⚠️ 警告: 未检测到 HOPSWORKS_API_KEY")
if "HF_TOKEN" not in os.environ:
    print("⚠️ 警告: 未检测到 HF_TOKEN")

# ==========================================
# 🧠 模型部分 (ML)
# ==========================================
def get_weather_forecast():
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    params = {
        "latitude": 39.4532, # Flores
        "longitude": -31.1274,
        "daily": ["temperature_2m_max", "precipitation_sum", "wind_gusts_10m_max", "wind_direction_10m_dominant"],
        "timezone": "Atlantic/Azores",
        "forecast_days": 7
    }
    
    try:
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        daily = response.Daily()
        
        df = pd.DataFrame({
            "date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s", utc = True),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s", utc = True),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            ),
            "temperature_max": daily.Variables(0).ValuesAsNumpy(),
            "precipitation": daily.Variables(1).ValuesAsNumpy(),
            "wind_gusts": daily.Variables(2).ValuesAsNumpy(),
            "wind_direction": daily.Variables(3).ValuesAsNumpy(),
        })
        df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
        return df
    except Exception as e:
        print(f"❌ Weather API Error: {e}")
        return pd.DataFrame()

def get_prediction_summary():
    print("🤖 Connecting to Hopsworks...")
    try:
        project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
        mr = project.get_model_registry()
        model = mr.get_model(name="azores_wind_model", version=1)
        model_dir = model.download()
        
        model_path = os.path.join(model_dir, "azores_wind_model.pkl")
        trained_model = joblib.load(model_path)
        
        df = get_weather_forecast()
        if df.empty: return "⚠️ Weather data unavailable."

        features = df[['temperature_max', 'precipitation', 'wind_gusts', 'wind_direction']]
        preds = trained_model.predict(features)
        
        summary = ""
        for date, wind, gust in zip(df['date_str'], preds, df['wind_gusts']):
            wind_kmh = max(0, wind)
            summary += f"- {date}: Predicted Wind {wind_kmh:.1f} km/h (Gusts {gust:.1f} km/h)\n"
        return summary
    except Exception as e:
        return f"Failed to fetch prediction data: {str(e)}"

# 初始化数据
print("⏳ Initialization...")
try:
    CACHE_FORECAST = get_prediction_summary()
    print("✅ Data ready!")
except Exception as e:
    CACHE_FORECAST = f"Error: {str(e)}"

# ==========================================
# 🗣️ LLM 部分 (修复 410 Gone 错误)
# ==========================================
def chatbot_response(message, history):
    # 1. 准备 Prompt
    system_prompt = f"""
    You are 'Captain Joao', a boat captain in Flores, Azores.
    
    Here is the WIND FORECAST for the next 7 days:
    {CACHE_FORECAST}
    
    Instructions:
    1. If wind > 30 km/h: Warn that the boat is cancelled.
    2. If wind < 30 km/h: Say it's safe to sail.
    3. Keep it short and use emojis (🌊, 🚤).
    """
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": str(message)})

    # 2. 关键修改：手动请求 router.huggingface.co
    # 这避开了 api-inference.huggingface.co 的 410 错误
    api_url = "https://router.huggingface.co/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN')}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": messages,
        "max_tokens": 500,
        "stream": False
    }

    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"⚠️ API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"⚠️ Connection Error: {str(e)}"

# ==========================================
# 🎨 UI
# ==========================================
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="🚤 Flores-Corvo Boat Forecaster",
    description="Architecture: Gradient Boosting Wind Prediction + LLM (Qwen-2.5)",
    examples=[
        "Will the boat run tomorrow?",
        "Is the weather good for sailing this weekend?",
        "What if the wind is too strong?",
    ],
    cache_examples=False,
    type="messages"  # Gradio 5.x 新参数，兼容性更好
)

if __name__ == "__main__":
    demo.launch()
