import gradio as gr
import hopsworks
import joblib
import pandas as pd
import os
import openmeteo_requests
import requests_cache
from retry_requests import retry
from huggingface_hub import InferenceClient

# ==========================================
# 🔐 安全配置
# ==========================================
# 在 Hugging Face Spaces 的 Settings -> Secrets 中配置这些密钥
# 这样代码在云端运行时会自动读取，不需要硬编码
if "HOPSWORKS_API_KEY" not in os.environ:
    print("⚠️ 警告: 未检测到 HOPSWORKS_API_KEY，可能会导致登录失败。")

if "HF_TOKEN" not in os.environ:
    print("⚠️ 警告: 未检测到 HF_TOKEN，LLM 可能无法响应。")

# ==========================================
# 🧠 模型部分: 获取天气并预测风速
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

def get_prediction_summary():
    """
    运行 ML 模型，返回未来7天的风速预测摘要文本
    """
    print("🤖 Connecting to Hopsworks to download model...")
    try:
        # 修改点：在 Spaces 中必须显式传入 project 和 api_key_value
        # 只要环境变量里设置了 HOPSWORKS_API_KEY，login() 通常会自动识别，
        # 但显式从 env 读取更稳妥。
        project = hopsworks.login(api_key_value=os.environ.get("HOPSWORKS_API_KEY"))
        
        mr = project.get_model_registry()
        model = mr.get_model(name="azores_wind_model", version=1)
        model_dir = model.download()
        
        model_path = os.path.join(model_dir, "azores_wind_model.pkl")
        trained_model = joblib.load(model_path)
        
        df = get_weather_forecast()
        features = df[['temperature_max', 'precipitation', 'wind_gusts', 'wind_direction']]
        
        preds = trained_model.predict(features)
        
        summary = ""
        for date, wind, gust in zip(df['date_str'], preds, df['wind_gusts']):
            wind_kmh = max(0, wind)
            summary += f"- {date}: Predicted Wind {wind_kmh:.1f} km/h (Gusts {gust:.1f} km/h)\n"
        
        return summary
    except Exception as e:
        return f"Failed to fetch prediction data: {str(e)}"

# 初始化
print("⏳ Initialization: Fetching latest data and model...")
# 注意：在 Space 构建阶段可能会失败，所以加个简单的异常处理
try:
    CACHE_FORECAST = get_prediction_summary()
except:
    CACHE_FORECAST = "Waiting for secrets configuration..."
print("✅ Data ready!")

# ==========================================
# 🗣️ LLM 部分
# ==========================================
def chatbot_response(message, history):
    client = InferenceClient(
        "Qwen/Qwen2.5-7B-Instruct", 
        token=os.environ.get("HF_TOKEN")
    )
    
    system_prompt = f"""
    You are 'Captain Joao', an experienced and humorous speedboat captain in Flores, Azores.
    
    Here is the REAL wind forecast for the next 7 days (based on ML predictions):
    {CACHE_FORECAST}
    
    **Rules:**
    1. Answer based on the data above.
    2. If predicted wind > 30 km/h: Be apologetic and warn that the boat is cancelled due to high waves.
    3. If predicted wind < 30 km/h: Be cheerful and say it's a perfect day for sailing!
    4. Keep answers short, nautical, and use emojis like 🌊, 🚤, ⚓.
    """

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    
    # Append history
    messages.extend(history)
    messages.append({"role": "user", "content": message})

    partial_message = ""
    try:
        for token in client.chat_completion(messages, max_tokens=500, stream=True):
            if token.choices[0].delta.content:
                partial_message += token.choices[0].delta.content
                yield partial_message
    except Exception as e:
        yield f"⚠️ Error: {str(e)}"

# ==========================================
# 🎨 Gradio 界面
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
    cache_examples=False
)

if __name__ == "__main__":
    demo.launch()
