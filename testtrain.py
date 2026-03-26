import streamlit as st
import yfinance as yf
from prophet import Prophet
from plotly import graph_objs as go
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

# ตั้งค่าหน้าจอแบบ Wide
st.set_page_config(layout="wide", page_title="NVDA: Prophet vs Linear Regression")

# Header สวยๆ
st.title("🥊 AI vs Statistical Line: ใครแม่นกว่ากัน?")
st.markdown("""
การเปรียบเทียบระหว่าง **Prophet (AI)** ที่มีความยืดหยุ่น กับ **Linear Regression** ที่เป็นเส้นตรง
โดยใช้หุ้น **NVIDIA (NVDA)** เป็นสนามประลอง
""")

# ---------------------------------------------------------
# 1. Load Data & Prepare
# ---------------------------------------------------------
@st.cache_data(ttl=3600) # เพิ่ม ttl=3600 (1 ชม.) เพื่อไม่ให้จำ Cache พังๆ ไว้นานเกินไป
def load_data(ticker):
    try:
        # ปล่อยให้ yfinance จัดการ Session เองตามคำแนะนำของ Error
        stock = yf.Ticker(ticker)
        data = stock.history(start="2024-01-01", end="2025-01-01")
        
        # เช็คว่าดึงข้อมูลมาได้หรือไม่
        if data.empty:
            return pd.DataFrame()
        
        # ดึงเฉพาะคอลัมน์ Close และดึง Date ออกมาจาก Index
        df = data[['Close']].reset_index()
        
        # เปลี่ยนชื่อคอลัมน์ให้เข้าฟอร์แมตที่ Prophet ต้องการ
        df.columns = ["ds", "y"]
        
        # ตัด Timezone ออก (Prophet ไม่รองรับข้อมูลที่มี Timezone)
        if df["ds"].dt.tz is not None:
            df["ds"] = df["ds"].dt.tz_localize(None)
        
        return df

    except Exception as e:
        # ดักจับ Error ไว้ไม่ให้แอปพัง
        st.warning(f"⚠️ เกิดข้อผิดพลาดในการดึงข้อมูลจาก Yahoo Finance: {e}")
        return pd.DataFrame()

ticker = "NVDA"
df = load_data(ticker)

if df.empty:
    st.error("ไม่พบข้อมูล: อาจเกิดจากปัญหาการเชื่อมต่ออินเทอร์เน็ต หรือ Yahoo Finance บล็อคคำขอ")
    st.stop()

# แบ่งข้อมูล (Train 10 เดือน / Test 2 เดือน)
train_df = df[(df["ds"] >= "2024-01-01") & (df["ds"] <= "2024-10-31")].copy()
actual_test_df = df[(df["ds"] >= "2024-11-01") & (df["ds"] <= "2024-12-31")].copy()

# ---------------------------------------------------------
# 2. Modeling
# ---------------------------------------------------------

# --- Model A: Prophet (The Hero) ---
model_prophet = Prophet(daily_seasonality=False, weekly_seasonality=True)
model_prophet.fit(train_df)
future = model_prophet.make_future_dataframe(periods=61)
forecast_prophet = model_prophet.predict(future)
pred_prophet = forecast_prophet[(forecast_prophet["ds"] >= "2024-11-01") & (forecast_prophet["ds"] <= "2024-12-31")][["ds", "yhat", "yhat_lower", "yhat_upper"]]

# --- Model B: Linear Regression (The Villain) ---
train_df['date_ordinal'] = train_df['ds'].map(pd.Timestamp.toordinal)
model_lr = LinearRegression()
model_lr.fit(train_df[['date_ordinal']], train_df['y'])

pred_lr = actual_test_df[["ds"]].copy()
pred_lr['date_ordinal'] = pred_lr['ds'].map(pd.Timestamp.toordinal)
pred_lr['yhat_lr'] = model_lr.predict(pred_lr[['date_ordinal']])

# รวมผลลัพธ์
comparison_df = pd.merge(actual_test_df, pred_prophet, on="ds", how="inner")
comparison_df = pd.merge(comparison_df, pred_lr[['ds', 'yhat_lr']], on="ds", how="inner")

# ---------------------------------------------------------
# 3. Metrics Calculation
# ---------------------------------------------------------
# Prophet Metrics
mae_p = mean_absolute_error(comparison_df['y'], comparison_df['yhat'])
mape_p = mean_absolute_percentage_error(comparison_df['y'], comparison_df['yhat']) * 100

# LR Metrics
mae_lr = mean_absolute_error(comparison_df['y'], comparison_df['yhat_lr'])
mape_lr = mean_absolute_percentage_error(comparison_df['y'], comparison_df['yhat_lr']) * 100

# ---------------------------------------------------------
# 4. Display Metrics (สวยงาม)
# ---------------------------------------------------------
st.markdown("### 📊 ผลลัพธ์ความแม่นยำ (Scoreboard)")

col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    st.subheader("🟢 Prophet")
    st.metric("ความคลาดเคลื่อน (MAE)", f"${mae_p:.2f}")
    st.metric("คิดเป็น % (MAPE)", f"{mape_p:.2f}%")

with col2:
    st.subheader("🔴 Linear Reg")
    st.metric("ความคลาดเคลื่อน (MAE)", f"${mae_lr:.2f}", delta=f"{mae_lr-mae_p:.2f} (แย่กว่า)", delta_color="inverse")
    st.metric("คิดเป็น % (MAPE)", f"{mape_lr:.2f}%", delta=f"{mape_lr-mape_p:.2f}%", delta_color="inverse")

with col3:
    st.info(f"""
    **📝 คำอธิบายความหมาย:**
    * **MAE (Mean Absolute Error):** โดยเฉลี่ยแล้ว โมเดลทายราคาผิดไปกี่ดอลลาร์
      - Prophet ทายผิดเฉลี่ย **${mae_p:.2f}**
      - Linear Reg ทายผิดเฉลี่ย **${mae_lr:.2f}**
    * **MAPE (Percentage Error):** ความผิดพลาดคิดเป็นกี่ % ของราคาจริง
      - Prophet พลาดไปแค่ **{mape_p:.2f}%** (ถือว่าแม่นยำมากสำหรับหุ้น)
    """)

st.markdown("---")

# ---------------------------------------------------------
# 5. Advanced Visualization (สวยและดูง่าย)
# ---------------------------------------------------------
fig = go.Figure()

# 5.1 Training Data (สีจางๆ ให้รู้ว่าเป็นอดีต)
fig.add_trace(go.Scatter(
    x=train_df["ds"], y=train_df["y"],
    mode='lines', name="History (Train)",
    line=dict(color="gray", width=1), opacity=0.3
))

# 5.2 Actual Price (ช่วง Test - สีดำเข้ม ให้เห็นชัดๆ)
fig.add_trace(go.Scatter(
    x=actual_test_df["ds"], y=actual_test_df["y"],
    mode='lines', name="Actual Price (เฉลย)",
    line=dict(color="black", width=3)
))

# 5.3 Linear Regression (เส้นประสีแดง)
fig.add_trace(go.Scatter(
    x=pred_lr["ds"], y=pred_lr["yhat_lr"],
    mode='lines', name="Linear Regression",
    line=dict(color="#FF4136", width=3, dash='dot')
))

# 5.4 Prophet (เส้นสีเขียว + พื้นที่ระบาย)
# พื้นที่ขอบเขตล่าง (โปร่งใส)
fig.add_trace(go.Scatter(
    x=pred_prophet["ds"], y=pred_prophet["yhat_lower"],
    mode='lines', line=dict(width=0), showlegend=False, hoverinfo='skip'
))
# พื้นที่ขอบเขตบน (ระบายสีเขียวจางๆ)
fig.add_trace(go.Scatter(
    x=pred_prophet["ds"], y=pred_prophet["yhat_upper"],
    mode='lines', line=dict(width=0), fill='tonexty',
    fillcolor='rgba(0, 204, 150, 0.2)', name="Prophet Confidence", hoverinfo='skip'
))
# เส้นหลัก Prophet
fig.add_trace(go.Scatter(
    x=pred_prophet["ds"], y=pred_prophet["yhat"],
    mode='lines', name="Prophet Forecast",
    line=dict(color="#00CC96", width=4)
))

# เส้นแบ่งจุดเริ่มต้น Test
fig.add_vline(x=pd.to_datetime("2024-11-01").timestamp() * 1000, 
              line_width=2, line_dash="dash", line_color="gray")

# Annotation บอกจุดเริ่ม
fig.add_annotation(
    x=pd.to_datetime("2024-11-01"), y=train_df['y'].max(),
    text="จุดเริ่มทำนาย (Start Testing)", showarrow=True, arrowhead=1
)

# Layout สวยๆ
fig.update_layout(
    title="📈 การเปรียบเทียบกราฟช่วง Test (Nov - Dec 2024)",
    xaxis_title="Date", yaxis_title="Price (USD)",
    height=600,
    template="plotly_white",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(
        rangeslider=dict(visible=True), # เพิ่ม Slider ด้านล่าง
        type="date"
    )
)

# Default Zoom ให้โฟกัสช่วง 4 เดือนล่าสุด (เห็นปลาย Train นิดหน่อย + Test ทั้งหมด)
fig.update_xaxes(range=["2024-09-01", "2025-01-01"])

st.plotly_chart(fig, use_container_width=True)
