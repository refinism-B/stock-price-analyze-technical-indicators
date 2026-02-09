import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # 新增：用於繪製子圖
import requests
import datetime
import json
from openai import OpenAI
import google.generativeai as genai
from secret import FMP_KEY, GOOGLE_KEY, OPENAI_KEY


# --- 頁面基本設定 ---
st.set_page_config(
    page_title="AI 股票趨勢分析系統 (Pro)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 輔助函數區 ---


@st.cache_data(ttl=3600)
def get_stock_data(symbol, api_key, start_date, end_date):
    """
    從 FMP API 獲取指定日期範圍的股票歷史數據
    """
    s_date = start_date.strftime('%Y-%m-%d')
    e_date = end_date.strftime('%Y-%m-%d')

    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?symbol={symbol}&from={s_date}&to={e_date}&apikey={api_key}"

    try:
        response = requests.get(url, timeout=15)
        try:
            data = response.json()
        except json.JSONDecodeError:
            return None, f"API 回傳非 JSON 格式 (Status: {response.status_code})"

        if isinstance(data, dict) and "Error Message" in data:
            return None, f"FMP API 錯誤: {data['Error Message']}"

        if response.status_code != 200:
            return None, f"HTTP 請求失敗 (代碼: {response.status_code})"

        df = None
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
        elif isinstance(data, dict) and 'historical' in data:
            df = pd.DataFrame(data['historical'])
        elif isinstance(data, dict) and symbol in data:
            df = pd.DataFrame(data[symbol])

        if df is None or df.empty:
            return None, f"該日期區間 ({s_date} ~ {e_date}) 無交易數據，或股票代碼錯誤。"

        df.columns = [c.lower() for c in df.columns]
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]

        if missing_cols:
            return None, f"數據缺少必要欄位: {', '.join(missing_cols)}"

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        return df, None

    except requests.exceptions.RequestException as e:
        return None, f"網路連線錯誤: {str(e)}"
    except Exception as e:
        return None, f"程式處理錯誤: {str(e)}"


def calculate_technical_indicators(df, rsi_days=14):
    """
    計算移動平均線與 RSI 技術指標
    """
    df = df.copy()

    # 1. 計算 MA
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

    # 2. 計算 RSI (新增功能)
    # 價格變化
    delta = df['close'].diff()

    # 分離漲跌
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)

    # 計算平均漲跌 (使用 Wilder's Smoothing，效果比簡單平均好)
    avg_gain = gain.ewm(com=rsi_days - 1, min_periods=rsi_days).mean()
    avg_loss = loss.ewm(com=rsi_days - 1, min_periods=rsi_days).mean()

    # 計算 RS
    rs = avg_gain / avg_loss

    # 計算 RSI
    df['RSI'] = 100 - (100 / (1 + rs))

    return df


def filter_data_by_date(df, start_date, end_date):
    """
    根據使用者選擇的日期範圍過濾數據
    """
    mask = (df['date'].dt.date >= start_date) & (
        df['date'].dt.date <= end_date)
    return df.loc[mask].reset_index(drop=True)


def create_chart(df, symbol):
    """
    使用 Plotly 繪製 K 線圖 (上) 與 RSI 指標圖 (下)
    """
    # 建立包含兩個子圖的圖表 (Row 1: K線, Row 2: RSI)
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],  # 上圖佔 70%, 下圖佔 30%
        subplot_titles=(f'{symbol} 股價走勢', 'RSI 相對強弱指標')
    )

    # --- 主圖：K線與均線 ---
    # K線
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K線',
        increasing_line_color='#26A69A',
        decreasing_line_color='#EF5350'
    ), row=1, col=1)

    # 移動平均線
    colors = {'MA5': '#FF9800', 'MA10': '#2196F3',
              'MA20': '#9C27B0', 'MA60': '#607D8B'}
    for ma, color in colors.items():
        if ma in df.columns:
            ma_data = df.dropna(subset=[ma])
            if not ma_data.empty:
                fig.add_trace(go.Scatter(
                    x=ma_data['date'],
                    y=ma_data[ma],
                    mode='lines',
                    name=ma,
                    line=dict(color=color, width=1.5)
                ), row=1, col=1)

    # --- 子圖：RSI ---
    if 'RSI' in df.columns:
        rsi_data = df.dropna(subset=['RSI'])
        if not rsi_data.empty:
            # RSI 線條
            fig.add_trace(go.Scatter(
                x=rsi_data['date'],
                y=rsi_data['RSI'],
                mode='lines',
                name='RSI',
                line=dict(color='#2962FF', width=2)  # 藍色線條
            ), row=2, col=1)

            # 超買線 (70) - 紅色虛線
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="超買 (70)", annotation_position="top left", row=2, col=1)

            # 超賣線 (30) - 綠色虛線
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="超賣 (30)", annotation_position="bottom left", row=2, col=1)

            # 填充背景色 (選用，增加視覺辨識度)
            # 這裡簡單處理，Plotly 對於區間填色較複雜，我們先保持線條清晰

    # --- 圖表佈局設定 ---
    fig.update_layout(
        title=f'{symbol} 技術分析圖表',
        yaxis_title='價格 (USD)',
        yaxis2_title='RSI',  # 第二個 Y 軸標題
        xaxis2_title='日期',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,  # 增加高度以容納兩個圖表
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        xaxis_rangeslider_visible=False  # 隱藏預設的 range slider 避免混亂
    )

    # 設定 RSI Y軸範圍固定在 0-100
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    return fig

# --- AI 分析相關函數 ---


def get_ai_prompts(symbol, df, start_date, end_date):
    """
    產生包含 RSI 分析的 System Prompt 與 User Prompt
    """
    # 準備最近幾筆數據 (包含 RSI)
    recent_cols = ['date', 'open', 'high', 'low',
                   'close', 'volume', 'MA5', 'MA20', 'MA60', 'RSI']
    # 確保欄位存在
    cols_to_use = [c for c in recent_cols if c in df.columns]

    recent_data = df.tail(5)[cols_to_use].to_dict(orient='records')

    # 數據格式化
    for record in recent_data:
        record['date'] = record['date'].strftime('%Y-%m-%d')
        for key, value in record.items():
            if pd.isna(value):
                record[key] = "N/A"
            elif isinstance(value, float):
                record[key] = round(value, 2)  # 數值保留兩位小數

    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    price_change = ((end_price - start_price) / start_price) * 100

    # 取得最新一筆 RSI
    latest_rsi = df.iloc[-1]['RSI'] if 'RSI' in df.columns and not pd.isna(
        df.iloc[-1]['RSI']) else "N/A"

    data_json = json.dumps(recent_data, indent=2)

    system_prompt = """
    你是一位專業的技術分析師，專精於股票技術分析，特別擅長結合「價格趨勢」與「RSI 動能指標」進行綜合研判。
    
    你的職責：
    1. 解讀 K 線型態與均線排列。
    2. **重點分析 RSI 指標**：判斷是否背離、是否處於超買(>70)或超賣(<30)區域、動能強弱。
    3. 提供客觀的支撐阻力位分析。
    4. 輸出純教育性的分析報告。

    重要原則：
    - **絕對不提供投資建議**。
    - 語氣客觀、專業、溫暖。
    - 使用繁體中文。
    - 必須明確指出 RSI 當前的數值意義。
    """

    user_prompt = f"""
    請基於以下數據進行深度技術分析：

    ### 1. 基本概況
    - 股票代號：{symbol}
    - 期間：{start_date} 至 {end_date}
    - 漲跌幅：{price_change:.2f}%
    - **最新 RSI (14)：{latest_rsi}**

    ### 2. 近 5 日詳細數據
    {data_json}

    ### 3. 分析架構要求 (請依此結構輸出)

    #### (1) 趨勢與均線分析
    - 目前的價格趨勢（多頭/空頭/盤整）。
    - 均線系統的排列狀態。

    #### (2) RSI 動能分析 (重點)
    - 目前 RSI 數值 ({latest_rsi}) 代表的市場狀態（超買/超賣/中性）。
    - 近期 RSI 走勢是否出現「背離」訊號（例如股價創高但 RSI 未創高）。
    - 動能是增強還是減弱？

    #### (3) 價格行為與量能
    - 關鍵支撐與壓力位置。
    - 成交量變化配合情況。

    #### (4) 綜合技術總結
    - 短期技術面觀察重點。
    - 風險提示。

    分析目標：{symbol}
    """
    return system_prompt, user_prompt


def generate_ai_analysis(model_provider, model_name, api_key, symbol, df, start_date, end_date):
    """
    統一的 AI 分析入口
    """
    system_prompt, user_prompt = get_ai_prompts(
        symbol, df, start_date, end_date)

    try:
        if model_provider == "OpenAI":
            openai_model = "gpt-4o-mini" if model_name == "o4-mini" else model_name
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            return response.choices[0].message.content

        elif model_provider == "Google":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt
            )
            generation_config = genai.types.GenerationConfig(temperature=0.3)
            response = model.generate_content(
                user_prompt, generation_config=generation_config)
            return response.text

    except Exception as e:
        return f"AI 分析生成失敗 ({model_provider}): {str(e)}"

# --- 主程式介面設計 ---


def main():
    # 標題區
    st.title("AI 股票趨勢分析系統 Pro 🚀")
    st.divider()

    # --- 側邊欄設定 ---
    with st.sidebar:
        st.header("📊 分析設定")
        st.divider()

        # 1. 股票代碼
        stock_symbol = st.text_input("股票代碼 (例如: AAPL)", value="AAPL").upper()

        # 2. API Keys
        fmp_api_key = st.text_input(
            "FMP API Key", type="password", value=FMP_KEY)

        st.markdown("---")
        st.subheader("🤖 AI 模型設定")
        model_option = st.selectbox(
            "選擇 AI 模型",
            options=["o4-mini", "gemini-3-flash-preview"],
            index=0
        )

        ai_api_key = ""
        model_provider = ""
        if model_option == "o4-mini":
            model_provider = "OpenAI"
            ai_api_key = st.text_input(
                "OpenAI API Key", type="password", value=OPENAI_KEY)
        elif "gemini" in model_option:
            model_provider = "Google"
            ai_api_key = st.text_input(
                "Google Gemini API Key", type="password", value=GOOGLE_KEY)

        st.markdown("---")

        # 3. 技術指標參數 (新增)
        st.subheader("⚙️ 指標參數")
        rsi_days = st.number_input(
            "RSI 計算天數", min_value=5, max_value=60, value=14, step=1)

        st.markdown("---")

        # 4. 日期選擇
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=120)  # 預設拉長一點以便觀察 RSI
        start_date_input = st.date_input("起始日期", value=default_start)
        end_date_input = st.date_input("結束日期", value=today)

        if start_date_input > end_date_input:
            st.error("起始日期不能晚於結束日期！")

        analyze_btn = st.button("🚀 開始分析", type="primary",
                                use_container_width=True)

        # 免責聲明
        st.markdown("---")
        st.markdown("""
        ### 📢 免責聲明
        本系統僅供學術研究與教育用途，AI 提供的數據與分析結果僅供參考，**不構成投資建議或財務建議**。
        """)

    # --- 主要執行邏輯 ---
    if analyze_btn:
        if not stock_symbol or not fmp_api_key or not ai_api_key:
            st.warning(f"請確保已輸入股票代碼、FMP Key 以及 {model_provider} API Key。")
        else:
            with st.spinner(f"正在獲取 {stock_symbol} 數據並計算 RSI ({rsi_days}日)..."):

                # 緩衝區處理 (為了計算 RSI 和 MA)
                buffer_days = rsi_days + 100
                api_start_date = start_date_input - \
                    datetime.timedelta(days=buffer_days)

                raw_df, error_msg = get_stock_data(
                    stock_symbol, fmp_api_key, api_start_date, end_date_input)

                if error_msg:
                    st.error(error_msg)
                else:
                    # 計算技術指標 (包含傳入自訂的 rsi_days)
                    processed_df = calculate_technical_indicators(
                        raw_df, rsi_days)

                    # 過濾回使用者想看的日期
                    final_df = filter_data_by_date(
                        processed_df, start_date_input, end_date_input)

                    if final_df.empty:
                        st.warning("選定的日期範圍內沒有數據。")
                    else:
                        st.success(f"成功分析 {stock_symbol}！")

                        # --- 基本資訊 ---
                        st.subheader("📈 基本統計資訊")
                        col1, col2, col3, col4 = st.columns(4)  # 增加一欄顯示 RSI

                        start_price = final_df.iloc[0]['close']
                        end_price = final_df.iloc[-1]['close']
                        price_diff = end_price - start_price
                        pct_change = (price_diff / start_price) * 100

                        # 取得最新 RSI
                        current_rsi = final_df.iloc[-1]['RSI'] if 'RSI' in final_df.columns else 0

                        col1.metric("起始價格", f"${start_price:.2f}")
                        col2.metric("結束價格", f"${end_price:.2f}")
                        col3.metric(
                            "期間變化", f"${price_diff:.2f}", f"{pct_change:.2f}%")
                        col4.metric(
                            f"RSI ({rsi_days})", f"{current_rsi:.1f}", delta=None, help=">70 超買, <30 超賣")

                        # --- 圖表顯示 (含 RSI 子圖) ---
                        st.subheader("📊 價量趨勢與 RSI 指標")
                        chart_fig = create_chart(final_df, stock_symbol)
                        st.plotly_chart(chart_fig, use_container_width=True)

                        # --- AI 分析 ---
                        st.subheader(f"🤖 AI 深度技術解讀 ({model_option})")
                        with st.spinner("AI 正在觀察 K 線與計算動能..."):
                            ai_insight = generate_ai_analysis(
                                model_provider, model_option, ai_api_key,
                                stock_symbol, final_df,
                                start_date_input, end_date_input
                            )
                            st.markdown(ai_insight)

                        # --- 數據表格 ---
                        st.subheader("📋 詳細交易數據")
                        display_cols = ['date', 'open', 'high', 'low',
                                        'close', 'volume', 'MA5', 'MA20', 'RSI']
                        valid_cols = [
                            c for c in display_cols if c in final_df.columns]

                        table_df = final_df[valid_cols].sort_values(
                            'date', ascending=False).head(10).copy()
                        table_df['date'] = table_df['date'].dt.date

                        # 格式化 RSI 顯示
                        if 'RSI' in table_df.columns:
                            table_df['RSI'] = table_df['RSI'].map(
                                '{:.2f}'.format)

                        st.dataframe(table_df, use_container_width=True)


if __name__ == "__main__":
    main()
