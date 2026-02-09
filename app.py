import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import datetime
import json
from openai import OpenAI
import google.generativeai as genai

# 請確保 secret.py 檔案存在並包含正確的 KEY，或是直接在環境變數中設定
try:
    from secret import FMP_KEY, GOOGLE_KEY, OPENAI_KEY
except ImportError:
    # 若無 secret 檔案，預設為空，請使用者在介面輸入
    FMP_KEY = ""
    GOOGLE_KEY = ""
    OPENAI_KEY = ""

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


def calculate_technical_indicators(df, rsi_days=14, kd_days=9):
    """
    計算移動平均線、RSI 與 KD 指標
    KD 參數預設: 9, 3, 3
    """
    df = df.copy()

    # 1. 計算 MA (移動平均線)
    df['MA5'] = df['close'].rolling(window=5).mean()
    df['MA10'] = df['close'].rolling(window=10).mean()
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA60'] = df['close'].rolling(window=60).mean()

    # 2. 計算 RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.ewm(com=rsi_days - 1, min_periods=rsi_days).mean()
    avg_loss = loss.ewm(com=rsi_days - 1, min_periods=rsi_days).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # 3. 計算 KD (Stochastic Oscillator)
    # RSV 公式: (今日收盤 - 最近n天最低) / (最近n天最高 - 最近n天最低) * 100
    low_min = df['low'].rolling(window=kd_days).min()
    high_max = df['high'].rolling(window=kd_days).max()

    # 避免分母為 0
    df['RSV'] = 100 * (df['close'] - low_min) / (high_max - low_min)
    df['RSV'] = df['RSV'].fillna(50)  # 補值避免初期計算錯誤

    # 計算 K 與 D
    # 公式: K = 1/3 * RSV + 2/3 * 前一日K
    # 這等同於 pandas 的 ewm(alpha=1/3)
    # 我們設定 adjust=False 來模擬遞迴計算

    df['K'] = df['RSV'].ewm(alpha=1/3, adjust=False).mean()
    df['D'] = df['K'].ewm(alpha=1/3, adjust=False).mean()

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
    使用 Plotly 繪製 K 線圖 (上)、RSI (中)、KD (下)
    """
    # 建立包含三個子圖的圖表
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],  # K線佔50%, RSI 25%, KD 25%
        subplot_titles=(f'{symbol} 股價走勢', 'RSI 相對強弱', 'KD 隨機指標')
    )

    # --- Row 1: K線與均線 ---
    fig.add_trace(go.Candlestick(
        x=df['date'],
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='K線',
        increasing_line_color='#26A69A', decreasing_line_color='#EF5350'
    ), row=1, col=1)

    colors = {'MA5': '#FF9800', 'MA10': '#2196F3',
              'MA20': '#9C27B0', 'MA60': '#607D8B'}
    for ma, color in colors.items():
        if ma in df.columns:
            ma_data = df.dropna(subset=[ma])
            if not ma_data.empty:
                fig.add_trace(go.Scatter(
                    x=ma_data['date'], y=ma_data[ma],
                    mode='lines', name=ma, line=dict(color=color, width=1)
                ), row=1, col=1)

    # --- Row 2: RSI ---
    if 'RSI' in df.columns:
        rsi_data = df.dropna(subset=['RSI'])
        if not rsi_data.empty:
            fig.add_trace(go.Scatter(
                x=rsi_data['date'], y=rsi_data['RSI'],
                mode='lines', name='RSI', line=dict(color='#2962FF', width=2)
            ), row=2, col=1)

            # 輔助線 (新增超買/超賣提示文字)
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                          annotation_text="超買(70)", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                          annotation_text="超賣(30)", row=2, col=1)

            # 背景色區域 (新增 RSI 超買/超賣背景，比照 KD 風格)
            fig.add_shape(
                type="rect", xref="x2", yref="y2",
                x0=rsi_data['date'].iloc[0], x1=rsi_data['date'].iloc[-1],
                y0=70, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0
            )
            fig.add_shape(
                type="rect", xref="x2", yref="y2",
                x0=rsi_data['date'].iloc[0], x1=rsi_data['date'].iloc[-1],
                y0=0, y1=30, fillcolor="green", opacity=0.1, layer="below", line_width=0
            )

    # --- Row 3: KD 指標 (新增) ---
    if 'K' in df.columns and 'D' in df.columns:
        kd_data = df.dropna(subset=['K', 'D'])
        if not kd_data.empty:
            # K線 (快線) - 藍色
            fig.add_trace(go.Scatter(
                x=kd_data['date'], y=kd_data['K'],
                mode='lines', name='K值 (快)', line=dict(color='#2979FF', width=1.5)
            ), row=3, col=1)

            # D線 (慢線) - 橘色/深藍
            fig.add_trace(go.Scatter(
                x=kd_data['date'], y=kd_data['D'],
                mode='lines', name='D值 (慢)', line=dict(color='#FF6D00', width=1.5)
            ), row=3, col=1)

            # 輔助線
            fig.add_hline(y=80, line_dash="dash", line_color="red",
                          annotation_text="超買(80)", row=3, col=1)
            fig.add_hline(y=20, line_dash="dash", line_color="green",
                          annotation_text="超賣(20)", row=3, col=1)

            # 背景色區域 (超買/超賣)
            # 改用 add_shape 避免覆蓋 RSI 的設定
            fig.add_shape(
                type="rect", xref="x3", yref="y3",
                x0=kd_data['date'].iloc[0], x1=kd_data['date'].iloc[-1],
                y0=80, y1=100, fillcolor="red", opacity=0.1, layer="below", line_width=0
            )
            fig.add_shape(
                type="rect", xref="x3", yref="y3",
                x0=kd_data['date'].iloc[0], x1=kd_data['date'].iloc[-1],
                y0=0, y1=20, fillcolor="green", opacity=0.1, layer="below", line_width=0
            )

    # --- 版面設定 ---
    fig.update_layout(
        height=900,  # 增加高度
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )

    # Y軸範圍固定
    fig.update_yaxes(range=[0, 100], row=2, col=1)  # RSI
    fig.update_yaxes(range=[0, 100], row=3, col=1)  # KD

    return fig

# --- AI 分析相關函數 ---


def get_ai_prompts(symbol, df, start_date, end_date):
    """
    產生包含 RSI 與 KD 分析的 Prompt
    """
    # 準備欄位
    recent_cols = ['date', 'close', 'volume',
                   'MA5', 'MA20', 'MA60', 'RSI', 'K', 'D']
    cols_to_use = [c for c in recent_cols if c in df.columns]

    # 取最後 5 筆資料 (包含今天與過去4天)
    recent_data = df.tail(5)[cols_to_use].to_dict(orient='records')

    # 格式化
    for record in recent_data:
        record['date'] = record['date'].strftime('%Y-%m-%d')
        for key, value in record.items():
            if isinstance(value, float):
                record[key] = round(value, 2)

    # KD 狀態計算 (用於 Prompt 輔助)
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]

    latest_k = last_row['K'] if 'K' in df.columns else 50
    latest_d = last_row['D'] if 'D' in df.columns else 50
    prev_k = prev_row['K'] if 'K' in df.columns else 50
    prev_d = prev_row['D'] if 'D' in df.columns else 50

    # 簡單交叉判斷 (提供給 AI 參考)
    kd_signal = "無特殊交叉"
    if prev_k < prev_d and latest_k > latest_d:
        kd_signal = "疑似黃金交叉 (K向上突破D)"
    elif prev_k > prev_d and latest_k < latest_d:
        kd_signal = "疑似死亡交叉 (K向下跌破D)"

    kd_status = "中性區間"
    if latest_k > 80:
        kd_status = "高檔鈍化/超買區"
    if latest_k < 20:
        kd_status = "低檔鈍化/超賣區"

    start_price = df.iloc[0]['close']
    end_price = df.iloc[-1]['close']
    price_change = ((end_price - start_price) / start_price) * 100

    data_json = json.dumps(recent_data, indent=2)

    system_prompt = """
    你是一位專業的股票技術分析師，擅長結合「價格趨勢」、「RSI」與「KD 隨機指標」進行綜合研判。

    ### 分析重點與邏輯：
    1. **趨勢與均線**：解讀 MA 排列與價格位置。
    2. **RSI 指標**：判斷動能強弱與背離。
    3. **KD 指標 (Stochastic Oscillator) 分析要求**：
       - **數值解讀**：觀察 K, D 值是否位於超買區 (>80) 或超賣區 (<20)。
       - **交叉訊號**：
         * 黃金交叉 (K由下往上突破D)：通常視為買進訊號，若發生在低檔 (<20) 準確度較高。
         * 死亡交叉 (K由上往下跌破D)：通常視為賣出訊號，若發生在高檔 (>80) 準確度較高。
       - **背離型態 (Divergence)**：
         * 高檔頂背離 (股價創高但 KD 未創高) -> 看空/反轉預警。
         * 低檔底背離 (股價創低但 KD 未創低) -> 看多/反彈預警。
       - **鈍化現象**：
         * 高檔鈍化 (K值連續 3 天 > 80)：代表強勢多頭，趨勢可能延續。
         * 低檔鈍化 (K值連續 3 天 < 20)：代表極弱勢空頭，可能跌深不見底。

    ### 輸出規範：
    - 語氣客觀、專業、溫暖。
    - 使用繁體中文。
    - **必須包含一個獨立章節：「KD 指標深度解析」**。
    - 嚴禁提供投資建議 (Buy/Sell)，僅做教學分析。
    """

    user_prompt = f"""
    請基於以下數據進行深度技術分析：

    ### 1. 概況
    - 標的：{symbol}
    - 期間漲跌：{price_change:.2f}%
    - **最新 K值：{latest_k:.2f} / D值：{latest_d:.2f}**
    - **最新 KD 狀態：{kd_status} / {kd_signal}**

    ### 2. 近 5 日數據 (含 MA, RSI, K, D)
    {data_json}

    ### 3. 分析報告架構 (請依此輸出)

    #### (1) 趨勢結構分析
    - 均線排列與多空方向。
    - 價格支撐與壓力觀察。

    #### (2) RSI 動能解讀
    - RSI 目前位置與意義。
    - 動能消長情況。

    #### (3) KD 指標深度解析 (重點)
    - **當前數值與位置**：K/D 值是否處於極端區域？
    - **交叉訊號**：是否有黃金交叉或死亡交叉？有效性如何？
    - **型態觀察**：是否有「背離」或「鈍化」現象？這代表什麼市場心理？

    #### (4) 綜合技術總結
    - 短線技術面觀察重點。
    - 風險提示 (例如：高檔背離風險、均線乖離過大等)。

    分析目標：{symbol}
    """
    return system_prompt, user_prompt


def generate_ai_analysis(model_provider, model_name, api_key, symbol, df, start_date, end_date):
    """
    執行 AI 分析
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

# --- 主程式 ---


def main():
    st.title("AI 股票趨勢分析系統 🚀")
    st.divider()

    # --- 側邊欄 ---
    with st.sidebar:
        st.header("📊 分析設定")
        st.divider()

        stock_symbol = st.text_input("股票代碼", value="AAPL").upper()
        fmp_api_key = st.text_input(
            "FMP API Key", type="password", value=FMP_KEY)

        st.markdown("---")
        st.subheader("🤖 AI 模型")
        model_option = st.selectbox(
            "選擇模型", ["o4-mini", "gemini-3-flash-preview"], index=0)

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
        st.subheader("⚙️ 指標參數")
        rsi_days = st.number_input("RSI 週期", value=14)

        # 新增 KD 參數輸入
        kd_days = st.number_input(
            "KD 計算天數 (RSV週期)", min_value=5, max_value=60, value=9, step=1)

        st.markdown("---")
        today = datetime.date.today()
        default_start = today - datetime.timedelta(days=120)
        start_date_input = st.date_input("起始日期", value=default_start)
        end_date_input = st.date_input("結束日期", value=today)

        analyze_btn = st.button("🚀 開始分析", type="primary",
                                use_container_width=True)

        st.markdown("---")
        st.markdown("### 📢 免責聲明\n本系統僅供教育研究，**不構成投資建議**。")

    # --- 執行邏輯 ---
    if analyze_btn:
        if not stock_symbol or not fmp_api_key or not ai_api_key:
            st.warning(f"請輸入完整 API Key 資訊。")
        else:
            with st.spinner(f"正在獲取 {stock_symbol} 數據並計算 KD/RSI..."):

                # 拉長緩衝區以確保 KD/MA 計算準確
                buffer_days = max(rsi_days, kd_days, 60) + 50
                api_start_date = start_date_input - \
                    datetime.timedelta(days=buffer_days)

                raw_df, error_msg = get_stock_data(
                    stock_symbol, fmp_api_key, api_start_date, end_date_input)

                if error_msg:
                    st.error(error_msg)
                else:
                    # 計算指標 (傳入 KD 參數)
                    processed_df = calculate_technical_indicators(
                        raw_df, rsi_days, kd_days)
                    final_df = filter_data_by_date(
                        processed_df, start_date_input, end_date_input)

                    if final_df.empty:
                        st.warning("選定範圍無數據。")
                    else:
                        st.success(f"分析完成：{stock_symbol}")

                        # --- 統計資訊 ---
                        st.subheader("📈 關鍵指標")
                        c1, c2, c3, c4, c5 = st.columns(5)  # 增加欄位顯示 KD

                        start_p = final_df.iloc[0]['close']
                        end_p = final_df.iloc[-1]['close']
                        chg = end_p - start_p
                        pct = (chg / start_p) * 100

                        cur_rsi = final_df.iloc[-1]['RSI'] if 'RSI' in final_df.columns else 0
                        cur_k = final_df.iloc[-1]['K'] if 'K' in final_df.columns else 0
                        cur_d = final_df.iloc[-1]['D'] if 'D' in final_df.columns else 0

                        c1.metric("價格", f"${end_p:.2f}", f"{pct:.2f}%")
                        c2.metric("RSI (14)", f"{cur_rsi:.1f}")
                        c3.metric("K值 (快)", f"{cur_k:.1f}",
                                  help=">80超買, <20超賣")
                        c4.metric("D值 (慢)", f"{cur_d:.1f}")

                        # 簡單訊號顯示
                        signal = "中性"
                        if cur_k > 80:
                            signal = "超買區"
                        elif cur_k < 20:
                            signal = "超賣區"
                        c5.metric("KD 狀態", signal)

                        # --- 圖表 ---
                        st.subheader("📊 價量與技術指標")
                        fig = create_chart(final_df, stock_symbol)
                        st.plotly_chart(fig, use_container_width=True)

                        # --- AI 分析 ---
                        st.subheader(f"🤖 AI 技術解讀 ({model_option})")
                        with st.spinner("AI 正在分析 KD 交叉與背離訊號..."):
                            ai_res = generate_ai_analysis(
                                model_provider, model_option, ai_api_key,
                                stock_symbol, final_df, start_date_input, end_date_input
                            )
                            st.markdown(ai_res)

                        # --- 數據表 ---
                        st.subheader("📋 詳細數據")
                        cols = ['date', 'close', 'volume',
                                'MA5', 'MA20', 'RSI', 'K', 'D']
                        show_df = final_df[[c for c in cols if c in final_df.columns]].sort_values(
                            'date', ascending=False).head(10)

                        # 格式化
                        if 'K' in show_df.columns:
                            show_df['K'] = show_df['K'].map('{:.2f}'.format)
                        if 'D' in show_df.columns:
                            show_df['D'] = show_df['D'].map('{:.2f}'.format)
                        if 'RSI' in show_df.columns:
                            show_df['RSI'] = show_df['RSI'].map(
                                '{:.2f}'.format)

                        st.dataframe(show_df, use_container_width=True)


if __name__ == "__main__":
    main()
