import yfinance as yf
import pandas as pd
import datetime

def get_stock_data(symbol, start_date, end_date):
    """
    從 yfinance 獲取指定日期範圍的股票歷史數據
    """
    try:
        # yfinance 接受 str 或是 datetime 物件
        # 為了保險起見，這裡不需特別轉 str，但若有需要可轉
        # start_date, end_date 通常是 datetime.date
        
        # 下載數據 (auto_adjust=True 會讓 Close = Adj Close, 下載後只有 Close)
        # 若需要 Open/High/Low 也一併修正，通常 auto_adjust=True 較佳，或是依舊習慣
        # 這裡為了對齊原邏輯，我們使用 standard 下載，但要注意 yfinance 的 column 結構
        df = yf.download(symbol, start=start_date, end=end_date, progress=False, multi_level_index=False)

        if df is None or df.empty:
            return None, f"該日期區間或股票代碼 ({symbol}) 無交易數據。"

        # 處理 Index (Date)
        df = df.reset_index()
        
        # 處理欄位名稱 (統一轉小寫)
        df.columns = [c.lower() for c in df.columns]
        
        # 檢查必要欄位
        required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [c for c in required_cols if c not in df.columns]
        
        if missing_cols:
             # 有時候 yfinance 可能回傳 'adj close'，如果沒有 'close'
             if 'adj close' in df.columns and 'close' not in df.columns:
                 df.rename(columns={'adj close': 'close'}, inplace=True)
                 missing_cols.remove('close')
        
        if missing_cols:
            return None, f"數據缺少必要欄位: {', '.join(missing_cols)}"

        # 確保日期格式
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date', ascending=True).reset_index(drop=True)

        return df, None

    except Exception as e:
        return None, f"yfinance 資料獲取失敗: {str(e)}"

def calculate_technical_indicators(df, rsi_days=14, kd_days=9):
    """
    計算移動平均線、RSI 與 KD 指標
    KD 參數預設: 9, 3, 3
    """
    if df is None or df.empty:
        return df

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
    if df is None or df.empty:
        return df
        
    mask = (df['date'].dt.date >= start_date) & (
        df['date'].dt.date <= end_date)
    return df.loc[mask].reset_index(drop=True)
