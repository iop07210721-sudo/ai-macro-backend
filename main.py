from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import pandas as pd
import numpy as np
import xgboost as xgb
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_TRANSLATION = {
    "Dollar_Index": {"name": "美元指數 (當前強度)", "desc": "美元的即時強弱。大宗商品多以美元計價，強勢美元會直接對資產價格造成下行壓力。"},
    "Dollar_Index_SMA20": {"name": "美元指數 (月線趨勢)", "desc": "美元近一個月的中期趨勢。用來檢視資金是否在中期持續回流美元體系。"},
    "Dollar_Index_SMA60": {"name": "美元指數 (季線趨勢)", "desc": "美元的長線走勢。反映宏觀貨幣政策的大方向。"},
    "Dollar_Index_RSI": {"name": "美元指數 (短期動能)", "desc": "衡量美元短期的超買或超賣狀態，判斷趨勢是否過熱。"},
    "US_10Y": {"name": "美債10年期殖利率 (當前報價)", "desc": "全球資產定價的無風險利率基準。攀升會增加持有不孳息大宗商品的機會成本。"},
    "US_10Y_SMA20": {"name": "美債10年期殖利率 (中線)", "desc": "反映市場對未來通膨與升息路徑的中期預期。"},
    "US_10Y_SMA60": {"name": "美債10年期殖利率 (長線)", "desc": "長期資金成本趨勢。攀升會吸引資金離開原物料市場。"},
    "Silver_SMA20": {"name": "白銀 (月線趨勢)", "desc": "白銀的中期資金動能，常作為商品板塊整體情緒的風向球。"},
    "Silver_MACD": {"name": "白銀 (MACD 動能)", "desc": "白銀波動大、敏銳度高，其動能常作為貴金屬與工業板塊的先行情緒指標。"},
    "VIX_SMA60": {"name": "恐慌指數 (長線底氣)", "desc": "衡量股市長期波動預期的基期水準。"},
    "Target_RSI": {"name": "分析標的 (短期動能)", "desc": "當前標的本身的近期超買/超賣狀態。"},
    "Target_Dist_SMA60": {"name": "分析標的 (季線乖離率)", "desc": "當前價格距離長線平均成本有多遠，用來評估是否漲跌過度。"}
}

def fetch_raw_data(target_ticker):
    MACRO = {"Dollar_Index": "DX-Y.NYB", "US_10Y": "^TNX", "VIX": "^VIX"}
    if target_ticker == "SI=F":
        MACRO["Copper"] = "HG=F"
    else:
        MACRO["Silver"] = "SI=F"
        
    tickers = {"Target": target_ticker, **MACRO}
    clean_dfs = []
    
    for name, ticker in tickers.items():
        data = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
        if not data.empty:
            df = data[['Close']].copy()
            df.columns = [name]
            df.index = pd.to_datetime(df.index).tz_localize(None).floor('D')
            df = df.groupby(df.index).last()
            clean_dfs.append(df)
            
    all_data = pd.concat(clean_dfs, axis=1).sort_index()
    all_data = all_data.dropna(how='all').ffill().bfill()
    all_data = all_data.loc["2000-01-01":]
    return all_data

def create_features(df):
    features = df.copy()
    for col in df.columns:
        features[f'{col}_SMA5'] = features[col].rolling(5).mean()
        features[f'{col}_SMA20'] = features[col].rolling(20).mean()
        features[f'{col}_SMA60'] = features[col].rolling(60).mean()
        features[f'{col}_Dist_SMA60'] = features[col] / features[f'{col}_SMA60'] - 1
        delta = features[col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        features[f'{col}_RSI'] = 100 - (100 / (1 + gain / loss))
        ema12 = features[col].ewm(span=12, adjust=False).mean()
        ema26 = features[col].ewm(span=26, adjust=False).mean()
        features[f'{col}_MACD'] = ema12 - ema26
    return features

@app.get("/api/forecast")
def get_forecast(ticker: str = "GC=F"):
    if ticker == "GC=F": asset_name = "黃金 (Gold)"
    elif ticker == "CL=F": asset_name = "原油 (WTI Crude Oil)"
    else: asset_name = "白銀 (Silver)"

    raw_data = fetch_raw_data(ticker)
    featured_data = create_features(raw_data)
    train_data = featured_data.copy()
    train_data['Target_Direction'] = (train_data['Target'].shift(-1) > train_data['Target']).astype(int)
    train_data['Target_Diff'] = train_data['Target'].shift(-1) - train_data['Target']
    train_data = train_data.dropna()

    X = train_data.drop(columns=['Target_Direction', 'Target_Diff'])
    y_class = train_data['Target_Direction']
    y_reg = train_data['Target_Diff']

    latest_features = X.iloc[[-1]]
    
    # 1. 訓練分類器
    clf_model = xgb.XGBClassifier(n_estimators=500, learning_rate=0.03, random_state=42, use_label_encoder=False, eval_metric='logloss')
    clf_model.fit(X, y_class)
    probabilities = clf_model.predict_proba(latest_features)[0]
    prob_down, prob_up = float(round(probabilities[0] * 100, 1)), float(round(probabilities[1] * 100, 1))

    if prob_up >= 60: signal_text = "🟢 強烈看多"
    elif prob_up >= 53: signal_text = "🟢 偏多推演"
    elif prob_up > 47: signal_text = "⚪ 中立觀望"
    elif prob_up > 40: signal_text = "🔴 偏空推演"
    else: signal_text = "🔴🔴 強烈看空"

    importances = clf_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(5)
    
    xai_results = []
    for _, row in importance_df.iterrows():
        f_key = row['Feature']
        name = FEATURE_TRANSLATION.get(f_key, {}).get('name', f_key.replace('_', ' '))
        default_desc = "模型偵測到該特徵對次日趨勢具備關鍵引導力。"
        if "SMA" in f_key: default_desc = "反映該數據的移動平均趨勢線扣抵變化，作為支撐阻力參考。"
        elif "RSI" in f_key or "MACD" in f_key: default_desc = "捕捉超買超賣與動能背離訊號，用於推演趨勢轉折。"
        desc = FEATURE_TRANSLATION.get(f_key, {}).get('desc', default_desc)
        xai_results.append({"name": name, "weight": float(round(row['Importance']*100, 1)), "desc": desc})

    # 2. 訓練回歸器
    reg_model = xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, random_state=42)
    reg_model.fit(X, y_reg)

    current_raw = raw_data.copy()
    future_dates = pd.bdate_range(start=raw_data.index[-1] + pd.Timedelta(days=1), periods=7)
    
    forecast_path = []
    for date in future_dates:
        latest_feats = create_features(current_raw).iloc[[-1]][X.columns]
        pred_diff = reg_model.predict(latest_feats)[0]
        next_price = current_raw['Target'].iloc[-1] + pred_diff
        forecast_path.append({"date": date.strftime('%Y-%m-%d'), "price": float(round(next_price, 2))})
        new_row = pd.DataFrame({'Target': [next_price]}, index=[date])
        current_raw = pd.concat([current_raw, new_row]).ffill()

    # ==========================================
    # 🔥 新增功能：打包前端畫圖需要的歷史資料
    # ==========================================
    
    # [提供給頁籤一] 過去一年的歷史軌跡
    one_year_ago = raw_data.index[-1] - pd.Timedelta(days=365)
    history_1y_df = raw_data.loc[one_year_ago:]
    history_1y = {
        "dates": history_1y_df.index.strftime('%Y-%m-%d').tolist(),
        "prices": history_1y_df['Target'].astype(float).round(2).tolist()
    }

    # [提供給頁籤二] 自 2000 年以來的總經多軸歷史
    # (為了避免 JSON 太大導致網頁當機，我們把龐大的時間序列資料轉成 List 送出)
    macro_history = {
        "dates": raw_data.index.strftime('%Y-%m-%d').tolist(),
        "target_prices": raw_data['Target'].astype(float).round(2).tolist(),
        "usd_index": raw_data['Dollar_Index'].astype(float).round(2).tolist(),
        "us_10y": raw_data['US_10Y'].astype(float).round(3).tolist()
    }

    # [提供給頁籤三] 2022 年大熊市壓力測試計算
    bt_data = train_data.loc["2022-01-01":"2022-12-31"].copy()
    if len(bt_data) > 50:
        bt_preds = clf_model.predict(bt_data.drop(columns=['Target_Direction', 'Target_Diff']))
        bt_returns = bt_data['Target'].pct_change().shift(-1).fillna(0)
        
        buy_hold_eq = (1 + bt_returns).cumprod()
        ai_eq = (1 + (bt_returns * bt_preds)).cumprod()
        
        # 抓出 AI 判定為「跌」(預測值為 0) 的紅色風險區間
        risk_zones = []
        for i in range(len(bt_preds) - 1):
            if bt_preds[i] == 0:
                risk_zones.append({
                    "start": bt_data.index[i].strftime('%Y-%m-%d'),
                    "end": bt_data.index[i+1].strftime('%Y-%m-%d')
                })

        backtest_2022 = {
            "dates": bt_data.index[:-1].strftime('%Y-%m-%d').tolist(),
            "buy_hold_eq": buy_hold_eq[:-1].astype(float).round(3).tolist(),
            "ai_strategy_eq": ai_eq[:-1].astype(float).round(3).tolist(),
            "risk_zones": risk_zones,
            "final_buy_hold": float(round(buy_hold_eq.iloc[-2], 3)),
            "final_ai": float(round(ai_eq.iloc[-2], 3))
        }
    else:
        backtest_2022 = None

    current_price = float(round(raw_data['Target'].iloc[-1], 2))

    # ==========================================
    # 最終回傳超大包 JSON
    # ==========================================
    return {
        "status": "success",
        "asset": asset_name,
        "current_price": current_price,
        "prediction": {
            "signal": signal_text,
            "prob_up": prob_up,
            "prob_down": prob_down
        },
        "xai_top_features": xai_results,
        "forecast_7_days": forecast_path,
        "history_1y": history_1y,
        "macro_history": macro_history,
        "backtest_2022": backtest_2022
    }