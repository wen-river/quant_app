import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

import akshare as ak
import pandas_ta as ta
import streamlit as st

# ================== 页面设置 ==================
st.set_page_config(
    page_title="量化选股Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .main .block-container {padding: 1rem 0.5rem;}
    .stButton>button {
        width:100%; background:#f63366; color:white; font-weight:bold;
        border-radius:8px; padding:0.6rem;
    }
    .stButton>button:hover {background:#e62e5c;}
</style>
""", unsafe_allow_html=True)

# ================== 策略配置 ==================
DEFAULT_CONFIG = {
    "trend": {
        "ema": {
            "enabled": True,
            "fast": 8, "slow": 21,
            "score_ema_fast_above_slow": 0.4,
            "score_price_above_fast": 0.3,
            "score_macd_hist_positive": 0.3
        }
    },
    "momentum": {
        "rsi": {
            "enabled": True,
            "lower": 40, "upper": 65,
            "score_in_range_min": 0.3,
            "score_in_range_max": 0.6,
            "score_30_40": 0.2
        },
        "kdj": {
            "enabled": True,
            "score_k_above_d": 0.2,
            "score_j_above_k": 0.1
        }
    },
    "volume": {
        "enabled": True,
        "ratio_2_0": 1.0,
        "ratio_1_5": 0.8,
        "ratio_1_2": 0.6,
        "ratio_1_0": 0.3,
        "ratio_below": 0.1
    },
    "risk": {
        "atr": {
            "enabled": True,
            "penalty_5pct": 0.3,
            "penalty_3pct": 0.15
        }
    },
    "extra": {
        "wr": {"enabled": False, "threshold": -80, "score": 0.15},
        "bollinger": {"enabled": False, "score": 0.2},
        "ema55_bullish": {"enabled": False, "score": 0.1}
    },
    "weights": {
        "trend": 0.35, "momentum": 0.30,
        "volume": 0.25, "risk": 0.10
    },
    "buy_threshold": 0.65
}

# ================== 核心计算引擎 ==================
class IndicatorCalculator:
    @staticmethod
    def calc_all(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['ema8'] = ta.ema(df['close'], length=8)
        df['ema21'] = ta.ema(df['close'], length=21)
        df['ema55'] = ta.ema(df['close'], length=55)
        macd = ta.macd(df['close'])
        if isinstance(macd, pd.DataFrame):
            df['macd'] = macd.iloc[:, 0]
            df['macd_signal'] = macd.iloc[:, 2] if macd.shape[1] > 2 else macd.iloc[:, 1]
        else:
            df['macd'] = macd
        df['macd_hist'] = df['macd'] - df['macd_signal']
        df['rsi'] = ta.rsi(df['close'])
        low_min = df['low'].rolling(9).min()
        high_max = df['high'].rolling(9).max()
        rsv = (df['close'] - low_min) / (high_max - low_min + 1e-9) * 100
        df['kdj_k'] = rsv.ewm(com=2, adjust=False).mean()
        df['kdj_d'] = df['kdj_k'].ewm(com=2, adjust=False).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
        tr = np.maximum(df['high'] - df['low'],
                        np.maximum(abs(df['high'] - df['close'].shift(1)),
                                   abs(df['low'] - df['close'].shift(1))))
        df['atr'] = tr.rolling(14).mean()
        df['vol_ma20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['vol_ma20'].replace(0, np.nan)
        df['wr'] = ta.willr(df['high'], df['low'], df['close'])
        bbands = ta.bbands(df['close'])
        if isinstance(bbands, pd.DataFrame):
            for col in bbands.columns:
                df[col] = bbands[col]
        return df.dropna()

class SignalEngine:
    @staticmethod
    def score(row, config):
        # 趋势得分
        t = 0.0
        if config['trend']['ema']['enabled']:
            c = config['trend']['ema']
            if row['ema8'] > row['ema21']:
                t += c['score_ema_fast_above_slow']
            if row['close'] > row['ema8']:
                t += c['score_price_above_fast']
            if row['macd_hist'] > 0:
                t += c['score_macd_hist_positive']
        t = min(t, 1.0)

        # 动量得分
        m = 0.0
        rsi_cfg = config['momentum']['rsi']
        if rsi_cfg['enabled']:
            rsi = row['rsi']
            if rsi_cfg['lower'] <= rsi <= rsi_cfg['upper']:
                pos = (rsi - rsi_cfg['lower']) / (rsi_cfg['upper'] - rsi_cfg['lower'])
                m += rsi_cfg['score_in_range_min'] + pos * (rsi_cfg['score_in_range_max'] - rsi_cfg['score_in_range_min'])
            elif 30 <= rsi < rsi_cfg['lower']:
                m += rsi_cfg['score_30_40']
        if config['momentum']['kdj']['enabled']:
            if row['kdj_k'] > row['kdj_d']:
                m += config['momentum']['kdj']['score_k_above_d']
                if row['kdj_j'] > row['kdj_k']:
                    m += config['momentum']['kdj']['score_j_above_k']
        m = min(m, 1.0)

        # 量能得分
        v = 0.0
        if config['volume']['enabled']:
            vr = row['volume_ratio']
            c = config['volume']
            if vr >= 2.0:   v = c['ratio_2_0']
            elif vr >= 1.5: v = c['ratio_1_5']
            elif vr >= 1.2: v = c['ratio_1_2']
            elif vr >= 1.0: v = c['ratio_1_0']
            else:           v = c['ratio_below']

        # 风险惩罚
        p = 0.0
        if config['risk']['atr']['enabled']:
            atr_pct = row['atr'] / row['close'] if row['close'] else 0
            if atr_pct > 0.05:
                p += config['risk']['atr']['penalty_5pct']
            elif atr_pct > 0.03:
                p += config['risk']['atr']['penalty_3pct']
        p = min(p, 1.0)

        # 额外加分
        extra = 0.0
        ex = config.get('extra', {})
        if ex.get('wr', {}).get('enabled') and row['wr'] <= ex['wr']['threshold']:
            extra += ex['wr']['score']
        if ex.get('bollinger', {}).get('enabled'):
            bbu = [c for c in row.index if c.startswith('BBU')]
            if bbu and row['close'] > row[bbu[0]]:
                extra += ex['bollinger']['score']
        if ex.get('ema55_bullish', {}).get('enabled') and row['ema8'] > row['ema21'] > row['ema55']:
            extra += ex['ema55_bullish']['score']

        w = config['weights']
        tfbi = (w['trend'] * t + w['momentum'] * m +
                w['volume'] * v - w['risk'] * p) + extra
        tfbi = round(max(0.0, min(tfbi, 1.0)), 3)

        return {
            'TFBI': tfbi,
            'trend': round(t, 2), 'momentum': round(m, 2),
            'volume': round(v, 2), 'risk': round(p, 2),
            'extra': round(extra, 2),
            'buy': tfbi >= config['buy_threshold'],
            'close': row['close']
        }

# ================== 数据获取 ==================
@st.cache_data(ttl=600)
def get_hot_stocks(n=100):
    try:
        df = ak.stock_zh_a_spot_em()
        df = df.sort_values('总市值', ascending=False).head(n)
        return list(zip(df['代码'], df['名称']))
    except:
        return [('000001','平安银行'), ('000858','五粮液'), ('300750','宁德时代'),
                ('600519','贵州茅台'), ('601318','中国平安'), ('000333','美的集团')]

@st.cache_data(ttl=3600)
def get_stock_name(code):
    try:
        df = ak.stock_info_a_code_name()
        match = df[df['code'] == code]
        return match.iloc[0]['name'] if not match.empty else ""
    except:
        return ""

@st.cache_data(ttl=600)
def fetch_indicators(code):
    try:
        end = datetime.now().strftime('%Y%m%d')
        start = (datetime.now() - timedelta(days=120)).strftime('%Y%m%d')
        df = ak.stock_zh_a_hist(symbol=code, period="daily",
                                start_date=start, end_date=end, adjust="qfq")
        if df.empty: return None
        df = df.rename(columns={'日期':'date','开盘':'open','收盘':'close',
                                '最高':'high','最低':'low','成交量':'volume'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = IndicatorCalculator.calc_all(df)
        return df.iloc[-1] if len(df) >= 30 else None
    except:
        return None

# ================== Session State 初始化 ==================
if 'custom_stocks' not in st.session_state:
    st.session_state.custom_stocks = []
if 'config' not in st.session_state:
    st.session_state.config = DEFAULT_CONFIG.copy()
if 'saved_configs' not in st.session_state:
    st.session_state.saved_configs = {"默认": DEFAULT_CONFIG.copy()}

# ================== 主界面 ==================
st.title("📈 量化选股Pro")
st.caption("趋势共振突破 · 无错误稳定版")

tab1, tab2, tab3, tab4 = st.tabs(["📋 股票池", "⚡ 扫描", "🔍 单股", "⚙️ 策略"])

# ---------- 股票池管理（使用表单，无需 rerun）----------
with tab1:
    st.subheader("自选股票池")

    # 添加股票 - 使用表单
    with st.form("add_stock_form", clear_on_submit=True):
        cols = st.columns([3, 1])
        with cols[0]:
            new_code = st.text_input("股票代码", placeholder="如 600519")
        with cols[1]:
            submitted = st.form_submit_button("➕ 添加")
        if submitted and new_code:
            code = new_code.strip().zfill(6)
            if code in [c for c,n in st.session_state.custom_stocks]:
                st.warning("已在池中")
            else:
                name = get_stock_name(code)
                if not name:
                    st.error("未获取到名称，请手动输入")
                else:
                    st.session_state.custom_stocks.append((code, name))
                    st.success(f"添加成功：{code} {name}")

    # 显示股票池
    if st.session_state.custom_stocks:
        st.subheader(f"共 {len(st.session_state.custom_stocks)} 只")
        for i, (code, name) in enumerate(st.session_state.custom_stocks):
            c1, c2, c3 = st.columns([2, 3, 1])
            c1.write(f"**{code}**")
            c2.write(name)
            if c3.button("🗑", key=f"del_{i}"):
                st.session_state.custom_stocks.pop(i)
                st.rerun()  # 删除操作简化，无冲突
        if st.button("🗑 清空全部"):
            st.session_state.custom_stocks.clear()
            st.rerun()
    else:
        st.info("股票池为空，请添加关注股票")

# ---------- 批量扫描（优化进度条，无 empty 冲突）----------
with tab2:
    st.subheader("选择扫描池")
    src = st.radio("来源", ["📁 我的股票池", "🔥 热门股 Top N"], horizontal=True)
    targets = []
    label = ""
    if src == "🔥 热门股 Top N":
        n = st.slider("扫描数量", 20, 200, 80, 10)
        targets = get_hot_stocks(n)
        label = f"热门股 Top{len(targets)}"
    else:
        targets = st.session_state.custom_stocks.copy()
        label = f"自选池({len(targets)}只)"
        if not targets:
            st.warning("自选池为空，请先添加或切换热门股")

    if st.button("🔍 开始扫描", use_container_width=True):
        if not targets:
            st.error("无待扫描股票")
        else:
            # 使用单个容器动态更新进度，不销毁元素
            progress_container = st.empty()
            progress_bar = progress_container.progress(0)
            status_text = progress_container.empty()

            results = []
            total = len(targets)
            cfg = st.session_state.config
            for i, (code, name) in enumerate(targets):
                status_text.text(f"({i+1}/{total}) 分析 {code} {name}")
                row = fetch_indicators(code)
                if row is not None:
                    res = SignalEngine.score(row, cfg)
                    if res['buy']:
                        results.append((code, name, res))
                progress_bar.progress((i+1)/total)
            progress_container.empty()  # 完成后移除进度条

            if results:
                st.success(f"✅ {label} 发现 {len(results)} 个买入信号")
                results.sort(key=lambda x: x[2]['TFBI'], reverse=True)
                for code, name, r in results:
                    with st.expander(f"✅ {code} {name}"):
                        c1, c2, c3 = st.columns(3)
                        c1.metric("TFBI", r['TFBI'])
                        c2.metric("趋势", f"{r['trend']:.0%}")
                        c3.metric("动量", f"{r['momentum']:.0%}")
                        st.write(f"量能 {r['volume']:.0%} | 风险 {r['risk']:.0%} | 额外 +{r['extra']:.0%}")
                        st.caption(f"最新价 {r['close']:.2f}")
            else:
                st.info(f"在 {label} 中未发现买入信号")

    if st.button("🧹 清除缓存（获取最新行情）"):
        st.cache_data.clear()
        st.success("缓存已清除，下次扫描将获取最新数据")

# ---------- 单股检测 ----------
with tab3:
    st.subheader("快速诊断")
    code = st.text_input("输入代码", "000001")
    if st.button("检测", use_container_width=True):
        row = fetch_indicators(code.strip().zfill(6))
        if row is None:
            st.error("获取数据失败，请检查代码")
        else:
            res = SignalEngine.score(row, st.session_state.config)
            st.subheader(f"诊断结果：{code}")
            c1, c2 = st.columns(2)
            c1.metric("TFBI评分", res['TFBI'])
            c2.metric("最新价", f"{res['close']:.2f}")
            t_col, m_col, v_col, r_col = st.columns(4)
            t_col.metric("趋势", f"{res['trend']:.0%}")
            m_col.metric("动量", f"{res['momentum']:.0%}")
            v_col.metric("量能", f"{res['volume']:.0%}")
            r_col.metric("风险", f"{res['risk']:.0%}", delta="-", delta_color="inverse")
            if res['buy']:
                st.success("🎯 触发买入信号！")
            else:
                st.info("未触发买入信号")

# ---------- 策略配置 ----------
with tab4:
    st.subheader("策略参数")
    saved = list(st.session_state.saved_configs.keys())
    sel = st.selectbox("加载方案", saved, index=saved.index("默认"))
    if st.button("加载该方案"):
        st.session_state.config = st.session_state.saved_configs[sel]
        st.rerun()

    cfg = st.session_state.config
    with st.expander("趋势"):
        cfg['trend']['ema']['enabled'] = st.checkbox("启用", value=cfg['trend']['ema']['enabled'])
        if cfg['trend']['ema']['enabled']:
            cfg['trend']['ema']['fast'] = st.slider("快线", 3,20, cfg['trend']['ema']['fast'])
            cfg['trend']['ema']['slow'] = st.slider("慢线", 5,50, cfg['trend']['ema']['slow'])
    with st.expander("动量"):
        cfg['momentum']['rsi']['enabled'] = st.checkbox("RSI", value=cfg['momentum']['rsi']['enabled'])
        if cfg['momentum']['rsi']['enabled']:
            cfg['momentum']['rsi']['lower'] = st.slider("下限", 10,50, cfg['momentum']['rsi']['lower'])
            cfg['momentum']['rsi']['upper'] = st.slider("上限", 50,90, cfg['momentum']['rsi']['upper'])
        cfg['momentum']['kdj']['enabled'] = st.checkbox("KDJ", value=cfg['momentum']['kdj']['enabled'])
    with st.expander("量能 & 风控"):
        cfg['volume']['enabled'] = st.checkbox("成交量", value=cfg['volume']['enabled'])
        cfg['risk']['atr']['enabled'] = st.checkbox("波动率惩罚", value=cfg['risk']['atr']['enabled'])
    with st.expander("权重 & 阈值"):
        cfg['weights']['trend'] = st.slider("趋势权重", 0.0,1.0, cfg['weights']['trend'], 0.05)
        cfg['weights']['momentum'] = st.slider("动量权重", 0.0,1.0, cfg['weights']['momentum'], 0.05)
        cfg['weights']['volume'] = st.slider("量能权重", 0.0,1.0, cfg['weights']['volume'], 0.05)
        cfg['weights']['risk'] = st.slider("风控权重", 0.0,1.0, cfg['weights']['risk'], 0.05)
        cfg['buy_threshold'] = st.slider("买入阈值", 0.3,0.9, cfg['buy_threshold'], 0.01)

    col_save, col_reset = st.columns(2)
    with col_save:
        new_name = st.text_input("保存为", "自定义")
        if st.button("💾 保存方案"):
            st.session_state.saved_configs[new_name] = cfg.copy()
            st.success(f"方案“{new_name}”已保存")
    with col_reset:
        if st.button("🔄 恢复默认"):
            st.session_state.config = DEFAULT_CONFIG.copy()
            st.session_state.saved_configs["默认"] = DEFAULT_CONFIG.copy()
            st.rerun()

st.divider()
st.caption("⚠️ 本App仅供学习研究，不构成投资建议。投资有风险，入市需谨慎。")
st.caption(f"🕒 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")