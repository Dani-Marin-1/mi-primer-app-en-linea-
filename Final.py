import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# ───────────────── Configuración de página ─────────────────
st.set_page_config(
    page_title="Análisis Financiero ",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────── CSS Personalizado ─────────────────
st.markdown("""
<style>
    /* Fondo principal BLANCO */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #F8F9FA !important;
    }
    
    /* Títulos y texto - CAMBIO: Azul más profesional y gris */
    h1, h2, h3, h4, h5, h6 {
        color: #2C3E50 !important;  /* Azul grisáceo más profesional */
        font-family: 'Arial', sans-serif;
    }
    
    .stMarkdown {
        color: #34495E !important;  /* Gris azulado más profesional */
    }
    
    /* Métricas - CAMBIO: Azul más fuerte */
    [data-testid="metric-container"] {
        background-color: #F8F9FA;
        border: 1px solid #E0E0E0;
        border-radius: 10px;
        padding: 15px;
    }
    
    [data-testid="metric-label"] {
        color: #7F8C8D !important;  /* Gris más profesional */
        font-size: 14px !important;
    }
    
    [data-testid="metric-value"] {
        color: #2980B9 !important;  /* Azul más fuerte y profesional */
        font-size: 24px !important;
        font-weight: bold;
    }
    
    [data-testid="metric-delta"] {
        font-size: 14px !important;
    }
    
    /* Tabs - CAMBIO: Azul más profesional */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #F8F9FA;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 20px;
        color: #7F8C8D;  /* Gris profesional */
        border: 1px solid #E0E0E0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #2980B9 !important;  /* Azul más fuerte */
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    /* Inputs */
    .stTextInput>div>div>input {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #BDC3C7;  /* Gris más profesional */
    }
    
    .stSelectbox>div>div {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #BDC3C7;  /* Gris más profesional */
    }
    
    .stTextArea>div>div>textarea {
        background-color: #FFFFFF;
        color: #000000;
        border: 1px solid #BDC3C7;  /* Gris más profesional */
    }
    
    /* Dataframes - CAMBIO: Azul más profesional */
    .dataframe {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    .dataframe th {
        background-color: #34495E !important;  /* Azul grisáceo profesional */
        color: #FFFFFF !important;
        font-weight: bold;
    }
    
    .dataframe td {
        background-color: #F8F9FA !important;
        color: #000000 !important;
    }
    
    /* Botones y controles - CAMBIO: Azul más fuerte */
    .stButton>button {
        background-color: #2980B9;  /* Azul más fuerte */
        color: #FFFFFF;
        border: none;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .stButton>button:hover {
        background-color: #2471A3;  /* Azul más oscuro al hover */
        color: #FFFFFF;
    }
    
    /* Info boxes - CAMBIO: Tonos más profesionales */
    .stInfo {
        background-color: #EBF5FB;
        border: 1px solid #3498DB;
        color: #2471A3;
    }
    
    .stWarning {
        background-color: #FEF9E7;
        border: 1px solid #F39C12;
        color: #B7950B;
    }
    
    .stError {
        background-color: #FDEDEC;
        border: 1px solid #E74C3C;
        color: #A93226;
    }
    
    .stSuccess {
        background-color: #E8F5E8;
        border: 1px solid #27AE60;
        color: #229954;
    }
    
    /* Separadores */
    hr {
        border-color: #BDC3C7;  /* Gris más profesional */
        margin: 20px 0;
    }
    
    /* Cards personalizadas */
    .custom-card {
        background-color: #F8F9FA;
        border: 1px solid #BDC3C7;  /* Gris más profesional */
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* Texto en sidebar */
    .css-1aumxhk {
        color: #2C3E50 !important;  /* Gris azulado profesional */
    }
    
    /* Tabla de comparación */
    .comparison-table {
        background-color: #FFFFFF;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid #E0E0E0;
    }
    
    .comparison-header {
        background-color: #34495E !important;
        color: #FFFFFF !important;
        font-weight: bold !important;
    }
    
    .comparison-cell {
        text-align: center !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Análisis Financiero ")
st.markdown("---")

# ───────────────── Sidebar ─────────────────
with st.sidebar:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.header("🎯 Controles Principales")
    stonk = st.text_input("**Símbolo de la Acción (ticker):**", value="JNJ").upper()

    st.subheader("📊 Período histórico")
    periodo_historico = st.selectbox(
        "**Período Histórico:**",
        ["1M", "3M", "6M", "1A", "3A", "5A", "Máximo"],
        index=3
    )

    st.subheader("🔍 Comparación de Empresas")
    tickers_comparacion = st.text_area(
        "**Símbolos para comparar (separados por coma):**",
        "AAPL, MSFT, GOOGL, AMZN"
    )
    comparacion_tickers = [t.strip().upper() for t in tickers_comparacion.split(",") if t.strip()]
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("💡 Usa tickers válidos (ej. SPY, AAPL, TSLA, MSFT).")

# ───────────────── Utilidades ─────────────────
PERIODO_YF = {
    "1M": "1mo", "3M": "3mo", "6M": "6mo",
    "1A": "1y", "3A": "3y", "5A": "5y", "Máximo": "max"
}

@st.cache_data(ttl=3600)
def get_stock_data(ticker: str, period_key: str):
    """Descarga datos desde yfinance de forma robusta."""
    try:
        yf_t = yf.Ticker(ticker)
        period = PERIODO_YF.get(period_key, "1y")
        
        # Descargar datos
        df = yf_t.history(period=period)
        
        if df is None or df.empty:
            return pd.DataFrame(), {}
        
        # Reset index para tener Date como columna
        df = df.reset_index()
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
        
        # Asegurar columnas numéricas
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Eliminar filas sin precio
        df = df.dropna(subset=['Close'])
        
        if df.empty:
            return pd.DataFrame(), {}

        # Obtener información
        info = {}
        try:
            info_dict = yf_t.info
            if info_dict:
                # Información básica
                info.update({
                    'currentPrice': info_dict.get('currentPrice'),
                    'regularMarketPreviousClose': info_dict.get('regularMarketPreviousClose'),
                    'marketCap': info_dict.get('marketCap'),
                    'trailingPE': info_dict.get('trailingPE'),
                    'volume': info_dict.get('volume'),
                    'sector': info_dict.get('sector'),
                    'industry': info_dict.get('industry'),
                    'longName': info_dict.get('longName'),
                    'shortName': info_dict.get('shortName'),
                    'fiftyTwoWeekHigh': info_dict.get('fiftyTwoWeekHigh'),
                    'fiftyTwoWeekLow': info_dict.get('fiftyTwoWeekLow'),
                    'dividendYield': info_dict.get('dividendYield'),
                    'beta': info_dict.get('beta'),
                    'priceToBook': info_dict.get('priceToBook'),
                    'priceToSalesTrailing12Months': info_dict.get('priceToSalesTrailing12Months'),
                    'enterpriseToEbitda': info_dict.get('enterpriseToEbitda'),
                    'enterpriseValue': info_dict.get('enterpriseValue'),
                    'returnOnEquity': info_dict.get('returnOnEquity'),
                    'returnOnAssets': info_dict.get('returnOnAssets'),
                    'grossMargins': info_dict.get('grossMargins'),
                    'operatingMargins': info_dict.get('operatingMargins'),
                    'profitMargins': info_dict.get('profitMargins'),
                    'debtToEquity': info_dict.get('debtToEquity'),
                    'dividendRate': info_dict.get('dividendRate'),
                    'averageVolume50Day': info_dict.get('averageVolume50Day'),
                    'fullTimeEmployees': info_dict.get('fullTimeEmployees'),
                    'longBusinessSummary': info_dict.get('longBusinessSummary'),
                    'country': info_dict.get('country'),
                    'website': info_dict.get('website'),
                    'city': info_dict.get('city'),
                    'startYear': info_dict.get('startYear'),
                    'companyOfficers': info_dict.get('companyOfficers')
                })
        except Exception as e:
            st.warning(f"Información limitada para {ticker}")

        return df, info
        
    except Exception as e:
        st.error(f"Error obteniendo datos para {ticker}: {e}")
        return pd.DataFrame(), {}

def safe_format(value, format_str="%.2f", default="N/A"):
    """Formatea valores de forma segura evitando errores con None."""
    if value is None:
        return default
    try:
        if isinstance(value, (int, float)):
            return format_str % value
        else:
            return str(value)
    except (TypeError, ValueError):
        return default

def format_number(n):
    """Formatea números grandes a formato legible."""
    if n is None:
        return "N/A"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "N/A"
    
    if abs(n) >= 1e12:
        return f"${n/1e12:.2f}T"
    elif abs(n) >= 1e9:
        return f"${n/1e9:.2f}B"
    elif abs(n) >= 1e6:
        return f"${n/1e6:.2f}M"
    elif abs(n) >= 1e3:
        return f"${n/1e3:.2f}K"
    else:
        return f"${n:.2f}"

def safe_percent(x):
    """Convierte a porcentaje de forma segura."""
    if x is None:
        return "N/A"
    try:
        return f"{float(x)*100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"

def safe_int_format(x):
    """Formatea enteros de forma segura."""
    if x is None:
        return "N/A"
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "N/A"

def calc_returns(data: pd.DataFrame):
    """Calcula rendimientos para diferentes períodos."""
    if data is None or data.empty:
        return {}
    
    close = data["Close"]
    if close.empty:
        return {}

    today = close.index[-1]
    returns = {}

    # Períodos predefinidos
    periods = {
        "1M": pd.DateOffset(months=1),
        "3M": pd.DateOffset(months=3),
        "6M": pd.DateOffset(months=6),
        "YTD": pd.Timestamp(year=today.year, month=1, day=1),
        "1A": pd.DateOffset(years=1),
        "3A": pd.DateOffset(years=3),
        "5A": pd.DateOffset(years=5),
    }

    for period_name, period_date in periods.items():
        try:
            if period_name == "YTD":
                start_data = close[close.index >= period_date]
            else:
                start_date = today - period_date
                start_data = close[close.index >= start_date]
            
            if len(start_data) > 0:
                start_price = start_data.iloc[0]
                current_price = close.iloc[-1]
                returns[period_name] = ((current_price - start_price) / start_price) * 100
            else:
                returns[period_name] = 0.0
        except Exception:
            returns[period_name] = 0.0

    return returns

def create_returns_chart(data: pd.DataFrame, ticker: str):
    """Crea gráfica de rendimientos acumulados base 0."""
    if data.empty or 'Close' not in data.columns:
        return None
    
    try:
        # Calcular rendimiento acumulado base 0
        initial_price = data['Close'].iloc[0]
        cumulative_returns = ((data['Close'] - initial_price) / initial_price) * 100
        
        fig = go.Figure()
        
        # Gráfica de área para rendimientos - CAMBIO A AZUL
        fig.add_trace(go.Scatter(
            x=data.index,
            y=cumulative_returns,
            fill='tozeroy',
            mode='lines',
            name='Rendimiento Acumulado',
            line=dict(color="#1E88E5", width=3),  # CAMBIADO A AZUL
            fillcolor='rgba(30, 136, 229, 0.3)'   # CAMBIADO A AZUL TRANSPARENTE
        ))
        
        # Línea en 0% para referencia
        fig.add_hline(y=0, line_dash="dash", line_color="#F44336", opacity=0.5)  # ROJO
        
        # Configurar layout claro
        fig.update_layout(
            title=f"Rendimientos Acumulados - {ticker}",
            xaxis_title="Fecha",
            yaxis_title="Rendimiento Acumulado (%)",
            height=400,
            yaxis=dict(
                tickformat=".1f%"
            ),
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfica de rendimientos: {e}")
        return None

def create_period_returns_chart(returns_dict: dict, ticker: str):
    """Crea gráfica de barras para rendimientos por período."""
    if not returns_dict:
        return None
    
    try:
        periods = list(returns_dict.keys())
        values = list(returns_dict.values())
        
        # Colores: VERDE para ganancias, ROJO para pérdidas
        colors = ['#F44336' if x < 0 else "#4CAF50" for x in values]  # ROJO y VERDE
        
        fig = go.Figure(data=[
            go.Bar(
                x=periods,
                y=values,
                marker_color=colors,
                text=[f"{x:.1f}%" for x in values],
                textposition='auto',
                textfont=dict(color='white')
            )
        ])
        
        # Línea en 0% para referencia
        fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.5)
        
        fig.update_layout(
            title=f"Rendimientos por Período - {ticker}",
            xaxis_title="Período",
            yaxis_title="Rendimiento (%)",
            height=400,
            yaxis=dict(
                tickformat=".1f%"
            ),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfica de períodos: {e}")
        return None

def descripcion_500(info: dict) -> str:
    """Devuelve un texto de empresa de EXACTOS 500 caracteres."""
    import re
    
    base = info.get("longBusinessSummary") or ""
    if not base:
        nombre = info.get("longName") or info.get("shortName") or "La compañía"
        sector = info.get("sector") or "N/D"
        industria = info.get("industry") or "N/D"
        pais = info.get("country") or "N/D"
        base = (f"{nombre} opera en el sector {sector} dentro de la industria {industria}. "
                f"Con presencia en {pais}, la empresa busca crecimiento rentable y sostenible.")
    
    # Limpiar y truncar a 500 caracteres
    texto = re.sub(r'\s+', ' ', str(base)).strip()
    if len(texto) <= 500:
        return texto
    else:
        truncated = texto[:500]
        last_dot = truncated.rfind('. ')
        if last_dot > 400:
            return truncated[:last_dot + 1]
        else:
            return truncated

def calculate_risk_metrics(data: pd.DataFrame, info: dict, period_key: str):
    """Calcula métricas de riesgo ajustadas al período seleccionado."""
    if data.empty or 'Close' not in data.columns:
        return {}
    
    try:
        returns = data['Close'].pct_change().dropna()
        
        if len(returns) < 10:  # Mínimo de datos para cálculo
            return {}
        
        # Volatilidad (desviación estándar anualizada)
        volatility = returns.std() * np.sqrt(252) * 100  # En porcentaje
        
        # Máxima caída (Drawdown)
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        
        # VaR (Value at Risk) 95%
        var_95 = np.percentile(returns, 5) * 100  # En porcentaje
        
        # Beta (volatilidad vs mercado)
        beta = info.get('beta', 1.0)
        if beta is None:
            beta = 1.0
        
        # Deuda/Capital
        debt_to_equity = info.get('debtToEquity', 0)
        if debt_to_equity is None:
            debt_to_equity = 0
        
        # AJUSTAR PARÁMETROS SEGÚN EL PERÍODO
        period_adjustments = {
            "1M": {"vol_multiplier": 0.7, "drawdown_multiplier": 0.5, "weight_volatility": 0.35},
            "3M": {"vol_multiplier": 0.8, "drawdown_multiplier": 0.7, "weight_volatility": 0.37},
            "6M": {"vol_multiplier": 0.9, "drawdown_multiplier": 0.85, "weight_volatility": 0.39},
            "1A": {"vol_multiplier": 1.0, "drawdown_multiplier": 1.0, "weight_volatility": 0.4},
            "3A": {"vol_multiplier": 1.1, "drawdown_multiplier": 1.1, "weight_volatility": 0.42},
            "5A": {"vol_multiplier": 1.2, "drawdown_multiplier": 1.2, "weight_volatility": 0.45},
            "Máximo": {"vol_multiplier": 1.3, "drawdown_multiplier": 1.3, "weight_volatility": 0.5}
        }
        
        adjustment = period_adjustments.get(period_key, period_adjustments["1A"])
        
        # CALIFICACIÓN DE RIESGO CON AJUSTE TEMPORAL
        
        # 1. VOLATILIDAD (Peso ajustado por período)
        adjusted_volatility = volatility * adjustment["vol_multiplier"]
        if adjusted_volatility < 20:
            volatility_score = 2
        elif adjusted_volatility < 35:
            volatility_score = 5
        else:
            volatility_score = 8
        
        # 2. DRAWDOWN MÁXIMO (Peso: 25%)
        adjusted_drawdown = abs(max_drawdown) * adjustment["drawdown_multiplier"]
        if adjusted_drawdown < 15:
            drawdown_score = 2
        elif adjusted_drawdown < 30:
            drawdown_score = 5
        else:
            drawdown_score = 8
        
        # 3. BETA (Peso: 20%)
        if beta < 0.8:
            beta_score = 2
        elif beta < 1.2:
            beta_score = 5
        else:
            beta_score = 8
        
        # 4. DEUDA/CAPITAL (Peso: 15%)
        if debt_to_equity < 0.5:
            debt_score = 2
        elif debt_to_equity < 1.0:
            debt_score = 5
        else:
            debt_score = 8
        
        # Score ponderado con ajustes temporales
        weight_volatility = adjustment["weight_volatility"]
        weight_drawdown = 0.25
        weight_beta = 0.2
        weight_debt = 0.15
        
        risk_score = (volatility_score * weight_volatility + 
                     drawdown_score * weight_drawdown + 
                     beta_score * weight_beta + 
                     debt_score * weight_debt)
        
        # Ajuste adicional para períodos muy cortos (más conservador)
        if period_key in ["1M", "3M"]:
            risk_score = min(risk_score + 1, 10)  # Períodos cortos = más riesgo percibido
        
        risk_score = min(10, max(1, risk_score))
        
        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'beta': beta,
            'debt_to_equity': debt_to_equity,
            'risk_score': risk_score,
            'volatility_score': volatility_score,
            'drawdown_score': drawdown_score,
            'beta_score': beta_score,
            'debt_score': debt_score,
            'period_adjustment': adjustment
        }
        
    except Exception as e:
        st.error(f"Error calculando métricas de riesgo: {e}")
        return {}

def create_volatility_chart(data: pd.DataFrame, ticker: str):
    """Crea gráfico de volatilidad histórica."""
    if data.empty or 'Close' not in data.columns:
        return None
    
    try:
        # Calcular volatilidad móvil (30 días)
        returns = data['Close'].pct_change()
        volatility_30d = returns.rolling(window=30).std() * np.sqrt(252) * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatility_30d.index,
            y=volatility_30d.values,
            mode='lines',
            name='Volatilidad 30d',
            line=dict(color="#F43636", width=2),  # ROJO
            fillcolor='rgba(244, 67, 54, 0.3)',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f"Volatilidad Histórica (30 días) - {ticker}",
            xaxis_title="Fecha",
            yaxis_title="Volatilidad Anualizada (%)",
            height=400,
            yaxis=dict(range=[0, max(volatility_30d.max() * 1.1, 10)])
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico de volatilidad: {e}")
        return None

def create_drawdown_chart(data: pd.DataFrame, ticker: str):
    """Crea gráfico de drawdown histórico."""
    if data.empty or 'Close' not in data.columns:
        return None
    
    try:
        returns = data['Close'].pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            name='Drawdown',
            line=dict(color='#F44336', width=2),  # ROJO
            fillcolor='rgba(244, 67, 54, 0.3)',
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=f"Drawdown Histórico - {ticker}",
            xaxis_title="Fecha",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creando gráfico de drawdown: {e}")
        return None

def get_key_metrics(ticker: str):
    """Obtiene las métricas clave para comparación."""
    data, info = get_stock_data(ticker, "1y")
    
    if not info:
        return None
    
    current_price = info.get('currentPrice', data['Close'].iloc[-1] if not data.empty else 0)
    
    # Calcular rendimiento YTD
    ytd_return = 0
    if not data.empty and 'Close' in data.columns:
        try:
            current_year = datetime.now().year
            year_start = data[data.index >= f'{current_year}-01-01']
            if len(year_start) > 0:
                start_price = year_start['Close'].iloc[0]
                ytd_return = ((current_price - start_price) / start_price) * 100
        except:
            ytd_return = 0
    
    return {
        'ticker': ticker,
        'nombre': info.get('longName', ticker),
        'precio_actual': current_price,
        'market_cap': info.get('marketCap'),
        'pe_ratio': info.get('trailingPE'),
        'dividend_yield': info.get('dividendYield'),
        'beta': info.get('beta'),
        'sector': info.get('sector', 'N/A'),
        'ytd_return': ytd_return,
        'volumen_promedio': info.get('averageVolume50Day')
    }

def create_comparison_table(metrics_list):
    """Crea una tabla de comparación de métricas clave."""
    if not metrics_list:
        return None
    
    # Crear DataFrame
    data = []
    for metrics in metrics_list:
        if metrics:
            data.append({
                'Ticker': metrics['ticker'],
                'Nombre': metrics['nombre'][:30] + '...' if len(metrics['nombre']) > 30 else metrics['nombre'],
                'Precio': f"${metrics['precio_actual']:.2f}" if metrics['precio_actual'] else 'N/A',
                'Market Cap': format_number(metrics['market_cap']),
                'P/E Ratio': safe_format(metrics['pe_ratio']),
                'Dividend Yield': safe_percent(metrics['dividend_yield']),
                'Beta': safe_format(metrics['beta']),
                'YTD Return': f"{metrics['ytd_return']:.2f}%",
                'Sector': metrics['sector']
            })
    
    df = pd.DataFrame(data)
    return df

# ───────────────── Tabs ─────────────────
if stonk:
    # NUEVO ORDEN DE TABS
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Resumen General",
        "🏢 Información Empresarial", 
        "📋 Métricas Financieras",
        "🔍 Comparación",
        "📈 Análisis Técnico",
        "⚠️ Análisis de Riesgo"
    ])

    # ───────── Tab 1: Resumen ─────────
    with tab1:
        st.header(f"📊 Resumen General - {stonk}")
        data, info = get_stock_data(stonk, periodo_historico)
        
        if data.empty:
            st.warning(f"No se encontraron datos para {stonk}. Verifica el símbolo.")
        else:
            # Métricas principales
            col1, col2, col3, col4 = st.columns(4)
            
            current_price = info.get("currentPrice", data["Close"].iloc[-1] if not data.empty else 0)
            prev_close = info.get("regularMarketPreviousClose", data["Close"].iloc[-2] if len(data) > 1 else current_price)
            
            try:
                delta = ((current_price - prev_close) / prev_close) * 100 if current_price and prev_close else 0
            except Exception:
                delta = 0
            
            with col1:
                price_display = f"${current_price:.2f}" if current_price else "N/A"
                delta_display = f"{delta:+.2f}%" if current_price else "N/A"
                # VERDE para ganancias, ROJO para pérdidas
                delta_color = "normal" if delta >= 0 else "inverse"
                st.metric("Precio Actual", price_display, delta_display, delta_color=delta_color)
            
            with col2:
                st.metric("Market Cap", format_number(info.get("marketCap")))
            
            with col3:
                pe = info.get("trailingPE")
                st.metric("P/E Ratio", f"{pe:.2f}" if pe else "N/A")
            
            with col4:
                vol = data["Volume"].iloc[-1] if not data["Volume"].empty else info.get("volume")
                st.metric("Volumen", safe_int_format(vol))

            # Variables para las gráficas (se mantienen)
            returns = calc_returns(data)
            per_order = ["1M", "3M", "6M", "YTD", "1A", "3A", "5A"]

            # Gráfica de rendimientos por período (SE MANTIENE)
            st.subheader("📊 Rendimientos por Período")
            returns_chart = create_period_returns_chart(returns, stonk)
            if returns_chart:
                st.plotly_chart(returns_chart, use_container_width=True)

            # Gráfica de rendimientos acumulados (SE MANTIENE)
            st.subheader("📈 Rendimientos Acumulados ")
            cumulative_chart = create_returns_chart(data, stonk)
            if cumulative_chart:
                st.plotly_chart(cumulative_chart, use_container_width=True)

            # Gráfico de precios (SE MANTIENE)
            st.subheader("💹 Evolución del Precio")
            
            if not data.empty and 'Close' in data.columns:
                try:
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Precio',
                        line=dict(color='#1E88E5', width=2)  # AZUL para el precio
                    ))
                    
                    fig.update_layout(
                        title=f"Evolución del Precio - {stonk}",
                        xaxis_title="Fecha",
                        yaxis_title="Precio ($)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error al crear gráfico: {e}")

    # ───────── Tab 2: Información Empresarial ─────────
    with tab2:
        st.header("🏢 Información Empresarial")
        data, info = get_stock_data(stonk, periodo_historico)
        
        if not info:
            st.warning("No hay información empresarial disponible.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📝 Información Básica")
                st.write(f"**Nombre:** {info.get('longName', 'N/A')}")
                st.write(f"**Sector:** {info.get('sector', 'N/A')}")
                st.write(f"**Industria:** {info.get('industry', 'N/A')}")
                st.write(f"**País:** {info.get('country', 'N/A')}")
                st.write(f"**Empleados:** {safe_int_format(info.get('fullTimeEmployees'))}")
                
                website = info.get('website', 'N/A')
                if website != 'N/A':
                    st.write(f"**Sitio Web:** [{website}]({website})")
                else:
                    st.write(f"**Sitio Web:** {website}")
            
            with col2:
                st.subheader("📅 Rango de Precios (52 semanas)")
                st.write(f"**Máximo:** ${safe_format(info.get('fiftyTwoWeekHigh'))}")
                st.write(f"**Mínimo:** ${safe_format(info.get('fiftyTwoWeekLow'))}")
                
                current = info.get('currentPrice', data['Close'].iloc[-1] if not data.empty else 0)
                high_52 = info.get('fiftyTwoWeekHigh')
                if current and high_52:
                    from_high = ((current - high_52) / high_52) * 100
                    st.write(f"**Desde Máximo:** {from_high:+.2f}%")
                
                st.subheader("📊 Trading")
                st.write(f"**Volumen Promedio (50d):** {safe_int_format(info.get('averageVolume50Day'))}")
            
            st.subheader("📖 Descripción de la Empresa")
            descripcion = descripcion_500(info)
            st.info(descripcion)

    # ───────── Tab 3: Métricas Financieras ─────────
    with tab3:
        st.header("📋 Métricas Financieras")
        data, info = get_stock_data(stonk, periodo_historico)
        
        if not info:
            st.warning("No hay información financiera disponible.")
        else:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📊 Valoración")
                st.metric("P/E Ratio", safe_format(info.get("trailingPE")))
                st.metric("P/B Ratio", safe_format(info.get("priceToBook")))
                st.metric("P/S Ratio", safe_format(info.get("priceToSalesTrailing12Months")))
                st.metric("EV/EBITDA", safe_format(info.get("enterpriseToEbitda")))
            
            with col2:
                st.subheader("💰 Rentabilidad")
                st.metric("Margen Bruto", safe_percent(info.get("grossMargins")))
                st.metric("Margen Operativo", safe_percent(info.get("operatingMargins")))
                st.metric("Margen Neto", safe_percent(info.get("profitMargins")))
                st.metric("ROE", safe_percent(info.get("returnOnEquity")))
                st.metric("ROA", safe_percent(info.get("returnOnAssets")))
            
            with col3:
                st.subheader("🏛️ Estructura")
                st.metric("Deuda/Capital", safe_format(info.get("debtToEquity")))
                st.metric("Beta", safe_format(info.get("beta")))
                st.metric("Dividend Yield", safe_percent(info.get("dividendYield")))
                st.metric("Tasa de Dividendo", safe_format(info.get("dividendRate")))

    # ───────── Tab 4: Comparación ─────────
    with tab4:
        st.header("🔍 Comparación de Empresas")
        
        all_tickers = [stonk] + [t for t in comparacion_tickers if t != stonk]
        comparison_data = {}
        
        # Gráfico de comparación de rendimientos (primero)
        st.subheader("📈 Comparación de Rendimientos ")
        
        comparison_data = {}
        for ticker in all_tickers:
            data, _ = get_stock_data(ticker, "1y")
            if not data.empty and 'Close' in data.columns:
                comparison_data[ticker] = data['Close']
        
        if len(comparison_data) >= 2:
            try:
                fig = go.Figure()
                
                colors = ["#356996", '#4CAF50', '#F44336', '#9C27B0', '#FF9800', '#00BCD4']
                
                for i, (ticker, prices) in enumerate(comparison_data.items()):
                    if not prices.empty:
                        normalized = (prices / prices.iloc[0]) * 100
                        fig.add_trace(go.Scatter(
                            x=normalized.index,
                            y=normalized.values,
                            mode='lines',
                            name=ticker,
                            line=dict(width=2, color=colors[i % len(colors)])
                        ))
                
                fig.update_layout(
                    title="Comparación de Rendimientos Normalizados (Base 100)",
                    xaxis_title="Fecha",
                    yaxis_title="Rendimiento (%)",
                    height=500,
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al crear gráfico de comparación: {e}")
        
        # Comparación de sectores
        st.subheader("🏭 Distribución por Sectores")
        
        # Obtener métricas para análisis de sectores
        with st.spinner("Obteniendo datos para comparación..."):
            metrics_list = []
            for ticker in all_tickers:
                metrics = get_key_metrics(ticker)
                if metrics:
                    metrics_list.append(metrics)
        
        if metrics_list:
            sector_counts = {}
            for metrics in metrics_list:
                sector = metrics.get('sector', 'N/A')
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            
            if sector_counts:
                fig_sector = px.pie(
                    values=list(sector_counts.values()),
                    names=list(sector_counts.keys()),
                    title="Distribución de Empresas por Sector"
                )
                st.plotly_chart(fig_sector, use_container_width=True)

        # Comparación de volatilidad (Beta)
        st.subheader("📉 Comparación de Riesgo (Beta)")
        
        if metrics_list:
            beta_data = []
            for metrics in metrics_list:
                if metrics.get('beta'):
                    beta_data.append({
                        'Ticker': metrics['ticker'],
                        'Beta': metrics['beta']
                    })
            
            if beta_data:
                beta_df = pd.DataFrame(beta_data)
                fig_beta = px.bar(
                    beta_df,
                    x='Ticker',
                    y='Beta',
                    title="Comparación de Beta (Volatilidad vs Mercado)",
                    color='Beta',
                    color_continuous_scale=['#4CAF50', '#FF9800', '#F44336']
                )
                fig_beta.update_layout(height=400)
                st.plotly_chart(fig_beta, use_container_width=True)
                
                # Análisis de beta
                st.info("""
                **Interpretación del Beta:**
                - **Beta < 1:** Menos volátil que el mercado
                - **Beta = 1:** Misma volatilidad que el mercado  
                - **Beta > 1:** Más volátil que el mercado
                - **Beta < 0:** Movimiento inverso al mercado
                """)

        # TABLA DE MÉTRICAS CLAVE (AHORA AL FINAL)
        if len(metrics_list) >= 2:
            st.subheader("📊 Métricas Clave Comparativas")
            
            # Mostrar tabla de comparación
            comparison_df = create_comparison_table(metrics_list)
            if comparison_df is not None:
                st.dataframe(
                    comparison_df,
                    use_container_width=True,
                    height=400
                )
                
                # Mostrar insights de la comparación
                st.subheader("💡 Insights de la Comparación")
                
                # Encontrar mejores valores en cada métrica
                if len(metrics_list) > 1:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # Mejor P/E Ratio (más bajo)
                        pe_ratios = [(m['ticker'], m['pe_ratio']) for m in metrics_list if m['pe_ratio'] and m['pe_ratio'] > 0]
                        if pe_ratios:
                            best_pe = min(pe_ratios, key=lambda x: x[1])
                            st.metric("Mejor P/E Ratio", f"{best_pe[0]}: {best_pe[1]:.1f}")
                    
                    with col2:
                        # Mejor Dividend Yield (más alto)
                        dividend_yields = [(m['ticker'], m['dividend_yield']) for m in metrics_list if m['dividend_yield'] and m['dividend_yield'] > 0]
                        if dividend_yields:
                            best_div = max(dividend_yields, key=lambda x: x[1])
                            st.metric("Mejor Dividend Yield", f"{best_div[0]}: {safe_percent(best_div[1])}")
                    
                    with col3:
                        # Mejor YTD Return (más alto)
                        ytd_returns = [(m['ticker'], m['ytd_return']) for m in metrics_list]
                        if ytd_returns:
                            best_ytd = max(ytd_returns, key=lambda x: x[1])
                            st.metric("Mejor YTD", f"{best_ytd[0]}: {best_ytd[1]:.1f}%")
            
            else:
                st.warning("No se pudieron obtener métricas para comparación.")
        else:
            st.warning("Se necesitan al menos 2 tickers válidos para la comparación.")

    # ───────── Tab 5: Análisis Técnico ─────────
    with tab5:
        st.header("📈 Análisis Técnico")
        data, info = get_stock_data(stonk, periodo_historico)
        
        if data.empty:
            st.warning("No hay datos para análisis técnico.")
        else:
            chart_type = st.selectbox(
                "Tipo de Gráfico:",
                ["Línea", "Velas", "Área"],
                key="tech_chart"
            )
            
            try:
                if chart_type == "Velas":
                    fig = go.Figure(data=[go.Candlestick(
                        x=data.index,
                        open=data['Open'],
                        high=data['High'],
                        low=data['Low'],
                        close=data['Close'],
                        name='Velas',
                        # VERDE para ganancias, ROJO para pérdidas
                        increasing_line_color='#4CAF50',
                        decreasing_line_color='#F44336'
                    )])
                elif chart_type == "Área":
                    fig = go.Figure(data=[go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        fill='tozeroy',
                        name='Precio',
                        line=dict(color='#1E88E5'),
                        fillcolor='rgba(30, 136, 229, 0.3)'
                    )])
                else:  # Línea
                    fig = go.Figure(data=[go.Scatter(
                        x=data.index,
                        y=data['Close'],
                        mode='lines',
                        name='Precio',
                        line=dict(color='#1E88E5', width=2)
                    )])
                
                fig.update_layout(
                    title=f"Gráfico de {chart_type} - {stonk}",
                    xaxis_title="Fecha",
                    yaxis_title="Precio ($)",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error al crear gráfico técnico: {e}")

            # Gráfico de volumen
            st.subheader("📊 Volumen de Trading")
            if 'Volume' in data.columns and not data['Volume'].isna().all():
                try:
                    fig_vol = go.Figure(data=[go.Bar(
                        x=data.index,
                        y=data['Volume'],
                        name='Volumen',
                        marker_color='#FF9800'  # NARANJA
                    )])
                    
                    fig_vol.update_layout(
                        title="Volumen de Trading",
                        xaxis_title="Fecha",
                        yaxis_title="Volumen",
                        height=300
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                except Exception as e:
                    st.error(f"Error al crear gráfico de volumen: {e}")

    # ───────── Tab 6: Análisis de Riesgo ─────────
    with tab6:
        st.header("⚠️ Análisis de Riesgo")
        data, info = get_stock_data(stonk, periodo_historico)
        
        if data.empty:
            st.warning("No hay datos para análisis de riesgo.")
        else:
            risk_metrics = calculate_risk_metrics(data, info, periodo_historico)
            
            if risk_metrics:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    # Indicador de riesgo visual
                    risk_score = risk_metrics.get('risk_score', 5)
                    risk_color = "#4CAF50" if risk_score <= 3 else "#FF9800" if risk_score <= 7 else "#F44336"
                    
                    st.markdown(
                        f"""
                        <div style='text-align: center; padding: 20px; border-radius: 10px; background-color: #F8F9FA; border: 2px solid {risk_color};'>
                            <h3 style='color: {risk_color}; margin: 0;'>Nivel de Riesgo</h3>
                            <h1 style='color: {risk_color}; font-size: 48px; margin: 10px 0;'>{risk_score:.1f}/10</h1>
                            <p style='color: #666666; margin: 0;'>
                                {"🟢 Bajo" if risk_score <= 3 else "🟡 Medio" if risk_score <= 7 else "🔴 Alto"}
                            </p>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                
                with col2:
                    st.metric("Volatilidad Anual", f"{risk_metrics.get('volatility', 0):.1f}%")
                    st.metric("Máximo Drawdown", f"{risk_metrics.get('max_drawdown', 0):.1f}%")
                
                with col3:
                    st.metric("Beta", safe_format(risk_metrics.get('beta')))
                    st.metric("VaR 95%", f"{risk_metrics.get('var_95', 0):.1f}%")
                
                with col4:
                    st.metric("Deuda/Capital", safe_format(risk_metrics.get('debt_to_equity')))
                
                # Gráficos de riesgo
                st.subheader("📊 Análisis de Volatilidad")
                vol_chart = create_volatility_chart(data, stonk)
                if vol_chart:
                    st.plotly_chart(vol_chart, use_container_width=True)
                
                st.subheader("📉 Análisis de Drawdown")
                drawdown_chart = create_drawdown_chart(data, stonk)
                if drawdown_chart:
                    st.plotly_chart(drawdown_chart, use_container_width=True)
                
                # Desglose del score de riesgo
                st.subheader("🔍 Desglose del Score de Riesgo")
                risk_cols = st.columns(4)
                
                with risk_cols[0]:
                    st.metric("Volatilidad", f"{risk_metrics.get('volatility_score', 0)}/8")
                with risk_cols[1]:
                    st.metric("Drawdown", f"{risk_metrics.get('drawdown_score', 0)}/8")
                with risk_cols[2]:
                    st.metric("Beta", f"{risk_metrics.get('beta_score', 0)}/8")
                with risk_cols[3]:
                    st.metric("Deuda", f"{risk_metrics.get('debt_score', 0)}/8")

# ───────────────── Footer ─────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666666;'>"
    " Análisis Financiero | Desarrollado con Yahoo Finance | Daniela Marin"
    "</div>", 
    unsafe_allow_html=True
)
