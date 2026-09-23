import streamlit as st
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FLOODGRAPH-RAG",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* Main background */

.stApp {
    background:
        radial-gradient(circle at 10% 10%, #dff6ff 0%, transparent 30%),
        radial-gradient(circle at 90% 20%, #e8f4ff 0%, transparent 30%),
        linear-gradient(135deg, #f5fbff 0%, #eef6fb 100%);
}


/* Main content */

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #06283d 0%,
        #0b4f71 55%,
        #136f91 100%
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}


/* Hero */

.hero {
    padding: 38px;
    border-radius: 25px;
    background:
        linear-gradient(
            135deg,
            rgba(4, 55, 78, 0.97),
            rgba(12, 104, 145, 0.95)
        );
    color: white;
    box-shadow: 0 10px 35px rgba(0, 65, 100, 0.20);
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 48px;
    margin: 0;
    font-weight: 800;
}

.hero h3 {
    margin-top: 10px;
    font-weight: 400;
    opacity: 0.95;
}

.hero p {
    font-size: 17px;
    line-height: 1.7;
}


/* Section titles */

.section-title {
    font-size: 28px;
    font-weight: 800;
    color: #063b57;
    margin-top: 25px;
    margin-bottom: 18px;
}


/* Cards */

.card {
    background: rgba(255,255,255,0.92);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #dcecf4;
    box-shadow: 0 8px 25px rgba(0,70,100,0.08);
    height: 100%;
}

.card h3 {
    color: #075985;
}

.card p {
    color: #4b6472;
    line-height: 1.7;
}


/* Metric cards */

.metric-card {
    background: white;
    padding: 22px;
    border-radius: 18px;
    border: 1px solid #dcecf4;
    box-shadow: 0 6px 20px rgba(0,70,100,0.07);
    text-align: center;
}

.metric-title {
    color: #6b7c86;
    font-size: 14px;
}

.metric-value {
    color: #075985;
    font-size: 30px;
    font-weight: 800;
    margin-top: 5px;
}


/* Risk card */

.risk-card {
    padding: 30px;
    border-radius: 22px;
    background: white;
    border: 2px solid #d9edf7;
    box-shadow: 0 8px 25px rgba(0,70,100,0.10);
    text-align: center;
}

.risk-number {
    font-size: 46px;
    font-weight: 800;
    color: #075985;
}


/* Workflow */

.workflow {
    text-align: center;
    padding: 22px;
    background: white;
    border-radius: 18px;
    border: 1px solid #dcecf4;
    min-height: 150px;
}


/* Info box */

.info-box {
    background: #eff9ff;
    border-left: 5px solid #0ea5e9;
    padding: 18px;
    border-radius: 12px;
    margin-bottom: 15px;
}


/* Footer */

.footer {
    margin-top: 50px;
    padding: 25px;
    text-align: center;
    color: #68808d;
    border-top: 1px solid #d9e8ef;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

PREDICTION_FILE = "outputs/flood_risk_predictions.csv"

try:

    predictions = pd.read_csv(PREDICTION_FILE)

except FileNotFoundError:

    st.error(
        "❌ Prediction file not found.\n\n"
        "Required file:\n"
        "`outputs/flood_risk_predictions.csv`"
    )

    st.stop()


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Node_ID",
    "District",
    "Historical_Flood_Count",
    "Recent_Flood_Count",
    "Last_Flood_Year",
    "Years_Since_Last_Flood",
    "Flood_2023",
    "Flood_Probability",
    "Predicted_Flood",
    "Risk_Level"
]

missing = [
    c for c in required_columns
    if c not in predictions.columns
]

if missing:

    st.error(
        f"Missing columns in prediction file: {missing}"
    )

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🌊 FLOODGRAPH-RAG</h1>

<h3>AI-Powered Flood Prediction & Disaster Intelligence</h3>

<p>
A graph-based flood prediction system that combines
historical flood memory, GCN-based prediction,
retrieval-augmented disaster knowledge and live weather
information.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.markdown("""
<h1 style="text-align:center;">🌊</h1>

<h2 style="text-align:center;">
FLOODGRAPH-RAG
</h2>

<p style="text-align:center;">
Flood Intelligence Platform
</p>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 About",
        "🚨 Flood Risk Check",
        "🌦️ Live Weather",
        "🧠 RAG Assistant",
        "📊 District Explorer",
        "ℹ️ About the Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "AI Flood Risk Analysis Platform"
)


# ============================================================
# ABOUT PAGE
# ============================================================

if page == "🏠 About":

    st.markdown(
        '<div class="section-title">🌊 About FLOODGRAPH-RAG</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>What is FLOODGRAPH-RAG?</h3>

    <p>
    FLOODGRAPH-RAG is an AI-based flood intelligence system
    designed to analyze historical flood patterns and estimate
    flood risk at the district level.
    </p>

    <p>
    The system combines a Graph Convolutional Network (GCN)
    with historical flood information and a semantic-search
    assistant that retrieves relevant flood-safety guidance
    using sentence-transformer embeddings.
    </p>

    <p>
    Live weather information is also used to provide current
    rainfall context alongside the historical AI prediction.
    </p>

    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">⚙️ How the System Works</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
        <div class="workflow">

        <h2>📚</h2>
        <h3>Historical Data</h3>

        <p>
        Historical flood events are processed
        to create district-level flood memory.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="workflow">

        <h2>🕸️</h2>
        <h3>Graph + GCN</h3>

        <p>
        District information is represented
        as nodes in a graph for prediction.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="workflow">

        <h2>🚨</h2>
        <h3>Risk Assessment</h3>

        <p>
        The model produces flood probability
        and risk classification.
        </p>

        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="workflow">

        <h2>🧠</h2>
        <h3>Semantic Search</h3>

        <p>
        Relevant flood-safety guidance is retrieved
        by embedding similarity, to support
        disaster-related questions.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">📌 What You Can Do</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Use the navigation menu to check a district's flood risk, "
        "view live weather, explore district predictions, or ask "
        "the RAG disaster assistant."
    )


# ============================================================
# FLOOD RISK CHECK
# ============================================================

elif page == "🚨 Flood Risk Check":

    st.markdown(
        '<div class="section-title">🚨 Flood Risk Check</div>',
        unsafe_allow_html=True
    )

    districts = sorted(
        predictions["District"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_district = st.selectbox(
        "📍 Select a district",
        districts
    )

    data = predictions[
        predictions["District"].astype(str)
        == selected_district
    ]

    row = data.iloc[0]

    probability = float(
        row["Flood_Probability"]
    )

    risk = str(
        row["Risk_Level"]
    )

    predicted = int(
        row["Predicted_Flood"]
    )

    historical = int(
        row["Historical_Flood_Count"]
    )

    recent = int(
        row["Recent_Flood_Count"]
    )

    last_year = row["Last_Flood_Year"]

    years_since = int(
        row["Years_Since_Last_Flood"]
    )


    st.markdown(
        '<div class="section-title">📊 AI Prediction</div>',
        unsafe_allow_html=True
    )

    a, b, c, d = st.columns(4)

    with a:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">
        Flood Probability
        </div>
        <div class="metric-value">
        {probability:.1%}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">
        Risk Level
        </div>
        <div class="metric-value">
        {risk}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">
        Historical Floods
        </div>
        <div class="metric-value">
        {historical}
        </div>
        </div>
        """, unsafe_allow_html=True)

    with d:
        st.markdown(f"""
        <div class="metric-card">
        <div class="metric-title">
        Recent Floods
        </div>
        <div class="metric-value">
        {recent}
        </div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:

        st.markdown("""
        <div class="risk-card">

        <h2>🤖 GCN Flood Probability</h2>

        <div class="risk-number">
        """ + f"{probability:.1%}" + """
        </div>

        </div>
        """, unsafe_allow_html=True)

        st.progress(
            min(max(probability, 0), 1)
        )


    with right:

        st.markdown("""
        <div class="risk-card">

        <h2>🚨 Prediction</h2>

        <div class="risk-number">
        """ +
        ("⚠️ FLOOD" if predicted == 1
         else "✅ NO FLOOD") +
        """
        </div>

        </div>
        """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">📚 Historical Flood Memory</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric(
            "Historical Floods",
            historical
        )

    with h2:
        st.metric(
            "Recent Floods",
            recent
        )

    with h3:
        st.metric(
            "Last Flood Year",
            last_year
        )

    with h4:
        st.metric(
            "Years Since Last Flood",
            years_since
        )


# ============================================================
# LIVE WEATHER
# ============================================================

elif page == "🌦️ Live Weather":

    st.markdown(
        '<div class="section-title">🌦️ Live Weather Monitor</div>',
        unsafe_allow_html=True
    )

    districts = sorted(
        predictions["District"]
        .dropna()
        .astype(str)
        .unique()
    )

    selected_district = st.selectbox(
        "📍 Select district",
        districts
    )


    @st.cache_data(ttl=600)
    def get_coordinates(district):

        url = (
            "https://geocoding-api.open-meteo.com/v1/search"
        )

        params = {
            "name": district,
            "count": 1,
            "language": "en",
            "format": "json"
        }

        r = requests.get(
            url,
            params=params,
            timeout=10
        )

        r.raise_for_status()

        data = r.json()

        if "results" not in data:
            return None

        result = data["results"][0]

        return (
            result["latitude"],
            result["longitude"]
        )


    @st.cache_data(ttl=600)
    def get_weather(lat, lon):

        url = (
            "https://api.open-meteo.com/v1/forecast"
        )

        params = {
            "latitude": lat,
            "longitude": lon,
            "current":
                "temperature_2m,"
                "precipitation,"
                "rain,"
                "weather_code",
            "forecast_days": 1
        }

        r = requests.get(
            url,
            params=params,
            timeout=10
        )

        r.raise_for_status()

        return r.json()


    try:

        coordinates = get_coordinates(
            selected_district
        )

        if coordinates:

            lat, lon = coordinates

            weather = get_weather(
                lat,
                lon
            )

            current = weather["current"]

            temperature = current.get(
                "temperature_2m", 0
            )

            rainfall = current.get(
                "rain", 0
            )

            precipitation = current.get(
                "precipitation", 0
            )


            w1, w2, w3 = st.columns(3)

            with w1:
                st.metric(
                    "🌡️ Temperature",
                    f"{temperature} °C"
                )

            with w2:
                st.metric(
                    "🌧️ Rainfall",
                    f"{rainfall} mm"
                )

            with w3:
                st.metric(
                    "💧 Precipitation",
                    f"{precipitation} mm"
                )


            if rainfall == 0:
                status = "No Rain"
            elif rainfall < 5:
                status = "Light Rain"
            elif rainfall < 15:
                status = "Moderate Rain"
            elif rainfall < 30:
                status = "Heavy Rain"
            else:
                status = "Very Heavy Rain"


            st.markdown(
                f"""
                <div class="info-box">

                <b>Current Rainfall Status:</b>
                {status}

                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.warning(
                "Weather location could not be found."
            )

    except Exception as e:

        st.warning(
            f"Live weather unavailable: {e}"
        )


# ============================================================
# RAG ASSISTANT
# ============================================================

elif page == "🧠 RAG Assistant":

    st.markdown(
        '<div class="section-title">🧠 Flood Disaster Assistant</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="card">

    <h3>Ask the FLOODGRAPH-RAG Assistant</h3>

    <p>
    Ask questions about flood preparedness, flood response,
    historical flood memory, flood risk and recovery.
    </p>

    </div>
    """, unsafe_allow_html=True)


    knowledge_base = [

        {
            "title": "Flood Risk Assessment",
            "text":
            "Flood risk assessment uses historical flood frequency, "
            "recent flood activity, years since the last flood and "
            "AI-based flood probability to understand district-level risk."
        },

        {
            "title": "Flood Preparedness",
            "text":
            "Flood preparedness includes monitoring rainfall, "
            "preparing emergency supplies, protecting important documents, "
            "planning evacuation routes and following official warnings."
        },

        {
            "title": "Flood Response",
            "text":
            "During flooding, follow official emergency instructions, "
            "avoid moving through flood water, move to safer elevated "
            "locations and contact emergency services when required."
        },

        {
            "title": "Historical Flood Memory",
            "text":
            "Historical flood memory records previous flood events "
            "for a district. Historical flood count, recent flood count, "
            "last flood year and years since the last flood describe "
            "past flood patterns."
        },

        {
            "title": "Flood Recovery",
            "text":
            "Flood recovery includes damage assessment, restoring "
            "essential services, cleaning affected areas safely, "
            "supporting communities and rebuilding infrastructure."
        }

    ]


    @st.cache_resource
    def load_model():

        return SentenceTransformer(
            "all-MiniLM-L6-v2"
        )


    @st.cache_resource
    def load_knowledge_embeddings():
        # Computed once per app session and reused on every
        # click, instead of re-encoding the same 5 passages
        # every time the button is pressed.
        model = load_model()

        documents = [
            item["text"]
            for item in knowledge_base
        ]

        return model.encode(documents)


    try:

        model = load_model()
        doc_embeddings = load_knowledge_embeddings()

        question = st.text_area(
            "💬 Your question",
            placeholder=
            "Example: How should I prepare for a flood?"
        )

        if st.button(
            "🔎 Search Flood Knowledge",
            use_container_width=True
        ):

            if question.strip() == "":

                st.warning(
                    "Please enter a question."
                )

            else:

                query_embedding = model.encode(
                    [question]
                )

                scores = cosine_similarity(
                    query_embedding,
                    doc_embeddings
                )[0]

                # Get the single most relevant document
                best_index = scores.argmax()
                best_score = scores[best_index]

                item = knowledge_base[best_index]

                st.markdown("### 🔎 Retrieved Knowledge")

                st.markdown(
                        f"""
                       <div class="info-box">

                          <h4>📌 {item["title"]} · Relevance: {best_score:.0%}</h4>

                         <p>{item["text"]}</p>

                        </div>
                     """,
                    unsafe_allow_html=True
                )

    except Exception as e:

        st.error(
            f"RAG model could not be loaded: {e}"
        )


# ============================================================
# DISTRICT EXPLORER
# ============================================================

elif page == "📊 District Explorer":

    st.markdown(
        '<div class="section-title">📊 District Explorer</div>',
        unsafe_allow_html=True
    )


    risk_distribution = (
        predictions["Risk_Level"]
        .value_counts()
        .reset_index()
    )

    risk_distribution.columns = [
        "Risk Level",
        "Number of Districts"
    ]


    st.markdown(
        "### Risk Distribution"
    )

    st.bar_chart(
        risk_distribution.set_index(
            "Risk Level"
        )
    )


    st.markdown(
        "### District Prediction Data"
    )

    st.dataframe(
        predictions,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# MODEL PAGE
# ============================================================

elif page == "ℹ️ About the Model":

    st.markdown(
        '<div class="section-title">🧠 About the AI Model</div>',
        unsafe_allow_html=True
    )


    c1, c2 = st.columns(2)


    with c1:

        st.markdown("""
        <div class="card">

        <h3>🕸️ Graph Convolutional Network</h3>

        <p>
        The GCN processes district-level information
        represented as a graph. District flood characteristics
        are used to estimate flood probability.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with c2:

        st.markdown("""
        <div class="card">

        <h3>📚 Historical Flood Memory</h3>

        <p>
        Historical flood count, recent flood count,
        last flood year and years since the last flood
        provide historical context for each district.
        </p>

        </div>
        """, unsafe_allow_html=True)


    st.markdown("<br>", unsafe_allow_html=True)


    c3, c4 = st.columns(2)


    with c3:

        st.markdown("""
        <div class="card">

        <h3>🧠 Semantic Search</h3>

        <p>
        Sentence-transformer embeddings and cosine similarity
        retrieve the closest matching flood-safety passage for
        the user's question. This is retrieval only — there's
        no generation step writing a new answer.
        </p>

        </div>
        """, unsafe_allow_html=True)


    with c4:

        st.markdown("""
        <div class="card">

        <h3>🌦️ Live Weather</h3>

        <p>
        Live weather information provides current rainfall
        and precipitation context for the selected district.
        </p>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

🌊 <b>FLOODGRAPH-RAG</b>

<br><br>

AI Flood Prediction • Graph Neural Network •
Historical Flood Memory • RAG • Live Weather

<br><br>

<i>AI prototype for flood-risk analysis and disaster intelligence</i>

</div>
""", unsafe_allow_html=True)