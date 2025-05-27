"""
Streamlit app for sentiment analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from predict import SentimentPredictor, load_default_model
import joblib
from datetime import datetime
import re


# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 1rem 0;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .positive-sentiment {
        color: #28a745;
        font-weight: bold;
    }
    .negative-sentiment {
        color: #dc3545;
        font-weight: bold;
    }
    .neutral-sentiment {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the sentiment analysis model (cached)"""
    return load_default_model()


def get_sentiment_color(sentiment, confidence):
    """Get color based on sentiment and confidence"""
    if sentiment == "Positive":
        return "#28a745" if confidence > 0.7 else "#28a74580"
    else:
        return "#dc3545" if confidence > 0.7 else "#dc354580"


def create_confidence_gauge(confidence,