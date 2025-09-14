import streamlit as st
from PIL import Image

# Page config
st.set_page_config(
    page_title="Float Chat - Ocean Intelligence",
    page_icon="🌊",
    layout="wide"
)

# Pure black and white styling with Raleway and Bitcount Grid Double
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Raleway:wght@100;200;300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Bitcount+Grid+Double&display=swap');
    
    .stApp {
        background: #000000;
        color: #ffffff;
        font-family: 'Raleway', sans-serif;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Bitcount Grid Double for all h1 tags */
    h1, .stApp h1 {
        font-family: 'Bitcount Grid Double', monospace !important;
        font-weight: normal !important;
    }
    
    /* Pure white text with Raleway font */
    .gradient-text {
        color: #ffffff;
        font-family: 'Raleway', sans-serif;
        font-weight: 700;
        line-height: 1.2;
    }
    
    .hero-gradient {
        color: #ffffff;
        font-family: 'Bitcount Grid Double', monospace;
        font-size: 5rem;
        font-weight: normal;
        text-align: center;
        margin: 2rem 0;
        line-height: 1.1;
        text-shadow: 0 0 30px rgba(255, 255, 255, 0.5);
    }
    
    .subtitle-gradient {
        color: #ffffff;
        font-family: 'Raleway', sans-serif;
        font-size: 2rem;
        font-weight: 300;
        text-align: center;
        margin-bottom: 3rem;
    }
    
    /* Content styling */
    .main-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    
.description, p.description, div.description {
    font-size: 1.2rem;
    line-height: 1.8;
    color: #ffffff;
    text-align: center;
    margin: 2rem auto 4rem auto;
    max-width: 800px;
    border: 2px solid #ffffff;
    padding: 2rem;
    border-radius: 16px;
    background: #000000;
    font-family: 'Raleway', sans-serif;
    font-weight: 400;
}

    
    /* Pure black and white feature cards */
    .feature-card {
        background: #000000;
        border: 3px solid #ffffff;
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1rem 0;
        transition: all 0.4s ease;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 15px 50px rgba(255, 255, 255, 0.4);
        border-width: 4px;
    }
    
    .feature-title {
        color: #ffffff;
        font-family: 'Raleway', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    .feature-text {
        color: #ffffff;
        line-height: 1.7;
        font-size: 1.1rem;
        font-family: 'Raleway', sans-serif;
        font-weight: 400;
    }
    
    /* Pure white outline images */
    .image-container {
        border-radius: 20px;
        overflow: hidden;
        border: 3px solid #ffffff;
        transition: all 0.4s ease;
        margin: 1rem 0;
        box-shadow: 0 0 20px rgba(255, 255, 255, 0.2);
    }
    
    .image-container:hover {
        box-shadow: 0 15px 40px rgba(255, 255, 255, 0.4);
        transform: translateY(-5px);
        border-width: 4px;
    }
    
    /* Pure white CTA Button */
    .cta-button {
        background: #ffffff;
        color: #000000;
        padding: 1.5rem 3rem;
        border: 3px solid #ffffff;
        border-radius: 50px;
        font-size: 1.3rem;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.4s ease;
        display: inline-block;
        text-decoration: none;
        margin: 3rem auto;
        text-align: center;
        font-family: 'Raleway', sans-serif;
        box-shadow: 0 0 30px rgba(255, 255, 255, 0.3);
    }
    
    .cta-button:hover {
        transform: translateY(-5px) scale(1.05);
        box-shadow: 0 20px 60px rgba(255, 255, 255, 0.5);
        background: #000000;
        color: #ffffff;
    }
    
    /* Pure white footer */
    .footer-text {
        text-align: center;
        color: #ffffff;
        font-size: 1.1rem;
        margin: 4rem 0 2rem 0;
        border-top: 2px solid #ffffff;
        padding-top: 2rem;
        font-family: 'Raleway', sans-serif;
        font-weight: 400;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .hero-gradient {
            font-size: 3rem;
        }
        .subtitle-gradient {
            font-size: 1.5rem;
        }
        .main-content {
            padding: 1rem;
        }
        .feature-card {
            padding: 1.5rem;
        }
    }
    
    /* Enhanced white glow effects */
    .glow-text {
        text-shadow: 0 0 15px rgba(255, 255, 255, 0.6),
                     0 0 25px rgba(255, 255, 255, 0.4),
                     0 0 35px rgba(255, 255, 255, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hero Section - Fixed version
st.markdown("""
<div class="main-content">
    <h1 class="hero-gradient glow-text">Float Chat</h1>
    <p class="subtitle-gradient">Dive into Ocean Intelligence with AI</p>
</div>
""", unsafe_allow_html=True)

# Separate the description outside the main-content div
st.markdown("""
<p class="description">
    Explore the depths of our oceans through the lens of artificial intelligence. 
    Float Chat combines cutting-edge AI technology with real-time oceanographic data 
    from Argo floats to unlock the secrets of marine ecosystems worldwide.
</p>
""", unsafe_allow_html=True)

# Features Section
st.markdown('<h1 class="gradient-text glow-text" style="font-size: 3rem; text-align: center; margin: 4rem 0 3rem 0;">Powerful Features</h1>', unsafe_allow_html=True)

# Create feature cards in columns
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">🔍 Intelligent Database Queries</h3>
        <p class="feature-text">
            Ask natural language questions about historical oceanographic measurements 
            and receive instant, accurate answers from our comprehensive database.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">📊 NetCDF Data Analysis</h3>
        <p class="feature-text">
            Access and analyze processed float profiles including temperature, salinity, 
            and pressure data with advanced visualization capabilities.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">🤖 RAG-Powered Intelligence</h3>
        <p class="feature-text">
            Leverage Retrieval-Augmented Generation to extract meaningful insights 
            from complex oceanographic datasets with unprecedented accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h3 class="feature-title">📈 Interactive Visualizations</h3>
        <p class="feature-text">
            Explore temperature and salinity profiles across different regions with 
            our modern, interactive dashboard and real-time data visualization.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Argo Buoys Section
st.markdown('<h1 class="gradient-text glow-text" style="font-size: 3rem; text-align: center; margin: 5rem 0 3rem 0;">Argo Floats in Action</h1>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

# Image handling
try:
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        img1 = Image.open(r"pages/floating_buoy.jpeg")
        st.image(img1, caption="Autonomous Ocean Monitoring", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        img2 = Image.open(r"pages/floating_buoy_2.jpeg")
        st.image(img2, caption="Deep Sea Data Collection", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        img3 = Image.open(r"pages/floating_buoy_3.jpeg")
        st.image(img3, caption="Global Ocean Intelligence", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
except FileNotFoundError:
    st.markdown("""
    <div style="text-align: center; color: #ffffff; padding: 3rem; border: 3px solid #ffffff; border-radius: 16px; margin: 2rem 0; background: #000000;">
        <p style="font-size: 1.2rem;">📷 Argo buoy images will appear here once you add the image files to your pages directory</p>
    </div>
    """, unsafe_allow_html=True)

# Call to Action
st.markdown("""
<div style="text-align: center; margin: 5rem 0;">
    <div class="cta-button">
        Start Exploring the Oceans
    </div>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-text">
    <p><strong>Float Chat</strong> • Where AI meets Oceanography • Discover the Future of Marine Science</p>
    <p>🌊 Powered by Argo Float Data • Built By SIRA Team</p>
</div>
""", unsafe_allow_html=True)
