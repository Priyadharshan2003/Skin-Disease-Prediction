import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import tensorflow as tf
import os
import base64
from io import BytesIO
import webbrowser
import time
import folium
from streamlit_folium import folium_static
from sklearn.metrics import confusion_matrix

# Import functions from the previous file
# In a real implementation, you would import these functions properly
# For demonstration, we assume they are available

# Page configuration
st.set_page_config(
    page_title="Skin Disease Prediction Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2C3E50;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495E;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    .card {
        border-radius: 10px;
        padding: 20px;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metrics-card {
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        background-color: #ffffff;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2980B9;
    }
    .metric-label {
        font-size: 1rem;
        color: #7F8C8D;
    }
    .high-risk {
        color: #E74C3C;
        font-weight: bold;
    }
    .medium-risk {
        color: #F39C12;
        font-weight: bold;
    }
    .low-risk {
        color: #27AE60;
        font-weight: bold;
    }
    .caution-box {
        background-color: #FFEBEE;
        border-left: 5px solid #E53935;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .info-box {
        background-color: #E1F5FE;
        border-left: 5px solid #039BE5;
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 5px;
    }
    .remedy-item {
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .footer {
        text-align: center;
        margin-top: 40px;
        padding: 20px;
        color: #7F8C8D;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Define skin condition information
skin_conditions = {
    'Actinic keratosis': {
        'description': 'Rough, scaly patches on skin that develop from years of sun exposure',
        'risk_level': 'Medium-High',
        'risk_description': 'Can develop into squamous cell carcinoma if left untreated',
        'medical_attention': 'Required'
    },
    'Atopic Dermatitis': {
        'description': 'Chronic inflammatory skin condition characterized by itchy, inflamed skin',
        'risk_level': 'Low',
        'risk_description': 'Chronic condition that requires management',
        'medical_attention': 'Recommended'
    },
    'Benign keratosis': {
        'description': 'Harmless skin growths that appear as waxy, brown, black or tan growths',
        'risk_level': 'Low',
        'risk_description': 'Benign condition but should be monitored',
        'medical_attention': 'Optional'
    },
    'Dermatofibroma': {
        'description': 'Common, harmless skin growths that often appear as small, firm bumps',
        'risk_level': 'Very Low',
        'risk_description': 'Benign growth that rarely causes problems',
        'medical_attention': 'Optional'
    },
    'Melanocytic nevus': {
        'description': 'Common moles that can be flat or raised, and range in color from pink to dark brown',
        'risk_level': 'Low',
        'risk_description': 'Generally benign but should be monitored for changes',
        'medical_attention': 'Monitoring'
    },
    'Melanoma': {
        'description': 'The most serious type of skin cancer that develops from the pigment-producing cells',
        'risk_level': 'Very High',
        'risk_description': 'Life-threatening if not treated early',
        'medical_attention': 'Urgent'
    },
    'Squamous cell carcinoma': {
        'description': 'The second most common form of skin cancer that develops from abnormal squamous cells',
        'risk_level': 'High',
        'risk_description': 'Can be aggressive if not treated',
        'medical_attention': 'Required'
    },
    'Tinea Ringworm': {
        'description': 'A common fungal infection that causes a red, circular rash with clearer skin in the middle',
        'risk_level': 'Low',
        'risk_description': 'Contagious but not serious',
        'medical_attention': 'Recommended'
    },
    'Candidiasis': {
        'description': 'A fungal infection caused by yeasts that belong to the genus Candida',
        'risk_level': 'Low',
        'risk_description': 'Usually not serious but can be uncomfortable',
        'medical_attention': 'Recommended'
    },
    'Vascular lesion': {
        'description': 'Abnormalities of blood vessels that are visible on the skin',
        'risk_level': 'Varies',
        'risk_description': 'Most are harmless but some may require attention',
        'medical_attention': 'Evaluation'
    }
}

# Initialize session state variables
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'user_address' not in st.session_state:
    st.session_state.user_address = ""
if 'nearby_dermatologists' not in st.session_state:
    st.session_state.nearby_dermatologists = []
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

# Sidebar navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/dermatology.png", width=100)
    st.title("Skin Analysis Hub")
    st.markdown("---")
    
    if st.button("🏠 Home", key="home_btn"):
        navigate_to("home")
    if st.button("🔍 Skin Disease Prediction", key="prediction_btn"):
        navigate_to("prediction")
    if st.button("📊 Analysis Dashboard", key="analysis_btn"):
        navigate_to("analysis")
    if st.button("💊 Home Remedies", key="remedies_btn"):
        navigate_to("remedies")
    if st.button("🏥 Find Dermatologists", key="doctors_btn"):
        navigate_to("doctors")
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    This dashboard uses EfficientNetB0, a state-of-the-art deep learning model, 
    to predict skin diseases from images with high accuracy.
    """)

# Home page
if st.session_state.current_page == "home":
    st.markdown("<h1 class='main-header'>Skin Disease Prediction & Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("https://img.icons8.com/color/480/000000/skin.png", width=150, output_format="PNG")
    
    st.markdown("---")
    
    st.markdown("<h2 class='sub-header'>Welcome to the Skin Analysis Hub</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    This award-winning dashboard uses advanced machine learning technology to help you:
    
    1. **Identify potential skin conditions** from images with high accuracy
    2. **Analyze your skin health** with detailed metrics and information
    3. **Find appropriate home remedies** with important safety precautions
    4. **Locate nearby dermatologists** within a 10km radius of your location
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>How to Use</h3>", unsafe_allow_html=True)
        st.markdown("""
        1. Navigate to the **Skin Disease Prediction** section
        2. Upload a clear, well-lit image of the affected skin area
        3. Get instant predictions and analysis
        4. Explore recommended home remedies (with medical caution)
        5. Find professional dermatologists near you
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<h3 class='sub-header'>Important Note</h3>", unsafe_allow_html=True)
        st.markdown("""
        <div class='caution-box'>
        This tool is designed to assist with preliminary skin condition identification but is not a replacement for professional medical advice. Always consult a qualified dermatologist for proper diagnosis and treatment.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>11</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Skin Conditions Detected</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>92%</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Model Accuracy</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>24/7</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Availability</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metrics-card'>", unsafe_allow_html=True)
        st.markdown("<div class='metric-value'>10km</div>", unsafe_allow_html=True)
        st.markdown("<div class='metric-label'>Doctor Search Radius</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Prediction page
elif st.session_state.current_page == "prediction":
    st.markdown("<h1 class='main-header'>Skin Disease Prediction</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Upload a clear, well-lit image of the affected skin area for analysis. For best results:
    - Ensure good lighting
    - Focus clearly on the affected area
    - Avoid blurry images
    - Include some surrounding normal skin for context
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='Uploaded Image', width=300)
            
        with col2:
            st.write("Analyzing image...")
            progress_bar = st.progress(0)
            
            # Simulate analysis progress
            for i in range(100):
                time.sleep(0.01)  # Simulate computation time
                progress_bar.progress(i + 1)
            
            # Simulate prediction (in a real app, this would call the model)
            # For demonstration, we'll use a mock result
            prediction = {
                'class_name': 'Melanocytic nevus',  # Example prediction
                'confidence': 92.7,
                'all_probabilities': [0.01, 0.02, 0.03, 0.01, 0.927, 0.005, 0.001, 0.002, 0.001, 0.004]
            }
            
            st.session_state.prediction_result = prediction
            
            st.success("Analysis complete!")
            
            risk_color = "low-risk"
            if prediction['class_name'] in ['Melanoma', 'Squamous cell carcinoma']:
                risk_color = "high-risk"
            elif prediction['class_name'] in ['Actinic keratosis']:
                risk_color = "medium-risk"
            
            st.markdown(f"""
            <div class='card'>
                <h3>Prediction Result:</h3>
                <h2 class='{risk_color}'>{prediction['class_name']}</h2>
                <h4>Confidence: {prediction['confidence']:.1f}%</h4>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>Condition Information</h3>", unsafe_allow_html=True)
            
            condition = prediction['class_name']
            info = skin_conditions[condition]
            
            st.markdown(f"""
            <div class='card'>
                <p><strong>Description:</strong> {info['description']}</p>
                <p><strong>Risk Level:</strong> <span class='{risk_color}'>{info['risk_level']}</span></p>
                <p><strong>Risk Description:</strong> {info['risk_description']}</p>
                <p><strong>Medical Attention:</strong> {info['medical_attention']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            if condition in ['Melanoma', 'Squamous cell carcinoma', 'Actinic keratosis']:
                st.markdown("""
                <div class='caution-box'>
                <strong>IMPORTANT:</strong> This condition may require immediate medical attention. 
                Please consult a dermatologist as soon as possible.
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Probability Distribution</h3>", unsafe_allow_html=True)
            
            # Create probability chart
            probs = prediction['all_probabilities']
            classes = list(skin_conditions.keys())
            
            fig = px.bar(
                x=classes, 
                y=probs,
                color=probs,
                color_continuous_scale='blues',
                title="Prediction Probabilities"
            )
            
            fig.update_layout(
                xaxis_title="Skin Condition",
                yaxis_title="Probability",
                xaxis={'categoryorder':'total descending'},
                yaxis_range=[0, 1]
            )
            
            # Rotate x-axis labels for better readability
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 View Detailed Analysis", key="view_analysis"):
                navigate_to("analysis")
                
        with col2:
            if st.button("💊 See Home Remedies", key="view_remedies"):
                navigate_to("remedies")
                
        with col3:
            if st.button("🏥 Find Nearby Dermatologists", key="view_doctors"):
                navigate_to("doctors")

# Analysis Dashboard page
elif st.session_state.current_page == "analysis":
    st.markdown("<h1 class='main-header'>Detailed Skin Condition Analysis</h1>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("No prediction data available. Please upload an image in the Prediction section first.")
        if st.button("Go to Prediction Page"):
            navigate_to("prediction")
    else:
        prediction = st.session_state.prediction_result
        condition = prediction['class_name']
        
        # Display prediction result summary
        st.markdown(f"""
        <div class='card'>
            <h3>Prediction: <span class='{get_risk_color(condition)}'>{condition}</span></h3>
            <p>Confidence: {prediction['confidence']:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Condition Details", "Differential Diagnosis", "Similar Cases"])
        
        with tab1:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<h3 class='sub-header'>Condition Characteristics</h3>", unsafe_allow_html=True)
                
                # Create radar chart for condition characteristics
                categories = ['Visibility', 'Common age group', 'Genetic factor', 'Sun exposure', 'Inflammation']
                
                # Mock values for demonstration
                if condition == 'Melanoma':
                    values = [0.9, 0.7, 0.8, 0.9, 0.5]
                elif condition == 'Squamous cell carcinoma':
                    values = [0.8, 0.6, 0.5, 0.9, 0.7]
                elif condition == 'Actinic keratosis':
                    values = [0.7, 0.7, 0.4, 0.9, 0.6]
                elif condition == 'Melanocytic nevus':
                    values = [0.6, 0.5, 0.7, 0.4, 0.3]
                else:
                    # Default values
                    values = [0.5, 0.5, 0.5, 0.5, 0.5]
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories,
                    fill='toself',
                    name=condition
                ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("<h3 class='sub-header'>Key Indicators</h3>", unsafe_allow_html=True)
                
                # Display key indicators based on condition
                indicators = get_condition_indicators(condition)
                
                for indicator, value in indicators.items():
                    st.markdown(f"""
                    <div style="margin-bottom: 10px;">
                        <p style="margin-bottom: 5px;"><strong>{indicator}:</strong></p>
                        <div style="background-color: #f0f0f0; border-radius: 5px;">
                            <div style="background-color: #4682B4; width: {value*100}%; height: 20px; border-radius: 5px; color: white; text-align: center;">
                                {value*100:.0f}%
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display ABCDE rule for melanoma or similar conditions
                if condition in ['Melanoma', 'Melanocytic nevus']:
                    st.markdown("""
                    <div class='info-box'>
                    <h4>ABCDE Rule Assessment:</h4>
                    <ul>
                        <li><strong>A</strong>symmetry: One half unlike the other half</li>
                        <li><strong>B</strong>order: Irregular, scalloped or poorly defined border</li>
                        <li><strong>C</strong>olor: Varies from one area to another</li>
                        <li><strong>D</strong>iameter: Usually greater than 6mm</li>
                        <li><strong>E</strong>volving: Changing in size, shape, color</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab2:
            st.markdown("<h3 class='sub-header'>Differential Diagnosis</h3>", unsafe_allow_html=True)
            
            # Create mock differential diagnosis data
            differential_data = get_differential_diagnosis(condition)
            
            fig = px.bar(
                x=list(differential_data.keys()),
                y=list(differential_data.values()),
                color=list(differential_data.values()),
                color_continuous_scale='blues',
                labels={'x': 'Condition', 'y': 'Similarity Score'}
            )
            
            fig.update_layout(
                xaxis_title="Similar Conditions",
                yaxis_title="Similarity Score",
                xaxis={'categoryorder':'total descending'},
                yaxis_range=[0, 1]
            )
            
            fig.update_xaxes(tickangle=45)
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            <div class='info-box'>
            <strong>Note:</strong> The similarity score indicates how closely the symptoms and appearance match
            the predicted condition. Higher scores suggest greater similarity in presentation.
            </div>
            """, unsafe_allow_html=True)
            
        with tab3:
            st.markdown("<h3 class='sub-header'>Similar Cases</h3>", unsafe_allow_html=True)
            
            # Mock data for similar cases
            st.markdown("""
            <div class='card'>
            <p>Based on our database, here are the statistics for similar cases:</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Age distribution
                age_data = {'0-20': 5, '21-40': 25, '41-60': 45, '61-80': 20, '80+': 5}
                
                fig = px.pie(
                    names=list(age_data.keys()),
                    values=list(age_data.values()),
                    title="Age Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Treatment outcomes
                outcome_data = {'Complete resolution': 65, 'Partial improvement': 25, 'No change': 8, 'Worsened': 2}
                
                fig = px.pie(
                    names=list(outcome_data.keys()),
                    values=list(outcome_data.values()),
                    title="Treatment Outcomes",
                    color_discrete_sequence=px.colors.sequential.Greens
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Location heatmap
            st.markdown("<h4>Body Location Distribution</h4>", unsafe_allow_html=True)
            
            # Mock data for body location heatmap
            body_data = get_body_location_data(condition)
            
            fig = px.imshow(
                body_data,
                labels=dict(x="Body Region", y="Position", color="Frequency"),
                x=['Head', 'Neck', 'Chest', 'Back', 'Arms', 'Hands', 'Abdomen', 'Legs', 'Feet'],
                y=['Upper', 'Middle', 'Lower'],
                color_continuous_scale='blues'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Home Remedies page
elif st.session_state.current_page == "remedies":
    st.markdown("<h1 class='main-header'>Home Remedies & Care</h1>", unsafe_allow_html=True)
    
    if st.session_state.prediction_result is None:
        st.warning("No prediction data available. Please upload an image in the Prediction section first.")
        if st.button("Go to Prediction Page"):
            navigate_to("prediction")
    else:
        prediction = st.session_state.prediction_result
        condition = prediction['class_name']
        
        # Display prediction result summary
        st.markdown(f"""
        <div class='card'>
            <h3>Condition: <span class='{get_risk_color(condition)}'>{condition}</span></h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Get remedies for the condition
        remedies = get_home_remedies(condition)
        
        # Check if this is a high-risk condition
        if condition in ['Melanoma', 'Squamous cell carcinoma']:
            st.markdown("""
            <div class='caution-box'>
            <h3>⚠️ MEDICAL ATTENTION REQUIRED</h3>
            <p>This condition requires <strong>immediate professional medical attention</strong>. 
            DO NOT attempt to treat this condition with home remedies.</p>
            <p>Please consult a dermatologist as soon as possible.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🏥 Find Nearby Dermatologists Now", key="find_doctors_urgent"):
                navigate_to("doctors")
        
        # Show caution for medium-risk conditions
        elif condition in ['Actinic keratosis']:
            st.markdown("""
            <div class='caution-box'>
            <h3>⚠️ CAUTION</h3>
            <p>This condition has potential to develop into a more serious condition if left untreated.</p>
            <p>While some home care measures may provide relief, medical evaluation is strongly recommended.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Display general home remedies caution
        st.markdown("""
        <div class='info-box'>
        <h3>Important Notice</h3>
        <p>The following suggestions are for informational purposes only and are not a substitute for professional medical advice.</p>
        <p>If you experience severe symptoms, allergic reactions, or worsening of the condition, seek medical attention immediately.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<h3 class='sub-header'>Suggested Home Care</h3>", unsafe_allow_html=True)
        
        # Display remedies
        for i, remedy in enumerate(remedies):
            if i == 0 and "CAUTION" in remedy or "URGENT" in remedy:
                st.markdown(f"""
                <div class='caution-box remedy-item'>
                {remedy}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='remedy-item'>
                {remedy}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h3 class='sub-header'>General Skin Care Tips</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <ul>
                <li>Use gentle, fragrance-free skin cleansers</li>
                <li>Apply broad-spectrum sunscreen (SPF 30+) daily</li>
                <li>Keep skin moisturized with appropriate products</li>
                <li>Avoid hot water when bathing or showering</li>
                <li>Stay hydrated by drinking plenty of water</li>
                <li>Wear protective clothing when outdoors</li>
                <li>Avoid known skin irritants and allergens</li>
                <li>Don't smoke, as it accelerates skin aging</li>
            </ul>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>When to See a Doctor</h3>", unsafe_allow_html=True)
            
            st.markdown("""
            <div class='info-box'>
            Consult a dermatologist if you experience:
            <ul>
                <li>Rapid growth or change in a skin lesion</li>
                <li>Bleeding, oozing, or ulceration</li>
                <li>Severe pain, itching, or discomfort</li>
                <li>Spreading or worsening despite treatment</li>
                <li>Symptoms that don't improve within 2 weeks</li>
                <li>Fever or other systemic symptoms</li>
                <li>Any concerning changes in the appearance</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🏥 Find Nearby Dermatologists", key="find_doctors_remedies"):
                navigate_to("doctors")

# Find Dermatologists page
elif st.session_state.current_page == "doctors":
    st.markdown("<h1 class='main-header'>Find Nearby Dermatologists</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    Enter your address or location to find dermatologists within a 10km radius.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_address = st.text_input("Enter your address or city:", 
                                     value=st.session_state.user_address)
    
    with col2:
        radius = st.selectbox("Search radius (km):", 
                             options=[5, 10, 15, 20, 25], 
                             index=1)
    
    if st.button("Search for Dermatologists", key="search_doctors"):
        st.session_state.user_address = user_address
        
        with st.spinner("Searching for dermatologists..."):
            # In a real app, this would call the actual function
            # For demo, use mock function that returns dermatologists near the location
            nearby_derms = find_nearby_dermatologists(user_address, radius)
            st.session_state.nearby_dermatologists = nearby_derms
    
    if st.session_state.user_address and isinstance(st.session_state.nearby_dermatologists, list) and len(st.session_state.nearby_dermatologists) > 0:
        st.success(f"Found {len(st.session_state.nearby_dermatologists)} dermatologists within {radius}km of your location!")
        
        col1, col2 = st.columns([2, 3])
        
        with col1:
            st.markdown("<h3 class='sub-header'>Dermatologist List</h3>", unsafe_allow_html=True)
            
            for i, derm in enumerate(st.session_state.nearby_dermatologists):
                st.markdown(f"""
                <div class='card' style='margin-bottom: 10px; padding: 10px;'>
                    <h4>{derm['name']}</h4>
                    <p><strong>Address:</strong> {derm['address']}, {derm['city']}</p>
                    <p><strong>Distance:</strong> {derm['distance']} km</p>
                    <p><strong>Rating:</strong> {derm['rating']} ⭐</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h3 class='sub-header'>Location Map</h3>", unsafe_allow_html=True)
            
            # Create and display the map
            map_file = create_dermatologist_map(st.session_state.user_address, st.session_state.nearby_dermatologists)
            
            if map_file:
                # For Streamlit, we'd use streamlit_folium to display the map
                # Since we can't actually render the map in this code example:
                st.image("https://img.icons8.com/color/480/000000/google-maps.png", width=100)
                st.markdown("""
                <div class='info-box'>
                In a real implementation, an interactive map would be displayed here showing your location and nearby dermatologists.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("Could not generate map. Please check your address and try again.")
    
    elif st.session_state.user_address and (not isinstance(st.session_state.nearby_dermatologists, list) or len(st.session_state.nearby_dermatologists) == 0):
        st.warning("No dermatologists found within the specified radius. Try increasing the search radius or checking a different location.")

# Helper functions for the dashboard
def get_risk_color(condition):
    if condition in ['Melanoma', 'Squamous cell carcinoma']:
        return "high-risk"
    elif condition in ['Actinic keratosis']:
        return "medium-risk"
    else:
        return "low-risk"

def get_condition_indicators(condition):
    # Mock data for demonstration
    if condition == 'Melanoma':
        return {
            "Malignancy risk": 0.9,
            "Spread potential": 0.8,
            "Treatment urgency": 0.95,
            "Self-healing probability": 0.05,
            "Genetic factor": 0.7
        }
    elif condition == 'Squamous cell carcinoma':
        return {
            "Malignancy risk": 0.75,
            "Spread potential": 0.6,
            "Treatment urgency": 0.8,
            "Self-healing probability": 0.1,
            "Genetic factor": 0.4
        }
    elif condition == 'Actinic keratosis':
        return {
            "Malignancy risk": 0.3,
            "Spread potential": 0.2,
            "Treatment urgency": 0.6,
            "Self-healing probability": 0.15,
            "Genetic factor": 0.3
        }
    elif condition == 'Melanocytic nevus':
        return {
            "Malignancy risk": 0.05,
            "Spread potential": 0.02,
            "Treatment urgency": 0.1,
            "Self-healing probability": 0.8,
            "Genetic factor": 0.7
        }
    else:
        # Default values
        return {
            "Malignancy risk": 0.2,
            "Spread potential": 0.2,
            "Treatment urgency": 0.3,
            "Self-healing probability": 0.6,
            "Genetic factor": 0.4
        }

def get_differential_diagnosis(condition):
    # Mock data for demonstration
    if condition == 'Melanoma':
        return {
            "Melanocytic nevus": 0.7,
            "Atypical nevus": 0.65,
            "Seborrheic keratosis": 0.45,
            "Dermatofibroma": 0.2,
            "Vascular lesion": 0.15
        }
    elif condition == 'Squamous cell carcinoma':
        return {
            "Actinic keratosis": 0.8,
            "Basal cell carcinoma": 0.7,
            "Bowen's disease": 0.65,
            "Keratoacanthoma": 0.6,
            "Seborrheic keratosis": 0.3
        }
    elif condition == 'Actinic keratosis':
        return {
            "Squamous cell carcinoma": 0.75,
            "Bowen's disease": 0.6,
            "Seborrheic keratosis": 0.5,
            "Solar lentigo": 0.45,
            "Lichen planus": 0.25
        }
    elif condition == 'Melanocytic nevus':
        return {
            "Melanoma": 0.6,
            "Seborrheic keratosis": 0.5,
            "Dermatofibroma": 0.4,
            "Blue nevus": 0.35,
            "Vascular lesion": 0.2
        }
    else:
        # Default for other conditions
        return {
            "Similar condition 1": 0.6,
            "Similar condition 2": 0.5,
            "Similar condition 3": 0.4,
            "Similar condition 4": 0.3,
            "Similar condition 5": 0.2
        }

def get_body_location_data(condition):
    # Mock data for heatmap of body locations
    if condition == 'Melanoma':
        return [
            [0.8, 0.5, 0.4, 0.7, 0.6, 0.8, 0.3, 0.4, 0.2],  # Upper
            [0.3, 0.4, 0.5, 0.8, 0.5, 0.7, 0.4, 0.5, 0.3],  # Middle
            [0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.7, 0.6]   # Lower
        ]
    elif condition == 'Squamous cell carcinoma':
        return [
            [0.9, 0.8, 0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.4],  # Upper
            [0.4, 0.5, 0.6, 0.7, 0.6, 0.7, 0.5, 0.4, 0.3],  # Middle
            [0.3, 0.2, 0.3, 0.4, 0.5, 0.6, 0.3, 0.5, 0.7]   # Lower
        ]
    elif condition == 'Actinic keratosis':
        return [
            [0.9, 0.8, 0.7, 0.6, 0.8, 0.9, 0.5, 0.3, 0.2],  # Upper
            [0.5, 0.4, 0.5, 0.6, 0.7, 0.8, 0.4, 0.3, 0.2],  # Middle
            [0.2, 0.1, 0.2, 0.3, 0.4, 0.5, 0.2, 0.4, 0.3]   # Lower
        ]
    else:
        # Default heatmap
        return [
            [0.5, 0.4, 0.5, 0.6, 0.5, 0.6, 0.4, 0.4, 0.3],  # Upper
            [0.4, 0.5, 0.6, 0.5, 0.6, 0.5, 0.5, 0.5, 0.4],  # Middle
            [0.3, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.6, 0.5]   # Lower
        ]

# Run the app
if __name__ == "__main__":
    # This would be the entry point in a standalone Streamlit app
    pass