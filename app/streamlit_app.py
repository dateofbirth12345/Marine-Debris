from __future__ import annotations

import os
import json
import random
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import folium
from streamlit_folium import st_folium
import pandas as pd

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.inference import predict_on_images


st.set_page_config(
    page_title="CleanSea Vision",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ocean-themed custom styling
st.markdown(
    """
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {padding-top: 0rem;}
    .stApp {background: linear-gradient(135deg, #0F4C75 0%, #3282B8 25%, #0F4C75 50%, #1B4F72 75%, #2E86AB 100%);}
    
    
    
    /* Hero Section */
    .hero {
        background: linear-gradient(135deg, #0F4C75 0%, #3282B8 25%, #2E86AB 50%, #1B4F72 75%, #0F4C75 100%);
        color: beige;
        padding: 6rem 2rem;
        text-align: center;
        border-radius: 20px;
        margin: 2rem 0;
        position: relative;
        overflow: hidden;
        min-height: 60vh;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .hero::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="wave" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse"><path d="M0,50 Q25,25 50,50 T100,50 L100,100 L0,100 Z" fill="rgba(255,255,255,0.1)"/></pattern></defs><rect width="100" height="100" fill="url(%23wave)"/></svg>');
        opacity: 0.3;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 800px;
    }
    
    .hero h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 5.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.03em;
        line-height: 0.95;
        text-transform: uppercase;
    }
    
    .hero p {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        margin-bottom: 3rem;
        opacity: 0.95;
        font-weight: 400;
        line-height: 1.6;
        letter-spacing: 0.01em;
    }
    
    .cta-button {
        font-family: 'Space Grotesk', sans-serif;
        background: linear-gradient(45deg, #F4A261, #E76F51);
        color: white;
        padding: 1.2rem 3rem;
        border: none;
        border-radius: 50px;
        font-size: 1.2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        box-shadow: 0 6px 20px rgba(244, 162, 97, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(244, 162, 97, 0.5);
    }
    
    /* Cards */
    .card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.15);
    }
    
    .card-header {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .card-icon {
        width: 48px;
        height: 48px;
        background: linear-gradient(45deg, #0F4C75, #3282B8);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
    }
    
    .card-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0F4C75;
        margin: 0;
    }
    
    /* Stats */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0F4C75;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #2C5530;
        font-weight: 500;
    }
    
    /* Table Styling */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #0F4C75 0%, #1B4F72 50%, #2E86AB 100%);
        color: white;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem -1rem -1rem -1rem;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .nav-menu {display: none;}
        .hero h1 {font-size: 2.5rem;}
        .hero p {font-size: 1rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Navigation Bar
st.markdown(
    """
    <nav class="navbar">
        <div class="nav-container">
            <a href="#" class="nav-brand">
                🌊 CLEANSEA VISION
            </a>
            <ul class="nav-menu">
                <li class="nav-item">
                    <a href="#" class="nav-link">🏠 Home</a>
                </li>
                <li class="nav-item">
                    <a href="#" class="nav-link">💙 DONATE</a>
                    <div class="dropdown">
                        <a href="#donate" class="dropdown-item">💳 Make Donation</a>
                        <a href="#donate" class="dropdown-item">🏦 Payment Methods</a>
                        <a href="#donate" class="dropdown-item">📧 Receipt & Confirmation</a>
                    </div>
                </li>
                <li class="nav-item">
                    <a href="#" class="nav-link">📸 Help Cleanup</a>
                    <div class="dropdown">
                        <a href="#upload" class="dropdown-item">📷 Upload Photos</a>
                        <a href="#upload" class="dropdown-item">🗺️ View Map</a>
                        <a href="#upload" class="dropdown-item">⚙️ Detection Settings</a>
                    </div>
                </li>
                <li class="nav-item">
                    <a href="#" class="nav-link">📊 Updates</a>
                </li>
                <li class="nav-item">
                    <a href="#" class="nav-link">🌍 Polluted Waterbodies</a>
                </li>
            </ul>
        </div>
    </nav>
    """,
    unsafe_allow_html=True,
)

# Hero Section
st.markdown(
    """
    <div class="hero">
        <div class="hero-content">
            <h1>CLEANSEA VISION</h1>
            <p>Protecting our oceans through technology and community action. Join thousands of citizens worldwide in detecting and reporting marine debris to create a cleaner, healthier planet.</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Center DONATE button that navigates to donation tab
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("DONATE", type="primary", use_container_width=True, key="hero_donate"):
        st.session_state.active_tab = "💙 Donate"


# Stats Section
st.markdown(
    """
    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-number">2,847</div>
            <div class="stat-label">Debris Reports</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">156</div>
            <div class="stat-label">Cleanup Events</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">89%</div>
            <div class="stat-label">Detection Accuracy</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">47</div>
            <div class="stat-label">Countries</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

# Main Content Tabs
# Initialize session state for tab navigation
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "📸 Help Cleanup"

# Create tabs with default selection
tab_names = ["📸 Help Cleanup", "💙 Donate", "📊 Updates", "🌍 Polluted Waterbodies", "🤝 Join Team"]
default_index = tab_names.index(st.session_state.active_tab) if st.session_state.active_tab in tab_names else 0

tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_names)

with tab1:
    st.markdown("### Upload Photos to Detect Marine Debris")
    st.markdown("Help us identify and locate marine debris by uploading photos from drones, boats, or shorelines.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload images (JPG/PNG)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            help="Upload clear photos of suspected marine debris"
        )
        
        if uploaded_files:
            # Save uploads temporarily
            tmp_dir = "runs/ui_uploads"
            os.makedirs(tmp_dir, exist_ok=True)
            image_paths: List[str] = []
            for f in uploaded_files:
                img = Image.open(f).convert("RGB")
                out_path = os.path.join(tmp_dir, f.name)
                img.save(out_path)
                image_paths.append(out_path)

            with st.spinner("Detecting debris..."):
                results = predict_on_images(image_paths)

            for item in results:
                st.markdown(f"**{os.path.basename(item['image_path'])}**")
                if len(item["boxes"]) == 0:
                    st.info("No debris detected.")
                    st.image(item["image_path"], use_column_width=True)
                else:
                    vis = draw_boxes(item["image_path"], item["boxes"])
                    st.image(vis, use_column_width=True)
                    # Metrics row
                    st.markdown(
                        " ".join(
                            [
                                f"<span class='metric-pill'>{b['label']} · {b['conf']:.2f}</span>"
                                for b in item["boxes"][:6]
                            ]
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown("<div class='caption'>EXIF GPS: " + (f"{item['gps']}" if item['gps'] else "Not available") + "</div>", unsafe_allow_html=True)
                st.divider()
    
    with col2:
        st.markdown("### Detection Settings")
        conf = st.slider("Detection confidence", 0.05, 0.90, 0.25, 0.01)
        iou = st.slider("NMS IoU threshold", 0.10, 0.90, 0.45, 0.01)
        st.markdown("""
        - Lower confidence → more detections
        - Higher confidence → fewer, stronger detections
        """)
        
        if uploaded_files:
            gps_points = [it["gps"] for it in results if it["gps"] is not None]
            st.markdown("### Map – Reported Locations")
            if len(gps_points) > 0:
                first_lat, first_lon = gps_points[0]  # type: ignore[index]
                m = folium.Map(location=[first_lat, first_lon], tiles="OpenStreetMap", zoom_start=11)
                for it in results:
                    if it["gps"] is not None:
                        lat, lon = it["gps"]
                        folium.Marker(
                            location=[lat, lon],
                            popup=os.path.basename(it["image_path"]) or "Photo",
                            icon=folium.Icon(color="blue", icon="info-sign"),
                        ).add_to(m)
                st_folium(m, width=None, height=400)
            else:
                st.info("No GPS found in uploaded photos. Enable camera location services to appear on the map.")

with tab2:
    st.markdown("### 💙 Support Our Mission")
    st.markdown("Your donation helps us maintain detection technology, organize cleanup events, and protect marine ecosystems worldwide.")
    
    # Donation form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 💳 Make a Donation")
        
        # Donation amount selection
        donation_amount = st.selectbox(
            "Select Amount",
            ["₹100", "₹500", "₹1000", "₹2500", "₹5000", "Custom Amount"],
            help="Choose a donation amount or select custom"
        )
        
        custom_amount = None
        if donation_amount == "Custom Amount":
            custom_amount = st.number_input("Enter Amount (₹)", min_value=10, max_value=100000, value=1000)
            amount = custom_amount
        else:
            amount = int(donation_amount.replace("₹", ""))
        
        # Donor information
        st.markdown("#### Donor Information")
        donor_name = st.text_input("Full Name", placeholder="Enter your full name")
        donor_email = st.text_input("Email", placeholder="your.email@example.com")
        donor_phone = st.text_input("Phone (Optional)", placeholder="+91 98765 43210")
        
        # Anonymous donation option
        anonymous = st.checkbox("Donate anonymously")

    with col2:
        st.markdown("#### 🏦 Payment Methods")
        
        # Payment method selection
        payment_method = st.radio(
            "Choose Payment Method",
            ["UPI", "Credit/Debit Card", "Net Banking", "Wallet"],
            help="Select your preferred payment method"
        )
        
        if payment_method == "UPI":
            st.markdown("##### UPI Payment")
            st.markdown("**Scan QR Code or Use UPI ID:**")
            st.markdown("""
            <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px; margin: 1rem 0;">
                <p><strong>UPI ID:</strong> cleanseavision@paytm</p>
                <p><strong>QR Code:</strong></p>
                <div style="width: 200px; height: 200px; background: #e9ecef; margin: 0 auto; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                    <span style="color: #6c757d;">QR Code</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        elif payment_method == "Credit/Debit Card":
            st.markdown("##### Card Payment")
            card_number = st.text_input("Card Number", placeholder="1234 5678 9012 3456")
            col_exp, col_cvv = st.columns(2)
            with col_exp:
                expiry = st.text_input("Expiry (MM/YY)", placeholder="12/25")
            with col_cvv:
                cvv = st.text_input("CVV", placeholder="123", max_chars=4)
            card_name = st.text_input("Name on Card", placeholder="John Doe")
            
        elif payment_method == "Net Banking":
            st.markdown("##### Net Banking")
            bank = st.selectbox("Select Bank", [
                "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank", 
                "Kotak Mahindra Bank", "Punjab National Bank", "Bank of Baroda"
            ])
            
        elif payment_method == "Wallet":
            st.markdown("##### Digital Wallet")
            wallet = st.selectbox("Select Wallet", [
                "Paytm", "PhonePe", "Google Pay", "Amazon Pay", "Mobikwik"
            ])

    # Donation summary and submit
    if st.button("💙 Donate Now", type="primary", use_container_width=True):
        if not donor_name or not donor_email:
            st.error("Please fill in your name and email address.")
        else:
            # Simulate payment processing
            with st.spinner("Processing your donation..."):
                import time
                time.sleep(2)
            
            # Show success message
            st.success(f"🎉 Thank you for your donation of ₹{amount}! Your contribution helps protect our oceans.")
            
            # Store donation record (in a real app, this would go to a database)
            donation_record = {
                "amount": amount,
                "donor_name": "Anonymous" if anonymous else donor_name,
                "email": donor_email,
                "payment_method": payment_method,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.markdown("### 📧 Receipt")
            st.json(donation_record)
            
            # Send email confirmation (simulated)
            st.info("📧 A receipt has been sent to your email address.")

with tab3:
    st.markdown("### Latest Cleanup Updates")
    
    # Generate sample updates
    updates_data = [
        {
            "date": "2024-01-15",
            "location": "Pacific Garbage Patch",
            "description": "Removed 2.3 tons of plastic debris from the Great Pacific Garbage Patch",
            "participants": 45,
            "impact": "High"
        },
        {
            "date": "2024-01-12",
            "location": "Mediterranean Sea",
            "description": "Community cleanup along the French Riviera collected 1.8 tons of waste",
            "participants": 120,
            "impact": "Medium"
        },
        {
            "date": "2024-01-10",
            "location": "Caribbean Sea",
            "description": "Underwater cleanup removed 500kg of fishing gear and plastic",
            "participants": 25,
            "impact": "High"
        },
        {
            "date": "2024-01-08",
            "location": "North Sea",
            "description": "Beach cleanup in Netherlands collected 800kg of microplastics",
            "participants": 80,
            "impact": "Medium"
        }
    ]
    
    for update in updates_data:
        with st.expander(f"🗓️ {update['date']} - {update['location']}"):
            st.markdown(f"**Description:** {update['description']}")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Participants", update['participants'])
            with col2:
                st.metric("Impact Level", update['impact'])
            with col3:
                st.metric("Status", "✅ Completed")

with tab3:
    st.markdown("### Top 1000 Polluted Waterbodies")
    st.markdown("Data sourced from global monitoring programs and citizen reports.")
    
    # Generate sample polluted waterbodies data
    waterbodies_data = {
        "Rank": list(range(1, 21)),
        "Waterbody": [
            "Great Pacific Garbage Patch", "Mediterranean Sea", "Caribbean Sea",
            "North Sea", "Baltic Sea", "Gulf of Mexico", "South China Sea",
            "Indian Ocean", "Atlantic Ocean (North)", "Pacific Ocean (North)",
            "Yangtze River", "Ganges River", "Nile River", "Amazon River",
            "Mississippi River", "Thames River", "Seine River", "Rhine River",
            "Danube River", "Volga River"
        ],
        "Pollution Level": ["Critical"] * 5 + ["High"] * 8 + ["Medium"] * 7,
        "Debris Count": [random.randint(50000, 200000) for _ in range(20)],
        "Last Updated": [datetime.now() - timedelta(days=random.randint(1, 30)) for _ in range(20)]
    }
    
    df = pd.DataFrame(waterbodies_data)
    df['Last Updated'] = df['Last Updated'].dt.strftime('%Y-%m-%d')
    
    # Add filters
    col1, col2 = st.columns(2)
    with col1:
        pollution_filter = st.selectbox("Filter by Pollution Level", ["All", "Critical", "High", "Medium"])
    with col2:
        search_term = st.text_input("Search waterbodies", placeholder="Enter name...")
    
    # Apply filters
    if pollution_filter != "All":
        df = df[df['Pollution Level'] == pollution_filter]
    
    if search_term:
        df = df[df['Waterbody'].str.contains(search_term, case=False)]
    
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", width="small"),
            "Waterbody": st.column_config.TextColumn("Waterbody", width="large"),
            "Pollution Level": st.column_config.TextColumn("Pollution Level", width="medium"),
            "Debris Count": st.column_config.NumberColumn("Debris Count", format="%d"),
            "Last Updated": st.column_config.TextColumn("Last Updated", width="medium")
        }
    )

with tab5:
    st.markdown("### Join Our Cleanup Team")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon">👥</div>
                <h3 class="card-title">Volunteer Opportunities</h3>
            </div>
            <p>Join local cleanup events and make a direct impact in your community.</p>
            <ul>
                <li>Beach cleanups</li>
                <li>Underwater diving cleanups</li>
                <li>River and lake cleanups</li>
                <li>Educational workshops</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <div class="card-icon">📢</div>
                <h3 class="card-title">Spread the Word</h3>
            </div>
            <p>Help us reach more people and create awareness about marine pollution.</p>
            <ul>
                <li>Share on social media</li>
                <li>Organize community events</li>
                <li>Educate schools and groups</li>
                <li>Partner with local organizations</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Contact Information")
    st.markdown("""
    - **Email:** info@cleanseavision.org
    - **Phone:** +1 (555) 123-4567
    - **Social Media:** @CleanSeaVision
    """)


def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    palette = [
        (39, 76, 119),  # deep blue
        (51, 92, 103),  # teal
        (237, 125, 49), # orange
        (120, 94, 240), # purple
        (35, 166, 213), # sky
        (23, 190, 187), # cyan
    ]
    return palette[cls_id % len(palette)]


def draw_boxes(image_path: str, boxes: List[dict]) -> Image.Image:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
        color = _color_for_class(int(b["cls"]))
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        label = f"{b['label']} {b['conf']:.2f}"
        tw, th = draw.textlength(label, font=font), 16
        # label background
        draw.rectangle([(x1, max(0, y1 - th - 4)), (x1 + int(tw) + 8, y1)], fill=(0, 0, 0, 160))
        draw.text((x1 + 4, y1 - th - 2), label, fill=(255, 255, 255), font=font)
    return img


# Footer
st.markdown(
    """
    <div class="footer">
        <h3>🌊 CleanSea Vision</h3>
        <p>Protecting our oceans through technology and community action</p>
        <div style="margin-top: 2rem;">
            <p><strong>Contact Us:</strong> info@cleanseavision.org | +1 (555) 123-4567</p>
            <p><strong>Follow Us:</strong> @CleanSeaVision on all social platforms</p>
            <p style="margin-top: 1rem; opacity: 0.8;">
                © 2024 CleanSea Vision. All rights reserved. | 
                <a href="#" style="color: #93c5fd;">Privacy Policy</a> | 
                <a href="#" style="color: #93c5fd;">Terms of Service</a>
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



