# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import json
with open("country_config.json", "r") as f:
    country_map = json.load(f)
from streamlit_option_menu import option_menu   
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import base64
import random
import io
import pandas as pd
import qrcode
from concurrent.futures import ThreadPoolExecutor
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from fpdf import FPDF
import nltk
from nltk.util import ngrams
from collections import Counter
import plotly.express as px
import random
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from streamlit.column_config import TextColumn
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
from fpdf import FPDF
from sklearn.cluster import KMeans
import warnings
import requests
import datetime
from languages import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import plotly.express as px
from google_play_scraper import reviews, Sort
import os
from google_play_scraper import Sort, reviews_all
from app_store_scraper import AppStore
import seaborn as sns
import pycountry
from wordcloud import WordCloud, STOPWORDS
from langdetect import detect
import matplotlib.pyplot as plt
from nltk.util import ngrams
from PIL import Image
from collections import Counter
from googletrans import Translator
from languages import *
warnings.filterwarnings('ignore')
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from streamlit_plotly_events import plotly_events
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
import pandas as pd
import warnings
import matplotlib.pyplot as plt

# from transformers import pipeline
from nltk.corpus import stopwords
from streamlit_autorefresh import st_autorefresh 

import nltk
# AUTO-FIX NLTK - Runs once
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    nltk.download('punkt', quiet=True)
# st.set_page_config(page_title="Apotheke Customer Sentiment Analyzer!!!", page_icon=":sparkles:",layout="wide")
st.set_page_config(page_title="Apotheke Customer Sentiment Analyzer!!!", page_icon="apotheke.png", layout="wide")


# st.title(" :sparkles: Sentiment Anaylzer")
st.markdown('<style>div.block-container{padding-top:0rem;text-align: center}</style>',unsafe_allow_html=True)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
#load the style sheet
local_css("custom_style.css")



# Load and encode image

# Load and encode image safely
dir_path = os.path.dirname(__file__)
filename = os.path.join(dir_path, "apotheke.png")
try:
    with open(filename, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    
    # Clean sidebar image display
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 5px;">
            <img src="data:image/png;base64,{encoded_image}" 
                 style="width: 50%; max-width: 50px; border-radius: 1px;">
        </div>
        """, 
        unsafe_allow_html=True
    )
except FileNotFoundError:
    st.sidebar.warning("apotheke.png missing")
except Exception as e:
    st.sidebar.error(f"Image read error: {e}")


# st.markdown("""
# <style>
# /* Consistent metric card sizing and alignment */
# .metric-card {
#     padding: 1rem;
#     border-radius: 10px;
#     margin: 0.5rem;
#     height: 160px;              
#     display: flex;
#     flex-direction: column;
#     justify-content: center;    
#     align-items: center;       
#     text-align: center;
#     box-sizing: border-box;
# }


# /* Clamp long heading to prevent overflow */
# .metric-card h2 {
#     margin: 6px 0 2px 0;
#     font-size: 1.6rem;
#     line-height: 1.2;
#     max-width: 100%;
#     overflow: hidden;
#     text-overflow: ellipsis;
#     white-space: nowrap;      
# }

 
# .metric-card h3 {
#     margin: 0;
#     font-size: 1.0rem;
#     line-height: 1.2;
# }



# .metric-card p {
#     margin: 2px 0 0 0;
#     font-size: 0.95rem;
# }

# /* Optional: ensure equal column height behavior in Streamlit columns */

# .block-container .row-widget stHorizontalBlock > div {

#     display: flex;

# }

# </style>

# """, unsafe_allow_html=True)

 

 

 



 

 

# st.markdown("""

# <style>

# /* Sidebar header compact */

# section[data-testid="stSidebar"] h1,

# section[data-testid="stSidebar"] h2,

# section[data-testid="stSidebar"] h3 {

#     text-align: left !important;

#     padding-left: 4px !important;

#     margin-top: 0px !important;

#     margin-bottom: 4px !important;

# }

 

# /* Remove extra spacing between buttons */

# section[data-testid="stSidebar"] div.stButton {

#     margin: 0px !important;       /* Remove margin around button container */

#     padding: 0px !important;      /* Remove padding inside container */

#     line-height: 1 !important;    /* Compact line height */

# }

 

# /* Make buttons inline-block to reduce gaps */

# section[data-testid="stSidebar"] div.stButton > button {

#     display: block !important;    /* Ensure full width */

#     background-color: transparent !important;

#     color: #0066cc !important;

#     border: none !important;

#     font-size: 2px !important;

#     text-align: left !important;

#     justify-content: flex-start !important;

#     padding: .5px 4px !important;  /* Minimal padding */

#     width: 100% !important;

#     margin: 0px !important;       /* Remove extra margin */

# }

 

# /* Hover and active states */

# section[data-testid="stSidebar"] div.stButton > button:hover {

#     color: #ff6600 !important;

#     text-decoration: underline !important;

# }

# section[data-testid="stSidebar"] div.stButton.active > button {

#     font-weight: bold !important;

#     color: #ff6600 !important;

# }

# </style>

# """, unsafe_allow_html=True)

 

 

 

# st.markdown("""

#     <style>

#     .nav-link:last-child {

#         background: linear-gradient(90deg, #ffdd00, #ffa500) !important;

#         color: #000 !important;

#         font-weight: bold !important;

#     }

#     </style>

# """, unsafe_allow_html=True)

 

 

 

# st.markdown("""

#     <style>

#     div[data-baseweb="select"] > div {

#         text-align: left !important;

#     }

#     </style>

#     """, unsafe_allow_html=True)

 

 

st.markdown("# **Customer Sentiment Analyzer**", unsafe_allow_html=True)
st.markdown("---")  # Divider 

 

def show_timed_warning_generic(message, duration=3):

    col1, col2, col3 = st.columns([1, 3, 1])

    with col2:

        warning_placeholder = st.empty()

        warning_placeholder.markdown(

            f"""

            <div class="timed-warning-box">

                <strong>{message}</strong>

            </div>

            """,

            unsafe_allow_html=True

        )

        time.sleep(duration)

        warning_placeholder.empty()

 

# app_url = 

 

# # Generate the QR code

# qr = qrcode.QRCode(

#     version=1,

#     box_size=10,

#     border=5

# )

# qr.add_data(app_url)

# qr.make(fit=True)

 

# # Create an image

# img = qr.make_image(fill="black", back_color="white")

 

# # Save to file

# img.save("app_qr_code.png")

 

#count = st_autorefresh(interval=3600000, key="fizzbuzzcounter")
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)


sia = SentimentIntensityAnalyzer()

stop_words = set(stopwords.words('english'))

# st.cache_data.clear()

# print(st.__version__)

translator = Translator()

sid = SentimentIntensityAnalyzer()

 

@st.cache_data

def iso2_to_name(code):

    try:

        return pycountry.countries.get(alpha_2=code).name

    except:

        return None

 

@st.cache_data

def name_to_iso3(name):

    try:

        return pycountry.countries.get(name=name).alpha_3

    except:

        return None

 

# --- Android Review Fetch ---

@st.cache_data(ttl=86400, show_spinner=False)

def load_android_data(app_id, country, app_name):

    reviews = reviews_all(

        app_id,

        sleep_milliseconds=0,

        lang='en',

        country=country,

        sort=Sort.NEWEST,

    )

    df = pd.DataFrame(np.array(reviews), columns=['review'])

    df = df.join(pd.DataFrame(df.pop('review').tolist()))

    columns_to_drop = ['reviewId', 'thumbsUpCount', 'reviewCreatedVersion', 'repliedAt', 'userImage']

    df = df.drop(columns=[c for c in columns_to_drop if c in df.columns], errors="ignore")

    df['AppName'] = app_name

    df['Country'] = country.lower()

 

    df.rename(columns={

        'content': 'review',

        'userName': 'UserName',

        'score': 'rating',

        'at': 'TimeStamp',

        'replyContent': 'WU_Response'

    }, inplace=True)

    return df

 

 

def show_progress():

    progress = st.progress(0)

    status = st.empty()

    for i in range(100):

        time.sleep(0.005)

        progress.progress(i + 1)

        status.text(f"Loading Data... {i + 1}%")

    progress.empty()

    status.empty()

 

def fetch_all_android(app_details):

    frames = []

    with ThreadPoolExecutor(max_workers=10) as executor:

        futures = [executor.submit(load_android_data, app_id, country, app_name)

                   for app_id, country, app_name in app_details]

        for future in futures:

            try:

                result = future.result()

                frames.append(result)

            except Exception as e:

                print(f"Android fetch failed: {e}")

                frames.append(pd.DataFrame())

    if frames:

        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()

 

 

 

def fetch_ios_reviews(app_id, country_code):

    """

    Fetch iOS app reviews using Apple's official RSS feed.

    """

    url = f"https://itunes.apple.com/{country_code}/rss/customerreviews/id={app_id}/sortBy=mostRecent/json"
    reviews = []

    try:

        resp = requests.get(url, timeout=10)

        resp.raise_for_status()

        feed = resp.json().get('feed', {})

        entries = feed.get('entry', [])

 

        if len(entries) <= 1:

            print(f"No reviews found for App ID: {app_id} in {country_code}")

            return pd.DataFrame()

 

        for item in entries[1:]:

            reviews.append({

                "rating": item.get('im:rating', {}).get('label'),

                "date": item.get('updated', {}).get('label'),

                "review": item.get('content', {}).get('label'),

                "UserName": item.get('author', {}).get('name', {}).get('label'),

                "AppName": "iOS",

                "Platform": "iOS",

                "Country": country_code,

                "AppID": app_id

            })

 

    except Exception as e:

        print(f"Error fetching reviews for {app_id}-{country_code}: {e}")

 

    return pd.DataFrame(reviews)

 

 

 

def fetch_all_ios(app_country_list):

    frames = []

    with ThreadPoolExecutor(max_workers=5) as executor:

        futures = [executor.submit(fetch_ios_reviews, app_id, cc)

                   for app_id, cc in app_country_list]

        for future in futures:

            df = future.result()

            if not df.empty:

                df["date"] = pd.to_datetime(df["date"], errors="coerce")

                df["TimeStamp"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")             

                frames.append(df)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

 

 

# --- All App Details, as before ---

app_details = [

 

    ('shop.shop_apotheke.com.shopapotheke', 'us', 'Android')      

]

 

app_country_list = [

    ("1104967519", "us"),


]



@st.cache_data(ttl=86400, show_spinner=False)

def get_all_reviews(app_details, app_country_list):

    finaldfandroid = fetch_all_android(app_details)

    finaldfios = fetch_all_ios(app_country_list)  # ✅ Removed pages argument

    if not finaldfandroid.empty and not finaldfios.empty:

        finaldf = pd.concat([finaldfandroid, finaldfios], ignore_index=True)

    elif not finaldfandroid.empty:

        finaldf = finaldfandroid

    elif not finaldfios.empty:

        finaldf = finaldfios

    else:

        finaldf = pd.DataFrame()

    return finaldf

 

 

with st.spinner("Fetching Android & iOS reviews..."):

    finaldf = get_all_reviews(app_details, app_country_list)

 

 

finaldf.columns = finaldf.columns.str.strip("'")

finaldf.columns = [c.replace(' ', '_') for c in finaldf.columns]

 

# Convert TimeStamp to datetime for filtering

finaldf["TimeStamp"] = pd.to_datetime(finaldf["TimeStamp"])

finaldf["DateTimeStamp"] = finaldf["TimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

 

today = datetime.date.today()

default_end = today

default_start = today.replace(day=1)

 

# Initialize session state

if "start_date" not in st.session_state:

    st.session_state.start_date = default_start

if "end_date" not in st.session_state:

    st.session_state.end_date = default_end

if "show_custom_dates" not in st.session_state:

    st.session_state.show_custom_dates = False

 

# --- Add buttons for quick date range selection ---

col_btn1, col_btn2, col_btn3, col_btn4, col_btn5, col_btn6 = st.columns(6)

 

with col_btn1:

    if st.button("3 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=90)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn2:

    if st.button("6 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=180)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn3:

    if st.button("9 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=270)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn4:

    if st.button("12 Months"):

        st.session_state.start_date = today - datetime.timedelta(days=365)

        st.session_state.end_date = today

        st.session_state.show_custom_dates = False

 

with col_btn5:

    if st.button("Custom"):

        st.session_state.show_custom_dates = True

 

with col_btn6:

    if st.button("Reset"):

        st.session_state.start_date = default_start

        st.session_state.end_date = default_end

        st.session_state.show_custom_dates = False

 

# --- Date Range Selection Header ---

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(

    "<h5 style='margin-bottom: 10px;'>Date Range Selection</h5>",

    unsafe_allow_html=True

)

 

# --- Show Date Inputs only if "Custom" is selected ---

if st.session_state.show_custom_dates:

    col1, col2 = st.columns((2))

    with col1:

        date1 = st.date_input("**Start Date**", value=st.session_state.start_date)

        st.session_state.start_date = date1

    with col2:

        date2 = st.date_input("**End Date**", value=st.session_state.end_date)

        st.session_state.end_date = date2

else:

    date1 = st.session_state.start_date

    date2 = st.session_state.end_date

 

# --- Convert to datetime.datetime for comparison ---

date1_dt = datetime.datetime.combine(date1, datetime.datetime.min.time())

date2_dt = datetime.datetime.combine(date2, datetime.datetime.max.time())

 

# --- Validation: Start date must be before end date ---

if date1_dt > date2_dt:

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.error("⚠️ Start Date must be before End Date.")

    st.stop()

    df = pd.DataFrame()

else:

    # --- Filter the dataframe based on selected date range ---

    try:

        filtered_df = finaldf[(finaldf["TimeStamp"] >= date1_dt) & (finaldf["TimeStamp"] <= date2_dt)]

        df = filtered_df.copy()

    except KeyError:

        df = pd.DataFrame()

 

    # --- Show selected date range summary ---

    if not st.session_state.show_custom_dates:

        st.success(f"📅 Showing Data from **{date1.strftime('%d-%b-%Y')}** to **{date2.strftime('%d-%b-%Y')}**")

 

 

# Create list of full country names for dropdown

country_names = [country_map.get(code, code) for code in df["Country"].unique()]


# Sidebar country selection

selected_country_names = st.sidebar.multiselect(

    "**Country Selection**",

    options=sorted(country_names),

    placeholder="Select Country/s"

)


# Convert selected country names back to codes

selected_country_codes = [code for code, name in country_map.items() if name in selected_country_names]


# Filter by country

if not selected_country_codes:

    df1 = df.copy()

else:

    df1 = df[df["Country"].isin(selected_country_codes)]

 

# country = st.sidebar.multiselect("**Select the Countries**", df["Country"].unique(),placeholder="")

# if not country:

#     df1 = df.copy()

# else:

#     df1 = df[df["Country"].isin(country)]

 

region = st.sidebar.multiselect("**Select the App Type**", df["AppName"].unique(),placeholder="Select the App Type")

if not region:

    df2 = df1.copy()

else:

    df2 = df1[df1["AppName"].isin(region)]

 

if not selected_country_codes and not region :

    filtered_df = df

else:

    filtered_df=df2

 

 

if 'rating' in filtered_df.columns:

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce') # convert rating to numeric (int/float)

    rating = st.sidebar.slider("**Filter by Ratings Range**", 1, 5, (1, 5))

    if rating:

        filtered_df = filtered_df[(filtered_df['rating'] >= rating[0]) & (filtered_df['rating'] <= rating[1])]

 

 

filtered_df["Platform"] = filtered_df["AppName"].apply(

    lambda x: "Android" if "android" in str(x).lower() else "iOS"

)

filtered_df["CountryName"] = filtered_df["Country"].apply(iso2_to_name)

 

country_platform_avg = (

    filtered_df.groupby(["CountryName", "Platform"])["rating"]

               .mean()

               .reset_index()

)

 

 

# Function to apply row-wise styling

def highlight_rating(row):

    try:

        rating = int(row['rating']) # Convert to int

    except:

        return [''] * len(row) # No styling if conversion fails

 

    if rating >= 4:

        return ['background-color: lightgreen'] * len(row)

    elif rating == 3:

        return ['background-color: yellow'] * len(row)

    else:

        return ['background-color: salmon'] * len(row)

 

 

def get_random_score_by_rating(rating):

    rating_ranges = {

        5: (0.81, 1.0),

        4: (0.61, 0.80),

        3: (0.41, 0.60),

        2: (0.21, 0.40),

        1: (0.0, 0.20)

    }

    return round(random.uniform(*rating_ranges.get(rating, (0.0, 0.20))), 2)

 

 

 

def get_score_by_rating(rating):

    rating_ranges = {

        5: (0.81, 1.0),

        4: (0.61, 0.80),

        3: (0.41, 0.60),

        2: (0.21, 0.40),

        1: (0.0, 0.20)

    }

    low, high = rating_ranges.get(rating, (0.0, 0.20))

    return round((low + high) / 2, 2)  # midpoint instead of random

 

 

def get_sentiment_score(row):

    review = str(row['review']).strip()

    rating = int(row['rating'])

 

    # If empty or single-word review → fallback to rating-based score

    if not review or len(review.split()) == 1:

        return get_score_by_rating(rating)

 

    # VADER sentiment

    sentiment = sid.polarity_scores(review)

    normalized_score = (sentiment['compound'] + 1) / 2  # 0–1 scale

 

    # Weight rating more heavily

    rating_score = get_score_by_rating(rating)

    final_score = (normalized_score * 0.3) + (rating_score * 0.7)

    return round(final_score, 2)

 

 

# Updated sentiment label logic

def get_sentiment_label(row):

    review = str(row['review']).strip()

    rating = row['rating']

 

    if review == '':

        if rating in [4, 5]:

            return 'Positive'

        elif rating == 3:

            return 'Neutral'

        elif rating in [1, 2]:

            return 'Negative'

        else:

            return 'Neutral' 

    else:

        if rating in [0, 1]:

            return 'Negative'

        elif rating == 3:

            return 'Neutral'

        elif rating in [4, 5]:

            return 'Positive'

        else:

            return 'Neutral'  # Default instead of 'Unknown'

 

def get_sentiment_emoticon(row):

    """Get emoticon based on sentiment"""

    review = str(row['review']).strip()

    rating = row['rating']

 

    if review == '':

        if rating ==5 :

            return '😍'

        elif rating == 4:

            return '😃'

        elif rating == 3:

            return '😐'

        elif rating ==2:

            return '😢'

        else:

            return '😭'

    else:

        if rating ==1:

            return '😭'

        elif rating == 2:

            return '😢'

        elif rating == 3:

            return '😐'

        elif rating ==4 :

            return '😃'

        else:

            return '😍'

 

 

 

 

 

 

def get_emoji_stars(rating):

    if pd.isnull(rating):

        return ""

    rating = int(rating)

    if rating == 5:

        return "🟩🟩🟩🟩🟩"  # Dark green squares for 5 stars

    elif rating == 4:

        return "🟩🟩🟩🟩"    # Green squares for 4 stars

    elif rating == 3:

        return "🟨🟨🟨"      # Yellow squares for 3 stars

    elif rating == 2:

        return "🟧🟧"        # Red squares for 2 stars

    elif rating == 1:

        return "🟥"          # Dark red square for 1 star

    return ""

 

 

 

def show_centered_warning(message="⚠️ No records found within the specified date range"):

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.warning(message)

 

 

def plot_bar(subplot,filtered_df):

    plt.subplot(1,2,subplot)

    axNewest=sns.barplot(y='Country',x='rating',hue='AppName',data=filtered_df, color='slateblue')

    plt.title('Ratings vs country',fontsize=70)

    # plt.xlabel('Ratings vs Country',fontsize=50)

    plt.ylabel(None)

    # plt.xticks(fontsize=40)

    # plt.yticks(fontsize=40)

    # sns.despine(left=True)

    axNewest.grid(False)

    axNewest.tick_params(bottom=True,left=False)

    return None

 

 

if filtered_df.empty:

  #st.warning("No records found within the specified date range")

   show_centered_warning()

else:

    filtered_df = filtered_df.reset_index(drop=True)

    filtered_df.index += 1  # Start index from 1

    filtered_df.index.name = "S.No."  # Give index column a name


# Apply updated sentiment score and label logic

    filtered_df['sentiment_score'] = filtered_df.apply(get_sentiment_score, axis=1)

    filtered_df['sentiment_label'] = filtered_df.apply(get_sentiment_label, axis=1)

    filtered_df['HappinessIndex'] = filtered_df.apply(get_sentiment_emoticon, axis=1)

    # Country mapping

    filtered_df["CountryName"] = filtered_df["Country"].str.lower().map(country_map).fillna(filtered_df["Country"])

    #filtered_df["CustomerRating"] = filtered_df["rating"].apply(lambda r: "★" * int(r) if pd.notnull(r) else "")

    filtered_df["CustomerRating"] = filtered_df["rating"].apply(get_emoji_stars)

    # Reorder columns

    filtered_df = filtered_df.reindex([ 'DateTimeStamp',  'review','CustomerRating', 'sentiment_score',

                                        'HappinessIndex', 'CountryName', 'AppName',

                                        'appVersion','rating','sentiment_label'], axis=1)

 

# 'UserName'

 

def format_column_label(s):

    # Split by underscores and capitalize each part

    return '_'.join(word.capitalize() for word in s.split('_'))

 

 

def format_column_label(col):

    custom_labels = {

        "TimeStamp": "Date",    

        "AppName": "App Type",

        "UserName": "User Name",

        "appVersion":"Version",

        "sentiment_score":"Sentiment Score",

        "sentiment_label":"Sentiment Label"

    }

    if col in custom_labels:

        return custom_labels[col]

    return '_'.join(word.capitalize() for word in col.split('_'))

 

 

def to_title_case_with_underscores(s):

    return '_'.join(word.capitalize() for word in s.split('_'))

 

 

column_config = {

    col: st.column_config.TextColumn(label=to_title_case_with_underscores(col))

    for col in filtered_df.columns

}

 

column_config = {

    col: st.column_config.TextColumn(label=format_column_label(col))

    for col in filtered_df.columns

}

 

 

# --- Caching grouped data computation

@st.cache_data(show_spinner=False)

def compute_grouped_data(df):

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    #df["CountryName"] = df["Country"].apply(iso2_to_name)

    df["ISO3"] = df["CountryName"].apply(name_to_iso3)

    grouped = (

        df.groupby(["CountryName", "ISO3"])

        .agg(avg_rating=("rating", "mean"), review_count=("rating", "count"))

        .reset_index()

    )

    return grouped

 

@st.cache_data(show_spinner=False)

def generate_figures(grouped):

    figures = []

    font_size = 12

 

    def rating_to_color(rating):

        if rating < 2.5:

            return "#B22222"  # Dark Red

        elif rating < 4.0:

            return "#FF8C00"  # Dark Orange

        else:

            return "#228B22"  # Forest Green

 

    for _, row in grouped.iterrows():

        fill_color = rating_to_color(row["avg_rating"])

 

        fig = go.Figure()

 

        # Country fill

        fig.add_trace(go.Choropleth(

            locations=[row["ISO3"]],

            z=[row["avg_rating"]],

            locationmode="ISO-3",

            colorscale=[[0, fill_color], [1, fill_color]],

            showscale=False,

            marker_line_color="gray",

            marker_line_width=0.5,

            hoverinfo="skip"

        ))

 

        # Annotation box in bottom center

        annotation_text = (

            f"<b>{row['CountryName']}</b><br>"

            f"⭐ Rating: {row['avg_rating']:.2f}<br>"

            f"📝 Reviews: {row['review_count']}"

        )

       

 

        fig.update_layout(

            annotations=[

                dict(

                    x=0.5,

                    y=0.01,

                    xref='paper',

                    yref='paper',

                    showarrow=False,

                    align='center',

                    text=annotation_text,

                    font=dict(size=font_size, color="black"),

                    bgcolor="white",

                    bordercolor="gray",

                    borderwidth=3,

                    opacity=0.98 

                )

            ],

            title={

                "text": f"🌐 Ratings for {row['CountryName']}",

                "x": 0.5,

                "xanchor": "center",

                "font": dict(size=18, family="Arial Black", color="black")

            },

            margin=dict(l=0, r=0, t=50, b=0),

            paper_bgcolor='white',

            plot_bgcolor='white',

            geo=dict(

                showcoastlines=True,

                coastlinecolor="LightGray",

                showland=True,

                landcolor="whitesmoke",

                showocean=True,

                oceancolor="aliceblue",

                showlakes=True,

                lakecolor="lightblue",

                showrivers=True,

                rivercolor="lightblue",

                showcountries=True,

                countrycolor="gray",

                projection_type="equirectangular",

                bgcolor='white',

                resolution=50,               

                showsubunits=True,

                subunitcolor="lightgray",

                showframe=True,

                framecolor="black",

               

                center=dict(lat=20, lon=0),

                projection_scale=1  # Zoom level 

            )

        )

        figures.append(fig)

 

    return figures

 

 

def show_world_map(filtered_df, date1, date2):

    st.markdown(

        """

        <style>

        .date-range-text {

            font-size: 14px;

            font-weight: bold;

            text-align: center;

            margin-bottom: 10px;

            z-index: 9999;

        }

        </style>

        """,

        unsafe_allow_html=True,

    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    with st.spinner("⏳ Please wait while the Bar chart is getting ready..."):

        # Average rating by country

        filtered_df["ISO3"] = filtered_df["CountryName"].apply(name_to_iso3)

        mean_ratings = filtered_df.groupby('ISO3')['rating'].mean().reset_index()

 

        figNewer = plt.figure(figsize=(15, 5))

        axar = sns.barplot(x='ISO3', y='rating', data=mean_ratings, palette='Pastel1')

        st.markdown("<br><b>Average Rating By Country</b><br>", unsafe_allow_html=True)

        axar.set(xlabel='Country', ylabel='Average Rating', title='')

        for container in axar.containers:

            axar.bar_label(container, fmt='%.2f')

        st.pyplot(figNewer)

       

        st.markdown("<br><br>", unsafe_allow_html=True)

 

       

 

        # ...existing code...

            # ---- Dumbbell chart: Android vs iOS per country ----

        if not country_platform_avg.empty:

            db = (

                country_platform_avg

                .pivot(index="CountryName", columns="Platform", values="rating")

                .reset_index()

            )

 

            # Ensure Android / iOS columns exist to avoid KeyError when one platform is missing

            for col in ["Android", "iOS"]:

                if col not in db.columns:

                    db[col] = np.nan

 

            # sort by combined mean so order is stable (handles missing cols)

            rating_cols = [c for c in ["Android", "iOS"] if c in db.columns]

            if rating_cols:

                db["mean_rating"] = db[rating_cols].mean(axis=1, skipna=True)

            else:

                db["mean_rating"] = np.nan

 

            # If there's no data for either platform, show friendly message and skip plotting

            if db[["Android", "iOS"]].isna().all(axis=None):

                st.info("No platform-specific rating data available to display the Android vs iOS chart.")

            else:

                fig_height = max(6, 0.35 * len(db))

                fig, ax = plt.subplots(figsize=(10, fig_height))

 

                # expand limits a bit beyond 1 and 5 so there is room for text

                x_min, x_max = 0.8, 5.2

                ax.set_xlim(x_min, x_max)

 

                # lines only when both platforms exist for that row

                mask_both = db["Android"].notna() & db["iOS"].notna()

                if mask_both.any():

                    ax.hlines(

                        y=db.loc[mask_both, "CountryName"],

                        xmin=db.loc[mask_both, "Android"],

                        xmax=db.loc[mask_both, "iOS"],

                        color="lightgray",

                        linewidth=2,

                        zorder=1,

                    )

 

                # Android points (only non-na)

                if db["Android"].notna().any():

                    ax.scatter(

                        db["Android"], db["CountryName"],

                        color="#ff9999", label="Android", s=50, zorder=3,

                    )

 

                # iOS points (only non-na)

                if db["iOS"].notna().any():

                    ax.scatter(

                        db["iOS"], db["CountryName"],

                        color="#99c2ff", label="iOS", s=50, zorder=3,

                    )

 

                ax.set_xlabel("Average Rating")

                ax.set_ylabel("")

                ax.set_title("Average Rating by Country – (Android vs iOS)")

 

                # padding for labels – fraction of axis range

                axis_span = x_max - x_min

                pad = 0.05 * axis_span

 

                for _, row in db.iterrows():

                    y = row["CountryName"]

                    a = row.get("Android")

                    i = row.get("iOS")

 

                    if pd.notna(a):

                        x_text = max(x_min + pad, a - pad)

                        ax.text(

                            x_text,

                            y,

                            f"{a:.2f}",

                            ha="right",

                            va="center",

                            fontsize=7,

                            color="#b30000",

                            clip_on=False,

                        )

 

                    if pd.notna(i):

                        x_text = min(x_max - pad, i + pad)

                        ax.text(

                            x_text,

                            y,

                            f"{i:.2f}",

                            ha="left",

                            va="center",

                            fontsize=7,

                            color="#004c99",

                            clip_on=False,

                        )

 

                ax.legend(loc="lower right")

 

                # extra padding around edges so border never cuts text

                plt.subplots_adjust(left=0.18, right=0.98, top=0.92, bottom=0.08)

                st.pyplot(fig)

                plt.close(fig)

 

 

 

 

def show_sunburst_chart(filtered_df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    filtered_df['rating'] = pd.to_numeric(filtered_df['rating'], errors='coerce')

    date_diff = (date2 - date1).days

 

    if date_diff > 365:

        show_timed_warning_generic("⚠️ Sunburst Chart is disabled for date ranges longer than 12 months", duration=3)

        return

 

    # Show progress bar before rendering chart

    progress = st.progress(0)

    status = st.empty()

    for i in range(100):

        time.sleep(0.005)

        progress.progress(i + 1)

        status.text(f"Loading Sunburst Chart... {i + 1}%")

    progress.empty()

    status.empty()


    

    st.write("### Sunburst Chart")

    fig = px.sunburst(

        filtered_df,

        path=['CountryName', 'AppName', 'rating', 'review'],

        values='rating',

        color='rating',

        color_continuous_scale='RdBu',

        color_continuous_midpoint=np.average(filtered_df['rating'], weights=filtered_df['rating']),

        title=""

    )

    fig.update_traces(hovertemplate="")

    fig.update_layout(width=800, height=800, coloraxis_showscale=False)

    st.plotly_chart(fig, use_container_width=True)

 

 

 

# Custom CSS for left alignment inside expanders

st.markdown("""

    <style>

    .streamlit-expanderContent p, .streamlit-expanderContent ul {

        text-align: left !important;

        margin-left: 0 !important;

        padding-left: 0 !important;

    }

    </style>

""", unsafe_allow_html=True)

 

 

def show_topic_modeling(filtered_df):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    with st.spinner("🔍 Performing Topic Modeling... Please wait"):

        # --- Separate reviews by rating ---

        positive_reviews = filtered_df[filtered_df['rating'] >= 4]['review']

        negative_reviews = filtered_df[filtered_df['rating'] <= 2]['review']

        neutral_reviews = filtered_df[filtered_df['rating'] == 3]['review']

 

        def extract_topics(reviews, n_topics=5, n_keywords=5):

            vectorizer = CountVectorizer(stop_words='english', max_df=1.0, min_df=1)

            doc_term_matrix = vectorizer.fit_transform(reviews)

            lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)

            lda.fit(doc_term_matrix)

            topics = []

            for idx, topic in enumerate(lda.components_):

                keywords = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_keywords:]]

                topics.append((f"Topic {idx+1}", keywords))

            return topics, lda.transform(doc_term_matrix)

 

        def get_representative_sentences(reviews, topic_distributions, n_sentences=1):

            sentences = []

            for topic_idx in range(topic_distributions.shape[1]):

                topic_scores = topic_distributions[:, topic_idx]

                top_indices = topic_scores.argsort()[-n_sentences:]

                topic_sentences = reviews.iloc[top_indices].tolist()

                sentences.append(topic_sentences)

            return sentences

 

        def display_topic_summary(topics, sentences, section_title):

            st.markdown(f"### {section_title}")

            for i, (topic_name, keywords) in enumerate(topics):

                keywords_str = ', '.join(keywords)

                with st.expander(keywords_str):

                    st.markdown("**Example Sentences:**")

                    for sent in sentences[i]:

                        st.markdown(f"- {sent}")

 

        # --- Run topic modeling ---

        positive_topics, positive_sentences = [], []

        negative_topics, negative_sentences = [], []

        neutral_topics, neutral_sentences = [], []

        st.markdown("<br>", unsafe_allow_html=True)

 

        if len(positive_reviews) > 0:

            positive_topics, positive_topic_distributions = extract_topics(positive_reviews)

            positive_sentences = get_representative_sentences(positive_reviews, positive_topic_distributions)

            display_topic_summary(positive_topics, positive_sentences, "Top 5 Best Aspects (Rating ≥ 4)")

            st.divider()

 

        if len(negative_reviews) > 0:

            negative_topics, negative_topic_distributions = extract_topics(negative_reviews)

            negative_sentences = get_representative_sentences(negative_reviews, negative_topic_distributions)

            display_topic_summary(negative_topics, negative_sentences, "Top 5 Issues (Rating ≤ 2)")

            st.divider()

 

        if len(neutral_reviews) > 0:

            neutral_topics, neutral_topic_distributions = extract_topics(neutral_reviews)

            neutral_sentences = get_representative_sentences(neutral_reviews, neutral_topic_distributions)

            display_topic_summary(neutral_topics, neutral_sentences, "Top 5 Neutral Topics (Rating = 3)")

            st.divider()

 

        # --- Generate PDF summary --- 

 

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font("Arial", size=12)

        pdf.set_auto_page_break(auto=True, margin=15)

 

        # --- Add Image at Top and Centered ---

        page_width = pdf.w  # Total page width

        image_width = 60    # Smaller image width in mm

        x_position = (page_width - image_width) / 2  # Center horizontally

        y_position = 10     # Top margin (adjust as needed)

 

        pdf.image(filename, x=x_position, y=y_position, w=image_width)

        pdf.ln(30)  # Add space after image before text starts (adjust based on image height)

 

        # --- Function to add topic sections ---

        def add_topic_section_to_pdf(pdf, section_title, topics, sentences):

            pdf.set_font("Arial", 'B', 14)

            pdf.cell(200, 10, txt=section_title, ln=True, align='C')  # Center section title

            pdf.set_font("Arial", size=12)

            for i, (topic_name, keywords) in enumerate(topics):

                pdf.set_font("Arial", 'B', 12)

                pdf.cell(200, 10, txt=f"{topic_name}", ln=True)

                pdf.set_font("Arial", size=12)

                pdf.multi_cell(0, 10, txt=f"Keywords: {', '.join(keywords)}")

                pdf.multi_cell(0, 10, txt="Example Sentences:")

                for sent in sentences[i]:

                    clean_sent = ''.join(char if ord(char) < 256 else ' ' for char in sent)

                    pdf.multi_cell(0, 10, txt=f"- {clean_sent}")

                pdf.ln(5)

 

        # Add sections

        add_topic_section_to_pdf(pdf, "Top 5 Best Aspects", positive_topics, positive_sentences)

        add_topic_section_to_pdf(pdf, "Top 5 Issues", negative_topics, negative_sentences)

        add_topic_section_to_pdf(pdf, "Top 5 Neutral Aspects", neutral_topics, neutral_sentences)

 

        # Output PDF

        pdf.output("topic_modeling_summary.pdf")

 

 

 

    # --- Show download button after spinner completes ---

    if positive_topics or negative_topics or neutral_topics:

        st.subheader("Topic Modeling")

        with open("topic_modeling_summary.pdf", "rb") as f:

            st.download_button(

                label="📄 Download Topic Modeling Summary as PDF",

                data=f,

                file_name="topic_modeling_summary.pdf",

                mime="application/pdf"

            )

    else:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

 

 

 

def remove_emojis(text):

    # This function removes emojis from the input text

    return text.encode('ascii', 'ignore').decode('ascii')

 
@st.cache_data(ttl=300)
def fast_sentiment(df):
    """Safe sentiment - handles existing columns"""
    df = df.copy()
    
    # Compute score
    df['sentiment_score'] = df['review'].apply(
        lambda x: sia.polarity_scores(str(x))['compound']
    )
    
    # FIXED: Safe label logic - check if column exists + type
    if 'sentiment_label' not in df.columns or df['sentiment_label'].dtype == 'object':
        df['sentiment_label'] = df['sentiment_score'].apply(
            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')
        )
    
    return df


@st.cache_data(ttl=300)
def generate_wordcloud_fast(text, mask=None):
    """Cached wordcloud generation"""
    stopwords_set = set(STOPWORDS)
    
    def redcare_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(["#FF6B35", "#000000", "#FFA726"])
    
    wc = WordCloud(
        stopwords=stopwords_set,
        max_words=25,      # Balanced speed/quality
        width=500,         # Smaller = 3x faster
        height=350,
        background_color='white',
        color_func=redcare_color_func,
        mask=mask,
        contour_width=1,   # Faster render
        collocations=False,
        random_state=42    # Consistent
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.close(fig)
    return fig

def show_word_cloud(filtered_df):
    if filtered_df.empty:
        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)
        return

    # FAST MASK
    mask = None
    try:
        mask = np.array(Image.open("Image/apotheke.png"))
    except:
        pass  # Silent fail

    st.subheader("Word Cloud")
    st.markdown("<h4 style='text-align: center; font-weight: bold;'>Select Sentiment</h4>", unsafe_allow_html=True)
    
    # INLINE SAFE SENTIMENT (no cache needed)
    if 'sentiment_score' not in filtered_df.columns:
        filtered_df = filtered_df.copy()
        filtered_df['sentiment_score'] = filtered_df['review'].apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
        filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(
            lambda x: 'Positive' if pd.notna(x) and x > 0.2 
                     else ('Negative' if pd.notna(x) and x < -0.2 else 'Neutral')
        )

    sentiment_option = st.selectbox("", 
        sorted(filtered_df['sentiment_label'].value_counts().index.tolist()))
    
    # FAST FILTER
    df_sent = filtered_df[filtered_df['sentiment_label'] == sentiment_option]
    if df_sent.empty:
        st.warning("No reviews for selected sentiment.")
        return

    # ULTRA-FAST TEXT (truncate)
    text = " ".join(df_sent['review'].dropna().astype(str).str[:120].tolist())

    # INLINE WORDCLOUD (no cache crash)
    stopwords_set = set(STOPWORDS)
    
    def redcare_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return np.random.choice(["#FF6B35", "#000000", "#FFA726"])

    try:
        wc = WordCloud(
            stopwords=stopwords_set,
            max_words=30,      # Fast
            width=480, height=320,  # Compact
            background_color='white',
            color_func=redcare_color_func,
            mask=mask,
            contour_width=1,
            collocations=False,
            random_state=42
        ).generate(text)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        plt.tight_layout(pad=0)
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"WordCloud render failed: {str(e)[:100]}")
        st.info("Try shorter date range or fewer reviews.")

 


 

def show_treemap_chart(filtered_df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    date_diff = (date2 - date1).days

    if date_diff > 365:

        show_timed_warning_generic("⚠️ TreeMap Chart is disabled for date ranges longer than 12 months", duration=3)

        return

 

    # Remove artificial progress simulation for large datasets

    st.write("### TreeMap Chart")

 

    # Limit the data size for rendering to avoid server overload (adjust threshold as needed)

    MAX_ROWS = 2000

    if len(filtered_df) > MAX_ROWS:

        filtered_df = filtered_df.sample(n=MAX_ROWS, random_state=42)

 

    filtered_df = filtered_df.fillna('end_of_hierarchy')

 

    # Use Streamlit spinner only for chart generation

    with st.spinner("🌳 Generating TreeMap Chart..."):

        try:

            fig3 = px.treemap(

                filtered_df,

                path=["CountryName", "AppName", "rating", "review"],

                hover_data=["rating"],

                color="review"

            )

            fig3.update_traces(hovertemplate='')

            st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:

            st.warning(f"Error generating TreeMap: {e}")

 

 

 

def show_visual_charts(filtered_df, df, date1, date2):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=3)

        return

 

    with st.spinner("⏳ Loading Charts, please wait..."):

        st.markdown("<br><br>", unsafe_allow_html=True)

        st.markdown(

            "<div style='text-align: center; font-size: 18px;'><b>Consolidated Sentiment across Countries</b></div>",

            unsafe_allow_html=True

        )

        st.markdown("<br><br>", unsafe_allow_html=True)

 

        # Pie chart for sentiment distribution

        fig = px.pie(

            filtered_df,

            names='sentiment_label',

            color='sentiment_label',

            color_discrete_map={

                'Positive': 'green',

                'Negative': 'red',

                'Neutral': 'yellow'

            },

            hole=0.2

        )

        fig.update_traces(textposition='inside', textinfo='percent+label')

 

        # Disable legend interactivity

        fig.update_layout(

            legend_itemclick=False,

            legend_itemdoubleclick=False

        )

 

        st.plotly_chart(fig, use_container_width=True)

 

        st.markdown("<br><b>Consolidated Ratings across Countries</b><br>", unsafe_allow_html=True)

 

        # Ensure all ratings from 1 to 5 are present

        rating_counts = filtered_df['rating'].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)

 

        # Convert to DataFrame for plotting

        plot_df = pd.DataFrame({'rating': rating_counts.index, 'count': rating_counts.values})

 

        # Plot

        fig = plt.figure(figsize=(15, 5))

        ax = sns.barplot(x='rating', y='count', data=plot_df, palette='turbo')

 

        # Add bar labels

        for container in ax.containers:

            ax.bar_label(container)

 

        # Hide Y-axis label if you want

        ax.set_ylabel("") 

        ax.set_yticks([]) 

 

        st.pyplot(fig)

 

        # Funnel chart for issue keywords

        issue_keywords = {
            "Delivery Delay": ["delivery late", "late delivery", "never delivered", "not arrived", "waiting days", "delayed", "not on time"],
            "Payment Failed": ["payment failed", "paypal", "pay pal", "card declined", "payment error", "cant pay", "transaction failed"],
            "App Crashes": ["crash", "crashing", "crashes", "freeze", "frozen", "stuck", "keeps closing", "buggy", "error"],
            "Login OTP": ["login", "log in", "otp", "verification code", "cant login", "password", "authentication"],
            "Slow Loading": ["slow", "lag", "loading", "takes long", "unresponsive", "hangs"],
            "Pricing Scam": ["expensive", "price higher", "scam", "rip off", "cheaper browser", "fee", "charge"],
            "Ads Popups": ["ad", "ads", "popup", "advertisement", "offer spam"],
            "Customer Service": ["support", "customer service", "helpdesk", "no response"]
        }


 

       

 

    funnel_labels = ['All Reviews', 'Filtered Negatives'] + list(issue_keywords.keys()) + ['Other Issues']

 

    # Filter negative reviews with rating 1, 2, 3

    filtered_negatives = filtered_df[

        (filtered_df['sentiment_label'].str.lower() == 'negative') &

        (filtered_df['rating'].isin([1, 2, 3]))

    ]

 

    # Initial counts

    all_reviews = len(filtered_df)

    counts = [all_reviews, len(filtered_negatives)]

 

    # Track indices for each stage

    stage_indices = {

        'All Reviews': df.index.tolist(),

        'Filtered Negatives': filtered_negatives.index.tolist(),

    }

 

    covered_indices = set()

 

    # Loop through issues and count matches

    for issue, keywords in issue_keywords.items():

        if isinstance(keywords, str):

            keywords = [keywords]

 

        mask = filtered_negatives['review'].fillna("").str.lower().apply(

            lambda text: any(kw.lower() in text for kw in keywords)

        )

 

        indices = filtered_negatives[mask].index.tolist()

        counts.append(len(indices))

        stage_indices[issue] = indices

        covered_indices.update(indices)

 

    # Other issues

    other_issues_indices = filtered_negatives.drop(index=list(covered_indices)).index.tolist()

    counts.append(len(other_issues_indices))

    stage_indices['Other Issues'] = other_issues_indices

 

    # Colors for funnel stages

    custom_colors = [

        "#1f77b4", "#d62728", "#ff7f0e", "#2ca02c",

        "#9467bd", "#8c564b", "#bcbd22", "#7f7f7f"

    ]

 

    # Build funnel chart

    fig = go.Figure(go.Funnel(

        y=funnel_labels,

        x=counts,

        textinfo="value+percent initial",

        marker=dict(color=custom_colors)

    ))

    fig.update_layout(

        title=dict(text="Issue Funnel", x=0.5, xanchor="center"),

        margin=dict(l=160, r=40, t=60, b=20)

    )

 

    st.markdown("<br><b>Click Keywords below to see detailed Customer Reviews<b><br>", unsafe_allow_html=True)

    selected = plotly_events(fig, click_event=True, hover_event=False)

 

    # Show reviews for selected stage

    if selected:

        idx = selected[0]['pointIndex']

        label = funnel_labels[idx]

        st.markdown(f"**Reviews for Stage: {label}**")

        indices = stage_indices.get(label, [])

        if indices:

            with st.spinner("⏳ Loading Customer Reviews..."):

                st.dataframe(filtered_df.loc[indices].reset_index(drop=True))

        else:

            st.info('No reviews for this stage.')

    else:

        st.markdown("", unsafe_allow_html=True)

 

 

def show_keyword_analysis(filtered_df, stop_words):

    if filtered_df.empty:

        show_timed_warning_generic("⚠️ No records found within the specified date range", duration=4)

        return

 

    st.subheader("Keyword and N-gram Analysis")

 

    with st.spinner("🔍 Performing Keyword & N-gram Analysis... Please wait"):

        # Sentiment Analysis

        filtered_df['sentiment_score'] = filtered_df['review'].apply(lambda x: sia.polarity_scores(str(x))['compound'])

        filtered_df['sentiment_label'] = filtered_df['sentiment_score'].apply(

            lambda x: 'Positive' if x > 0.2 else ('Negative' if x < -0.2 else 'Neutral')

        )

 

        # Filters

        selected_sentiment = st.selectbox("Filter by Sentiment", filtered_df['sentiment_label'].unique().tolist() + ['All'])

        selected_apptype = st.selectbox("Filter by App Type", filtered_df['AppName'].dropna().unique().tolist() + ['All'])

        ngram_count = st.slider("Number of Top N-grams to Display", 5, 30, 10)

 

        df_filtered = filtered_df.copy()

        if selected_sentiment != 'All':

            df_filtered = df_filtered[df_filtered['sentiment_label'] == selected_sentiment]

        if selected_apptype != 'All':

            df_filtered = df_filtered[df_filtered['AppName'] == selected_apptype]

 

        # Tokenization

        all_text = " ".join(df_filtered['review'].astype(str).tolist()).lower()

        tokens = [word for word in nltk.word_tokenize(all_text) if word.isalpha() and word not in stop_words]

 

        # N-gram Frequencies

        unigram_freq = Counter(tokens)

        bigram_freq = Counter(ngrams(tokens, 2))

        trigram_freq = Counter(ngrams(tokens, 3))

 

        # DataFrames

        top_unigrams = pd.DataFrame(unigram_freq.most_common(ngram_count), columns=['Unigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

        top_bigrams = pd.DataFrame(bigram_freq.most_common(ngram_count), columns=['Bigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

        top_trigrams = pd.DataFrame(trigram_freq.most_common(ngram_count), columns=['Trigram', 'Count']).sort_values(by='Count').reset_index(drop=True)

 

        # Convert text columns

        top_unigrams['Unigram'] = top_unigrams['Unigram'].astype(str)

        top_bigrams['Bigram'] = top_bigrams['Bigram'].apply(lambda x: ' '.join(x))

        top_trigrams['Trigram'] = top_trigrams['Trigram'].apply(lambda x: ' '.join(x))

 

        # Ensure Count is numeric

        top_unigrams['Count'] = pd.to_numeric(top_unigrams['Count'])

        top_bigrams['Count'] = pd.to_numeric(top_bigrams['Count'])

        top_trigrams['Count'] = pd.to_numeric(top_trigrams['Count'])

 

   

    # Charts using go.Figure with .tolist()

    col1, col2, col3 = st.columns(3)

 

    with col1:

        st.write("### Top Unigrams")

        st.dataframe(top_unigrams)

        fig_uni = go.Figure(go.Bar(x=top_unigrams['Unigram'].tolist(), y=top_unigrams['Count'].tolist(), marker_color='blue'))

        fig_uni.update_layout(title='Top Unigrams', xaxis_title='Unigram', yaxis_title='Count')

        fig_uni.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_uni, use_container_width=True)

 

    with col2:

        st.write("### Top Bigrams")

        st.dataframe(top_bigrams)

        fig_bi = go.Figure(go.Bar(x=top_bigrams['Bigram'].tolist(), y=top_bigrams['Count'].tolist(), marker_color='blue'))

        fig_bi.update_layout(title='Top Bigrams', xaxis_title='Bigram', yaxis_title='Count')

        fig_bi.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_bi, use_container_width=True)

 

    with col3:

        st.write("### Top Trigrams")

        st.dataframe(top_trigrams)

        fig_tri = go.Figure(go.Bar(x=top_trigrams['Trigram'].tolist(), y=top_trigrams['Count'].tolist(), marker_color='blue'))

        fig_tri.update_layout(title='Top Trigrams', xaxis_title='Trigram', yaxis_title='Count')

        fig_tri.update_xaxes(tickangle=-90)

        st.plotly_chart(fig_tri, use_container_width=True)

 

    st.markdown("<br><br>", unsafe_allow_html=True)

 

 

 

def show_translation_widget(languages):

    from googletrans import Translator

 

    source_text = st.text_area("**Enter Text to translate:**", height=100)

 

    default_language = 'English'

    default_index = languages.index(default_language) if default_language in languages else 0

 

    target_language = st.selectbox("**Select target language:**", languages, index=default_index)

    st.markdown("<br>", unsafe_allow_html=True)

 

    if st.button('Translate'):

        translator = Translator()

        try:

            out = translator.translate(source_text, dest=target_language)

            st.write(out.text)

        except Exception as e:

            st.error(f"Translation failed: {e}")

 

@st.cache_data

def process_complaints(filtered_df):

    """Extract complaints directly from filtered_df"""

    if filtered_df.empty or 'review' not in filtered_df.columns:

         return pd.DataFrame(), {}

 

    df_neg = filtered_df[

        (filtered_df.get('rating', pd.Series(5)) <= 2) |

        (filtered_df.get('sentiment_score', pd.Series(0.5)) < 0.5) |

        (filtered_df['sentiment_label'] == 'Negative')

    ].copy()

 

    if df_neg.empty:

        return df_neg, {}

 

    # Ensure a proper datetime column

    if 'DateTimeStamp' in df_neg.columns:

        df_neg['DateTimeStamp'] = pd.to_datetime(df_neg['DateTimeStamp'], errors='coerce')

    else:

        df_neg['DateTimeStamp'] = pd.to_datetime(df_neg.get('TimeStamp', pd.Series()), errors='coerce')

 

    # Monthly period as timestamp for reliable plotting/grouping

    df_neg['month_year'] = df_neg['DateTimeStamp'].dt.to_period('M').dt.to_timestamp()

 

    df_neg['App_Version'] = df_neg.get('appVersion', 'Unknown').astype(str).str.extract(r'(\d+\.\d+)').fillna('Unknown')

    df_neg['CountryName'] = df_neg.get('CountryName', df_neg.get('Country', 'Unknown')).fillna('Unknown')

 

    # Issue keyword detection (case insensitive)

    issue_keywords = {

        'App Crash': ['crash', 'crashing', 'frozen', 'freezing', 'lag', 'lagging', 'slow', 'buffering', 'stuck'],

        'Ads/Popup': ['ad', 'advertisement', 'popup', 'offer', 'ads'],

        'Transaction Error': ['error', 'failed', 'cancel', 'timeout', 'declined', 'c2002'],

        'Fees/Expensive': ['fee', 'charge', 'expensive', 'cost', 'rip off', 'overcharge'],

        'Verification': ['verify', 'verification', 'id', 'selfie', 'face', 'document'],

        'Refund': ['refund', 'return', 'money back', 'stuck money'],

        'Customer Service': ['support', 'customer service', 'help', 'call center']

    }

 

    for issue, keywords in issue_keywords.items():

        df_neg[issue] = df_neg['review'].str.lower().str.contains(

            '|'.join(keywords), case=False, na=False

        ).astype(int)

 

    # Churn signals

    churn_keywords = {

        'moneygram': ['moneygram','money gram'],

        'xoom': ['xoom'],

        'paypal': ['paypal', 'pay pal'],

        'remitly': ['remitly']

    }

    churn_data = {}

    for service, keywords in churn_keywords.items():

        churn_data[service] = df_neg['review'].str.lower().str.contains('|'.join(keywords), case=False, na=False).sum()

 

    return df_neg, {'issues': issue_keywords, 'churn': churn_data}

 

 

def get_priority_score(df_neg, issue_name):

    """Calculate priority: frequency × severity (safe with zero counts)"""

    freq = int(df_neg.get(issue_name, pd.Series(dtype=int)).sum())

    if freq == 0:

        return 0.0

    severity_reviews = df_neg[df_neg[issue_name] == 1]

    # handle missing sentiment_score carefully

    mean_sent = severity_reviews.get('sentiment_score', pd.Series([0.3])).mean()

    if pd.isna(mean_sent):

        mean_sent = 0.3

    severity = 1 - mean_sent

    return float(freq * severity)

 

@st.cache_data(show_spinner=False)

def compute_top_issues(df_neg, issues_dict, total_complaints):

    """Return sorted top issues dataframe (cached to avoid repeated recompute & flicker)."""

    if df_neg is None or df_neg.empty or not issues_dict or total_complaints == 0:

        return pd.DataFrame(columns=['Issue','Count','% of Total','Priority Score'])

    issues = list(issues_dict.keys())

    counts = [int(df_neg[issue].sum()) for issue in issues]

    priorities = [get_priority_score(df_neg, issue) for issue in issues]

    pct = [f"{(c/total_complaints*100):.1f}%" for c in counts]

    df = pd.DataFrame({

        'Issue': issues,

        'Count': counts,

        '% of Total': pct,

        'Priority Score': priorities

    })

    return df.sort_values('Priority Score', ascending=False).reset_index(drop=True)

 

 

def show_customer_insights(all_df):

    """

    Analyze full historical dataset vs last 1 year and produce actionable charts/tables.

    Call with finaldf (all-time reviews).

    """

    if all_df.empty:

        show_timed_warning_generic("⚠️ No records available for Customer Insights", duration=4)

        return

 

    # Defensive copy & types

    all_df = all_df.copy()

    all_df["TimeStamp"] = pd.to_datetime(all_df.get("TimeStamp", pd.Series()), errors="coerce")

    all_df["rating"] = pd.to_numeric(all_df.get("rating", pd.Series(dtype=float)), errors="coerce")

 

    # Ensure a CountryName column exists (prefer country_map lookup from ISO2 -> full name)

    if "CountryName" not in all_df.columns:

        if "Country" in all_df.columns:

            # map lower-case ISO2 codes to friendly names when possible, else keep original

            try:

                all_df["CountryName"] = all_df["Country"].astype(str).str.lower().map(country_map).fillna(all_df["Country"].astype(str))

            except Exception:

                all_df["CountryName"] = all_df["Country"].astype(str)

        else:

            all_df["CountryName"] = "Unknown"

   

    cutoff = pd.Timestamp.today() - pd.Timedelta(days=365)

    last_year = all_df[all_df["TimeStamp"] >= cutoff]

    historical = all_df[all_df["TimeStamp"] < cutoff]

 

    st.markdown("## Customer Insights — Historical vs Last 12 months")

    st.markdown(f"**Data range:** {all_df['TimeStamp'].min().date() if not all_df['TimeStamp'].isna().all() else 'N/A'} → {all_df['TimeStamp'].max().date()}")

    st.markdown("---")

 

    # Summary KPIs

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric("Total reviews (all-time)", f"{len(all_df):,}")

    with col2:

        st.metric("Reviews last 12 months", f"{len(last_year):,}", delta=f"{len(last_year) - len(historical):,}")

    with col3:

        avg_all = all_df['rating'].dropna().astype(float).mean()

        st.metric("Avg rating (all-time)", f"{avg_all:.2f}" if not np.isnan(avg_all) else "N/A")

    with col4:

        avg_last = last_year['rating'].dropna().astype(float).mean()

        avg_hist = historical['rating'].dropna().astype(float).mean()

        delta = (avg_last - avg_hist) if not (np.isnan(avg_last) or np.isnan(avg_hist)) else np.nan

        st.metric("Avg rating (last 12m)", f"{avg_last:.2f}" if not np.isnan(avg_last) else "N/A", delta=f"{delta:+.2f}" if not np.isnan(delta) else "")

 

 

    st.markdown("---")

 

    # Country-level improvement/regression: compare avg rating per country (last year vs earlier)

    try:

        agg_last = last_year.groupby("CountryName")["rating"].mean().rename("avg_last").reset_index()

        agg_hist = historical.groupby("CountryName")["rating"].mean().rename("avg_hist").reset_index()

        cmp = pd.merge(agg_hist, agg_last, on="CountryName", how="outer")

        cmp["avg_hist"] = cmp["avg_hist"].fillna(np.nan)

        cmp["avg_last"] = cmp["avg_last"].fillna(np.nan)

        cmp["delta"] = cmp["avg_last"] - cmp["avg_hist"]

        cmp["ISO3"] = cmp["CountryName"].apply(name_to_iso3)

 

        # show top regressions and improvements

        top_regress = cmp.sort_values("delta").head(10).loc[:, ["CountryName", "avg_hist", "avg_last", "delta"]]

        top_improve = cmp.sort_values("delta", ascending=False).head(10).loc[:, ["CountryName", "avg_hist", "avg_last", "delta"]]

 

        c1, c2 = st.columns(2)

        with c1:

            st.subheader("Top regressions (avg rating ↓)")

            st.dataframe(top_regress.style.format({"avg_hist":"{:.2f}", "avg_last":"{:.2f}", "delta":"{:+.2f}"}))

        with c2:

            st.subheader("Top improvements (avg rating ↑)")

            st.dataframe(top_improve.style.format({"avg_hist":"{:.2f}", "avg_last":"{:.2f}", "delta":"{:+.2f}"}))

 

    except Exception as e:

        st.warning(f"Country comparison failed: {e}")

 

    st.markdown("---")

 

    # Issue detection: simple keyword buckets

    buckets = {

        "Crashes / freezes": ["crash", "crashes", "crashing", "freeze", "freezes", "stuck", "shut down", "keeps closing", "keeps stopping"],

        "OTP / verification": ["otp", "verification", "code", "sms", "message not received", "can't receive code", "verification failed"],

        "Login / auth": ["log in", "login", "sign in", "can't sign", "can't log", "password", "c2016", "c9999", "authentication"],

        "Performance / slow": ["slow", "lag", "loading", "takes long", "time out", "unresponsive"],

        "Pricing / fees / exchange": ["fee", "fees", "exchange rate", "rate", "price", "expensive"],

        "UI / UX": ["ui", "user friendly", "confusing", "not intuitive", "navigation"]

    }

 

    # Count bucket mentions (all-time and last-year)

    issue_counts = []

    for name, keys in buckets.items():

        pattern = "(" + "|".join([kw.replace(" ", r"\s+") for kw in keys]) + ")"

        mask_all = all_df['review'].fillna("").str.lower().str.contains(pattern, regex=True)

        mask_last = last_year['review'].fillna("").str.lower().str.contains(pattern, regex=True)

        issue_counts.append({"issue": name, "all_time": int(mask_all.sum()), "last_year": int(mask_last.sum())})

 

    issues_df = pd.DataFrame(issue_counts).sort_values("all_time", ascending=False)

 

    st.dataframe(issues_df.set_index("issue"))

 

    #st.markdown("---")

 

    # AppVersion impact on rating (last year vs historical)

    try:

        v_last = last_year.groupby("appVersion")["rating"].agg(["mean","count"]).reset_index().rename(columns={"mean":"avg_last","count":"count_last"})

        v_hist = historical.groupby("appVersion")["rating"].agg(["mean","count"]).reset_index().rename(columns={"mean":"avg_hist","count":"count_hist"})

        vcmp = pd.merge(v_hist, v_last, on="appVersion", how="outer").fillna(0)

        vcmp["delta"] = vcmp["avg_last"] - vcmp["avg_hist"]

        vcmp = vcmp.sort_values("count_last", ascending=False).head(20)

        #st.subheader("Top App Versions (impact on rating)")

        fig_ver = px.bar(vcmp, x="appVersion", y=["avg_hist","avg_last"], barmode="group", title="Avg rating by appVersion (hist vs last 12m)")

        #st.plotly_chart(fig_ver, use_container_width=True)

        #st.dataframe(vcmp.loc[:, ["appVersion","avg_hist","avg_last","count_hist","count_last","delta"]].style.format({"avg_hist":"{:.2f}","avg_last":"{:.2f}","delta":"{:+.2f}"}))

    except Exception as e:

        st.warning(f"AppVersion analysis failed: {e}")

 

    st.markdown("---")

 

    # Actionable recommendations box (derived from above signals)

    st.subheader("Actionable recommendations")

    recs = []

    # pricing

    if issues_df.loc[issues_df['issue'].str.contains("Pricing", case=False), "last_year"].sum() > 0:

        recs.append("- Review pricing / exchange rate policy for countries where pricing mentions are high.")

    # crashes

    if issues_df.loc[issues_df['issue'].str.contains("Crashes", case=False), "last_year"].sum() > 0:

        recs.append("- Prioritise stability fixes (crash/freeze) for top affected appVersion/countries.")

    # otp

    if issues_df.loc[issues_df['issue'].str.contains("OTP", case=False), "last_year"].sum() > 0:

        recs.append("- Investigate OTP delivery for mobile operators in countries with repeated OTP failures.")

    # version regressions

    if not vcmp.empty and (vcmp["delta"] < -0.1).any():

        recs.append("- Roll back or hotfix app versions with significant rating drops (delta <= -0.1).")

    if len(recs) == 0:

        recs.append("- No strong automated signals found. Consider deeper manual review or upload competitor data for benchmarking.")

 

    for r in recs:

        st.markdown(r)

 

    # st.markdown("---")

    # st.info("Upload competitor review datasets (CSV with columns: TimeStamp, rating, review, CountryName) to compare trends across providers — feature placeholder.")

 

 

def show_complaint_analytics(filtered_df, date1, date2):

    """🚨 COMPLAINT ANALYTICS - Works with existing filtered_df"""

    st.markdown("""

    <style>

    .complaint-header {color: #d32f2f; font-size: 2.5em; font-weight: bold; text-align: center;}

    .metric-card {background: linear-gradient(135deg, #ff6b6b, #ee5a52); padding: 1rem; border-radius: 10px; margin: 0.5rem;}

    </style>

    """, unsafe_allow_html=True)

   

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown('<div class="complaint-header">🚨 Complaints Dashboard</div>', unsafe_allow_html=True)

   

    st.markdown("<br><br>", unsafe_allow_html=True)

  

    # Process complaints from existing filtered_df

    df_neg, analysis_data = process_complaints(filtered_df)

   

    if df_neg.empty:

        st.warning("⚠️ No negative reviews found in selected filters")

        st.info("💡 Try: Lower rating filter (1-2⭐), longer date range, or different countries")

        return

   

    total_complaints = len(df_neg)

    st.metric("📊 Total Complaints Analyzed", total_complaints)

   

    # # === 1. TIME-SERIES TRENDS ===

    # col1, col2 = st.columns(2)

    # with col1:

    #     st.markdown("### 📈 Monthly Complaint Trends")

    #     # require at least 2 months to show trend

    #     if 'month_year' in df_neg.columns and df_neg['month_year'].notna().sum() > 1:

    #         # group by month (using timestamp in month_year)

    #         issues = list(analysis_data['issues'].keys())

    #         trends = (

    #             df_neg

    #             .set_index('month_year')

    #             .groupby(pd.Grouper(freq='M'))[issues]

    #             .sum()

    #             .reset_index()

    #         )

    #         if not trends.empty:

    #             # convert to readable month label and ensure chronological order

    #             trends['month_label'] = pd.to_datetime(trends['month_year']).dt.strftime('%Y-%m')

    #             trends = trends.sort_values('month_year')

    #             fig_line = px.line(

    #                 trends,

    #                 x='month_label',

    #                 y=issues,

    #                 title="Complaint Volume Over Time",

    #                 color_discrete_sequence=px.colors.sequential.Oranges

    #             )

    #             fig_line.update_xaxes(tickangle=45)

    #             st.plotly_chart(fig_line, use_container_width=True, height=400)

    #         else:

    #             st.info("ℹ️ Not enough monthly aggregated data to show trends")

    #     else:

    #         st.info("ℹ️ Need more time range for trends (select at least 2 months)")

 

    # with col2:

    #     st.markdown("### 🎯 Issue Priority Matrix")

    #     issues = list(analysis_data.get('issues', {}).keys())

    #     issue_stats = pd.DataFrame({

    #         'issue': issues,

    #         'frequency': [int(df_neg[issue].sum()) for issue in issues],

    #         'priority': [get_priority_score(df_neg, issue) for issue in issues]

    #     })

    #     # avoid divide by zero and produce sensible severity

    #     issue_stats['severity'] = issue_stats.apply(

    #         lambda r: (r['priority'] / r['frequency']) if r['frequency'] > 0 else 0.0,

    #         axis=1

    #     )

 

    #     # if no frequencies, show info

    #     if issue_stats['frequency'].sum() == 0:

    #         st.info("No detected issue mentions in the selected filters/date range.")

    #     else:

    #         fig_bubble = px.scatter(

    #             issue_stats,

    #             x='frequency',

    #             y='severity',

    #             size='priority',

    #             hover_name='issue',

    #             size_max=50,

    #             color='priority',

    #             color_continuous_scale='Reds',

    #             title="Priority: Size = Frequency × Severity"

    #         )

    #         st.plotly_chart(fig_bubble, use_container_width=True, height=400)

 

    # === 3. HEATMAP ===

   

    st.markdown("### 🌍 Country × App Version Heatmap")

 

    if 'App_Version' in df_neg.columns and 'CountryName' in df_neg.columns:

        heatmap_data = (

            df_neg.groupby(['CountryName', 'App_Version'])

            .size()

            .reset_index(name='complaints')

        )

 

        fig_heatmap = px.density_heatmap(

            heatmap_data,

            x='App_Version',

            y='CountryName',

            z='complaints',

            title="Complaint Density",

            color_continuous_scale='Reds'

        )

 

        # ✅ Center the title

        fig_heatmap.update_layout(

            title_x=0.45,  # 0.5 = center

            height=500

        )

       

        fig_heatmap.update_layout(

            title_font=dict(size=22, family='Arial', color='black')

       )


        st.plotly_chart(fig_heatmap, use_container_width=True)        

    

    

    st.markdown("### 📋 Top Issues Ranked")

 

    top_issues = compute_top_issues(df_neg, analysis_data.get('issues', {}), total_complaints)

 

    if top_issues.empty:

        st.info("No top issues detected for the selected filters/date range.")

    else:

        display_df = top_issues.copy()

 

        # 1) Round Priority Score (e.g., 32.4000 → 32.4)

        display_df['Priority Score'] = (

            display_df['Priority Score']

            .astype(float)

            .round(1)

        )

 

        # 2) Add S.No. as the first column, starting from 1

        display_df.insert(0, 'S.No.', range(1, len(display_df) + 1))

 

        # 3) Build centered, full-width HTML table

        table_html = display_df.to_html(index=False, classes='top-issues-table')

 

        st.markdown(

            """

            <style>

            /* Center the table block and make it full width */

            .top-issues-wrapper {

                display: flex;

                justify-content: center;   /* center the block */

                width: 100%;

            }

            .top-issues-table {

                width: 100%;               /* expand to fill width */

                border-collapse: collapse;

                font-family: system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;

            }

            .top-issues-table th, .top-issues-table td {

                padding: 0.5rem 0.75rem;

                border-bottom: 1px solid rgba(0,0,0,0.08);

                text-align: left;          /* keep text left-aligned for readability */

            }

            .top-issues-table th {

                font-weight: 600;

                background: rgba(0,0,0,0.02);

            }

            </style>

            """,

            unsafe_allow_html=True

        )

 

        st.markdown(

            f"""

            <div class="top-issues-wrapper">

                {table_html}

            </div>

            """,

            unsafe_allow_html=True

        )

 

    st.markdown("<br><br>", unsafe_allow_html=True)

   

    # # === 5. CHURN SIGNALS ===

    # if analysis_data.get('churn'):

   

    #     st.markdown("### ⚠️ Churn Risk (Competitor Mentions)")

    #     churn_df = pd.DataFrame(list(analysis_data['churn'].items()),

    #                            columns=['Competitor', 'Mentions']).sort_values('Mentions', ascending=False)

 

    #     fig_churn = px.bar(

    #         churn_df,

    #         x='Competitor',

    #         y='Mentions',

    #         title="Users Threatening to Switch",

    #         color='Mentions',

    #         color_continuous_scale='OrRd',

    #         hover_data={'Mentions': True}

    #      )

    #     # explicit hover template to show the Mentions count

    #     fig_churn.update_traces(hovertemplate='Competitor: %{x}<br>Mentions: %{y}<extra></extra>')

    #     fig_churn.update_layout(coloraxis_showscale=False)

    #     st.plotly_chart(fig_churn, use_container_width=True)

   

    

    # === 6. INTERACTIVE REVIEW EXPLORER ===

   

 

    st.markdown("### 🔍 Explore Raw Complaints")

 

    issues_list = top_issues['Issue'].tolist() if not top_issues.empty else []

 

    # --- Centered dropdown ---

    center_cols = st.columns([1, 2, 1])

    with center_cols[1]:

        selected_issue = st.selectbox(

            "Filter by Issue:",

            options=issues_list,

            index=0 if issues_list else None

        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Table directly below the dropdown ---

    if not issues_list:

        st.info("No issue categories to explore.")

    elif selected_issue is None:

        st.info("Please select an issue to explore complaints.")

    else:

        # Filter reviews for the selected issue

        issue_reviews = df_neg[df_neg[selected_issue] == 1][

            ['DateTimeStamp', 'review', 'CountryName', 'rating', 'sentiment_label']

        ].copy()

 

        if issue_reviews.empty:

            st.info("No reviews for the selected issue.")

        else:

            # Rename columns for display

            issue_reviews = issue_reviews.rename(columns={

                'DateTimeStamp': 'Date',

                'review': 'Complaint',

                'CountryName': 'Country',

                'rating': 'Rating',

                'sentiment_label': 'Sentiment'

            })

 

            # Reset and add S.No. (1-based)

            issue_reviews = issue_reviews.reset_index(drop=True)

            issue_reviews.index = issue_reviews.index + 1

            issue_reviews.index.name = "S.No."

 

            # --- Pagination ---

           

            # --- Compute page slice as before ---

           

            # --- Pagination (10 rows per page) ---

            page_size = 5

            total_rows = len(issue_reviews)

            total_pages = (total_rows + page_size - 1) // page_size

            total_pages = max(total_pages, 1)  # ensure at least 1

 

            # Unique page key per selected issue to avoid collisions across issues

            page_key = f"explorer_page_{selected_issue}"

 

            # Reset page to 1 if the selected issue changed since last render

            # (So you don't carry a larger page index from a previous issue)

            if st.session_state.get("last_selected_issue") != selected_issue:

                st.session_state[page_key] = 1

                st.session_state["last_selected_issue"] = selected_issue

 

            # Initialize page if not present

            if page_key not in st.session_state:

                st.session_state[page_key] = 1

 

            # Clamp to valid range

            page = int(st.session_state[page_key])

            page = max(1, min(page, total_pages))

 

            # Compute slice

            start = (page - 1) * page_size

            end = min(start + page_size, total_rows)

 

            # --- Display table (remove index column) ---

            page_slice = issue_reviews.iloc[start:end].copy()

            # Optional: add S.No. column if you like

            # page_slice.insert(0, "S.No.", range(start + 1, start + 1 + len(page_slice)))

 

            # Prefer HTML to guarantee no index column. If you want st.dataframe interactivity,

            # replace the two lines below with: st.dataframe(page_slice, use_container_width=True)

            table_html = page_slice.to_html(index=False)

            st.markdown(table_html, unsafe_allow_html=True)

 

            st.caption(

                f"Showing {min(start+1, total_rows)}–{end} of {total_rows} rows — Page {page}/{total_pages}"

            )

 

            # --- Centered pagination control: render slider ONLY if we have >1 page ---

            cols_nav = st.columns([1, 2, 1])

            with cols_nav[1]:

                if total_pages > 1:

                    st.slider(

                        "Page",

                        min_value=1,

                        max_value=total_pages,

                        value=page,

                        key=page_key

                    )

                else:

                    st.markdown(

                        "<div style='text-align:center; font-size:0.9rem;'>Only one page</div>",

                        unsafe_allow_html=True

                    )

 

 

            # --- Centered download button (no index in CSV) ---

            cols_dl = st.columns([1, 2, 1])

            with cols_dl[1]:

                csv_all = issue_reviews.to_csv(index=False).encode('utf-8')

                st.download_button(

                    label=f"Download All Results ({selected_issue})",

                    data=csv_all,

                    file_name=f"complaints_{selected_issue}_all.csv",

                    mime='text/csv'

                )

 


    

    # === SUMMARY CARDS ===

    st.markdown("---")

    if top_issues.empty:

        st.info("Summary metrics not available (no detected issues).")

    else:

        col1, col2, col3, col4 = st.columns(4)

        top0 = top_issues.iloc[0]

 

        # Safely compute top country and its complaint count

        if 'CountryName' in df_neg.columns and not df_neg.empty:

            country_series = (

                df_neg['CountryName']

                .astype(str).str.strip()

                .replace({'': 'Unknown'}).fillna('Unknown')

            )

            vc = country_series.value_counts()

            top_country = vc.index[0] if len(vc) > 0 else "N/A"

            top_country_count = int(vc.iloc[0]) if len(vc) > 0 else 0

        else:

            top_country, top_country_count = "N/A", 0

 

        # Average severity (sentiment score)

        avg_sev_series = df_neg.get("sentiment_score", pd.Series(dtype=float))

        avg_sev = float(avg_sev_series.mean()) if not avg_sev_series.empty else 0.0

 

        with col1:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #66bb6a, #388e3c);">

                <h3 style='color:white;'>Top Issue</h3>

                <h2 style='color:white;'>{top0['Issue']}</h2>

                <p style='color:white;'>{int(top0['Count'])} cases</p>

            </div>

            """, unsafe_allow_html=True)

 

       

        with col2:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #ffa726, #fb8c00);">

                <h3 style='color:white;'>Highest Priority</h3>

                <h2 style='color:white;'>{top0['Issue']}</h2>

                <p style='color:white;'>Score: {float(top0['Priority Score']):.1f}</p>

                <p style='color:white; font-size:0.85em;'>(Most severe & frequent)</p>

            </div>

            """, unsafe_allow_html=True)

 

 

        with col3:

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #42a5f5, #1976d2);">

                <h3 style='color:white;'>Most Complaints</h3>

                <h2 style='color:white;'>{top_country}</h2>

                <p style='color:white;'>{top_country_count} complaints</p>

            </div>

            """, unsafe_allow_html=True)

 

       

        with col4:

            # Interpret severity

            severity_label = (

                "Very Negative" if avg_sev < 0.3 else

                "Negative" if avg_sev < 0.5 else

                "Moderate"

            )

            st.markdown(f"""

            <div class="metric-card" style="background: linear-gradient(135deg, #ff6b6b, #ee5a52);">

                <h3 style='color:white;'>Avg Severity</h3>

                <h2 style='color:white;'>{avg_sev:.2f}</h2>

                <p style='color:white;'>Overall sentiment: {severity_label}</p>

            </div>

            """, unsafe_allow_html=True)

 

 

 

    st.markdown("<br>", unsafe_allow_html=True)

# Define links

links = {

    "Global Ranking": "worldmap",

    "Interactive Sunburst": "sunburst",

    "Word Cloud": "wordcloud",

    "Visual Charts": "visualcharts",

    "Interactive TreeMap": "treemap",

    "Keyword Analysis": "keyword",

    "Topic Modeling": "topic",

    "LanguageTranslation": "translation",

    "Complaint Analytics": "insights",

 

}

 

# Menu options: charts + Reset at the end

menu_options = list(links.keys()) + ["---","🔄 RESET"]

menu_icons = [

    "globe", "sun", "cloud", "bar-chart", "tree",

    "search", "book", "translate", "arrow-repeat"

]

 

# Sidebar menu with bold links and no background on selection

with st.sidebar:

    selected = option_menu(

        menu_title="",  # No title

        options=menu_options,

        icons=menu_icons,

        menu_icon="cast",

        default_index=len(menu_options) - 1,  # Default selection is "🔄 Reset"

        styles={

            "container": {

                "padding": "2px",

                "background-color": "#f8f9fa"

            },

            "icon": {

                "color": "#007BFF",

                "font-size": "14px"

            },

            "nav-link": {

                "font-size": "12px",

                "text-align": "left",

                "margin": "1px 0",

                "font-weight": "bold",  # Bold for all items

           

            },

            "nav-link-selected": {

                "color": "#000000",       # Black text

                "font-weight": "bold",     # Bold for selected

                "background-color": "#ffdd00",

            },

        }

    )

 

# Handle selection

if selected == "🔄 Reset":

    st.session_state.selected_chart = None

else:

    st.session_state.selected_chart = links.get(selected)

   

 

# Main container

main_container = st.container()

with main_container:

    if st.session_state.get("selected_chart") is None:

        st.markdown("<br>", unsafe_allow_html=True)

        search_query = st.text_input("**Search Reviews :**")

        st.markdown("<br>", unsafe_allow_html=True)

 

        # Filter logic

        if search_query:

            with st.spinner("🔍 Fetching reviews..."):

                filtered_df = filtered_df[filtered_df['review'].str.contains(search_query, case=False, na=False)]

                placeholder = st.empty()

                progress_bar = st.progress(0)

                for i in range(100):

                    time.sleep(0.01)

                    progress_bar.progress(i + 1)

                    placeholder.text(f"Fetching reviews... {i+1}%")

                placeholder.empty()

                progress_bar.empty()

 

 

        if not filtered_df.empty:

            # Format timestamp

            filtered_df["DateTimeStamp"] = pd.to_datetime(filtered_df["DateTimeStamp"])

            filtered_df["DateTimeStamp"] = filtered_df["DateTimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

 

            # Pagination

            page_size = 100

            total_pages = max((len(filtered_df) - 1) // page_size + 1, 1)

            page_num = st.session_state.get("page_slider", 1)

            page_num = max(1, min(page_num, total_pages))

 

            start_idx = (page_num - 1) * page_size

            end_idx = min(start_idx + page_size, len(filtered_df))

            column_config = {

                "TimeStamp": st.column_config.DatetimeColumn(

                    label="📅 Date",

                    width="small",

                    format="YYYY-MM-DD"

                ),

                "review": st.column_config.TextColumn(

                    label="💬 Review",

                    width="large"

                ),

                "rating": st.column_config.NumberColumn(

                    label="⭐ Rating",

                    width="small",

                    format="%.0f ⭐"

                ),

                "sentiment_score": st.column_config.ProgressColumn(

                    label="📊 Score",

                    width="small",

                    format="%.2f",

                    min_value=0,

                    max_value=1,

                ),

                "sentiment_label": st.column_config.TextColumn(

                    label="😊 Sentiment",

                    width="small"

                ),

                "HappinessIndex": st.column_config.TextColumn(

                    label="Happiness Index",

                    width="small"

                ),

                "CountryName": st.column_config.TextColumn(

                    label="🌍 Country",

                    width="small"

                ),

                "AppName": st.column_config.TextColumn(

                    label="📱 App",

                    width="small"

                ),

                "appVersion": st.column_config.TextColumn(

                    label="🔢 Version",

                    width="small"

                ),

                "UserName": st.column_config.TextColumn(

                    label="👤 User",

                    width="small"

                )

            }

 

            st.dataframe(

                filtered_df.iloc[start_idx:end_idx],

                column_config=column_config,

                height=275,

                use_container_width=True

            )

            st.caption("ℹ️ Sentiment Scores are between 0 to 1.")

            if total_pages > 1:

                col_left, col_center, col_right = st.columns([1, 2, 1])

                with col_center:

                    st.markdown("<div style='text-align: center; font-size: 14px;'>Navigate Pages</div>", unsafe_allow_html=True)

                    page_num = st.slider(

                        label="",

                        min_value=1,

                        max_value=total_pages,

                        value=page_num,

                        key="page_slider"

                    )

            else:

                st.markdown("<div style='text-align: center; font-size: 8px;'>Only one page of results</div>", unsafe_allow_html=True)

 

            st.markdown("<br>", unsafe_allow_html=True)

            st.caption(f"Showing page {page_num} of {total_pages} — rows {start_idx + 1} to {end_idx}")

            st.success(f"✅ Displaying {len(filtered_df)} reviews.")

            st.markdown("<br>", unsafe_allow_html=True)

 

            # Download button

            csv = filtered_df.to_csv(index=False).encode('utf-8-sig')

            st.download_button('Download Data', data=csv, file_name="Data.csv", mime="text/csv")

 

    else:

        # Render selected chart

        #st.write(f"✅ You selected: {st.session_state.selected_chart}")

        if st.session_state.selected_chart == "worldmap":

            show_world_map(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "sunburst":

            show_sunburst_chart(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "wordcloud":

            show_word_cloud(filtered_df)

        elif st.session_state.selected_chart == "visualcharts":

            show_visual_charts(filtered_df, df, date1, date2)

        elif st.session_state.selected_chart == "treemap":

            show_treemap_chart(filtered_df, date1, date2)

        elif st.session_state.selected_chart == "keyword":

            show_keyword_analysis(filtered_df, stop_words)

        elif st.session_state.selected_chart == "insights":

            show_complaint_analytics(filtered_df, date1, date2)

            show_customer_insights(finaldf)

        elif st.session_state.selected_chart == "topic":

            show_topic_modeling(filtered_df)

        elif st.session_state.selected_chart == "translation":

            #st.subheader("🌐 Language Translation - Filtered Reviews")

 

            # Show filtered DataFrame first

            if not filtered_df.empty:

                # Format timestamp

                filtered_df["DateTimeStamp"] = pd.to_datetime(filtered_df["DateTimeStamp"])

                filtered_df["DateTimeStamp"] = filtered_df["DateTimeStamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Pagination setup

                page_size = 100

                total_pages = max((len(filtered_df) - 1) // page_size + 1, 1)

                page_num = st.session_state.get("page_slider_translation", 1)

                page_num = max(1, min(page_num, total_pages))

 

                start_idx = (page_num - 1) * page_size

                end_idx = min(start_idx + page_size, len(filtered_df))

 

                # Column configuration

                column_config = {

                    "TimeStamp": st.column_config.DatetimeColumn(label="📅 Date", width="small", format="YYYY-MM-DD"),

                    "review": st.column_config.TextColumn(label="💬 Review", width="large"),

                    # "rating": st.column_config.NumberColumn(label="⭐ Rating", width="small", format="%.0f ⭐"),

                    "CustomerRating": st.column_config.TextColumn(label="⭐ Rating", width="small"),

                    "CountryName": st.column_config.TextColumn(label="🌍 Country", width="small"),

                    "AppName": st.column_config.TextColumn(label="📱 App", width="small"),

                    "appVersion": st.column_config.TextColumn(label="🔢 Version", width="small"),

                    "UserName": st.column_config.TextColumn(label="👤 User", width="small")

                }

 

                # Display DataFrame

                st.dataframe(filtered_df.iloc[start_idx:end_idx], column_config=column_config, height=275, use_container_width=True)

                #st.caption("ℹ️ Showing filtered reviews for translation.")

 

                # Pagination slider

                if total_pages > 1:

                    col_left, col_center, col_right = st.columns([1, 2, 1])

                    with col_center:

                        st.markdown("<div style='text-align: center; font-size: 14px;'>Navigate Pages</div>", unsafe_allow_html=True)

                        page_num = st.slider("", min_value=1, max_value=total_pages, value=page_num, key="page_slider_translation")

                # else:

                #     st.markdown("<div style='text-align: center; font-size: 8px;'>Only one page of results</div>", unsafe_allow_html=True)

               

                st.markdown("<br>", unsafe_allow_html=True)

                st.caption(f"Showing page {page_num} of {total_pages} — rows {start_idx + 1} to {end_idx}")

               

                st.success(f"✅ Displaying {len(filtered_df)} reviews.")

 

                # Download button

                csv = filtered_df.to_csv(index=False).encode('utf-8-sig')

                st.download_button('Download Data', data=csv, file_name="Filtered_Translation_Data.csv", mime="text/csv")

            else:

                st.warning("⚠️ No filtered reviews available for translation.")

 

            st.markdown("---")  # Separator

            # Show translation widget below DataFrame

            show_translation_widget(languages)

 

 

 

 

 

 

# qr_img = Image.open('app_qr_code.png')

# # Add vertical space or put this block at the very end of your app

# # Convert QR image to base64

# buffered = io.BytesIO()

# qr_img.save(buffered, format="PNG")

# img_str = base64.b64encode(buffered.getvalue()).decode()




st.markdown("""
<div style="text-align: center; margin-top: 5rem; font-size: 0.9rem; color: #666; font-weight: bold;">
    <em>*Customer Review Data will only be visible when either Language Translation/Reset Button is clicked</em><br>
    <em>*SunBurst and TreeMap are displayed for data upto 12 months</em>
</div>
""", unsafe_allow_html=True)


st.markdown("""
<div style="position: fixed; right: 20px; bottom: 20px; z-index: 1000; text-align: right;">
    <a href="mailto:srivastava.jaideep@gmail.com?subject=Feedback%20%7C%20Issue%20%7C%20Suggestion" 
       class="bottom-link" 
       style="font-size: 14px; color: #003366; font-weight: bold; 
              text-decoration: none; background: white; 
              padding: 8px 16px; border-radius: 5px; 
              box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        📧 Feedback
    </a>
</div>
<style>
.bottom-link:hover {
    background: #ffdd00 !important;
    color: black !important;
    text-decoration: none !important;
}
</style>
""", unsafe_allow_html=True)
