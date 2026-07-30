# Import the libraries used to load, prepare and display predictions.
import joblib
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from pathlib import Path

APP_DIRECTORY = Path(__file__).resolve().parent

# Set the browser tab details and use a wider layout for the form.
st.set_page_config(
    page_title="Heart Disease Decision-Support Tool",
    page_icon= str(APP_DIRECTORY / "icons" / "heart_symbol.svg"),
    layout="wide",
)


# Apply consistent colours, spacing and contrast across both themes.
st.markdown(
    """
    <style>
    /* Store colours that are reused throughout the page. */
    :root {
        --navy: #17324d;
        --blue: #2c6e9b;
        --red: #b42318;
        --green: #087443;
        --surface: #ffffff;
        --border: #d8e2ec;
        --muted: #4d6173;
    }

    /* Match the page colours to Streamlit's active theme. */
    .stApp {
        background:
            url("/app/static/heart-shape-stethoscope-medical-subjects.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* Keep the page content centred and comfortably spaced. */
    .block-container {
        max-width: 950px;
        padding: 200px 28px 56px;
    }

    /* Keep the top header red in both themes. */
    [data-testid="stHeader"] {
        background: #b42318 !important;
    }

    /* Keep every toolbar button visible against the red header. */
    [data-testid="stToolbar"] button,
    [data-testid="stToolbar"] button:hover,
    [data-testid="stToolbar"] button:focus {
        color: #ffffff !important;
    }

    /* Keep the toolbar icons white. */
    [data-testid="stToolbar"] button svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Keep the main menu button white in every interaction state. */
    [data-testid="stMainMenu"],
    [data-testid="stMainMenu"]:hover,
    [data-testid="stMainMenu"]:focus {
        color: #ffffff !important;
    }

    /* Keep the three-dot menu icon white. */
    [data-testid="stMainMenu"] svg {
        fill: #ffffff !important;
        color: #ffffff !important;
    }

    /* Use readable heading colours for both themes. */
    h1, h2, h3 {
        color: light-dark(#17324d, #fafafa) !important;
        letter-spacing: 1px;
    }

    /* Add a red divider below the main heading. */
    h1 {
        border-bottom: 3px solid #b42318;
        padding-bottom: 8px;
    }

    /* Give the introduction equal space above and below. */
    .intro-copy {
        margin: 16px 0 !important;
    }

    /* Keep form labels readable against each form background. */
    [data-testid="stWidgetLabel"] p {
        color: light-dark(#17324d, #fafafa) !important;
        font-weight: 600;
    }

    /* Keep all input surfaces white in both themes. */
    [data-baseweb="select"] > div,
    [data-baseweb="input"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* Remove nested borders from number input fields. */
    [data-baseweb="input"] > div,
    [data-baseweb="input"] input {
        background: #ffffff !important;
        border: none !important;
    }

    /* Keep entered values dark against the white input surfaces. */
    [data-baseweb="input"] input,
    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div * {
        color: #1a2633 !important;
        -webkit-text-fill-color: #1a2633 !important;
    }

    /* Keep dropdown arrows dark and visible. */
    [data-baseweb="select"] svg {
        fill: #1a2633 !important;
        color: #1a2633 !important;
    }

    /* Make example placeholders lighter than entered values. */
    [data-baseweb="input"] input::placeholder {
        color: rgba(26, 38, 51, 0.42) !important;
        opacity: 1;
        -webkit-text-fill-color: rgba(26, 38, 51, 0.42) !important;
    }

    /* Make unselected dropdown examples lighter than selected values. */
    [data-baseweb="select"]:has(input[value=""])
    > div > div > div:last-child {
        color: rgba(26, 38, 51, 0.42) !important;
        -webkit-text-fill-color: rgba(26, 38, 51, 0.42) !important;
    }

    /* Keep opened dropdown options white with dark text. */
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        background: #ffffff !important;
        color: #1a2633 !important;
        -webkit-text-fill-color: #1a2633 !important;
    }

    /* Keep the number adjustment buttons light. */
    [data-testid="stNumberInput"] button {
        background: #f4f7fb !important;
        color: #17324d !important;
    }

    /* Keep the number adjustment symbols dark. */
    [data-testid="stNumberInput"] button svg {
        fill: #17324d !important;
        color: #17324d !important;
    }

    /* Highlight focused fields without changing their base colours. */
    [data-baseweb="select"] > div:focus-within,
    [data-baseweb="input"]:focus-within {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(44, 110, 155, 0.15) !important;
    }

    /* Use white for the light form and black for the dark form. */
    div[data-testid="stForm"] {
        background: light-dark(#ffffff, #0e1117);
        border: 1px solid light-dark(#d8e2ec, #3b414b);
        border-radius: 14px;
        padding: 16px 18px 16px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.14);
    }

    /* Use white in light mode and a softer grey in dark mode. */
    [data-testid="stExpander"] {
        background: light-dark(#ffffff, #4a4a4a);
        border: 1px solid light-dark(#d8e2ec, #5a5a5a);
        border-radius: 9px;
        overflow: hidden;
    }

    /* Match the probability card to the prediction details surface. */
    [data-testid="stMetric"] {
        background: light-dark(#ffffff, #4a4a4a);
        border: 1px solid light-dark(#d8e2ec, #5a5a5a);
        border-radius: 9px;
        padding: 16px;
    }

    /* Keep the probability label and value readable in each theme. */
    [data-testid="stMetric"] * {
        color: light-dark(#17324d, #fafafa) !important;
        -webkit-text-fill-color: light-dark(#17324d, #fafafa) !important;
    }

    /* Keep the expander text readable in each theme. */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] * {
        color: light-dark(#17324d, #fafafa) !important;
        -webkit-text-fill-color: light-dark(
            #17324d,
            #fafafa
        ) !important;
    }

    /* Keep the screening warning background consistent. */
    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) {
        background: #fff3cd !important;
        border-radius: 9px !important;
    }

    /* Keep the screening warning text and icon consistent. */
    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) p,
    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) svg {
        color: #5f4300 !important;
        fill: #5f4300 !important;
        -webkit-text-fill-color: #5f4300 !important;
    }

    /* Style the prediction button as the main action. */
    div[data-testid="stFormSubmitButton"] > button {
        background: #b42318;
        border: 0;
        border-radius: 9px;
        color: white;
        font-size: 16px;
        font-weight: 700;
        min-height: 48px;
    }

    /* Keep every part of the button label white in both themes. */
    div[data-testid="stFormSubmitButton"] > button,
    div[data-testid="stFormSubmitButton"] > button p,
    div[data-testid="stFormSubmitButton"] > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    /* Darken the prediction button when it is hovered. */
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #8f1c13;
        color: white;
    }

    /* Keep supporting text readable in both themes. */
    .helper-text {
        color: light-dark(#374d5e, #fafafa);
        font-size: 16px;
    }

    /* Separate the full interface from the photographic background. */
    .block-container {
        max-width: 1050px;
        margin-top: 54px;
        margin-bottom: 42px;
        padding: 34px 38px 42px;
        background: rgba(248, 250, 252, 0.97);
        border: 1px solid rgba(216, 226, 236, 0.95);
        border-radius: 22px;
        box-shadow: 0 18px 55px rgba(23, 50, 77, 0.24);
    }

    /* Keep all content text dark against the light interface. */
    .block-container,
    .block-container p,
    .block-container li {
        color: #1a2633;
    }

    /* Override theme heading colours on the light content panel. */
    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4 {
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
    }

    /* Keep captions readable while retaining their supporting role. */
    [data-testid="stCaptionContainer"] {
        color: #43576a !important;
        opacity: 0.88;
    }

    /* Present the introduction inside a clear white card. */
    .hero-card {
        padding: 26px 30px;
        background: linear-gradient(135deg, #ffffff 0%, #f8eceb 100%);
        border: 1px solid #e7c8c4;
        border-left: 6px solid var(--red);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.10);
        margin-bottom: 18px;
    }

    /* Style the small label above the title. */
    .hero-eyebrow {
        margin: 0 0 6px !important;
        color: var(--red) !important;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 1.6px;
        text-transform: uppercase;
    }

    /* Keep the main title clear across screen sizes. */
    .hero-title {
        margin: 0;
        color: var(--navy) !important;
        font-size: clamp(30px, 4vw, 45px);
        font-weight: 800;
        letter-spacing: -0.8px;
        line-height: 1.12;
    }

    /* Keep the introduction readable without stretching too widely. */
    .hero-copy {
        max-width: 760px;
        margin: 12px 0 0 !important;
        color: #34495c !important;
        font-size: 17px;
        line-height: 1.6;
    }

    /* Keep every form section heading visible on the white form. */
    .form-section {
        margin: 14px 0 6px;
        padding: 10px 14px;
        background: #eef3f7;
        border-left: 4px solid var(--blue);
        border-radius: 8px;
        color: var(--navy);
        font-size: 17px;
        font-weight: 800;
    }

    /* Keep the form fully opaque over the background image. */
    div[data-testid="stForm"] {
        background: #ffffff;
        border: 1px solid #d8e2ec;
        border-radius: 16px;
        padding: 20px 22px 22px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.11);
    }

    /* Prevent theme colours from reducing form-label contrast. */
    [data-testid="stWidgetLabel"] p {
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
    }

    /* Keep supporting panels white and their text dark. */
    [data-testid="stExpander"],
    [data-testid="stMetric"] {
        background: #ffffff;
        border-color: #d8e2ec;
    }

    [data-testid="stExpander"] *,
    [data-testid="stMetric"] * {
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
    }

    /* Override the stronger theme selector used by expander headings. */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: var(--navy) !important;
        -webkit-text-fill-color: var(--navy) !important;
    }

    /* Present the final prediction as one high-contrast card. */
    .result-card {
        margin-top: 12px;
        padding: 24px 26px;
        background: #ffffff;
        border: 1px solid #d8e2ec;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(23, 50, 77, 0.12);
    }

    /* Use red when further clinical assessment is advised. */
    .result-positive {
        border-left: 7px solid var(--red);
        background: #fff7f6;
    }

    /* Use green when heart disease is not indicated. */
    .result-negative {
        border-left: 7px solid var(--green);
        background: #f3fbf7;
    }

    /* Emphasise the main prediction statement. */
    .result-title {
        margin: 0 0 8px;
        color: var(--navy) !important;
        font-size: 24px;
        font-weight: 800;
    }

    /* Make the probability prominent without overpowering the result. */
    .result-probability {
        margin: 14px 0;
        color: var(--navy) !important;
        font-size: 19px;
        font-weight: 700;
    }

    /* Keep the result guidance clear on both card colours. */
    .result-copy {
        margin: 0 !important;
        color: #34495c !important;
        font-size: 16px;
        line-height: 1.55;
    }

    /* Finish the page with a clear but quiet disclaimer. */
    .footer-card {
        margin-top: 30px;
        padding-top: 18px;
        border-top: 1px solid #cbd7e2;
        color: #43576a !important;
        font-size: 13px;
        line-height: 1.5;
        text-align: center;
    }

    /* Leave a visible strip of background below the fixed header. */
    [data-testid="stMain"] {
        padding-top: 28px;
    }

    /* Provide a complete high-contrast design when dark mode is active. */
    @media (prefers-color-scheme: dark) {
        .stApp {
            background:
                linear-gradient(
                    rgba(5, 10, 18, 0.42),
                    rgba(5, 10, 18, 0.42)
                ),
                url("app/static/heart-shape-stethoscope-medical-subjects.jpg");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }

        /* Keep the complete interface opaque over the background. */
        .block-container {
            background: rgba(32, 33, 36, 0.98);
            border-color: #4a4d52;
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.55);
        }

        /* Use light text throughout the dark interface. */
        .block-container,
        .block-container p,
        .block-container li {
            color: #e5edf5;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3,
        .block-container h4 {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        [data-testid="stCaptionContainer"] {
            color: #c7d2df !important;
        }

        /* Keep the introduction distinct from the main dark panel. */
        .hero-card {
            background: linear-gradient(135deg, #2b2c30 0%, #302a2b 100%);
            border-color: #604345;
            border-left-color: #ef5b4f;
            box-shadow: 0 8px 26px rgba(0, 0, 0, 0.32);
        }

        .hero-eyebrow {
            color: #ff8075 !important;
            -webkit-text-fill-color: #ff8075 !important;
        }

        .hero-title {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        .hero-copy {
            color: #d7e0ea !important;
            -webkit-text-fill-color: #d7e0ea !important;
        }

        /* Separate each field group from the form background. */
        .form-section {
            background: #34363b;
            border-left-color: #38bdf8;
            color: #f1f5f9;
        }

        /* Keep the form and supporting panels solid in dark mode. */
        div[data-testid="stForm"],
        [data-testid="stExpander"],
        [data-testid="stMetric"] {
            background: #292a2e;
            border-color: #4f5258;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.30);
        }

        /* Keep labels and supporting-panel text clearly visible. */
        [data-testid="stWidgetLabel"] p,
        [data-testid="stExpander"] *,
        [data-testid="stMetric"] *,
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary * {
            color: #e5edf5 !important;
            -webkit-text-fill-color: #e5edf5 !important;
        }

        /* Give inputs dark surfaces with light entered values. */
        [data-baseweb="select"] > div,
        [data-baseweb="input"],
        [data-baseweb="input"] > div,
        [data-baseweb="input"] input {
            background: #36383d !important;
            border-color: #5a5d63 !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div * {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        /* Keep unselected examples lighter than entered values. */
        [data-baseweb="input"] input::placeholder,
        [data-baseweb="select"]:has(input[value=""])
        > div > div > div:last-child {
            color: #aab7c7 !important;
            opacity: 1;
            -webkit-text-fill-color: #aab7c7 !important;
        }

        /* Keep dropdown arrows and number controls visible. */
        [data-baseweb="select"] svg,
        [data-testid="stNumberInput"] button svg {
            fill: #e5edf5 !important;
            color: #e5edf5 !important;
        }

        [data-testid="stNumberInput"] button {
            background: #42454b !important;
            color: #e5edf5 !important;
        }

        /* Keep opened dropdown options consistent with dark inputs. */
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] [role="option"] {
            background: #36383d !important;
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        /* Keep the warning distinct without using low-contrast yellow. */
        [data-testid="stAlert"]:has(
            [data-testid="stAlertContentWarning"]
        ) {
            background: #3b2f0a !important;
            border: 1px solid #806a20 !important;
        }

        [data-testid="stAlert"]:has(
            [data-testid="stAlertContentWarning"]
        ) p,
        [data-testid="stAlert"]:has(
            [data-testid="stAlertContentWarning"]
        ) svg {
            color: #fde68a !important;
            fill: #fde68a !important;
            -webkit-text-fill-color: #fde68a !important;
        }

        /* Give result cards distinct dark surfaces. */
        .result-card {
            background: #292a2e;
            border-color: #4f5258;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.32);
        }

        .result-positive {
            background: #342527;
            border-left-color: #f87171;
        }

        .result-negative {
            background: #233229;
            border-left-color: #34d399;
        }

        .result-title,
        .result-probability {
            color: #f8fafc !important;
            -webkit-text-fill-color: #f8fafc !important;
        }

        .result-copy {
            color: #d7e0ea !important;
            -webkit-text-fill-color: #d7e0ea !important;
        }

        .footer-card {
            border-top-color: #475569;
            color: #bdc9d6 !important;
            -webkit-text-fill-color: #bdc9d6 !important;
        }
    }

    /* Follow Streamlit's selected Light or Dark theme. */
    .stApp {
        background:
            linear-gradient(
                light-dark(
                    rgba(255, 255, 255, 0.04),
                    rgba(5, 10, 18, 0.42)
                ),
                light-dark(
                    rgba(255, 255, 255, 0.04),
                    rgba(5, 10, 18, 0.42)
                )
            ),
            url("app/static/heart-shape-stethoscope-medical-subjects.jpg");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .block-container {
        background: light-dark(
            rgba(248, 250, 252, 0.97),
            rgba(32, 33, 36, 0.98)
        );
        border-color: light-dark(
            rgba(216, 226, 236, 0.95),
            #4a4d52
        );
        box-shadow: 0 18px 55px light-dark(
            rgba(23, 50, 77, 0.24),
            rgba(0, 0, 0, 0.55)
        );
    }

    .block-container,
    .block-container p,
    .block-container li {
        color: light-dark(#1a2633, #e5edf5);
    }

    .block-container h1,
    .block-container h2,
    .block-container h3,
    .block-container h4 {
        color: light-dark(#17324d, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#17324d, #f8fafc) !important;
    }

    [data-testid="stCaptionContainer"] {
        color: light-dark(#43576a, #c7d2df) !important;
    }

    .hero-card {
        background: linear-gradient(
            135deg,
            light-dark(#ffffff, #2b2c30) 0%,
            light-dark(#f8eceb, #302a2b) 100%
        );
        border-color: light-dark(#e7c8c4, #604345);
        border-left-color: light-dark(var(--red), #ef5b4f);
        box-shadow: 0 8px 25px light-dark(
            rgba(23, 50, 77, 0.10),
            rgba(0, 0, 0, 0.32)
        );
    }

    .hero-eyebrow {
        color: light-dark(var(--red), #ff8075) !important;
        -webkit-text-fill-color: light-dark(var(--red), #ff8075) !important;
    }

    .hero-title {
        color: light-dark(var(--navy), #f8fafc) !important;
        -webkit-text-fill-color: light-dark(var(--navy), #f8fafc) !important;
    }

    .hero-copy {
        color: light-dark(#34495c, #d7e0ea) !important;
        -webkit-text-fill-color: light-dark(#34495c, #d7e0ea) !important;
    }

    .form-section {
        background: light-dark(#eef3f7, #34363b);
        border-left-color: light-dark(var(--blue), #38bdf8);
        color: light-dark(var(--navy), #f1f5f9);
    }

    div[data-testid="stForm"],
    [data-testid="stExpander"],
    [data-testid="stMetric"] {
        background: light-dark(#ffffff, #292a2e);
        border-color: light-dark(#d8e2ec, #4f5258);
        box-shadow: 0 8px 24px light-dark(
            rgba(23, 50, 77, 0.11),
            rgba(0, 0, 0, 0.30)
        );
    }

    [data-testid="stWidgetLabel"] p,
    [data-testid="stExpander"] *,
    [data-testid="stMetric"] *,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: light-dark(var(--navy), #e5edf5) !important;
        -webkit-text-fill-color: light-dark(var(--navy), #e5edf5) !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="input"] input {
        background: light-dark(#ffffff, #36383d) !important;
        border-color: light-dark(var(--border), #5a5d63) !important;
        color: light-dark(#1a2633, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#1a2633, #f8fafc) !important;
    }

    [data-baseweb="select"] > div,
    [data-baseweb="select"] > div * {
        color: light-dark(#1a2633, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#1a2633, #f8fafc) !important;
    }

    [data-baseweb="input"] input::placeholder,
    [data-baseweb="select"]:has(input[value=""])
    > div > div > div:last-child {
        color: light-dark(
            rgba(26, 38, 51, 0.42),
            #aab7c7
        ) !important;
        opacity: 1;
        -webkit-text-fill-color: light-dark(
            rgba(26, 38, 51, 0.42),
            #aab7c7
        ) !important;
    }

    [data-baseweb="select"] svg,
    [data-testid="stNumberInput"] button svg {
        fill: light-dark(#1a2633, #e5edf5) !important;
        color: light-dark(#1a2633, #e5edf5) !important;
    }

    [data-testid="stNumberInput"] button {
        background: light-dark(#f4f7fb, #42454b) !important;
        color: light-dark(var(--navy), #e5edf5) !important;
    }

    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] [role="option"] {
        background: light-dark(#ffffff, #36383d) !important;
        color: light-dark(#1a2633, #f8fafc) !important;
        -webkit-text-fill-color: light-dark(#1a2633, #f8fafc) !important;
    }

    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) {
        background: light-dark(#fff3cd, #3b2f0a) !important;
        border: 1px solid light-dark(#ead992, #806a20) !important;
    }

    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) p,
    [data-testid="stAlert"]:has(
        [data-testid="stAlertContentWarning"]
    ) svg {
        color: light-dark(#5f4300, #fde68a) !important;
        fill: light-dark(#5f4300, #fde68a) !important;
        -webkit-text-fill-color: light-dark(#5f4300, #fde68a) !important;
    }

    .result-card {
        background: light-dark(#ffffff, #292a2e);
        border-color: light-dark(#d8e2ec, #4f5258);
        box-shadow: 0 8px 24px light-dark(
            rgba(23, 50, 77, 0.12),
            rgba(0, 0, 0, 0.32)
        );
    }

    .result-positive {
        background: light-dark(#fff7f6, #342527);
        border-left-color: light-dark(var(--red), #f87171);
    }

    .result-negative {
        background: light-dark(#f3fbf7, #233229);
        border-left-color: light-dark(var(--green), #34d399);
    }

    .result-title,
    .result-probability {
        color: light-dark(var(--navy), #f8fafc) !important;
        -webkit-text-fill-color: light-dark(var(--navy), #f8fafc) !important;
    }

    .result-copy {
        color: light-dark(#34495c, #d7e0ea) !important;
        -webkit-text-fill-color: light-dark(#34495c, #d7e0ea) !important;
    }

    .footer-card {
        border-top-color: light-dark(#cbd7e2, #475569);
        color: light-dark(#43576a, #bdc9d6) !important;
        -webkit-text-fill-color: light-dark(#43576a, #bdc9d6) !important;
    }

    /* Keep help icons separate from the number adjustment controls. */
    [data-testid="stTooltipIcon"] button {
        background: transparent !important;
        border: 0 !important;
        color: light-dark(#17324d, #e5edf5) !important;
        padding: 0 !important;
    }

    [data-testid="stTooltipIcon"] button svg {
        color: inherit !important;
        fill: none !important;
        stroke: currentColor !important;
    }

    /* Limit number-control styling to the increase and decrease buttons. */
    [data-testid="stNumberInput"] button[data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInput"] button[data-testid="stNumberInputStepUp"] {
        background: light-dark(#f4f7fb, #42454b) !important;
        color: light-dark(#17324d, #e5edf5) !important;
    }

    [data-testid="stNumberInput"] button[data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInput"] button[data-testid="stNumberInputStepUp"] svg {
        color: inherit !important;
        fill: currentColor !important;
        stroke: none !important;
    }

    /* Keep the detailed result panel visually separate from the result card. */
    .result-card {
        margin-bottom: 18px;
    }

    /* Present the probability as the focal point of the prediction result. */
    .result-metric {
        margin: 18px 0;
        text-align: center;
    }

    .result-probability-label {
        display: block;
        color: light-dark(#34495c, #d7e0ea) !important;
        font-size: 15px;
        font-weight: 700;
    }

    .result-probability-value {
        display: block;
        margin-top: 4px;
        color: light-dark(#17324d, #f8fafc) !important;
        font-size: clamp(34px, 5vw, 48px);
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }

    /* Match each chart with the existing light and dark grey panels. */
    [data-testid="stPlotlyChart"] {
        padding: 10px 12px;
        background: light-dark(#ffffff, #292a2e);
        border: 1px solid light-dark(#d8e2ec, #4f5258);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 24px light-dark(
            rgba(23, 50, 77, 0.10),
            rgba(0, 0, 0, 0.30)
        );
    }

    /* Keep field-level validation feedback visible in both themes. */
    .field-error {
        margin: 5px 0 0 !important;
        color: light-dark(#b42318, #fca5a5) !important;
        font-size: 13px;
        font-weight: 600;
        line-height: 1.35;
    }

    /* Reduce outer spacing on smaller displays. */
    @media (max-width: 700px) {
        [data-testid="stMain"] {
            padding-top: 18px;
        }

        .block-container {
            margin-top: 18px;
            padding: 22px 16px 30px;
            border-radius: 14px;
        }

        .hero-card {
            padding: 22px 20px;
        }

        [data-testid="stPlotlyChart"] {
            padding: 6px 4px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Prepare one patient's details using the saved training steps.
def prepare_input(input_data, model_bundle):
    """Apply the same preprocessing used when the final model was trained."""
    # Convert the entered details into the table format expected by the model.
    df_input = pd.DataFrame([input_data])

    # Record whether the major-vessel result was originally missing.
    df_input["ca_missing"] = df_input["ca"].isna().astype(int)

    # Fill missing numerical values with medians from the training data.
    numerical_columns = model_bundle["numerical_missing_columns"]
    df_input[numerical_columns] = df_input[numerical_columns].fillna(
        model_bundle["median_values"]
    )

    # Fill missing binary and discrete values with training modes.
    mode_columns = model_bundle["mode_columns"]
    df_input[mode_columns] = df_input[mode_columns].fillna(
        model_bundle["mode_values"]
    )

    # Keep missing slope and thal values as meaningful categories.
    missing_category_columns = model_bundle["missing_category_columns"]
    df_input[missing_category_columns] = df_input[missing_category_columns].fillna(
        "missing"
    )

    # Convert the categorical values into the model's numerical format.
    df_input = pd.get_dummies(
        df_input,
        columns=model_bundle["columns_to_encode"],
        drop_first=False,
        dtype=int,
    )

    # Match the input columns and order used to train the final model.
    return df_input.reindex(columns=model_bundle["encoded_columns"], fill_value=0)


def display_chart(figure, chart_key):
    """Display one responsive chart without unnecessary toolbar controls."""
    # Keep the chart background transparent so the grey panel remains visible.
    figure.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"family": "Arial, sans-serif"},
        dragmode=False,
    )

    # Prevent accidental zooming while keeping hover details available.
    figure.update_xaxes(fixedrange=True)
    figure.update_yaxes(fixedrange=True)

    # Fit the chart to the available page width.
    st.plotly_chart(
        figure,
        width="stretch",
        theme="streamlit",
        config={
            "displayModeBar": False,
            "responsive": True,
            "scrollZoom": False,
            "doubleClick": False,
        },
        key=chart_key,
    )


def create_probability_gauge(probability):
    """Create a gauge for the current patient's predicted probability."""
    # Convert the probability into the percentage shown on the gauge.
    probability_percentage = probability * 100

    # Match the gauge bar to the predicted side of the threshold.
    gauge_colour = "#b42318" if probability >= 0.5 else "#087443"

    # Build the gauge with the probability and decision threshold together.
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability_percentage,
            number={
                "suffix": "%",
                "valueformat": ".1f",
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticksuffix": "%",
                },
                "bar": {
                    "color": gauge_colour,
                    "thickness": 0.38,
                },
                "bgcolor": "rgba(0, 0, 0, 0)",
                "borderwidth": 0,
                "steps": [
                    {
                        "range": [0, 50],
                        "color": "rgba(20, 184, 166, 0.22)",
                    },
                    {
                        "range": [50, 100],
                        "color": "rgba(248, 113, 113, 0.20)",
                    },
                ],
                "threshold": {
                    "line": {
                        "color": "#f59e0b",
                        "width": 4,
                    },
                    "thickness": 0.80,
                    "value": 50,
                },
            },
        )
    )

    # Keep the gauge compact enough for desktop and mobile screens.
    figure.update_layout(
        height=310,
        margin={"l": 38, "r": 38, "t": 32, "b": 18},
    )

    # Return the completed gauge for display below the prediction.
    return figure


def create_measurement_comparison(input_data, measurement_reference):
    """Compare patient measurements with the training-data distribution."""
    # Store the labels and units used for each measurement row.
    measurement_details = {
        "age": ("Age", "years"),
        "trestbps": ("Resting blood pressure", "mm Hg"),
        "chol": ("Cholesterol", "mg/dL"),
        "thalch": ("Maximum heart rate", "bpm"),
        "oldpeak": ("ST depression", ""),
    }

    # Give every measurement its own horizontal comparison row.
    figure = make_subplots(
        rows=len(measurement_details),
        cols=1,
        vertical_spacing=0.12,
        subplot_titles=[
            (
                f"{label} ({unit})"
                if unit
                else label
            )
            for label, unit in measurement_details.values()
        ],
    )

    # Add the training range and patient value for each measurement.
    for row_number, (column, _) in enumerate(
        measurement_details.items(),
        start=1,
    ):
        # Retrieve the saved training statistics for this measurement.
        reference = measurement_reference[column]

        # Convert the submitted value into a consistent numerical type.
        patient_value = float(input_data[column])

        # Extend the axis when the patient value falls outside the saved range.
        lower_value = min(reference["minimum"], patient_value)
        upper_value = max(reference["maximum"], patient_value)

        # Add a small margin so the end markers are not cut off.
        padding = max((upper_value - lower_value) * 0.05, 1)

        # Show the complete range observed in the training data.
        figure.add_trace(
            go.Scatter(
                x=[reference["minimum"], reference["maximum"]],
                y=[0, 0],
                mode="lines",
                line={
                    "color": "#9ca3af",
                    "width": 4,
                },
                name="Training range",
                legendgroup="training_range",
                showlegend=row_number == 1,
                hovertemplate="Training range: %{x}<extra></extra>",
            ),
            row=row_number,
            col=1,
        )

        # Emphasise the middle half of the training measurements.
        figure.add_trace(
            go.Scatter(
                x=[
                    reference["first_quartile"],
                    reference["third_quartile"],
                ],
                y=[0, 0],
                mode="lines",
                line={
                    "color": "#14b8a6",
                    "width": 15,
                },
                name="Middle 50%",
                legendgroup="middle_range",
                showlegend=row_number == 1,
                hovertemplate="Middle 50%: %{x}<extra></extra>",
            ),
            row=row_number,
            col=1,
        )

        # Mark the training median and the submitted patient value.
        figure.add_trace(
            go.Scatter(
                x=[reference["median"]],
                y=[0],
                mode="markers",
                marker={
                    "color": "#f59e0b",
                    "size": 12,
                    "symbol": "diamond",
                },
                name="Training median",
                legendgroup="median",
                showlegend=row_number == 1,
                hovertemplate="Training median: %{x}<extra></extra>",
            ),
            row=row_number,
            col=1,
        )

        # Plot the submitted value as a separate patient marker.
        figure.add_trace(
            go.Scatter(
                x=[patient_value],
                y=[0],
                mode="markers",
                marker={
                    "color": "#b42318",
                    "line": {
                        "color": "#ffffff",
                        "width": 1,
                    },
                    "size": 15,
                    "symbol": "circle",
                },
                name="Patient value",
                legendgroup="patient",
                showlegend=row_number == 1,
                hovertemplate="Patient value: %{x}<extra></extra>",
            ),
            row=row_number,
            col=1,
        )

        # Use the same calculated range for every item in this row.
        figure.update_xaxes(
            range=[
                lower_value - padding,
                upper_value + padding,
            ],
            showgrid=False,
            zeroline=False,
            row=row_number,
            col=1,
        )

        # Hide the unused vertical axis to keep the chart uncluttered.
        figure.update_yaxes(
            range=[-1, 1],
            visible=False,
            row=row_number,
            col=1,
        )

    # Place one shared legend above all five comparison rows.
    figure.update_layout(
        height=640,
        margin={"l": 24, "r": 24, "t": 52, "b": 20},
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.08,
            "xanchor": "left",
            "x": 0,
        },
    )

    # Return the completed measurement comparison.
    return figure


def create_feature_importance_chart(model, encoded_columns):
    """Show the strongest overall features used by the final model."""
    # Replace encoded column names with clearer labels for the chart.
    readable_feature_names = {
        "age": "Age",
        "trestbps": "Resting blood pressure",
        "chol": "Cholesterol",
        "fbs": "Fasting blood sugar",
        "thalch": "Maximum heart rate",
        "exang": "Exercise-induced angina",
        "oldpeak": "ST depression",
        "ca": "Major vessels",
        "ca_missing": "Major vessels missing",
        "sex_Male": "Sex: Male",
        "cp_atypical angina": "Chest pain: Atypical angina",
        "cp_non-anginal": "Chest pain: Non-anginal",
        "cp_typical angina": "Chest pain: Typical angina",
        "restecg_normal": "Resting ECG: Normal",
        "restecg_st-t abnormality": "Resting ECG: ST-T abnormality",
        "slope_flat": "ST slope: Flat",
        "slope_missing": "ST slope: Missing",
        "slope_upsloping": "ST slope: Upsloping",
        "thal_missing": "Thalassemia: Missing",
        "thal_normal": "Thalassemia: Normal",
        "thal_reversable defect": "Thalassemia: Reversible defect",
    }

    # Pair each encoded feature with its percentage importance.
    feature_importance = pd.DataFrame(
        {
            "Feature": encoded_columns,
            "Importance": model.feature_importances_ * 100,
        }
    )

    # Apply the readable label wherever a matching name is available.
    feature_importance["Display Feature"] = feature_importance["Feature"].map(
        readable_feature_names
    )

    # Keep the original name if a readable label has not been defined.
    feature_importance["Display Feature"] = feature_importance[
        "Display Feature"
    ].fillna(feature_importance["Feature"])

    # Keep the ten strongest features and order them for a horizontal chart.
    top_features = (
        feature_importance
        .nlargest(10, "Importance")
        .sort_values("Importance")
    )

    # Use teal for the main bars and red for the strongest feature.
    bar_colours = ["#0f766e"] * len(top_features)
    bar_colours[-1] = "#b42318"

    # Draw the importance values beside their feature labels.
    figure = go.Figure(
        go.Bar(
            x=top_features["Importance"],
            y=top_features["Display Feature"],
            orientation="h",
            marker_color=bar_colours,
            text=top_features["Importance"].map(
                lambda value: f"{value:.1f}%"
            ),
            textposition="outside",
            cliponaxis=False,
            hovertemplate=(
                "%{y}<br>Overall importance: %{x:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    # Leave room for the labels and percentages around the bars.
    figure.update_layout(
        height=500,
        margin={"l": 18, "r": 58, "t": 22, "b": 56},
        xaxis_title="Overall model importance (%)",
        yaxis_title="",
        showlegend=False,
    )

    # Start the importance scale at zero and retain light guide lines.
    figure.update_xaxes(
        rangemode="tozero",
        showgrid=True,
        gridcolor="rgba(107, 114, 128, 0.22)",
        zeroline=False,
    )

    # Return the completed feature-importance chart.
    return figure


def create_confusion_matrix_chart(confusion_values):
    """Display the final model's held-out test results."""
    # Convert the saved values into a fixed numerical matrix.
    matrix = np.asarray(confusion_values, dtype=int)

    # Use shorter class labels so they remain readable on mobile screens.
    class_labels = [
        "No disease",
        "Heart disease",
    ]

    # Describe the meaning of each position in the matrix.
    cell_labels = [
        ["True negatives", "False positives"],
        ["False negatives", "True positives"],
    ]

    # Separate correct outcomes from errors for the two chart colours.
    correct_outcomes = np.asarray(
        [
            [1, 0],
            [0, 1],
        ]
    )

    # Reserve space for the label and count shown when hovering.
    hover_details = np.empty((2, 2, 2), dtype=object)

    # Store the matching description and patient count for every cell.
    for row_number in range(2):
        for column_number in range(2):
            hover_details[row_number, column_number] = [
                cell_labels[row_number][column_number],
                matrix[row_number, column_number],
            ]

    # Colour correct predictions teal and incorrect predictions red.
    figure = go.Figure(
        go.Heatmap(
            z=correct_outcomes,
            x=class_labels,
            y=class_labels,
            customdata=hover_details,
            colorscale=[
                [0.00, "#b42318"],
                [0.49, "#b42318"],
                [0.50, "#0f766e"],
                [1.00, "#0f766e"],
            ],
            zmin=0,
            zmax=1,
            showscale=False,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Actual: %{y}<br>"
                "Predicted: %{x}<br>"
                "Patients: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    # Place each patient count directly inside its matrix cell.
    for row_number, actual_label in enumerate(class_labels):
        for column_number, predicted_label in enumerate(class_labels):
            patient_count = matrix[row_number, column_number]

            figure.add_annotation(
                x=predicted_label,
                y=actual_label,
                text=f"<b>{patient_count}</b>",
                showarrow=False,
                font={
                    "color": "#ffffff",
                    "size": 18,
                },
            )

    # Label the prediction and actual-outcome axes.
    figure.update_layout(
        height=430,
        margin={"l": 30, "r": 30, "t": 24, "b": 30},
        xaxis_title="Predicted outcome",
        yaxis_title="Actual outcome",
    )

    # Show the no-disease class at the top of the matrix.
    figure.update_yaxes(autorange="reversed")

    # Return the completed test-performance chart.
    return figure


# Load the saved model and stop with a clear message if loading fails.
try:
    # Open the saved model bundle used by the notebook.
    model_bundle = joblib.load(APP_DIRECTORY / "model" / "heart_disease_model_gbt_rs.pkl")

    # Retrieve the trained classifier from the saved bundle.
    model = model_bundle["model"]

    # Confirm that the bundle contains the supporting graph information.
    required_graph_data = {
        "measurement_reference",
        "validation_confusion_matrix",
    }
    if not required_graph_data.issubset(model_bundle):
        raise KeyError("The saved graph information is incomplete.")
except FileNotFoundError:
    # Stop early when the model file cannot be found.
    st.error("The saved model file is missing. Please contact the application owner.")
    st.stop()
except Exception:
    # Keep unexpected loading errors clear without exposing technical details.
    st.error(
        "The prediction model could not be loaded. Please check the model file "
        "and the installed library versions."
    )
    st.stop()


# Introduce the tool inside a card that remains separate from the background.
st.markdown(
    """
    <div class="hero-card">
        <p class="hero-eyebrow">Clinical decision support</p>
        <h1 class="hero-title">Heart Disease Decision-Support Tool</h1>
        <p class="hero-copy">
            Use available patient assessment information to estimate whether
            heart disease may be present. Complete the required fields, then
            select <strong>Generate prediction</strong>.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Display a fixed safety reminder before the patient form.
st.warning(
    "This tool supports clinical decision-making only. It does not replace professional "
    "medical assessment, diagnosis or treatment."
)

# Provide brief instructions without taking up permanent page space.
with st.expander("How to use this tool", expanded=False):
    st.markdown(
        "1. Enter the patient details available from the assessment.  \n"
        "2. Select **Not recorded** only when an optional test result is unavailable.  \n"
        "3. Review the prediction and model-estimated probability with a qualified "
        "healthcare professional."
    )


# Map the displayed sex labels to the values used during training.
sex_options = {"Female": "Female", "Male": "Male"}

# Map the displayed chest pain labels to the dataset categories.
chest_pain_options = {
    "Typical angina": "typical angina",
    "Atypical angina": "atypical angina",
    "Non-anginal pain": "non-anginal",
    "Asymptomatic": "asymptomatic",
}

# Convert yes-or-no selections into the model's binary values.
yes_no_options = {"No": 0.0, "Yes": 1.0}

# Map the displayed ECG labels to the dataset categories.
resting_ecg_options = {
    "Normal": "normal",
    "ST-T abnormality": "st-t abnormality",
    "Left ventricular hypertrophy": "lv hypertrophy",
}

# Keep an unavailable slope result as a missing value.
slope_options = {
    "Not recorded": np.nan,
    "Upsloping": "upsloping",
    "Flat": "flat",
    "Downsloping": "downsloping",
}

# Keep an unavailable major-vessel result as a missing value.
major_vessels_options = {
    "Not recorded": np.nan,
    "0": 0.0,
    "1": 1.0,
    "2": 2.0,
    "3": 3.0,
}

# Map the displayed thalassemia labels to the dataset categories.
thal_options = {
    "Not recorded": np.nan,
    "Normal": "normal",
    "Fixed defect": "fixed defect",
    "Reversible defect": "reversable defect",
}


# Check every submitted value before it is passed to the prediction model.
def validate_inputs(input_values):
    """Return field-specific messages for missing or unsupported inputs."""
    # Collect each validation message under its matching field name.
    errors = {}

    # Define the accepted ranges and number formats for measured values.
    numerical_rules = {
        "age": {
            "label": "Age",
            "minimum": 28,
            "maximum": 77,
            "integer": True,
            "unit": "years",
        },
        "resting_blood_pressure": {
            "label": "Resting blood pressure",
            "minimum": 80,
            "maximum": 200,
            "integer": True,
            "unit": "mm Hg",
        },
        "cholesterol": {
            "label": "Serum cholesterol",
            "minimum": 85,
            "maximum": 603,
            "integer": True,
            "unit": "mg/dl",
        },
        "maximum_heart_rate": {
            "label": "Maximum heart rate achieved",
            "minimum": 60,
            "maximum": 202,
            "integer": True,
            "unit": "bpm",
        },
        "oldpeak": {
            "label": "ST depression after exercise",
            "minimum": -2.6,
            "maximum": 6.2,
            "integer": False,
            "unit": "",
        },
    }

    # Check every numerical field against the matching rule.
    for field_name, rule in numerical_rules.items():
        # Read the submitted value using the rule's field name.
        value = input_values[field_name]

        # Build one reusable message from the allowed training range.
        range_text = (
            f"Enter a value from {rule['minimum']} to {rule['maximum']}"
            f" {rule['unit']}."
        ).replace(" .", ".")

        # Treat an empty required measurement as a validation error.
        if value is None:
            errors[field_name] = range_text
            continue

        # Reject values that are not stored as supported number types.
        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            errors[field_name] = "Enter a valid number."
            continue

        # Convert valid number types before checking their format and range.
        numeric_value = float(value)

        # Reject infinite values before the remaining numerical checks.
        if not math.isfinite(numeric_value):
            errors[field_name] = "Enter a finite number."
        elif rule["integer"] and not numeric_value.is_integer():
            errors[field_name] = "Enter a whole number."
        elif not rule["integer"] and not math.isclose(
            (numeric_value - rule["minimum"]) / 0.1,
            round((numeric_value - rule["minimum"]) / 0.1),
            abs_tol=1e-9,
        ):
            errors[field_name] = "Enter a value in steps of 0.1."
        elif not rule["minimum"] <= numeric_value <= rule["maximum"]:
            errors[field_name] = range_text

    # Define the allowed choices and whether each selection is required.
    categorical_rules = {
        "sex_label": ("Sex", sex_options, True),
        "chest_pain_label": ("Chest pain type", chest_pain_options, True),
        "fasting_blood_sugar_label": (
            "Fasting blood sugar above 120 mg/dl",
            yes_no_options,
            True,
        ),
        "resting_ecg_label": ("Resting ECG result", resting_ecg_options, True),
        "exercise_angina_label": (
            "Exercise-induced angina",
            yes_no_options,
            True,
        ),
        "slope_label": ("Slope of the peak exercise ST segment", slope_options, False),
        "major_vessels_label": (
            "Major vessels seen by fluoroscopy",
            major_vessels_options,
            False,
        ),
        "thal_label": ("Thalassemia test result", thal_options, False),
    }

    # Check required selections and reject any unsupported option.
    for field_name, (label, options, required) in categorical_rules.items():
        value = input_values[field_name]
        if required and value is None:
            errors[field_name] = f"Select {label.lower()}."
        elif value is not None and value not in options:
            errors[field_name] = f"Select a recognised {label.lower()}."

    # Return an empty dictionary when every submitted value is valid.
    return errors


# Introduce the patient form and identify its required fields.
st.subheader("Patient assessment")
st.caption("Fields marked with * are required to generate a prediction.")

# Group all patient inputs so they are checked together on submission.
with st.form("heart_disease_form"):
    # Group the basic patient details at the start of the form.
    st.markdown(
        '<div class="form-section">1. Patient information</div>',
        unsafe_allow_html=True,
    )
    patient_column_1, patient_column_2, patient_column_3 = st.columns(3)

    with patient_column_1:
        # Collect the patient's age within the range found in the dataset.
        age = st.number_input(
            "Age (years) *",
            min_value=28,
            max_value=77,
            value=None,
            step=1,
            placeholder="e.g. 54",
            help="Enter an age from 28 to 77.",
        )
        age_error = st.empty()

    with patient_column_2:
        # Collect the sex category used by the trained model.
        sex_label = st.selectbox(
            "Sex *",
            options=sex_options.keys(),
            index=None,
            placeholder="e.g. Male",
        )
        sex_label_error = st.empty()

    with patient_column_3:
        # Collect the reported chest-pain category.
        chest_pain_label = st.selectbox(
            "Chest pain type *",
            options=chest_pain_options.keys(),
            index=None,
            placeholder="e.g. Asymptomatic",
        )
        chest_pain_label_error = st.empty()

    # Group the measured numerical and laboratory values.
    st.markdown(
        '<div class="form-section">2. Clinical measurements</div>',
        unsafe_allow_html=True,
    )
    measurement_column_1, measurement_column_2 = st.columns(2)

    with measurement_column_1:
        # Collect the resting blood-pressure measurement.
        resting_blood_pressure = st.number_input(
            "Resting blood pressure (mm Hg) *",
            min_value=80,
            max_value=200,
            value=None,
            step=1,
            placeholder="e.g. 130",
            help="Enter a value from 80 to 200 mm Hg.",
        )
        resting_blood_pressure_error = st.empty()

        # Collect the serum cholesterol measurement.
        cholesterol = st.number_input(
            "Serum cholesterol (mg/dl) *",
            min_value=85,
            max_value=603,
            value=None,
            step=1,
            placeholder="e.g. 240",
            help="Enter a value from 85 to 603 mg/dl.",
        )
        cholesterol_error = st.empty()

        # Record whether fasting blood sugar exceeds the dataset threshold.
        fasting_blood_sugar_label = st.selectbox(
            "Fasting blood sugar above 120 mg/dl *",
            options=yes_no_options.keys(),
            index=None,
            placeholder="e.g. No",
        )
        fasting_blood_sugar_label_error = st.empty()

    with measurement_column_2:
        # Collect the highest heart rate reached during assessment.
        maximum_heart_rate = st.number_input(
            "Maximum heart rate achieved (bpm) *",
            min_value=60,
            max_value=202,
            value=None,
            step=1,
            placeholder="e.g. 140",
            help="Enter a value from 60 to 202 beats per minute.",
        )
        maximum_heart_rate_error = st.empty()

        # Collect the measured ST depression after exercise.
        oldpeak = st.number_input(
            "ST depression after exercise *",
            min_value=-2.6,
            max_value=6.2,
            value=None,
            step=0.1,
            format="%.1f",
            placeholder="e.g. 0.5",
            help="Enter a value from -2.6 to 6.2.",
        )
        oldpeak_error = st.empty()

    # Group the remaining examination and test results.
    st.markdown(
        '<div class="form-section">3. Assessment results</div>',
        unsafe_allow_html=True,
    )
    assessment_column_1, assessment_column_2 = st.columns(2)

    with assessment_column_1:
        # Collect the resting ECG category.
        resting_ecg_label = st.selectbox(
            "Resting ECG result *",
            options=resting_ecg_options.keys(),
            index=None,
            placeholder="e.g. Normal",
        )
        resting_ecg_label_error = st.empty()

        # Record whether exercise produced angina symptoms.
        exercise_angina_label = st.selectbox(
            "Exercise-induced angina *",
            options=yes_no_options.keys(),
            index=None,
            placeholder="e.g. No",
        )
        exercise_angina_label_error = st.empty()

        # Allow an unavailable ST-slope result to remain missing.
        slope_label = st.selectbox(
            "Slope of the peak exercise ST segment (optional)",
            options=slope_options.keys(),
            index=None,
            placeholder="e.g. Flat",
        )
        slope_label_error = st.empty()

    with assessment_column_2:
        # Allow an unavailable fluoroscopy result to remain missing.
        major_vessels_label = st.selectbox(
            "Major vessels seen by fluoroscopy (optional)",
            options=major_vessels_options.keys(),
            index=None,
            placeholder="e.g. 0",
        )
        major_vessels_label_error = st.empty()

        # Allow an unavailable thalassemia result to remain missing.
        thal_label = st.selectbox(
            "Thalassemia test result (optional)",
            options=thal_options.keys(),
            index=None,
            placeholder="e.g. Normal",
        )
        thal_label_error = st.empty()

    # Submit every form field together for validation and prediction.
    submitted = st.form_submit_button(
        "Generate prediction",
        type="primary",
        use_container_width=True,
    )


# Keep the validation summary directly below the completed form.
validation_summary = st.empty()

# Validate every input before the preprocessing and prediction steps.
if submitted:
    # Group the displayed form values under their validation names.
    submitted_values = {
        "age": age,
        "sex_label": sex_label,
        "chest_pain_label": chest_pain_label,
        "resting_blood_pressure": resting_blood_pressure,
        "cholesterol": cholesterol,
        "fasting_blood_sugar_label": fasting_blood_sugar_label,
        "maximum_heart_rate": maximum_heart_rate,
        "oldpeak": oldpeak,
        "resting_ecg_label": resting_ecg_label,
        "exercise_angina_label": exercise_angina_label,
        "slope_label": slope_label,
        "major_vessels_label": major_vessels_label,
        "thal_label": thal_label,
    }

    # Run all field checks before preparing the model input.
    validation_errors = validate_inputs(submitted_values)

    # Match each field with the space reserved for its error message.
    error_placeholders = {
        "age": age_error,
        "sex_label": sex_label_error,
        "chest_pain_label": chest_pain_label_error,
        "resting_blood_pressure": resting_blood_pressure_error,
        "cholesterol": cholesterol_error,
        "fasting_blood_sugar_label": fasting_blood_sugar_label_error,
        "maximum_heart_rate": maximum_heart_rate_error,
        "oldpeak": oldpeak_error,
        "resting_ecg_label": resting_ecg_label_error,
        "exercise_angina_label": exercise_angina_label_error,
        "slope_label": slope_label_error,
        "major_vessels_label": major_vessels_label_error,
        "thal_label": thal_label_error,
    }

    # Display an error under every affected field.
    for field_name, placeholder in error_placeholders.items():
        if field_name in validation_errors:
            placeholder.markdown(
                f'<p class="field-error">{validation_errors[field_name]}</p>',
                unsafe_allow_html=True,
            )
        else:
            placeholder.empty()

    if validation_errors:
        # Use shorter field names in the combined validation summary.
        summary_labels = {
            "age": "Age",
            "sex_label": "Sex",
            "chest_pain_label": "Chest pain type",
            "resting_blood_pressure": "Resting blood pressure",
            "cholesterol": "Serum cholesterol",
            "fasting_blood_sugar_label": "Fasting blood sugar above 120 mg/dl",
            "maximum_heart_rate": "Maximum heart rate achieved",
            "oldpeak": "ST depression after exercise",
            "resting_ecg_label": "Resting ECG result",
            "exercise_angina_label": "Exercise-induced angina",
            "slope_label": "Slope of the peak exercise ST segment",
            "major_vessels_label": "Major vessels seen by fluoroscopy",
            "thal_label": "Thalassemia test result",
        }

        # Format every validation message as one readable list item.
        error_items = "\n".join(
            f"- **{summary_labels[field_name]}:** {message}"
            for field_name, message in validation_errors.items()
        )
        validation_summary.error(
            "Please correct the following before generating a prediction:\n\n"
            + error_items
        )
    else:
        # Convert validated selections into the values used during model training.
        input_data = {
            "age": age,
            "sex": sex_options[sex_label],
            "cp": chest_pain_options[chest_pain_label],
            "trestbps": resting_blood_pressure,
            "chol": cholesterol,
            "fbs": yes_no_options[fasting_blood_sugar_label],
            "restecg": resting_ecg_options[resting_ecg_label],
            "thalch": maximum_heart_rate,
            "exang": yes_no_options[exercise_angina_label],
            "oldpeak": oldpeak,
            "slope": slope_options.get(slope_label, np.nan),
            "ca": major_vessels_options.get(major_vessels_label, np.nan),
            "thal": thal_options.get(thal_label, np.nan),
        }

        # Prepare the input and generate its class and probability.
        try:
            # Apply the same preprocessing used during model training.
            prepared_input = prepare_input(input_data, model_bundle)

            # Generate the final binary prediction.
            prediction = int(model.predict(prepared_input)[0])

            # Retrieve the probability assigned to heart disease.
            probability = float(model.predict_proba(prepared_input)[0, 1])

            # Confirm that both returned values can be safely displayed.
            if prediction not in (0, 1) or not math.isfinite(probability):
                raise ValueError("The model returned an unsupported result.")
            if not 0 <= probability <= 1:
                raise ValueError("The model returned an invalid probability.")
        except Exception:
            # Replace prediction errors with one clear message for the user.
            validation_summary.error(
                "The prediction could not be generated. Please check the entered "
                "information and try again."
            )
        else:
            # Select the card style and guidance that match the prediction.
            if prediction == 1:
                result_class = "result-positive"
                result_title = "Heart disease may be present"
                result_guidance = (
                    "Arrange appropriate clinical review. This output supports "
                    "decision-making and is not a diagnosis."
                )
            else:
                result_class = "result-negative"
                result_title = "No heart disease indicated"
                result_guidance = (
                    "Continue to use clinical judgement and the full patient "
                    "assessment when deciding the next step."
                )

            # Display the prediction and probability together in one result card.
            st.subheader("Prediction result")
            st.markdown(
                f"""
                <div class="result-card {result_class}">
                    <p class="result-title">{result_title}</p>
                    <div class="result-metric">
                        <span class="result-probability-label">
                            Estimated probability of heart disease
                        </span>
                        <span class="result-probability-value">{probability:.1%}</span>
                    </div>
                    <p class="result-copy">{result_guidance}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Explain how the result should be used.
            with st.expander("Prediction details"):
                st.write(
                    "This result helps identify patients who may need further "
                    "clinical assessment. It supports clinical decision-making and should "
                    "still be reviewed by a qualified healthcare professional."
                )

            # Present the patient-specific and overall model visualisations.
            st.subheader("Prediction insights")

            # Show where the patient sits against the classification threshold.
            st.markdown("#### Patient-specific probability")
            display_chart(
                create_probability_gauge(probability),
                "patient_probability_gauge",
            )

            # Compare the submitted measurements with the training distribution.
            st.markdown(
                "#### Patient measurements compared with training data"
            )
            display_chart(
                create_measurement_comparison(
                    input_data,
                    model_bundle["measurement_reference"],
                ),
                "patient_measurement_comparison",
            )

            # Summarise the strongest features used across the complete model.
            st.markdown(
                "#### Overall model feature importance (not patient-specific)"
            )
            display_chart(
                create_feature_importance_chart(
                    model,
                    model_bundle["encoded_columns"],
                ),
                "overall_feature_importance",
            )

            # Keep the held-out test results inside an optional details panel.
            with st.expander("Model performance"):
                st.markdown("#### Held-out test-set confusion matrix")
                display_chart(
                    create_confusion_matrix_chart(
                        model_bundle["validation_confusion_matrix"]
                    ),
                    "validation_confusion_matrix",
                )


# Close the page with a concise purpose and safety statement.
st.markdown(
    """
    <div class="footer-card">
        Heart Disease Decision-Support Tool
    </div>
    """,
    unsafe_allow_html=True,
)
