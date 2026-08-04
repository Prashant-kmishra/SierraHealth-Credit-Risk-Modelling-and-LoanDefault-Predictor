import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pipeline import CreditRiskPipeline

# Page Config
st.set_page_config(page_title="Credit Risk Default Predictor", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #F8F9FA;
    }
    
    @keyframes flipIn {
        0% { transform: rotateY(90deg); opacity: 0; }
        100% { transform: rotateY(0deg); opacity: 1; }
    }
    
    /* Typography */
    h1, h2 {
        color: #1A1A1A !important;
    }
    h3 {
        color: #ED1C24 !important;
    }
    
    /* Form Submit Button */
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #ED1C24;
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 5px;
        width: 100%;
        transition: 0.3s;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #C8102E;
        color: white;
    }
    
    /* Style normal buttons (like View Reasons, Back) */
    button[kind="secondary"] {
        border-color: #ED1C24;
        color: #ED1C24;
    }
    button[kind="secondary"]:hover {
        border-color: #C8102E;
        color: #C8102E;
    }
</style>
</style>
""", unsafe_allow_html=True)

# Global Background Dust Injection
import streamlit.components.v1 as components
components.html("""
<script>
    const parentDoc = window.parent.document;
    const parentWin = window.parent;
    
    let canvas = parentDoc.getElementById('sierra-dust-canvas');
    if (!canvas) {
        canvas = parentDoc.createElement('canvas');
        canvas.id = 'sierra-dust-canvas';
        canvas.style.position = 'fixed';
        canvas.style.top = '0';
        canvas.style.left = '0';
        canvas.style.width = '100vw';
        canvas.style.height = '100vh';
        canvas.style.zIndex = '0'; 
        canvas.style.pointerEvents = 'none';
        
        let stApp = parentDoc.querySelector('.stApp');
        if (stApp) {
            stApp.insertBefore(canvas, stApp.firstChild);
        } else {
            parentDoc.body.appendChild(canvas);
        }
    }
    
    const ctx = canvas.getContext('2d');
    
    function resize() {
        canvas.width = parentWin.innerWidth;
        canvas.height = parentWin.innerHeight;
    }
    resize();
    parentWin.addEventListener('resize', resize);

    let particlesArray = [];
    let mouse = { x: null, y: null, radius: 120 };

    parentWin.addEventListener('mousemove', function(event){
        mouse.x = event.clientX;
        mouse.y = event.clientY;
    });
    parentWin.addEventListener('mouseout', function(){
        mouse.x = undefined;
        mouse.y = undefined;
    });

    class Particle {
        constructor(x, y, directionX, directionY, size, color) {
            this.x = x;
            this.y = y;
            this.directionX = directionX;
            this.directionY = directionY;
            this.size = size;
            this.color = color;
        }
        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2, false);
            ctx.fillStyle = this.color;
            ctx.fill();
        }
        update() {
            if (this.x > canvas.width || this.x < 0) this.directionX = -this.directionX;
            if (this.y > canvas.height || this.y < 0) this.directionY = -this.directionY;

            let dx = mouse.x - this.x;
            let dy = mouse.y - this.y;
            let distance = Math.sqrt(dx * dx + dy * dy);
            
            if (distance < mouse.radius) {
                const forceDirectionX = dx / distance;
                const forceDirectionY = dy / distance;
                const force = (mouse.radius - distance) / mouse.radius;
                const directionX = forceDirectionX * force * 2;
                const directionY = forceDirectionY * force * 2;
                this.x -= directionX;
                this.y -= directionY;
            } else {
                this.x += this.directionX;
                this.y += this.directionY;
            }
            this.draw();
        }
    }

    function init() {
        particlesArray = [];
        let numberOfParticles = (canvas.height * canvas.width) / 9000;
        if(numberOfParticles > 200) numberOfParticles = 200;
        
        for (let i = 0; i < numberOfParticles; i++) {
            let size = (Math.random() * 1.5) + 0.5;
            let x = Math.random() * canvas.width;
            let y = Math.random() * canvas.height;
            let directionX = (Math.random() * 0.8) - 0.4;
            let directionY = (Math.random() * 0.8) - 0.4;
            let color = 'rgba(237, 28, 36, 0.4)';
            particlesArray.push(new Particle(x, y, directionX, directionY, size, color));
        }
    }

    function connect() {
        let opacityValue = 1;
        for (let a = 0; a < particlesArray.length; a++) {
            for (let b = a; b < particlesArray.length; b++) {
                let distance = ((particlesArray[a].x - particlesArray[b].x) * (particlesArray[a].x - particlesArray[b].x)) + 
                               ((particlesArray[a].y - particlesArray[b].y) * (particlesArray[a].y - particlesArray[b].y));
                if (distance < (canvas.width/10) * (canvas.height/10)) {
                    opacityValue = 1 - (distance / 15000);
                    ctx.strokeStyle = 'rgba(237, 28, 36,' + (opacityValue * 0.15) + ')';
                    ctx.lineWidth = 0.5;
                    ctx.beginPath();
                    ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                    ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                    ctx.stroke();
                }
            }
        }
    }

    if(parentWin.dustAnimationId) {
        cancelAnimationFrame(parentWin.dustAnimationId);
    }

    function animate() {
        parentWin.dustAnimationId = requestAnimationFrame(animate);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        for (let i = 0; i < particlesArray.length; i++) {
            particlesArray[i].update();
        }
        connect();
    }

    init();
    animate();
</script>
""", height=0, width=0)

# Header with Logo
import os
import base64

logo_html = ""
# Resolve absolute path to the logo relative to this file
base_dir = os.path.dirname(os.path.abspath(__file__))
logo_path = os.path.join(base_dir, "logo.png")

if os.path.exists(logo_path):
    with open(logo_path, "rb") as img_file:
        logo_base64 = base64.b64encode(img_file.read()).decode('utf-8')
    logo_html = f'<img src="data:image/png;base64,{logo_base64}" height="45" style="border-radius: 5px;">'

components.html(f"""
<script>
    const parentDoc = window.parent.document;
    const header = parentDoc.querySelector('header[data-testid="stHeader"]');
    
    if (header && !parentDoc.getElementById("sierra-navbar-logo")) {{
        const logoDiv = parentDoc.createElement("div");
        logoDiv.id = "sierra-navbar-logo";
        logoDiv.style.display = "flex";
        logoDiv.style.alignItems = "center";
        logoDiv.style.position = "absolute";
        logoDiv.style.left = "1.5rem";
        logoDiv.style.top = "50%";
        logoDiv.style.transform = "translateY(-50%)";
        logoDiv.style.zIndex = "999999";
        
        logoDiv.innerHTML = `
            {logo_html}
            <div style="display: flex; flex-direction: column; justify-content: center; margin-left: 12px;">
                <p style="color: #ED1C24; font-size: 1.25rem; font-weight: bold; margin: 0; padding: 0; line-height: 1.1; font-family: sans-serif;">SierraFinance</p>
                <p style="color: #555555; font-size: 0.65rem; font-weight: 600; letter-spacing: 0.5px; margin: 0; padding: 0; text-transform: uppercase; margin-top: 2px; font-family: sans-serif;">AI POWERED CREDIT. BETTER DECISIONS.</p>
            </div>
        `;
        
        header.appendChild(logoDiv);
    }}
</script>
""", height=0, width=0)

st.markdown("<h2 style='color: #1A1A1A; text-align: center; margin-bottom: 0; margin-top: 20px;'>Credit Risk Modelling</h2>", unsafe_allow_html=True)
st.markdown("<h4 style='color: #555555; text-align: center; font-weight: normal; margin-top: 5px;'>Advanced Retail Banking & Underwriting Prediction Engine</h4>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Initialize the pipeline once using Streamlit caching
@st.cache_resource
def load_model():
    return CreditRiskPipeline()

try:
    pipeline = load_model()
except Exception as e:
    st.error(f"Error loading pipeline or artifacts: {e}")
    st.stop()

# Sync query params with session state for Chrome Back button support
if "page" in st.query_params:
    st.session_state.page = st.query_params["page"]
else:
    if 'page' not in st.session_state:
        st.session_state.page = "main"

if "step" in st.query_params:
    try:
        st.session_state.wizard_step = int(st.query_params["step"])
    except ValueError:
        st.session_state.wizard_step = 0
else:
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 0

def set_page(page_name):
    st.session_state.page = page_name
    st.query_params["page"] = page_name

def next_step():
    st.session_state.wizard_step += 1
    st.query_params["step"] = str(st.session_state.wizard_step)
def prev_step():
    st.session_state.wizard_step -= 1
    st.query_params["step"] = str(st.session_state.wizard_step)

def start_over():
    st.session_state.wizard_step = 0
    st.query_params["step"] = "0"

defaults = {
    'age': 30, 'residence_type': "Owned", 'income': 600000.0,
    'credit_utilization_ratio': 50.0, 'loan_purpose': "Personal",
    'loan_type': "Secured", 'loan_amount': 2000000.0,
    'loan_tenure_months': 24, 'total_loan_months': 36,
    'delinquent_months': 12, 'total_dpd': 15, 'number_of_open_accounts': 2
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


if st.session_state.page == "main":
    step = st.session_state.wizard_step

    with st.container():
        components.html("""
        <script>
            const iframe = window.frameElement;
            if (iframe) {
                const container = iframe.closest('div[data-testid="stVerticalBlock"]');
                if (container) {
                    container.style.backgroundColor = '#FFFFFF';
                    container.style.padding = '3rem';
                    container.style.borderRadius = '12px';
                    container.style.boxShadow = '0 8px 16px rgba(0, 0, 0, 0.08)';
                    container.style.borderTop = '6px solid #ED1C24';
                
                    // Force a CSS reflow to restart the animation on every step
                    container.style.animation = 'none';
                    container.offsetHeight; 
                    container.style.animation = 'flipIn 0.6s cubic-bezier(0.4, 0, 0.2, 1) forwards';
                    container.style.perspective = '1000px';
                }
            }
        </script>
        <style>
            @keyframes flipIn {
                0% { transform: rotateY(90deg); opacity: 0; }
                100% { transform: rotateY(0deg); opacity: 1; }
            }
        </style>
        """, height=0, width=0)

        if step == 0:
            st.subheader("Personal Details")
            st.markdown("<p style='color: #555;'>Enter your demographic and financial information.</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.age)
                st.session_state.income = st.number_input("Annual Income", min_value=0.0, value=st.session_state.income)
            with col2:
                st.session_state.residence_type = st.selectbox("Residence Type", ["Owned", "Rented", "Mortgage"], index=["Owned", "Rented", "Mortgage"].index(st.session_state.residence_type))
                st.session_state.credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=st.session_state.credit_utilization_ratio)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns([3, 1])
            with col_right:
                st.button("Next →", on_click=next_step, use_container_width=True, type="primary")

        elif step == 1:
            st.subheader("Loan Details")
            st.markdown("<p style='color: #555;'>Specify the requirements of the requested loan.</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.loan_purpose = st.selectbox("Loan Purpose", ["Personal", "Education", "Medical", "Home", "Auto"], index=["Personal", "Education", "Medical", "Home", "Auto"].index(st.session_state.loan_purpose))
                st.session_state.loan_amount = st.number_input("Loan Amount", min_value=0.0, value=st.session_state.loan_amount)
            with col2:
                st.session_state.loan_type = st.selectbox("Loan Type", ["Unsecured", "Secured"], index=["Unsecured", "Secured"].index(st.session_state.loan_type))
                st.session_state.loan_tenure_months = st.number_input("Loan Tenure (Months)", min_value=1, value=st.session_state.loan_tenure_months)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.button("← Back", on_click=prev_step, use_container_width=True)
            with col_right:
                st.button("Next →", on_click=next_step, use_container_width=True, type="primary")

        elif step == 2:
            st.subheader("Credit History")
            st.markdown("<p style='color: #555;'>Provide historic credit and delinquency data.</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.total_loan_months = st.number_input("Total Loan Months (History)", min_value=0, value=st.session_state.total_loan_months)
                st.session_state.delinquent_months = st.number_input("Delinquent Months", min_value=0, value=st.session_state.delinquent_months)
            with col2:
                st.session_state.total_dpd = st.number_input("Total Days Past Due (DPD)", min_value=0, value=st.session_state.total_dpd)
                st.session_state.number_of_open_accounts = st.number_input("Number of Open Accounts", min_value=0, value=st.session_state.number_of_open_accounts)
            
            st.markdown("<br>", unsafe_allow_html=True)
            col_left, col_right = st.columns([1, 1])
            with col_left:
                st.button("← Back", on_click=prev_step, use_container_width=True)
            with col_right:
                st.button("Predict Risk Engine", on_click=next_step, use_container_width=True, type="primary")

        elif step == 3:
            input_data = {
                "age": st.session_state.age,
                "residence_type": st.session_state.residence_type,
                "income": st.session_state.income,
                "credit_utilization_ratio": st.session_state.credit_utilization_ratio,
                "loan_purpose": st.session_state.loan_purpose,
                "loan_type": st.session_state.loan_type,
                "loan_amount": st.session_state.loan_amount,
                "loan_tenure_months": st.session_state.loan_tenure_months,
                "total_loan_months": st.session_state.total_loan_months,
                "delinquent_months": st.session_state.delinquent_months,
                "total_dpd": st.session_state.total_dpd,
                "number_of_open_accounts": st.session_state.number_of_open_accounts,
            }

            with st.spinner("Processing Data and Running Master Ensemble..."):
                try:
                    processed_df = pipeline.preprocess_input(input_data)
                    results = pipeline.predict(processed_df)
                
                    st.header("Engine Prediction Results")
                    pred = results["prediction"]
                    prob = results["probability"]
                
                    if pred == 1:
                        st.error("**DECLINED: High Risk of Default**")
                        st.write("Unfortunately, the application does not meet our approval criteria.")
                        st.session_state.processed_df = processed_df
                        col_b1, col_b2 = st.columns([1, 1])
                        with col_b1:
                            st.button("View Decline Reasons", on_click=set_page, args=("reasons",), use_container_width=True, type="primary")
                    else:
                        st.success("**CONGRATULATIONS: Application APPROVED!**")
                        st.write("The applicant is classified as Low Risk and is eligible for the loan.")
                        
                    st.metric(label="Calculated Default Probability", value=f"{prob*100:.2f}%")
                    
                    st.markdown("---")
                    col_left, col_right = st.columns([1, 1])
                    with col_left:
                        st.button("← Start Over", on_click=start_over, use_container_width=True)
                    with col_right:
                        st.button("View Model Breakdown", on_click=set_page, args=("breakdown",), use_container_width=True)
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.button("← Back", on_click=prev_step)

elif st.session_state.page == "reasons":
    components.html("""<script>
        window.parent.document.querySelectorAll('div[data-testid="stVerticalBlock"]').forEach(b => {
            b.style.backgroundColor = ''; b.style.boxShadow = ''; b.style.borderTop = ''; b.style.padding = '';
        });
    </script>""", height=0, width=0)
    st.button("← Back to Application", on_click=set_page, args=("main",))
    st.title("Decision Explanation (SHAP)")
    st.write("This chart shows the top factors that contributed to the model's 'Declined' decision.")
    
    if hasattr(st.session_state, 'processed_df'):
        import matplotlib.pyplot as plt
        import shap
        
        df_processed = st.session_state.processed_df
        explainer = pipeline.shap_explainer
        
        if explainer is not None:
            with st.spinner("Generating Explainability Report..."):
                explanation = explainer(df_processed)
                
                # Extract SHAP values
                shap_values = explanation[0].values
                feature_names = explanation[0].feature_names
                
                # Create a list of tuples (feature, shap_value) and sort by absolute impact
                contributions = list(zip(feature_names, shap_values))
                contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                st.subheader("Key Risk Drivers")
                st.write("The AI Engine identified the following top factors influencing this decision:")
                
                for feature, impact in contributions[:5]:
                    clean_name = feature.replace('_', ' ').title()
                    if impact > 0.5:
                        st.markdown(f"- 🔴 **{clean_name}**: Strongly increased the calculated default risk.")
                    elif impact > 0:
                        st.markdown(f"- 🟠 **{clean_name}**: Marginally increased the calculated default risk.")
                    elif impact < -0.5:
                        st.markdown(f"- 🟢 **{clean_name}**: Strongly decreased the calculated default risk.")
                    else:
                        st.markdown(f"- 🟡 **{clean_name}**: Marginally decreased the calculated default risk.")
                
                st.markdown("---")
                st.subheader("Detailed SHAP Waterfall Plot")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(explanation[0], show=False)
                st.pyplot(fig)
        else:
            st.error("SHAP Explainer was not found in the artifacts. Unable to generate reasons.")
    else:
        st.warning("No prediction data found. Please submit an application first.")

elif st.session_state.page == "breakdown":
    components.html("""<script>
        window.parent.document.querySelectorAll('div[data-testid="stVerticalBlock"]').forEach(b => {
            b.style.backgroundColor = ''; b.style.boxShadow = ''; b.style.borderTop = ''; b.style.padding = '';
        });
    </script>""", height=0, width=0)
    st.button("← Back to Application", on_click=set_page, args=("main",))
    st.title("Model Architecture & Detailed Breakdown")
    
    st.graphviz_chart('''
    digraph Architecture {
        rankdir=TD;
        node [shape=box, style="filled,rounded", color="#ED1C24", fontcolor="white", fontname="sans-serif", margin="0.2,0.1"];
        edge [color="#555555", fontname="sans-serif", fontsize=10];
        
        Data [label="Customer & Loan Data", shape=cylinder, color="#1A1A1A"];
        
        subgraph cluster_models {
            label = "Optuna Tuning + Imbalance Handling";
            style = dashed;
            color = "#1A1A1A";
            fontcolor = "#1A1A1A";
            fontname = "sans-serif";
            
            XGB [label="XGBoost\\n(scale_pos_weight)"];
            LGB [label="LightGBM\\n(SMOTETomek)"];
            CAT [label="CatBoost\\n(auto_class_weights)"];
        }
        
        Data -> XGB;
        Data -> LGB;
        Data -> CAT;
        
        subgraph cluster_calib {
            label = "Isotonic Calibration (20% Holdout)";
            style = dashed;
            color = "#1A1A1A";
            fontcolor = "#1A1A1A";
            fontname = "sans-serif";
            
            CalibXGB [label="Calibrated XGB"];
            CalibLGB [label="Calibrated LGB"];
            CalibCAT [label="Calibrated CAT"];
        }
        
        XGB -> CalibXGB;
        LGB -> CalibLGB;
        CAT -> CalibCAT;
        
        Ensemble [label="Master Ensemble\\n(Unweighted Soft Voting)"];
        
        CalibXGB -> Ensemble;
        CalibLGB -> Ensemble;
        CalibCAT -> Ensemble;
        
        Threshold [label="Business Thresholding\\n(Max F2-Score @ 12.9%)", shape=diamond, color="#1A1A1A"];
        
        Ensemble -> Threshold;
        
        Declined [label="Declined\\n(High Risk)", color="#ED1C24"];
        Approved [label="Approved\\n(Low Risk)", color="#28a745"];
        
        Threshold -> Declined [label=" >= 12.9%"];
        Threshold -> Approved [label=" < 12.9%"];
    }
    ''')
    
    st.markdown("""
    ### 1. Imbalance Handling & Hyperparameter Tuning
    Credit default datasets are notoriously imbalanced, where defaults represent a small fraction of overall loans. Standard models naturally bias toward the majority class (predicting 'No Default' for everyone to achieve high accuracy). To force the models to learn the complex patterns of defaulters, we implemented model-specific architectural strategies:
    
    *   **XGBoost (eXtreme Gradient Boosting):** We configured the `scale_pos_weight` parameter directly inside the objective function. This applies a massive mathematical penalty to the gradients whenever the model misclassifies a defaulter, forcing the decision trees to prioritize minority class splits.
    *   **LightGBM (Light Gradient Boosting Machine):** Because LightGBM builds trees leaf-wise (which is incredibly fast but prone to overfitting imbalanced data), we augmented its training data using **SMOTETomek** pipelines. This complex technique first oversamples defaulters using SMOTE (Synthetic Minority Over-sampling Technique) to create synthetic data points, and then cleans the noisy borders using Tomek Links (removing overlapping majority/minority pairs) to establish a razor-sharp decision boundary.
    *   **CatBoost (Categorical Boosting):** Known for its symmetric "oblivious" trees, CatBoost natively prevents overfitting. We leveraged its algorithmic `auto_class_weights='Balanced'` feature, which dynamically scales the loss function based on the exact frequency of classes in the training folds.
    
    **Optuna Optimization:** Instead of guessing parameters, we utilized the Optuna framework. Optuna ran 50+ trials using the Tree-structured Parzen Estimator (TPE) algorithm to intelligently explore the hyperparameter space (learning rates, tree depths, L2 regularization penalties, and min_child_weights). It optimized strictly for the **F2-Score** rather than standard accuracy.

    ### 2. Probability Calibration
    Raw machine learning models (especially tree-based ensembles) do not output true probabilities; they output abstract logits or scores that cluster near 0 or 1. 
    
    To make these scores interpretable for risk management, each model was individually passed through a **CalibratedClassifierCV**. We used a strict 20% holdout validation dataset (to prevent data leakage) and applied **Isotonic Regression**. Isotonic Regression fits a non-parametric, monotonically increasing piecewise step function to the raw scores. 
    
    *The Result:* If the calibrated model predicts a 15% probability, it means that historically, exactly 15 out of 100 applicants with those exact characteristics defaulted. This ensures the output is a true, trustworthy probability.

    ### 3. The Master Ensemble
    No single model is perfect. To achieve extreme robustness, the final architecture acts as an **Unweighted Averaging Ensemble (Soft Voting Classifier)**. 
    
    The pipeline executes all three calibrated models independently and mathematically averages their probability outputs:
    `P(Default) = [ P_xgb(x) + P_lgb(x) + P_cat(x) ] / 3`
    
    Because XGBoost builds trees depth-wise, LightGBM builds them leaf-wise, and CatBoost builds them symmetrically, they all make slightly different mathematical assumptions. By ensembling them together, we drastically reduce individual model variance and bias, creating a Master Model that ignores noise and captures the underlying truth of the applicant's financial behavior.

    ### 4. Custom Business Thresholding
    Standard machine learning models use a naive 50% probability cutoff to make a Yes/No decision. However, in banking, the financial cost of a **False Negative** (approving a loan that defaults) is massively higher than a **False Positive** (declining a good loan).
    
    To solve this, the Master Ensemble ignores the 50% default cutoff. Instead, the threshold was computationally determined by plotting a **Precision-Recall Curve** across all possible thresholds. We mathematically identified the exact threshold that maximizes the **F2-Score** (which weighs Recall / catching defaulters twice as heavily as Precision). 
    
    This resulted in an optimized, highly-sensitive business threshold (approx. ~12.9%). Any applicant whose predicted probability crosses this specific threshold is automatically classified as High Risk, maximizing the institution's financial safety.
    """)

