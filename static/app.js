// Application Client-Side Logic

// Set current timestamp in header
document.getElementById('current-time').textContent = new Date().toLocaleDateString(undefined, { 
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
});

// Tab Switching Mechanism
function switchTab(tabId) {
    // Hide all tab contents
    const contents = document.querySelectorAll('.tab-content');
    contents.forEach(content => content.classList.remove('active'));
    
    // Deactivate all navigation items
    const navItems = document.querySelectorAll('.nav-item');
    navItems.forEach(item => item.classList.remove('active'));
    
    // Show selected tab content
    document.getElementById(tabId).classList.add('active');
    
    // Activate corresponding navigation item
    if (tabId === 'predict-section') {
        document.getElementById('nav-predict').classList.add('active');
        document.getElementById('page-title').textContent = "Automated Approval System";
        document.getElementById('page-subtitle').textContent = "Input customer demographics & financial data to run optimized predictive risk assessment.";
    } else if (tabId === 'analytics-section') {
        document.getElementById('nav-analytics').classList.add('active');
        document.getElementById('page-title').textContent = "Model Performance Analytics";
        document.getElementById('page-subtitle').textContent = "Deep dive into accuracy, precision, recall, and hyperparameter tuning statistics.";
        // Refresh metrics if needed
        loadModelMetrics();
    } else if (tabId === 'pipeline-section') {
        document.getElementById('nav-pipeline').classList.add('active');
        document.getElementById('page-title').textContent = "End-to-End Pipeline";
        document.getElementById('page-subtitle').textContent = "Full technical schema outlining data collection, cleaning, scaling, and classification steps.";
    }
}

// Fetch model metrics on page load
document.addEventListener('DOMContentLoaded', () => {
    loadModelMetrics();
});

// Load Model Evaluation Metrics from Flask API
async function loadModelMetrics() {
    try {
        const response = await fetch('/api/metrics');
        if (!response.ok) {
            throw new Error("Could not fetch metrics from backend.");
        }
        const data = await response.json();
        
        // Populate analytics metrics card values
        document.getElementById('metric-accuracy').textContent = (data.accuracy * 100).toFixed(1) + '%';
        document.getElementById('metric-f1').textContent = (data.f1_score * 100).toFixed(1) + '%';
        document.getElementById('metric-roc-auc').textContent = data.roc_auc.toFixed(3);
        
        // Hyperparameters details
        const params = data.best_params;
        document.getElementById('metric-best-c').textContent = params.C;
        
        // Precision & Recall
        document.getElementById('metric-precision').textContent = (data.precision * 100).toFixed(1) + '%';
        document.getElementById('metric-recall').textContent = (data.recall * 100).toFixed(1) + '%';
        
        // Dataset Stats
        document.getElementById('stat-dataset-size').textContent = data.dataset_size;
        document.getElementById('stat-train-size').textContent = data.train_size;
        document.getElementById('stat-test-size').textContent = data.test_size;
        
        // Confusion Matrix Values
        const cm = data.confusion_matrix;
        // cm is [[TN, FP], [FN, TP]]
        document.querySelector('#cm-tn .cm-value').textContent = cm[0][0];
        document.querySelector('#cm-fp .cm-value').textContent = cm[0][1];
        document.querySelector('#cm-fn .cm-value').textContent = cm[1][0];
        document.querySelector('#cm-tp .cm-value').textContent = cm[1][1];
        
        // Render ROC Curve
        drawRocCurve(data.roc_curve);
        
    } catch (err) {
        console.error("Failed to load model metrics:", err);
    }
}

// Draw ROC Curve SVG path
function drawRocCurve(curveData) {
    if (!curveData || !curveData.fpr || !curveData.tpr) return;
    
    const fpr = curveData.fpr;
    const tpr = curveData.tpr;
    const pathElement = document.getElementById('roc-curve-path');
    
    // SVG chart dimensions mapping:
    // x ranges from 40 to 360 (0.0 to 1.0) -> width = 320
    // y ranges from 360 to 40 (0.0 to 1.0) -> height = 320
    let pathD = "M 40 360"; // Start at (0.0, 0.0) which is (40, 360)
    
    for (let i = 0; i < fpr.length; i++) {
        const x = 40 + (fpr[i] * 320);
        const y = 360 - (tpr[i] * 320);
        pathD += ` L ${x.toFixed(1)} ${y.toFixed(1)}`;
    }
    
    pathElement.setAttribute('d', pathD);
}

// Form Submission handling
async function submitApplication(e) {
    e.preventDefault();
    
    const form = document.getElementById('credit-form');
    const submitBtn = document.getElementById('btn-submit');
    const btnText = document.getElementById('btn-text');
    const spinner = document.getElementById('btn-spinner');
    
    // Toggle Loading State
    submitBtn.disabled = true;
    btnText.classList.add('hidden');
    spinner.classList.remove('hidden');
    
    // Collect Form Values
    const payload = {
        Gender: document.getElementById('Gender').value,
        Age: document.getElementById('Age').value,
        Debt: document.getElementById('Debt').value,
        Married: document.getElementById('Married').value,
        BankCustomer: document.getElementById('BankCustomer').value,
        EducationLevel: document.getElementById('EducationLevel').value,
        Ethnicity: document.getElementById('Ethnicity').value,
        YearsEmployed: document.getElementById('YearsEmployed').value,
        PriorDefault: document.getElementById('PriorDefault').value,
        Employed: document.getElementById('Employed').value,
        CreditScore: document.getElementById('CreditScore').value,
        DriversLicense: document.getElementById('DriversLicense').value,
        Citizen: document.getElementById('Citizen').value,
        ZipCode: document.getElementById('ZipCode').value,
        Income: document.getElementById('Income').value
    };
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error || "Prediction request failed.");
        }
        
        const result = await response.json();
        if (result.success) {
            renderPredictionResult(result.prediction, result.probability, payload);
        } else {
            alert("Error in prediction processing: " + result.error);
        }
        
    } catch (err) {
        console.error("Submission failed:", err);
        alert("An error occurred during verification: " + err.message);
    } finally {
        // Reset Loading State
        submitBtn.disabled = false;
        btnText.classList.remove('hidden');
        spinner.classList.add('hidden');
    }
}

// Render prediction result output visually
function renderPredictionResult(verdict, probability, inputs) {
    const card = document.getElementById('result-card');
    const placeholder = document.getElementById('result-placeholder');
    const content = document.getElementById('result-content');
    
    // Reveal main result layout
    placeholder.classList.add('hidden');
    card.classList.remove('placeholder-state');
    content.classList.remove('hidden');
    
    // Clear and set styling
    card.classList.remove('verdict-approved', 'verdict-denied');
    const verdictBox = document.getElementById('verdict-box');
    verdictBox.classList.remove('verdict-approved', 'verdict-denied');
    
    const verdictTitle = document.getElementById('verdict-title');
    const verdictIcon = document.getElementById('verdict-icon');
    
    // Decision Outcomes mapping
    if (verdict === 1) {
        card.classList.add('verdict-approved');
        verdictBox.classList.add('verdict-approved');
        verdictTitle.textContent = "APPROVED";
        verdictIcon.className = "fa-solid fa-circle-check";
    } else {
        card.classList.add('verdict-denied');
        verdictBox.classList.add('verdict-denied');
        verdictTitle.textContent = "REJECTED";
        verdictIcon.className = "fa-solid fa-circle-xmark";
    }
    
    // Animate radial confidence gauge
    // Circumference of radius 40 circle is 2 * pi * r = 251.2
    const fillElement = document.getElementById('gauge-fill');
    const textPercentage = document.getElementById('probability-value');
    
    const percentValue = Math.round(probability * 100);
    textPercentage.textContent = percentValue + '%';
    
    // For SVG stroke-dashoffset: 0 means full circle, 251.2 means empty circle
    const offset = 251.2 - (probability * 251.2);
    
    // Trigger smooth transition
    setTimeout(() => {
        fillElement.style.strokeDashoffset = offset;
    }, 100);
    
    // Generate Dynamic Risk Analysis Details based on features
    const riskBox = document.getElementById('risk-indicators');
    riskBox.innerHTML = ''; // Clear previous items
    
    // Prior Defaults check
    if (inputs.PriorDefault === 't') {
        createRiskItem(riskBox, "Prior defaults recorded in credit bureau (High Risk)", false);
    } else {
        createRiskItem(riskBox, "Clean credit history; no prior defaults", true);
    }
    
    // Income check
    const incomeNum = parseFloat(inputs.Income);
    if (incomeNum >= 3000) {
        createRiskItem(riskBox, `Annual income criteria met ($${incomeNum.toLocaleString()})`, true);
    } else {
        createRiskItem(riskBox, `Limited annual income ($${incomeNum.toLocaleString()})`, false);
    }
    
    // Debt Index check
    const debtNum = parseFloat(inputs.Debt);
    if (debtNum <= 5.0) {
        createRiskItem(riskBox, `Debt index is within safe parameters (${debtNum})`, true);
    } else {
        createRiskItem(riskBox, `High debt exposure index flagged (${debtNum})`, false);
    }
    
    // Credit Score check
    const creditNum = parseInt(inputs.CreditScore);
    if (creditNum >= 3) {
        createRiskItem(riskBox, `Active credit lines show positive rating (${creditNum} score)`, true);
    } else {
        createRiskItem(riskBox, `Low credit rating history (${creditNum} score)`, false);
    }
    
    // Years Employed check
    const yearsNum = parseFloat(inputs.YearsEmployed);
    if (yearsNum >= 1.5) {
        createRiskItem(riskBox, `Stable tenure of employment (${yearsNum} years)`, true);
    } else {
        createRiskItem(riskBox, `Short employment tenure recorded (${yearsNum} years)`, false);
    }
    
    // Scroll results into view if on mobile/small screen
    if (window.innerWidth <= 1024) {
        card.scrollIntoView({ behavior: 'smooth' });
    }
}

// Add risk bullet items helper
function createRiskItem(parent, text, isPassed) {
    const item = document.createElement('div');
    item.className = isPassed ? "risk-item pass" : "risk-item fail";
    
    const icon = document.createElement('i');
    icon.className = isPassed ? "fa-solid fa-circle-check" : "fa-solid fa-circle-exclamation";
    
    const label = document.createElement('span');
    label.textContent = text;
    
    item.appendChild(icon);
    item.appendChild(label);
    parent.appendChild(item);
}
