document.addEventListener('DOMContentLoaded', () => {
    
    const predictionForm = document.getElementById('prediction-form');
    const resetButton = document.getElementById('reset-button');
    const resultsContainer = document.getElementById('results-container');
    const predictionContainer = document.getElementById('prediction-container');
    const predictionResult = document.getElementById('prediction-result');
    const modelUsed = document.getElementById('model-used');
    const probabilityContainer = document.getElementById('probability-container');
    const approvedBar = document.getElementById('approved-bar');
    const rejectedBar = document.getElementById('rejected-bar');
    const approvedValue = document.getElementById('approved-value');
    const rejectedValue = document.getElementById('rejected-value');
    const modelInfoContent = document.getElementById('model-info-content');
    
    
    fetchModelInfo();
    
    
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        
        document.getElementById('predict-button').disabled = true;
        document.getElementById('predict-button').textContent = 'Predicting...';
        resultsContainer.innerHTML = '<p class="no-results">Processing, please wait...</p>';
        predictionContainer.classList.add('hidden');
        
        
        const formData = new FormData(predictionForm);
        const jsonData = {};
        
        
        for (const [key, value] of formData.entries()) {
            jsonData[key] = value;
        }
        
        try {
            
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(jsonData)
            });
            
            const data = await response.json();
            
            if (data.error) {
                
                resultsContainer.innerHTML = `<p class="error">Error: ${data.error}</p>`;
                predictionContainer.classList.add('hidden');
            } else {
                
                resultsContainer.innerHTML = '';
                predictionContainer.classList.remove('hidden');
                
                
                const isPredictionApproved = data.prediction === 'Approved';
                predictionResult.className = 'prediction-card ' + (isPredictionApproved ? 'approved' : 'rejected');
                predictionResult.textContent = data.prediction;
                
                
                modelUsed.textContent = `Model: ${getModelName(data.model_used)}`;
                
                
                if (data.probability) {
                    probabilityContainer.classList.remove('hidden');
                    
                    const approvedProb = data.probability.Approved * 100;
                    const rejectedProb = data.probability.Rejected * 100;
                    
                    
                    approvedBar.style.width = `${approvedProb}%`;
                    rejectedBar.style.width = `${rejectedProb}%`;
                    
                    
                    approvedValue.textContent = `${approvedProb.toFixed(1)}%`;
                    rejectedValue.textContent = `${rejectedProb.toFixed(1)}%`;
                } else {
                    probabilityContainer.classList.add('hidden');
                }
            }
        } catch (error) {
            resultsContainer.innerHTML = `<p class="error">Error: ${error.message}</p>`;
        } finally {
            
            document.getElementById('predict-button').disabled = false;
            document.getElementById('predict-button').textContent = 'Predict Loan Approval';
        }
    });
    
    
    resetButton.addEventListener('click', () => {
        predictionForm.reset();
        resultsContainer.innerHTML = '<p class="no-results">Fill the form and click Predict to see results</p>';
        predictionContainer.classList.add('hidden');
    });
    
    
    function getModelName(modelKey) {
        const modelNames = {
            'knn': 'K-Nearest Neighbors',
            'decision_tree': 'Decision Tree',
            'random_forest': 'Random Forest'
        };
        return modelNames[modelKey] || modelKey;
    }
    
    
    async function fetchModelInfo() {
        try {
            const response = await fetch('/model_info');
            const data = await response.json();
            
            let infoHTML = '';
            
            for (const [key, model] of Object.entries(data.models)) {
                const isAvailable = data.available[key];
                infoHTML += `
                    <div class="model-card">
                        <h4>${model.name} 
                            <span class="model-available ${isAvailable ? 'available' : 'unavailable'}">
                                ${isAvailable ? 'Available' : 'Unavailable'}
                            </span>
                        </h4>
                        <p>${model.description}</p>
                    </div>
                `;
            }
            
            modelInfoContent.innerHTML = infoHTML;
        } catch (error) {
            modelInfoContent.innerHTML = `<p class="error">Error loading model information: ${error.message}</p>`;
        }
    }
});