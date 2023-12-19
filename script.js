function predict() {
    var formData = new FormData(document.getElementById('upload-form'));
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('prediction-result').innerText = 'Prediction: ' + data.prediction;
        document.getElementById('feedback-section').style.display = 'block';
    })
    .catch(error => console.error('Error:', error));
}

function submitFeedback(event) {
    event.preventDefault();
    var formData = new FormData(document.getElementById('feedback-form'));
    fetch('/feedback', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('feedback-message').innerText = data.message;
    })
    .catch(error => console.error('Error:', error));
}