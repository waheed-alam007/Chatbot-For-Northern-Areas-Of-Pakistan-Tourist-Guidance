document.getElementById('send-btn').addEventListener('click', function() {
    const userInput = document.getElementById('user-input').value;
    if (userInput.trim() === "") return;

    // Display user's message in the chatbox
    appendMessage(userInput, 'user');

    // Send the message to Flask backend for processing
    fetch('/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `question=${encodeURIComponent(userInput)}`
    })
    .then(response => response.json())
    .then(data => {
        // Display the bot's response in the chatbox
        if (data.response) {
            appendMessage(data.response, 'bot');
        } else {
            appendMessage(data.error || "Sorry, there was an error. Please try again.", 'bot');
        }
    })
    .catch(error => {
        appendMessage("Sorry, there was an error. Please try again later.", 'bot');
    });

    // Clear input field
    document.getElementById('user-input').value = '';
});

// Function to append messages to the chatbox
function appendMessage(message, sender) {
    const chatBox = document.getElementById('chat-box');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-message' : 'bot-message');
    messageDiv.textContent = message;
    chatBox.appendChild(messageDiv);

    // Auto-scroll to the bottom of the chatbox
    chatBox.scrollTop = chatBox.scrollHeight;
}
