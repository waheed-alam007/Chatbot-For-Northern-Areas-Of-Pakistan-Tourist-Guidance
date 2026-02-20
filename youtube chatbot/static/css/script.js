// 

async function sendMessage() {
  const input = document.getElementById("user-input");
  const message = input.value.trim();
  if (!message) return;

  const chatBox = document.getElementById("chat-box");

  // Show user message
  const userDiv = document.createElement("div");
  userDiv.className = "user-message";
  userDiv.textContent = "You: " + message;
  chatBox.appendChild(userDiv);

  input.value = "";

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ question: message })
    });

    const data = await response.json();

    const botDiv = document.createElement("div");
    botDiv.className = "bot-message";
    botDiv.textContent = "Bot: " + data.answer;
    chatBox.appendChild(botDiv);

    chatBox.scrollTop = chatBox.scrollHeight;
  } catch (err) {
    console.error("Error:", err);
  }
}
