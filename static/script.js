document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // Отображение сообщения пользователя
        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = text;
        chatMessages.appendChild(userMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        userInput.value = '';
        sendButton.disabled = true;

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });

            if (!response.ok) {
                throw new Error('Ошибка сервера');
            }

            const data = await response.json();
            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message';
            
            let resultText = '<strong>Найденные ценности:</strong><ul>';
            data.categories.forEach(cat => {
                resultText += `<li>${cat.name}: ${cat.probability.toFixed(2)}%</li>`;
            });
            resultText += '</ul>';
            resultText += `<p><strong>Пояснение:</strong> ${data.explanation}</p>`;
            resultText += '<p><strong>Ключевые слова:</strong> ' + data.keywords.map(k => `${k.word} (${k.score.toFixed(2)})`).join(', ') + '</p>';

            botMessage.innerHTML = resultText;
            chatMessages.appendChild(botMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            const errorMessage = document.createElement('div');
            errorMessage.className = 'bot-message';
            errorMessage.textContent = `Ошибка: ${error.message}`;
            chatMessages.appendChild(errorMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        sendButton.disabled = false;
    }
});