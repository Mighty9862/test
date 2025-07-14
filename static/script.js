document.addEventListener('DOMContentLoaded', () => {
    const chatMessages = document.getElementById('chat-messages');
    const userInput = document.getElementById('user-input');
    const fileInput = document.getElementById('file-input');
    const sendButton = document.getElementById('send-button');
    const sendFileButton = document.getElementById('send-file-button');
    const loadingIndicator = document.getElementById('loading-indicator');

    sendButton.addEventListener('click', sendMessage);
    sendFileButton.addEventListener('click', sendFile);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = text;
        chatMessages.appendChild(userMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        userInput.value = '';
        sendButton.disabled = true;
        loadingIndicator.classList.remove('hidden');

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

            if (!data.categories || !Array.isArray(data.categories)) {
                botMessage.innerHTML = '<p>Ошибка: сервер не вернул категории.</p>';
            } else {
                let resultText = '<strong>Найденные ценности:</strong><ul>';
                data.categories.forEach(cat => {
                    resultText += `<li>${cat.name}: ${cat.probability.toFixed(2)}%</li>`;
                });
                resultText += '</ul>';
                botMessage.innerHTML = resultText;
            }

            chatMessages.appendChild(botMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            const errorMessage = document.createElement('div');
            errorMessage.className = 'bot-message';
            errorMessage.textContent = `Ошибка: ${error.message}`;
            chatMessages.appendChild(errorMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        loadingIndicator.classList.add('hidden');
        sendButton.disabled = false;
    }

    async function sendFile() {
        const file = fileInput.files[0];
        if (!file) return;

        const userMessage = document.createElement('div');
        userMessage.className = 'user-message';
        userMessage.textContent = `📄 Загружен файл: ${file.name}`;
        chatMessages.appendChild(userMessage);
        chatMessages.scrollTop = chatMessages.scrollHeight;

        sendFileButton.disabled = true;
        loadingIndicator.classList.remove('hidden');

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.detail || 'Ошибка при анализе файла');
            }

            const data = await response.json();

            const botMessage = document.createElement('div');
            botMessage.className = 'bot-message';

            if (!data.categories || !Array.isArray(data.categories)) {
                botMessage.innerHTML = '<p>Ошибка: сервер не вернул категории.</p>';
            } else {
                let resultText = '<strong>Результаты анализа файла:</strong><ul>';
                data.categories.forEach(cat => {
                    resultText += `<li>${cat.name}: ${cat.probability.toFixed(2)}%</li>`;
                });
                resultText += '</ul>';
                botMessage.innerHTML = resultText;
            }

            chatMessages.appendChild(botMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        } catch (error) {
            const errorMessage = document.createElement('div');
            errorMessage.className = 'bot-message';
            errorMessage.textContent = `Ошибка: ${error.message}`;
            chatMessages.appendChild(errorMessage);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        loadingIndicator.classList.add('hidden');
        sendFileButton.disabled = false;
    }
});
