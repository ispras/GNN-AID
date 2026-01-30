let sessionId = null;
let socket = null;

function startSession() {
    fetch('/start-session', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Failed to start session:', data.error);
                return;
            }
            sessionId = data.sessionId;
            console.log('Session started:', sessionId);

            // Establish WebSocket connection
            socket = new WebSocket('ws://localhost:8765');

            socket.onopen = function(e) {
                console.log("WebSocket connection established");
            };

            socket.onmessage = function(event) {
                let data = JSON.parse(event.data);
                console.log('Received from server:', data.result);
                // Handle the received data as needed
            };

            socket.onclose = function(event) {
                if (event.wasClean) {
                    console.log(`Connection closed cleanly, code=${event.code}, reason=${event.reason}`);
                } else {
                    console.log('Connection died');
                }
            };

            socket.onerror = function(error) {
                console.log(`WebSocket error: ${error.message}`);
            };
        });
}

function sendMessage(message) {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ sessionId: sessionId, message: message }));
    } else {
        console.log('WebSocket is not connected');
    }
}

function endSession() {
    if (sessionId) {
        if (socket) {
            socket.close();
        }
        navigator.sendBeacon('/end-session', JSON.stringify({ sessionId: sessionId }));
    }
}

window.addEventListener('load', startSession);
window.addEventListener('beforeunload', endSession);