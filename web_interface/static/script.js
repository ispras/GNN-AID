let socket;
let sessionId;

function openNewTab(url) {
    window.open(url, '_blank');
    // window.open(window.location.href, '_blank');
}

document.addEventListener('DOMContentLoaded', (event) => {
    console.log('DOMContentLoaded event')
    socket = io();

    socket.on('connect', () => {
        console.log('Connected to server');
    });

    socket.on('disconnect', () => {
        console.log('Disconnected from server');
        // document.getElementById('connection-status').textContent = 'Disconnected';
        clearInterval(1000);
    });

    socket.on('session_id', (data) => {
        sessionId = data.session_id;
        document.getElementById('session-id').textContent = sessionId;
        startHeartbeat();
    });

    socket.on('heartbeat_response', (data) => {
        document.getElementById('server-time').textContent =
            new Date(data.server_time * 1000).toLocaleTimeString();
    });

    updateClientTime();
    setInterval(updateClientTime, 1000);
});

function startHeartbeat() {
    setInterval(() => {
        socket.emit('heartbeat', { session_id: sessionId })
    }, 10000);
}

function updateClientTime() {
    document.getElementById('client-time').textContent = new Date().toLocaleTimeString();
}

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (socket) {
        socket.disconnect();
    }
})