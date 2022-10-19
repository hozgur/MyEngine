window.chrome.webview.addEventListener('message', onMessage);

function onMessage(msg) {
    console.log(msg.data)
    let ipath = document.getElementById("ipath");
    ipath.value = JSON.parse(msg.data).msg;    
}