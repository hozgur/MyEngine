
window.chrome.webview.addEventListener('message', onMessage);

function onMessage(msg) {
    console.log(msg.data)
    let p = document.createElement("p");
    p.innerText = msg.data + "\n";
    let msgwnd = document.getElementById("message");
    msgwnd.appendChild(p);
    p.scrollIntoView(false);
}

