window.chrome.webview.addEventListener('message', onMessage);
window.addEventListener('load', onLoad);


function loadThumb(path) {
    const img = document.createElement("img");
    img.src = path
    img.id = "image";
    img.style = "max-width:100px;max-height:100px";
    img.classList.add("right");
    document.body.appendChild(img)
}

messageHandlers = {
    "path": msg => document.getElementById("ipath").value = msg,
    "load": loadThumb
}

function onMessage(msg) {
    data = JSON.parse(msg.data);
    console.log(data);
    messageHandlers[data.id](data.msg);
}

function post(id, message) {
    window.chrome.webview.postMessage(JSON.stringify({ id: id.toString(), message: message }))
}


function onLoad() {
    post("ready", "web ready");
    document.getElementById("loadimage").onclick = onclickLoadImage;
}


function onclickLoadImage() {    
    post("load", document.getElementById("ipath").value)
}