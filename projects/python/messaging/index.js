window.chrome.webview.addEventListener('message', onMessage);
window.addEventListener('load', onLoad);

function onMessage(msg) {
    console.log(msg.data)
    let ipath = document.getElementById("ipath");
    ipath.value = JSON.parse(msg.data).msg;    
}

function post(id, message) {
    window.chrome.webview.postMessage(JSON.stringify({ id: id.toString(), message: message }))
}


function onLoad() {
    post("ready", "hello");
    document.getElementById("loadimage").onclick = onclickLoadImage;
}


function onclickLoadImage() {
    post("load", document.getElementById("ipath").value)
    if (document.getElementById("image"))
        return;

    const img = document.createElement("img");
    img.src = "./" + document.getElementById("ipath").value;
    img.id = "image";
    img.style = "max-width:100px;max-height:100px";
    img.classList.add("right");
    document.body.appendChild(img)
}