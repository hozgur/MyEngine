const gui = new dat.GUI();

const parameters = {
    color: [255, 0, 0],
    brushSize: 5,
    reload: function () { post('reload', '') },
    save:   function () { post('save', '') },
    load:   function () { post('load', '') }
};

function post(id, message) {
    window.chrome.webview.postMessage(JSON.stringify({ id: id, message: message }))
}
function onLoad() {    
    gui.add(parameters, 'reload').name('Reload Python Module');
    gui.add(parameters, 'save').name('Save Model');
    gui.add(parameters, 'load').name('Load Model');    
}

// add onload event listener
window.addEventListener('load', onLoad);

