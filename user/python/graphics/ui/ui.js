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
const paramCount = 16;
function onLoad() {
    for (let i = 0; i < paramCount; i++)
        parameters["param" + i] = 0
    gui.add(parameters, 'reload').name('Reload Python Module');
    gui.add(parameters, 'save').name('Save Model');
    gui.add(parameters, 'load').name('Load Model');
    gui.addColor(parameters, 'color').name('Color').onChange(function (value) {        
        post('color', JSON.stringify(value));
    });
    gui.add(parameters, 'brushSize', 1, 50).name('Brush Size').onChange(function (value) {
        post('brushSize', JSON.stringify(value));
    });
}

// add onload event listener
window.addEventListener('load', onLoad);

