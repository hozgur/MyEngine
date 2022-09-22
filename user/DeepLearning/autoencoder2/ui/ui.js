const gui = new dat.GUI();

const parameters = {    
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
    for (item in parameters)
        gui.add(parameters, item, -200, 200).onChange(function (newValue) { post(this.property, newValue.toString()) });
    
    
}

// add onload event listener
window.addEventListener('load', onLoad);

