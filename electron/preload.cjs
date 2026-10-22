const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('zemCleanliness', {
  request: (method, route, body, requestId) => ipcRenderer.invoke('cleanliness:request', method, route, body, requestId),
  chooseOpticalJson: () => ipcRenderer.invoke('cleanliness:chooseOpticalJson'),
  openPath: (target) => ipcRenderer.invoke('cleanliness:openPath', target),
  getRuntime: () => ipcRenderer.invoke('cleanliness:getRuntime'),
  getAcceptanceTest: () => ipcRenderer.invoke('cleanliness:getAcceptanceTest'),
  log: (level, event, details) => ipcRenderer.invoke('cleanliness:log', level, event, details)
})
