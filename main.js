const {app, BrowserWindow} = require('electron')

  // Keep a global reference of the window object, if you don't, the window will
  // be closed automatically when the JavaScript object is garbage collected.
  let win

  function createWindow () {
    win = new BrowserWindow({width: 800, height: 600, webPreferences:{nodeIntegration:true}})

    win.setMenu(null)
    win.loadFile('index.html')

    //win.webContents.openDevTools()

    win.on('closed', () => {
      win = null
    })
  }

  app.on('ready', createWindow)

  app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
      app.quit()
    }
  })

  app.on('activate', () => {
    if (win === null) {
      createWindow()
    }
  })


const path=require('path')

let pyProc = null
let pyPort = null


const createPyProc = () => {
  let script = path.join(__dirname, 'py', 'thrift_cv_server.py') //thrift_server
  pyProc = require('child_process').spawn('python', [script])
  if (pyProc != null) {
    console.log('child process success')
  }

  pyProc.stdout.on('data', data => {
      console.log('Python: '+data.toString())
  })
}


const exitPyProc = () => {
  pyProc.kill()
  pyProc = null
  pyPort = null
}

app.on('ready', createPyProc)
app.on('will-quit', exitPyProc)