module.exports = {
  run: [{
    // Update launcher scripts
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    // Remove existing app folder so we can re-fetch the latest from the remote
    when: "{{exists('app')}}",
    method: "fs.rm",
    params: {
      path: "app"
    }
  }, {
    // Re-fetch app code via sparse checkout (app is not a standalone git repo)
    method: "shell.run",
    params: {
      message: [
        "git clone --filter=blob:none --sparse https://github.com/Adutchguy/roop-unleashed-wip.git _app_tmp && git -C _app_tmp sparse-checkout set app && mv _app_tmp/app app && rm -rf _app_tmp"
      ]
    }
  }, {
    method: "shell.run",
    params: {
      venv: "env",
      path: "app",
      message: "uv pip install -r requirements.txt"
    }
  }, {
    method: "script.start",
    params: {
      uri: "torch.js",
      params: {
        venv: "env",
        path: "app",
      }
    }
  }]
}
