module.exports = {
  requires: {
    bundle: "ai",
  },
  run: [
    {
      when: "{{!exists('app')}}",
      method: "shell.run",
      params: {
        message: [
          "git clone --filter=blob:none --sparse https://github.com/Adutchguy/roop-unleashed-wip.git _app_tmp && git -C _app_tmp sparse-checkout set app && mv _app_tmp/app app && rm -rf _app_tmp"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    }
  ]
}
