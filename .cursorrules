# Development Guide for Pinokio Projects

## Non-Negotiable Execution Workflow

To guarantee every contribution follows this guide precisely, obey this checklist **before any edits** and **again before finalizing**. Do not skip or reorder.
1. **AGENTS Snapshot:** Re-open this file and write down (in your working notes or response draft) the exact sections relevant to the requested task. No work begins until this snapshot exists.
2. **Example Lock-in:** Identify the closest matching script in `C:\pinokio\prototype\system\examples`. Record its path and keep it open while editing. Every launcher change must mirror that reference unless the user explicitly instructs otherwise.
3. **Pre-flight Checklist:** Convert the applicable rules from this document and `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md into a task-specific checklist (install/start/reset/update structure, regex patterns, menu defaults, log checks, etc.). Confirm each item is ticked **before** making changes.
4. **Mid-task Verification:** Any time you touch a Pinokio script, cross-check the corresponding example line to ensure syntax and structure match. Document the reference (example path + line) in your reasoning.
5. **Exit Checklist:** Before responding to the user, revisit the pre-flight checklist and explicitly confirm every item is satisfied. If anything diverges from the example or these rules, fix it first.

If any step cannot be completed, stop immediately and ask the user how to proceed. These five steps are mandatory for every session.

### Critical Pattern Lock: Capturing Web UI URLs

When writing `start.js` (or any script that needs to surface a web URL for a server):

1. **Always copy the capture block from an example such as `system/examples/mochi/start.js`.**
```javascript
on: [{
  event: "/(http:\\/\\/[0-9.:]+)/",
  done: true
}]
```

2. **Set the local variable using the captured match exactly as below (The regex capture object is passed in as `input.event`, so need to use the index 1 inside the parenthesis):**
```javascript
{
  method: "local.set",
  params: {
    url: "{{input.event[1]}}"
  }
}
```

3. Always try to come up with the most generic regex.
4. During the exit checklist, explicitly confirm that the `url` local variable is set via `local.set` API by using the captured regex object as passed in as `input.event` from the previous `shell.run` step.

Deviation from this pattern requires written approval from the user.

- Make sure to keep this entire document and `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md in memory with high priority before making any decision. Pinokio is a system that makes it easy to write launchers through scripting by providing various cross-platform APIs, so whenever possible you should prioritize using Pinokio API over lower level APIs.
- When writing pinokio scripts, ALWAYS check the examples folder (in C:\pinokio\prototype\system\examples folder) to see if there are existing example scripts you can imitate, instead of assuming syntax.
- When implementing pinokio script APIs and you cannot infer the syntax just based on the examples, always search the API documentation `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md to use the correct syntax instead of assuming the syntax.
- When trying to fix something or figure out what's going on, ALWAYS start by checking the `logs` folder before doing anything else, as mentioned in the "Troubleshooting with Logs" section.
- Finally, make sure to ALWAYS follow all the items in the "best practices" section below.

## Determine User Intent
If the initial prompt is simply a URL and nothing else, check the website content and determine the intent, and ask the user to confirm. For example a URL may point to

1. A Tutorial: the intent may be to implement a demo for the tutorial and build a launcher.
2. A Demo: the intent may be a 1-click launcher for the demo
3. Open source project: the intent may be a 1-click launcher for the project 
4. Regular website: the intent may be to clone the website and a launcher.
5. There can be other cases, but try to guess.

## Project Structure

Pinokio projects normally follow a standardized structure with app logic separated from launcher scripts:

Pinokio projects follow a standardized structure with app logic separated from launcher scripts:

```
project-root/
├── app/                 # Self-contained app logic (can be standalone repo)
│   ├── package.json     # Node.js projects
│   ├── requirements.txt # Python projects
│   └── ...              # Other language-specific files
├── README.md            # Documentation
├── install.js           # Installation script
├── start.js             # Launch script
├── update.js            # Update script (for updating the scripts and app logic to the latest)
├── reset.js             # Reset dependencies script
├── pinokio.js           # UI generator script
└── pinokio.json         # Metadata (title, description, icon)
```

- Keep app code in `/app` folder only (never in root)
- Store all launcher files in project root (never in `/app`)
- `/app` folder should be self-contained and publishable


The only exceptions are serverless web apps---purely frontend only web applications that do NOT have a server component and connect to 3rd party API endpoints--in which case the folder structure looks like the following (No need for launcher scripts since the index.html will automatically launch. The only thing needed is the metadata file named pinokio.json):

```
project-root/
├── index.html           # The serverless web app entry point
├── ...
├── README.md            # Documentation
└── pinokio.json         # Metadata (title, description, icon)
```

IMPORTANT: ALWAYS try to follow the best practices in the examples folder (C:\pinokio\prototype\system\examples) instead of trying to come up with your own structure. The examples have been optimized for the best user experience.

## Launcher Project Working Directory

- The project working directory for a script is always the same directory as the script location.
- For example, when you run `shell.run` API inside `pinokio/start.js`, the default path for shell execution is `pinokio`.
- If the launcher files are in the project root path, then the default path for shell execution is the project root.
- Therefore, it is important to specify the correct `path` attribute when running `shell.run` API commands.

Example: in the following project structure:

```
project-root/
├── pinokio/                 # Pinokio launcher folder
│    ├── start.js             # Launch script
│    ├── pinokio.js           # UI generator script
│    └── pinokio.json         # Metadata (title, description, icon)
└─── backend/
     ├── requirements.txt          # App dependencies
     └── app.py                    # App code
```

The `pinokio/start.js` should use the correct path `../backend` as the `path` attribute, as follows:

```
{
  run: [{
    ...
  }, {
    method: "shell.run",
    params: {
      message: "python app.py",
      venv: "env",
      path: "../backend"
    }
  }, {
    ...
  }]
}
```

## Development Workflow

### 1. Understanding the Project
- Check `SPEC.md` in project root. If the file exists, use that to learn about the project details (what and how to build)
- If no `SPEC.md` exists, build based on user requirements
### 2. Modifying Existing Launcher Projects
If we are starting with existing launcher script files, work with the existing files instead of coming up with your own.
- **Preserve existing functionality:** Only modify necessary parts
- **Don't touch working scripts:** Unless adding/updating specific commands
- **Follow existing conventions:** Match the style and structure already present
### 3. Try to adopt from examples as much as possible
- If starting from scratch, first determine what type of project you will be building, and then check the examples folder (C:\pinokio\prototype\system\examples) to see if you can adopt them instead of coming up everything from scratch.
- Even if there are no relevant examples, check the examples to get inspiration for how you would structure the script files even if you have to write from scratch.
### 4. Writing from scratch as a last resort
If there are relevant examples to adopt from, write the scripts from scratch, but just make sure to follow the requirements in the next section.
### 5. Debugging
When the user reports something is not working, ALWAYS inspect the logs folder to get all the execution logs. For more info on how this works, check the "Troubleshooting with Logs" section below.

## Script Requirements

### 1. 1-click launchable
- The main purpose of Pinokio is to provide an easy interface to invoke commands, which may include launching servers, installing programs, etc. Make sure the final product provides ways to install, launch, reset, and update whatever is needed.

### 2. Write Documentation
- ALWAYS write a documentation. A documentation must be stored as `README.md` in the project root folder, along with the rest of the pinokio launcher script files. A documentation file must contain:
  - What the app does
  - How to use the app
  - API documentation for programmatically accessing the app's main features (Javascript, Python, and Curl)

## Types of launchers
## 1. Launching servers
- When an app requires launching a server, here are the commonly used scripts:
  - `install.js`: a script to install the app
  - `start.js`: a script to start the app
  - `reset.js`: a script to reset all the dependencies installed in the `install.js` step. used if the user wants to restart from scratch
  - `update.js`: a script to update the launcher AND the app in case there are new updates. Involves pulling in the relevant git repositories installed through `install.js` (often it's the script repo and some git repositories cloned through the install steps if any)
  - `pinokio.js`: the launcher script that ties all of the above scripts together by providing a UI that links to these scripts.
  - `pinokio.json`: For metadata

Here's a basic server launcher script example (`start.js`). Unless there's a special reason you need to use another pattern, this is the most recommended pattern. Use this or adopt it as needed, but NEVER try something else unless there's a good reason you should not take this approach:

```javascript
module.exports = {
  // By setting daemon: true, the script keeps running even after all items in the `run` array finishes running. Mandatory for launching servers, since otherwise the shells running the server process will get killed after the scripts finish running.
  daemon: true,
  run: [
    {
      // The "shell.run" API for running a shell session
      method: "shell.run",
      params: {
        // Edit 'venv' to customize the venv folder path
        venv: "env",
        // Edit 'env' to customize environment variables (see documentation)
        env: { },
        // Edit 'path' to customize the path to start the shell from
        path: "app",
        // Edit 'message' to customize the commands, or to run multiple commands
        message: [
          "python app.py",
        ],
        on: [{
          // The regular expression pattern to monitor.
          // Whenever each "event" pattern occurs in the shell terminal, the shell will return,
          // and the script will go onto the next step.
          // The regular expression match object will be passed on to the next step as `input.event`
          // Useful for capturing the URL at which the server is running (in case the server prints some message about where the server is running)
          "event": "/(http:\/\/\\S+)/", 

          // Use "done": true to move to the next step while keeping the shell alive.
          // Use "kill": true to move to the next step after killing the shell.
          "done": true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        // the input.event is the regular expression match object from the previous step
        // In this example, since the pattern was "/(http:\/\/\\S+)/", input.event[1] will include the exact http url match caputred by the parenthesis.
        // Therefore setting the local variable 'url'
        url: "{{input.event[1]}}"
      }
    }
  ]
}
```

## 2. Launching serverless web apps

- In case of purely static web apps WITHOUT servers or backends (for example an HTML based app that connects to 3rd party servers--either remote or localhost), we do NOT need the launcher scripts.
- In these cases, simply include `index.html` in the project root folder and everything should automatically work. No need for any of the pinokio launcher scripts. (Do 
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## 3. Launching quick scripts without web UI

- In many cases, we may not even need a web UI, but instead just a simple way to run scripts.
- This may include TUI (Terminal User Interface) apps, a simple launcher 
- In these cases, all we need is the launcher file `pinokio.js`, which may link to multiple scripts. In this case, there are no web apps (no serverless apsp, no servers), but instead just the default pinokio launcher UI that calls a bunch of scripts.
- Here are some examples:
  - A pinokio script to toggle the desktop theme between dark and light
    - Write some code (python or javascript or whatever)
    - Write a `toggle.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `toggle.js` so the user can simply click the "toggle" button to toggle back and forth between desktop themes
  - A pinokio script to fetch some file
    - Write some code (python or javascript or whatever)
    - Write a `fetch.js` pinokio script that executes the code
    - Write a `pinokio.js` launcher script to create a sidebar UI that displays the `fetch.js` so the user can simply click the "fetch" button to fetch some data.
- You still need to include the metadata file so they show up properly on pinokio:
  - `pinokio.json`: For metadata

## API

This section lists all the script APIs available on Pinokio. To learn the details of how they are used, you can:
1. Check the examples in the C:\pinokio\prototype\system\examples folder
2. Read the `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md further documentation on the full syntax

### Script API

These APIs can be used to describe each step in a pinokio script:
- shell.run: run shell commands
- input: accept user input
- filepicker: accept file upload
- fs.write: write to file
- fs.read: read from file
- fs.copy: copy files
- fs.download: download files
- fs.link: create a symbolic link (or junction on windows) for folders
- fs.open: open the system file explorer at a given path
- fs.cat: print file contents
- jump: jump to a specific step
- local.set: set local variables for the currently running script
- json.set: update a json file
- json.rm: remove keys from a json file
- json.get: get values from a json file
- log: print to the web terminal
- net: make network requests
- notify: display a notification
- script.download: download a script from a git uri
- script.start: start a script
- script.stop: stop a script
- script.return: return values if the current script was called by a caller script, so the caller script can utilize the return value as `input`
- web.open: open a url in web browser
- hf.download: huggingfac-cli download API
### Template variables
The following variables are accessible inside template expressions (example `{{args.command}` in scripts, resulting in dynamic behaviors of scripts:
- input: An input is a variable that gets passed from one RPC call to the next
- args: args is the parameter object that gets passed into the script (via pinokio.js `params`). Unlike `input` which takes the value passed in from the immediately previous step, `args` is a global value that is the same through out the entire script execution.
- local: local variable object that can be set with `local.set` API
- self: refers to the script file itself (which is JSON or JavaScript). For example if `start.js` that's currently running has `daemon: true` set, `{{self.daemon}}` will evaluate to true.
- uri: The current script uri
- port: The next available port. Very useful when you need to launch an app at a specific port without port conflicts.
- cwd: The current script execution folder path
- platform: The current operating system. May be one of the following: `darwin`, `win32`, `linux`
- arch: The current system architecture. May be one of the following: x32, x64, arm, arm64, s390, s390x, mipsel, ia32, mips, ppc, ppc64
- gpus: array of available GPUs on the machine (example: `['apple']`, `['nvidia']`)
- gpu: the first available GPU (example: `nvidia`)
- current: The current variable points to the index of the currently executing instruction within the run array.
- next: The next variable points to the index of the next instruction to be executed. (null if the current instruction is the final instruction in the run array)
- envs: You can access the environment variables of the currently running process with envs object.
- which: Check whether a command exists (example: `{{which('winget')}}`. Can be used in the `when` attribute of a script step to run commands or install first.
- exists: Check whether a file or folder exists at the specified relative path (example: `"when": "{{!exists('app')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- running: Check whether a script file is running (example: `"when": "{{!running('start.js')}}"`). Can be used with the `when` attribute to determine a path's existence and trigger custom logic. Use relative paths and it will resolve automatically to the current execution folder. 
- os: Pinokio exposes the node.js os module through the os variable.
- path: Pinokio exposes the node.js path module through the os variable (example: `{{path.resolve(...)}}`

## System Capabilities
### Package Management (Use in Order of Preference)
The following package managers come pre-installed with Pinokio, so whenever you need to install a 3rd party binary, remember that these are available. Also, you can assume these are available and include the following package manager commands in Pinokio scripts:
1. **UV** - For Python packages (preferred over pip)
2. **NPM** - For Node.js packages  
3. **Conda** - For cross-platform 3rd party binaries
4. **Brew** - Mac-only fallback when other options unavailable
5. **Git** - Full access to git is available.
**Important:** Include all install commands in the install script for reproducibility.
### HTTPS Proxy Support
- All HTTP servers automatically get HTTPS endpoints
- Convention: `http://localhost:<PORT>` → `https://<PORT>.localhost`
- Full proxy list available at: `http://localhost:2019/config/`
### Pterm Features:
- **Clipboard Access:** Read from or Write to system clipboard via pinokio Pterm CLI (`pterm clipboard` command.)
- **Notifications:** Send desktop alerts via pinokio pterm CLI (`pterm push` command.)
- **Script Testing:** Run launcher scripts via pinokio pterm CLI (`pterm start` command.)
- **File Selection:** Use built-in filepicker for user file/folder input (`pterm filepicker` command.)
- **Git Operations:** Clone repositories, push to GitHub
- **GitHub Integration:** Full GitHub CLI support (`gh` commands)

## Troubleshooting with Logs
Pinokio stores the logs for everything that happened in terminal at the following locations, so you can make use of them to determine what's going on:

### Log Structure
In case there is a `pinokio` folder in the project root folder, you should be able to find the logs folder here:

```
pinokio/
└── logs/   # Direct user interaction logs
    ├── api/     # Launcher script logs (install.js, start.js, etc.)
    ├── dev/     # AI coding tool logs (organized by tool)
    └── shell/   # Direct user interaction logs
```

Otherwise, the `logs` folder should be found at project root:

```
logs/
├── api/     # Launcher script logs (install.js, start.js, etc.)
├── dev/     # AI coding tool logs (organized by tool)
└── shell/   # Direct user interaction logs
```

### Log File Naming
- Unix timestamps for each session
- Special "latest" file contains most recent session logs
- **Default:** Use "latest" files for current issues
- **Historical:** Use timestamped files for pattern analysis and the full history.

## Best practices
### 0. Always reference the logs when debugging
- When the user asks to fix something, ALWAYS check the logs folder first to check what went wrong. Check the "Troubleshooting with Logs" section.
### 1. Shell commands for launching programs
- Launch flags related
  - Try as hard as possible to minimize launch flags and parameters when launching an app. For example, instead of `python app.py --port 8610`, try to do `python app.py` unless really necessary. The only exception is when the only way to launch the app is to specify the flags.
- Launch IP related
  - Always try to find a way to launch servers at 127.0.0.1 or localhost, often by specifying launch flags or using environment variables. Some apps launch apps at 0.0.0.0 by default but we do not want this.
- Launch Port related
  - In case the app itself automatically launches at the next available port by default (for example Gradio does this), do NOT specify port, since it's taken care of by the app itself. Always try to minimize the amount of code.
  - If the install instruction says to launch at a specific port, don't use the hardcoded port they suggest since there's a risk of port conflicts. Instead, use Pinokio's `{{port}}` template expression to automatically get the next available port.
  - For example, if the instruction says `python app.py --port 7860`, don't use that hardcoded port since there might be another app running at that port. Instead, automatically assign the next available port like this: `python app.py --port {{port}}`
  - Note that the `{{port}}` expression always returns the next immediately available port for each step, so if you have multiple steps in a script and use `{{port}}` in multiple steps, the value will be different. So if you want to launch at the next available port and then later reuse that port, you will need to first use `{{port}}` to get the next available port, and save the value in local variable using `local.set`, and then use the `{{local.<variable_name>}}` expression later.
### 2. shell.run API
- When writing `shell.run` API requests, always use relative paths (no absolute paths) for the `path` field. For example, if you need to run a command from `app` folder, the `path` attribute should simply be `app`, instead of its full absolute path.
### 2. Package managers
- When installing python packages, try best to use `uv` instead of `pip` even if the install instruction says to use pip. Instead of `pip install -r requirements.txt`, you can simply use `uv pip install -r requirements.txt` for example. Even if the project's own README says use pip or poetry, first check if there's a way to use uv instead.
- When you need to install some global package, try to use `conda` as much as possible. Even on macs, `brew` should be only used if there are no `conda` options.
### 3. Minimal Always
- If you are starting with existing script files, before modifying, creating, or removing any script files, first look at `pinokio.js` to understand which script files are actually used in the launcher. The only script files used are the ones mentioned in the `pinokio.js` file. The `pinokio.js` file is the file that constructs the UI dynamically.
- Do not create a redundant script file that does something that already exists. Instead modify the existing script file for the feature. For example, do not create an `install.json` file for installation if `install.js` already exists. Instead, modify the `install.js` file.
- Pinokio accepts both JSON and JS script files, so when determining whether a script for a specific purpose already exists, check both JSON and JS files mentioned in the `pinokio.js` file. Do not create script files for rendundant purpose.
- When building launchers for existing projects cloned from a repository, try to stay away from modifying the project folder (the `C:\pinokio\api\roop-unleashed` folder), even if installations are failing. Instead, try to work around it by creating additional files in the launcher folder, and using those files IN ADDITION to the default project.
  - The only exception when you may need to make changes to the project folder is when the user explicitly wants to modify the existing project. Otherwise if the purpose is to simply write a launcher, the app logic folder should never be touched.
- When running shell commands, take full advantage of the Pinokio `shell.run` API, which provides features like `env`, `venv`, `input`, `path`, `sudo`, `on`, etc. which can greatly reduce the amount of script code.
  - Python apps: Always use virtual environments via `venv` attribute. This attribute automatically creates a venv or uses if it already exists.
### 4. Try to support Cross-platform as much as possible
- Use cross-platform shell commands only.
- This means, prefer to use commands that work on all platforms instead of the current platform.
- If there are no cross platform commands, use Pinokio's template expressions to conditionally use commands depending on `platform`, `arch`, etc.
- Also try to utilize Pinokio Pterm APIs for various cross-platform system features.
- If it is impossible to implement a cross platform solution (due to the nature of the project itself), set the `platform`, `arch`, and/or `gpu` attributes of the `pinokio.json` file to declare the limitation.
- Pinokio provides various APIs for cross-platform way of calling commonly used system functions, or lets you selectively run commands depending on `platform`, `arch`, etc.
### 5. Do not make assumptions about Pinokio API
- Do NOT make assumptions about which Pinokio APIs exist. Check the documentation.
- Do NOT make assumptions about the Pinokio API syntax. Follow the documentation.
### 6. Scripts must be able to replicate install and launch steps 100%
- The whole point of the scripts is for others to easily download and invoke them via Pinokio interface with one click. Therefore, do not assume the end user's system state, and make everything self-contained.
- When a 3rd party package needs to be installed, or a 3rd party repository needs to be downloaded, include them in the scripts.
### 7 Dynamic UI rendering
- The `pinokio.js` launcher script can change dynamically depending on the current state of the script execution. Which means, depending on what the file returns, it can determine what the sidebar looks like at any given moment of the script cycle.
  - `info.exists(relative_path)`: The `info.exists` can be used to check whether a relative path (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.running(relative_path)`: The `info.running` can be used to check whether a script at a relative path is currently running (relative to the script root path) exists. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `info.local(relative_path)`: The `info.local` can be used to return all the local variables tied to a script that's currently running. The `pinokio.js` file can determine which menu items to return based on this value at any given moment.
  - `default`: set the `default` attribute on any menu item for whichever menu needs to be selected by default at a given step. Some example scenarios:
    - during the install process, the `install.js` menu item needs to be set as the `default`, so it automatically executes the script
    - when launching the `start.js` menu item needs to be set as the `default`, so it automatically executes the script
    - after the app has launched, the `default` needs to be set on the web UI URL, so the user is sent to the actual app automatically.
  - Check the examples in the C:\pinokio\prototype\system\examples folder to see how these are being used.
### 8. No need for stop scripts
- `pinokio.js` does NOT need a separate `stop` script. Every script that can be started can also be natively stopped through the Pinokio UI, therefore you do not need a separate stop script for start script
### 9. Writing launchers for existing projects
- When writing or modifying pinokio launcher scripts, figure out the install/launch steps by reading the project folder `app`.
- In most cases, the `README.md` file in the `C:\pinokio\api\roop-unleashed` folder contains the instructions needed to install and run the app, but if not, figure out by scanning the rest of the project files.
- Install scripts should work for each specific operating system, so ignore Docker related instructions. Instead use install/launch instructions for each platform.
### 10. Don't use Docker unless really necessary
- Some projects suggest docker as installation options. But even in these cases, try to find "development" options to launch the app without relying on Docker, as much as possible. We do not need Docker since we can automatically install and launch apps specifically for the user's platform, since we can write scripts that run cross platform.
### 11. pinokio.json
- Do not touch the `version` field since the version is the script schema version and the one pre-set in `pinokio.js` must be used.
- `icon`: It's best if we have a user friendly icon to represent the app, so try to get an image and link it from `pinokio.json`.
  - If the git repository for the `C:\pinokio\api\roop-unleashed` folder points to GitHub (for example https://github.com/<USERNAME>/<REPO_NAME>`, ask the user if they want to download the icon from GitHub, and if approved, get the `avatar_url` by fetching `https://api.github.com/users/<USERNAME>`, and then download the image to the root folder as `icon.png`, and set `icon.png` as the `icon` field of the `pinokio.json`. 
### 12. Gitignore
- When a launcher involves cloning 3rd party repositories, downloading files dynamically, or some files to be generated, these need to be included in the .gitignore file. This may include things like:
  - Cloning git repositories
  - Downloading files
  - Dynamically creating files during installation or running, such as Sqlite Databases, or environment variables, or anything specific to the user.
- Make sure these file paths are included in the .gitignore file, and if not, include them in .gitignore.

## AI Libraries (Pytorch, Xformers, Triton, Sageattention, etc.)
If the launcher has a dedicated built-in script named `torch.js`, it can be used as follows:

```
// install.js
module.exports = {
  run: [
    // Edit this step with your custom install commands
    {
      method: "shell.run",
      params: {
        venv: "venv",                // Edit this to customize the venv folder path
        path: "app",
        message: [
          "uv pip install -r requirements.txt"
        ],
      }
    },
    // Delete this step if your project does not use torch
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          path: "app",
          venv: "venv",                // Edit this to customize the venv folder path
          // xformers: true   // uncomment this line if your project requires xformers
          // triton: true   // uncomment this line if your project requires triton
          // sageattention: true   // uncomment this line if your project requires sageattention
          // flashattention: true   // uncomment this line if your project requires flashattention
        }
      }
    },
  ]
}
```

The `torch.js` script also includes ways to install pytorch dependent libraries such as xformers, triton, sagetattention. If any of these libraries need to be installed, use the torch.js to install in order to install them cross platform.


## Quick Reference
### Essential Documentation
- **Pinokio Programming:** See `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md → "Programming Pinokio" section
- **Dynamic Menus:** See `PINOKIO.md` at C:\pinokio\prototype\PINOKIO.md → "Dynamic menu rendering" section  
- **CLI Commands:** See `PTERM.md` at C:\pinokio\prototype\PTERM.md
### Common Patterns
- **Python Virtual Env:** `shell.run` with `venv` attribute
- **Cross-platform Commands:** Always test on multiple platforms
- **Error Handling:** Check logs/api for launcher issues
- **GitHub Operations:** Use `gh` CLI for advanced GitHub features
## Development Principles
1. **Minimize Shell Usage:** Leverage API parameters instead of raw commands
2. **Maintain Separation:** Keep app logic and launchers separate
3. **Follow Conventions:** Match existing project patterns
4. **Test Thoroughly:** Use CLI to verify launcher functionality
5. **Document Changes:** Update relevant metadata and documentation
