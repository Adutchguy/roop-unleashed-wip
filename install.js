const path = require('path');
const os = require('os');

const envPath = path.resolve(__dirname, '.env');

// insightface==0.7.3 has no published wheel for Windows, so pip would try (and fail)
// to build it from source without a compiler. Install C0untFloyd's prebuilt cp310
// wheel first so the later `pip install -r requirements.txt` sees it already satisfied.
// On macOS/Linux we instead make sure a compiler + cython are present so the source
// build in requirements.txt succeeds on its own.
function insightfaceStep(kernel) {
	if (kernel.platform === 'win32') {
		// insightface==0.7.3 has no PyPI wheel for Windows, and its original wheel
		// host (C0untFloyd/roop-unleashed) was disabled by GitHub. Rather than
		// depend on another external mirror that could disappear the same way,
		// the wheel is vendored in-repo at app/vendor/.
		return 'pip install vendor/insightface-0.7.3-cp310-cp310-win_amd64.whl';
	}
	return 'pip install cython==0.29.36';
}

// Matches the hardware matrix documented in README.md.
function torchStep(kernel) {
	const { platform, gpu } = kernel;

	if (gpu === 'nvidia') {
		return 'pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps';
	}
	if (platform === 'win32' && gpu === 'amd') {
		return 'pip install torch torch-directml torchvision torchaudio --force-reinstall';
	}
	if (platform === 'linux' && gpu === 'amd') {
		return 'pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/rocm6.3 --force-reinstall --no-deps';
	}
	if (platform === 'darwin') {
		return 'pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps';
	}
	return 'pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu --force-reinstall --no-deps';
}

// TensorRT is optional acceleration for NVIDIA GPUs, selectable in the app's
// own Settings tab (Provider: tensorrt). onnxruntime-gpu==1.19.0 requires
// TensorRT 10.2 specifically (see https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements) —
// without it, roop/core.py's own `import tensorrt` probe fails and the app
// silently falls back to plain CUDA even if TensorRT is selected.
function tensorrtStep(kernel) {
	return 'pip install tensorrt-cu12==10.2.0';
}

function onnxruntimeStep(kernel) {
	const { platform, gpu } = kernel;

	if (gpu === 'nvidia') {
		return 'pip install onnxruntime-gpu==1.19.0';
	}
	if (platform === 'win32' && gpu === 'amd') {
		return 'pip install onnxruntime-directml';
	}
	if (platform === 'linux' && gpu === 'amd') {
		return 'pip install https://repo.radeon.com/rocm/manylinux/rocm-rel-6.3/onnxruntime_rocm-1.19.0-cp310-cp310-linux_x86_64.whl';
	}
	if (platform === 'darwin') {
		if (os.arch() === 'arm64') {
			return 'pip install onnxruntime-silicon==1.16.3';
		}
		return 'pip install onnxruntime==1.17.1';
	}
	return 'pip install onnxruntime==1.17.1';
}

module.exports = async kernel => {
	const { platform, gpu } = kernel;

	const config = {
		run: [
			{
				// updates the conda binary Pinokio ships with
				method: 'shell.run',
				params: {
					message: 'conda install conda=25.5.1 --yes'
				}
			},
			{
				// creates the isolated project environment at ./.env
				// pinned to 3.10 to match the prebuilt Windows insightface wheel (cp310)
				method: 'shell.run',
				params: {
					message: 'conda install python=3.10 --yes',
					conda: {
						path: envPath
					}
				}
			},
			{
				method: 'shell.run',
				params: {
					message: 'conda install conda-forge::ffmpeg --yes',
					conda: {
						path: envPath
					}
				}
			},
			{
				// compiler + cmake so insightface (and any other source-only deps) can build
				when: '{{ platform !== "win32" }}',
				method: 'shell.run',
				params: {
					message: 'conda install conda-forge::cxx-compiler conda-forge::cmake --yes',
					conda: {
						path: envPath
					}
				}
			},
			{
				// self-contained CUDA runtime + cuDNN so onnxruntime-gpu works without a
				// system-wide CUDA Toolkit install
				when: '{{ (platform === "win32" || platform === "linux") && gpu === "nvidia" }}',
				method: 'shell.run',
				params: {
					message: 'conda install -c nvidia cuda-runtime=12.8 cudnn=9 --yes',
					conda: {
						path: envPath
					}
				}
			},
			{
				method: 'shell.run',
				params: {
					message: insightfaceStep(kernel),
					path: 'app',
					conda: {
						path: envPath
					},
					// pip's dependency-resolver notices ("ERROR: pip's dependency
					// resolver does not currently take into account...") print the
					// word "error" without actually failing the install. Without this,
					// Pinokio's default error-detection kills the whole script here and
					// every step after it silently never runs.
					on: [{
						event: '/error:/i',
						break: false
					}]
				}
			},
			{
				method: 'shell.run',
				params: {
					message: 'pip install -r requirements.txt',
					path: 'app',
					conda: {
						path: envPath
					},
					on: [{
						event: '/error:/i',
						break: false
					}]
				}
			},
			{
				method: 'shell.run',
				params: {
					message: torchStep(kernel),
					path: 'app',
					conda: {
						path: envPath
					},
					on: [{
						event: '/error:/i',
						break: false
					}]
				}
			},
			{
				method: 'shell.run',
				params: {
					message: onnxruntimeStep(kernel),
					path: 'app',
					conda: {
						path: envPath
					},
					on: [{
						event: '/error:/i',
						break: false
					}]
				}
			},
			{
				// makes the "tensorrt" provider in the app's Settings tab actually
				// work instead of silently falling back to CUDA (see tensorrtStep above)
				when: '{{ gpu === "nvidia" }}',
				method: 'shell.run',
				params: {
					message: tensorrtStep(kernel),
					path: 'app',
					conda: {
						path: envPath
					},
					on: [{
						event: '/error:/i',
						break: false
					}]
				}
			},
			{
				method: 'input',
				params: {
					title: 'Installation completed',
					description: 'Return to the dashboard to start roop-unleashed. Models (~2GB) will download automatically on first run.'
				}
			}
		]
	};

	return config;
};
