const path = require('path');

module.exports = () => {
	const config = {
		run: [
			{
				// only removes the Python environment; app code, downloaded models,
				// and saved settings under app/ are left untouched
				method: 'fs.rm',
				params: {
					path: path.resolve(__dirname, '.env')
				}
			}
		]
	};

	return config;
};
