const path = require('path');

module.exports = () => {
	const config = {
		daemon: true,
		run: [
			{
				method: 'shell.run',
				params: {
					message: 'python run.py',
					path: 'app',
					conda: {
						path: path.resolve(__dirname, '.env')
					},
					on: [{
						event: '/(http:\/\/[0-9.:]+)/',
						done: true
					}]
				}
			},
			{
				method: 'local.set',
				params: {
					url: '{{ input.event[0] }}'
				}
			}
		]
	};

	return config;
};
