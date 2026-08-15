module.exports = () => {
	const config = {
		run: [
			{
				method: 'shell.run',
				params: {
					message: 'git pull'
				}
			}
		]
	};

	return config;
};
