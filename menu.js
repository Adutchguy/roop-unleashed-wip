module.exports = async (kernel, info) => {
	const menu = [];

	if (!info.exists('.env')) {
		menu.push({
			icon: 'fa-solid fa-plug',
			text: 'Install',
			href: 'install.js',
			params: {
				run: true,
				fullscreen: true
			}
		});
		return menu;
	}

	if (info.running('start.js')) {
		const memory = info.local('start.js');

		if (memory && memory.url) {
			menu.push({
				icon: 'fa-solid fa-rocket',
				text: 'Open Web UI',
				href: memory.url,
				params: {
					run: true,
					fullscreen: true
				}
			});
		}
		menu.push({
			icon: 'fa-solid fa-desktop',
			text: 'Terminal',
			href: 'start.js',
			params: {
				run: true,
				fullscreen: true
			}
		});
	} else {
		[
			[ 'fa-solid fa-power-off', 'Start', 'start.js' ],
			[ 'fa-solid fa-rotate', 'Update', 'update.js' ],
			[ 'fa-regular fa-circle-xmark', 'Reset', 'reset.js' ]
		].forEach(([ icon, text, href ]) => {
			menu.push({
				icon,
				text,
				href,
				params: {
					run: true,
					fullscreen: true
				}
			});
		});
	}

	return menu;
};
