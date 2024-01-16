# Video Upscaler

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## Description
Quick project I hope to work on to help me upscale old home video footage from camcorders, etc.


## Features
- Upscales video using [Real-ESRGAN](https://github.com/sberbank-ai/Real-ESRGAN). Currently tested on Apple Silicon GPU (working, sort of)

## Installation

To install this project using Poetry, follow these steps:

1. Make sure you have Poetry installed. If not, you can install it by following the instructions [here](https://python-poetry.org/docs/#installation).
2. Clone the repository: `git clone https://github.com/nostoslabs/video_upscaler.git`
3. Navigate to the project directory: `cd video_upscaler`
4. Install the project dependencies: `poetry install`
5. Activate the virtual environment: `poetry shell`

You have now successfully installed the project using Poetry.



## Usage
```bash
poetry run upscale video.mp4 video_HD.mp4
```

## Contributing

Contributions are welcome! Follow these steps to contribute to the project:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Open a pull request to the main repository.

Please ensure that your code follows the project's coding conventions and style guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Contact me via an issue.

## Acknowledgements
This is just a wrapper of the excellent pyAV and https://github.com/sberbank-ai/Real-ESRGAN models.
