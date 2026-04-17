🎮 Arkanoid (Python Web Version)

A simple App for performing planks correctly, for those who train alone. runs in the browser using a local Python server.

🚀 Getting Started

Follow these steps to run the game locally:

1. Clone or download the repository
git clone <your-repository-url>
cd APP/Prancha

Or simply download and extract the ZIP file.

2. Start a local server

Make sure you have Python installed, then run:

python -m http.server 8000
3. Open in your browser

Go to:

http://localhost:8000

The index.html file will load automatically, and the game will start.

🕹️ How to Play
Use your keyboard (or mouse, depending on implementation) to control the paddle
Break all the bricks to win
Don’t let the ball fall!
⚙️ Requirements
Python 3.x
A modern web browser (Chrome, Edge, Firefox recommended)
📁 Project Structure
APP
└── Prancha/
    ├── index.html
    ├── (other game files)
📌 Notes
The game must be run through a local server (not by opening the file directly), otherwise some features may not work correctly.
If port 8000 is already in use, you can change it (e.g., python -m http.server 8080).
