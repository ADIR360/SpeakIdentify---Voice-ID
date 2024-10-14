# Voice Identification Tool

**This project is very much incomplete!**
**Plus Database Issue**
**Tinker conflict**
    <div id="header" align="center">
  <img src="https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExcHBxb2tncms0ODFya3VuZXQ3ZXhwOGIwNDN2bWNuanBlZW1pYjl2MSZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/bTzFnjHPuVvva/giphy.gif" width="100vh"/> </div>
## Overview

The Voice Identification Tool is a forensic application designed to recognize and analyze voice patterns. By leveraging audio processing libraries, this tool can identify speakers based on pre-recorded voice samples. This project aims to create a reliable method for identifying individuals using their unique voice characteristics.

## Features

- **Voice Recording**: Users can record their voice patterns and save them directly to a database.
- **Voice Analysis**: The tool analyzes recorded audio to identify the speaker based on their voice pattern.
- **Multi-Window User Interface**: The application provides a user-friendly interface, allowing users to seamlessly navigate between recording and analyzing functionalities.
- **Audio Processing**: Utilizes libraries like `librosa` for audio analysis and feature extraction.
- **Database Management**: The application can save and retrieve voice patterns from a SQLite database.
- **Future Enhancements**:
  - A more polished and responsive user interface.
  - An intuitive way to access and manage voice recordings.
  - Enhanced voice pattern analysis algorithms for improved accuracy.

## How It Works

The Voice Identification Tool operates in the following manner:

1. **Voice Recording**: 
   - The user initiates the recording process by selecting the "Record Voice to Database" option.
   - The application prompts the user to speak a specific letter or phrase.
   - The recorded audio is processed to extract voice features.

2. **Feature Extraction**:
   - The application uses the `librosa` library to analyze the recorded audio.
   - Mel-frequency cepstral coefficients (MFCCs) are computed from the audio data to capture the unique characteristics of the voice.
   - These features are saved in a database alongside the corresponding identifier (e.g., letter or name).

3. **Voice Analysis**:
   - The user selects the "Analyze Voice" option to identify a speaker.
   - The application records new audio input and extracts its features.
   - Using the features stored in the database, the application compares the new input against the existing voice patterns.
   - The tool identifies the speaker by finding the closest match based on voice patterns.

## Setup Instructions

### Prerequisites

- Python 3.x
- Required Python libraries listed in `requirements.txt`.

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/Voice-Identification-Tool.git
   cd Voice-Identification-Tool

2. Create a virtual environment:

   ```bash
   python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`


3. Install required libraries:
   ```bash
      pip install -r requirements.txt
4. Run the application:
   ```bash
   python3 main.py
### Contributing
Contributions are welcome! If you have suggestions for improvements or features, feel free to open an issue or submit a pull request.

### License
This project is licensed under the MIT License

### Explanation of Sections

1. **Overview**: Provides a brief introduction to the project, highlighting its purpose and goals.
2. **Features**: Lists the current functionalities and future enhancements you plan to implement.
3. **How It Works**: Describes the process of recording, extracting features, and analyzing voices in detail.
4. **Setup Instructions**: Guides users through the setup and installation process for the project.
5. **Contributing**: Encourages contributions and explains how others can help improve the project.
6. **License**: Mentions the licensing terms of the project.

Feel free to modify any part of the README to better fit your project's needs!
