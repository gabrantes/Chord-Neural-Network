# ChordNet [WIP]

ChordNet is a multi-output classification model for predicting chords. Given a key signature, type & voicings of the current chord, and type of the next chord, it can produce smooth voice-leading to the next chord, with 90% accuracy.

For more details, see [Background](#background) and [Development](#development).

## Table of Contents   
   * [Getting Started](#getting-started)
     * [Prerequisites](#prerequisites)
     * [Installing](#installing)
     * [Demo](#demo)
     * [Training (optional)](#training-\(optional\))
   * [Background](#background)
   * [Development](#development)
   * [Technologies](#technologies)
   * [License](#license)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* Python 3
* pip OR pipenv

### Installing

1. Clone repository.

   ```
   git clone https://github.com/gabrantes/ChordNet.git
   ``` 
  
2. Install requirements using pip OR pipenv.

   - Using pip:  
     ```
     pip install -r requirements.txt
     ```    
    
   - Using pipenv:  
     ```
     pipenv sync  
     ```
    
3. Add project to PYTHONPATH so modules can be found.

   - Windows
     ```
     set PYTHONPATH=path/to/ChordNet
     ```
     
   - Unix
     ```
     export PYTHONPATH=path/to/ChordNet
     ```
     
### Demo

1. Run predictions script.
   ```
   python scripts/predict.py
   ```

### Training (optional)

1. Preprocess data and save results as new text files in `data/`.
   ```
   python scripts/preprocess.py
   ```
   
2. Train model.
   ```
   python scripts/train.py
   ```

3. View metrics using TensorBoard.
   ```
   tensorboard --logdir=logs/
   ```
   
4. Run predictions script.
   ```
   python scripts/predict.py
   ```
   
## Background

In traditional music theory, there are many guidelines and rules when writing chord progressions for four voice parts. This is also referred to as four-part harmony. 

"Tonal Harmony" (Kostka & Payne) is a popular music theory textbook that was used as the basis for this project. For example, the vocal ranges for each part were set as dictated in the book, and over 300 example chord progresions were used as 'seeds' for the dataset.

## Development
TODO

## Technologies

* Python
* Keras (TensorFlow backend)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details
