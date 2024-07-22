
# 🎵 RC Stable Audio Tools

**Stable Audio Tools** provides training and inference tools for generative audio models from Stability AI. This repository is a fork with additional modifications to enhance functionality such as:

- **Dynamic Model Loading**: Enables dynamic model swaps of the base model and any future community finetune releases.

<p align="center">
  <img src="https://i.imgur.com/kB8CQ3J.gif" alt="Model Loader Gif" width="50%">
</p>


- **Random Prompt Button**: A one-click Random Prompt button presently tuned to my [Infinite Pianos Finetune](https://huggingface.co/RoyalCities/RC_Infinite_Pianos) - as more models are released this will be expanded.

<p align="center">
  <img src="https://i.imgur.com/fNEE8cR.gif" alt="Random Prompt Button Gif" width="95%">
</p>


- **BPM & Bar Selector**: BPM & Bar settings tied to the model's timing conditioning, which will auto-fill any prompt with the needed BPM/Bar info. You can also lock or unlock the BPM if you wish to randomize this as well with the Random Prompt button.

<p align="center">
  <img src="https://i.imgur.com/hcedPl5.png" alt="BPM and Bar Example Gif" width="50%">
</p>

- **Automatic Sample to MIDI Converter**: The fork will automatically convert all generated samples to .MID format, enabling users to have an infinite source of MIDI.

<p align="center">
  <img src="https://i.imgur.com/R9ipGiq.gif" alt="Midi Converter Example Gif" width="50%">
</p>

- **Automatic Sample Trimming**: The fork will automatically trim all generated samples to the exact length desired for easier importing into DAWs.

<p align="center">
  <img src="https://i.imgur.com/ApH5SOM.gif" alt="Midi Converter Example Gif" width="75%">
</p>

## 🚀 Installation

### 📥 Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/RoyalCities/RC-stable-audio-tools.git
cd RC-stable-audio-tools
```

### 🔧 Setup the Environment

#### 🌐 Create a Virtual Environment

It's recommended to use a virtual environment to manage dependencies:

- **Windows:**

  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

- **macOS and Linux:**

  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

#### 📦 Install the Required Packages

Install Stable Audio Tools and the necessary packages from `setup.py`:

```bash
pip install stable-audio-tools
pip install .
```

### 🪟 Additional Step for Windows Users

To ensure Gradio uses GPU/CUDA and not default to CPU, uninstall and reinstall `torch`, `torchvision`, and `torchaudio` with the correct CUDA version:

```bash
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## ⚙️ Configuration

A sample `config.json` is included in the root directory. Customize it to specify directories for custom models and outputs (.wav and .mid files will be stored here):

```json
{
    "model_directory": "models",
    "output_directory": "generations"
}
```

## 🖥️ Usage

### 🎚️ Running the Gradio Interface

Start the Gradio interface using a batch file or directly from the command line:

- **Batch file example:**

  ```batch
  @echo off
  cd /d path-to-your-venv/Scripts
  call activate
  cd /d path-to-your-stable-audio-tools
  python run_gradio.py --model-config models/path-to-config/example_config.json --ckpt-path models/path-to-config/example.ckpt
  pause
  ```

- **Command line:**

  ```bash
  python run_gradio.py --model-config models/path-to-config/example_config.json --ckpt-path models/path-to-config/example.ckpt
  ```

### 🎶 Generating Audio and MIDI

Input prompts in the Gradio interface to generate audio and MIDI files, which will be saved as specified in `config.json`.

The interface has been expanded with Bar/BPM settings (which modifies both the user prompt + sample length conditioning), MIDI display + conversion and also features Dynamic Model Loading. 

Models must be stored inside their own sub folder along with their accompanying config files. i.e. A single finetune could have multiple checkpoints. All related checkpoints could go inside of the same "model1" subfolder but its important their associated config file is included within the same folder as the checkpoint itself.

To switch models simply pick the model you want to load using the drop down and pick "Load Model". 

## 🛠️ Advanced Usage

For detailed instructions on training and inference commands, flags, and additional options, refer to the main GitHub documentation:
[Stable Audio Tools Detailed Usage](https://github.com/Stability-AI/stable-audio-tools)

---

I did my best to make sure the code is OS agnostic but I've only been able to test this with Windows / NVIDIA. Hopefully it works for other operating systems. 

If theres any other features or tooling that you may want let me know on here or by contacting me on [Twitter](https://x.com/RoyalCities). I'm just a hobbyist but if it can be done I'll see what I can do.

Have fun!
