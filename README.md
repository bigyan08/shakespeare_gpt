# Shakespeare GPT

Shakespeare GPT is a language model project designed to generate text in the style of William Shakespeare. It leverages modern machine learning techniques to create poetry, plays, and prose reminiscent of the Bard.

## Features

- Generate Shakespearean-style text
- Customizable prompts for plays, sonnets, and dialogues
- Easy-to-use command-line interface

## Installation

Clone the repository:

```bash
git clone https://github.com/bigyan08/shakespeare_gpt.git
cd shakespeare_gpt
```

Install dependencies:

```bash
pip install torch scikit-learn transformers 
```

## Usage

Run the main script with your prompt:

```bash
python main.py --prompt "To be, or not to be" --length 100
```

## Project Structure

- `main.py` - Entry point for text generation
- `src/` - Model training and inference code
- `data/` - Shakespearean text datasets
- `README.md` - Project documentation
- `notebook/` - Jupyter notebooks for experimentation 
- `models/` - Saved models as pth files

## Note
- Ensure you have a compatible GPU for optimal performance and enable it in `config.py`. In my case, I used cpu as I don't have a GPU.
- Train your own model and give a name, it will be saved in models directory.
- The default model is mini_gpt.pth, you can use your own trained model by changing the data_path of model in config.py.

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

