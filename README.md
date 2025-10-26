# rag-privacy

Code repo for [Follow My Instruction and Spill the Beans: Scalable Data Extraction from Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2402.17840) (ICLR 2025).

## Installing `pyserini`
You will need `pyserini` for RIC-LM.
First make sure that you've installed `torch` and `python>=3.10`. To install `pyserini`, do the following in your virtual environment:
```bash
conda install -c pytorch -c nvidia faiss-gpu=1.7.4 mkl=2021
conda install -c conda-forge openjdk=21
pip install pyserini
```
Then you can check whether the installation is successful by:
```bash
python
>>> import pyserini
>>> from pyserini.search.lucene import LuceneSearcher
```
If there is no error, then the installation has no problem.
However, you might encounter one error:
```bash
ImportError: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found (required by ...)
```
Then you should do the following first every time you run the scripts:
```bash
export LD_LIBRARY_PATH=/path/to/your/conda/envs/your_env_name/lib
```
Then things should be good now.


## For Henry:

### Set up
```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install pyserini and dependencies
uv pip install pyserini
uv pip install torch torchvision torchaudio  # PyTorch
uv pip install faiss-cpu  # or faiss-gpu if you have CUDA
uv pip install transformers datasets tqdm numpy together wandb nltk evaluate rouge_score sacrebleu bert_score accelerate

# Install a JDK 21 that registers cleanly on macOS
brew install --cask temurin@21

# Use it for this shell
export JAVA_HOME="$("/usr/libexec/java_home" -v 21)"
export PATH="$JAVA_HOME/bin:$PATH"
# If you previously set these, keep them in sync:
export DYLD_LIBRARY_PATH="$JAVA_HOME/lib/server:$JAVA_HOME/lib:$JAVA_HOME/lib/jli:${DYLD_LIBRARY_PATH}"

# sanity check
java -version   # should show 21.x
```

### Running the scripts

```bash
./construct_adversarial_prompt.sh
./main.sh
```


## For Emira

### Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv
source .venv/bin/activate

# Install pyserini and dependencies
uv pip install pyserini
uv pip install torch torchvision torchaudio  # PyTorch
uv pip install faiss-cpu
uv pip install transformers datasets tqdm numpy together wandb nltk evaluate rouge_score sacrebleu bert_score accelerate hf_transfer

# Install JDK 21
curl -LsSf https://astral.sh/uv/install.sh | sh
sudo apt install ./jdk-21_linux-x64_bin.deb

# Use it for this shell
export JAVA_HOME="$("/usr/libexec/java_home" -v 21)"
export PATH="$JAVA_HOME/bin:$PATH"
# If you previously set these, keep them in sync:
export DYLD_LIBRARY_PATH="$JAVA_HOME/lib/server:$JAVA_HOME/lib:$JAVA_HOME/lib/jli:${DYLD_LIBRARY_PATH}"

java -version   # should show 21.x
```


