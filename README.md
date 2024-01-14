# PicoGPT

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.


#### Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

[START_HERE](kv_cache.ipynb)



