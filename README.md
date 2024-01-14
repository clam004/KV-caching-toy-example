# KV-caching-toy-example

A minimal toy example of the KV-cache used to speed up large language model transformers using only numpy, composed from an edited version of https://github.com/jaymody/picoGPT/pull/7 with moving diagrams from https://medium.com/@joaolages/kv-caching-explained-276520203249 alongside code annotations

<img src="samples/KV_cache.gif">

#### Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

[START_HERE](kv_cache.ipynb)

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.




