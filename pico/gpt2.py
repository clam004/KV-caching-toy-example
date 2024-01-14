import numpy as np

from tqdm import tqdm

seq_len = 0

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    # normalize x to have mean=0 and var=1 over last axis
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  
    return g * x + b  # scale and offset with gamma/beta params

def linear(x, w, b):  # [m, in], [in, out], [out] -> [m, out]
    return x @ w + b

def ffn(x, c_fc, c_proj):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]
    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]
    return x

# Q, K, V -> A
def attention(Q, K, V, mask, verbose = False): 
    
    # [n_seq_q, n_embd], [n_seq_k, n_embd], [n_seq_k, n_embd], [n_seq_q, n_seq_k] -> [n_seq_q, n_embd]
    QK_T = Q @ K.T

    global seq_len
    if QK_T.shape[1] > seq_len:
        print("QK_T.shape", QK_T.shape)
        seq_len = QK_T.shape[1]

    A = softmax(QK_T / np.sqrt(Q.shape[-1]) + mask) @ V
    
    return A

def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]

    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):  # [n_seq] -> [n_seq, n_vocab]
    # token + positional embeddings
    x = wte[inputs] + wpe[range(len(inputs))]  # [n_seq] -> [n_seq, n_embd]

    # forward pass through n_layer transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [n_seq, n_embd] -> [n_seq, n_embd]

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T  # [n_seq, n_embd] -> [n_seq, n_vocab]


def generate(inputs, params, n_head, n_tokens_to_generate):
    
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids


def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):

    from utils import load_encoder_hparams_and_params

    # load encoder, hparams, and params from the released open-ai gpt-2 files
    tokenizer, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = tokenizer.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = tokenizer.decode(output_ids)

    return output_text

def mha_kv(x, c_attn, c_proj, n_head, kvcache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    # qkv projection
    # n_seq = 1 when we pass kvcache, so we will compute new_q, new_k and new_v
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    if kvcache:
        # qkv
        new_q, new_k, new_v = qkv  # new_q, new_k, new_v = [1, n_embd]
        old_k, old_v = kvcache
        k = np.vstack([old_k, new_k]) # k = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        v = np.vstack([old_v, new_v]) # v = [n_seq, n_embd], where n_seq = prev_n_seq + 1
        qkv = [new_q, k, v]

    current_cache = [qkv[1], qkv[2]]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [n_head, 3, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    if kvcache:
        causal_mask = np.zeros((1, k.shape[0]))
    else:
        causal_mask = (1 - np.tri(x.shape[0])) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [n_head, 3, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    
    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, current_cache

def transformer_block_kv(x, mlp, attn, ln_1, ln_2, n_head, kvcache=None):  # [n_seq, n_embd] -> [n_seq, n_embd]
    
    # multi-head causal self attention
    attn_out, kvcache_updated = mha_kv(layer_norm(x, **ln_1), **attn, n_head=n_head, kvcache=kvcache)
    x = x + attn_out  # [n_seq, n_embd] -> [n_seq, n_embd]

    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x, kvcache_updated

def gpt2_kv(inputs, wte, wpe, blocks, ln_f, n_head, kvcache = None):  # [n_seq] -> [n_seq, n_vocab]
    
    if not kvcache:
        kvcache = [None]*len(blocks)
        wpe_out = wpe[range(len(inputs))]
    else: # cache already available, only send last token as input for predicting next token
        wpe_out = wpe[[len(inputs)-1]]
        inputs = [inputs[-1]]

    # token + positional embeddings
    x = wte[inputs] + wpe_out  # [n_seq] -> [n_seq, n_embd]

    
    # forward pass through n_layer transformer blocks
    new_kvcache = []
    for block, kvcache_block in zip(blocks, kvcache):
        x, updated_cache = transformer_block_kv(x, **block, n_head=n_head, kvcache=kvcache_block)  # [n_seq, n_embd] -> [n_seq, n_embd]
        new_kvcache.append(updated_cache)  # TODO: inplace extend new cache instead of re-saving whole

    # projection to vocab
    x = layer_norm(x, **ln_f)  # [n_seq, n_embd] -> [n_seq, n_embd]
    return x @ wte.T, new_kvcache  # [n_seq, n_embd] -> [n_seq, n_vocab]

def generate_kv(inputs, params, n_head, n_tokens_to_generate):

    kvcache = None
    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits, kvcache = gpt2_kv(inputs, **params, n_head=n_head, kvcache=kvcache)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs = np.append(inputs, [next_id])  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

if __name__ == "__main__":
    
    import fire

    fire.Fire(main)
