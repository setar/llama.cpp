import pytest
from utils import *

server = ServerPreset.tinyllama2()

# ~226 tokens with the stories260K tokenizer (see test_ctx_shift.py)
LONG_TEXT = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
""".strip()

# shares the full LONG_TEXT prefix, diverges only in the tail
DIVERGENT_TEXT = LONG_TEXT + "\nThis is a completely different final sentence."

SLOT_FILE = "slot_ckpt.bin"


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()
    server.slot_save_path = "./tmp"
    server.temperature = 0.0
    server.n_ctx = 1024  # 512 per slot is not enough for LONG_TEXT + divergent tail


def tokenize_len(server, text):
    # /completion prepends BOS, so count special tokens the same way
    res = server.make_request("POST", "/tokenize", data={"content": text, "add_special": True})
    assert res.status_code == 200
    return len(res.body["tokens"])


def test_slot_save_restore_checkpoints():
    """SLOT_SAVE/SLOT_RESTORE must carry prompt.checkpoints alongside KV state.

    A divergent prompt whose common prefix lands on the checkpoint boundary must
    restore the context checkpoint in the target slot instead of forcing a full
    prompt re-processing.
    """
    global server
    server.start()

    n_long = tokenize_len(server, LONG_TEXT)
    n_divergent = tokenize_len(server, DIVERGENT_TEXT)
    assert n_long >= 2 * server.n_batch  # multi-batch prompt, checkpoint n_tokens > 0

    # full prefill in slot 1
    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_long

    # save slot 1 (KV state + tokens) and restore it into slot 0
    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": SLOT_FILE,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": SLOT_FILE,
    })
    assert res.status_code == 200

    # warm slot: the end-of-prompt checkpoint covers the whole prefix,
    # only the divergent tail is processed
    res = server.make_request("POST", "/completion", data={
        "prompt": DIVERGENT_TEXT,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_warm = res.body["timings"]["prompt_n"]
    assert n_warm < n_divergent  # checkpoint restore hit inside the slot

    # the restored slot must behave like the warm slot: restore the checkpoint
    # carried by the slot file instead of doing a full prefill
    res = server.make_request("POST", "/completion", data={
        "prompt": DIVERGENT_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    n_restored = res.body["timings"]["prompt_n"]
    assert n_restored < n_divergent       # no full re-processing
    assert n_restored == n_warm           # same amount of work as the warm slot


def test_slot_save_restore_without_checkpoints():
    """Backward compatibility: a slot file without a checkpoint sidecar restores
    with an empty checkpoint list and falls back to full prefill."""
    global server
    server.start()

    n_divergent = tokenize_len(server, DIVERGENT_TEXT)

    res = server.make_request("POST", "/completion", data={
        "prompt": LONG_TEXT,
        "id_slot": 1,
        "cache_prompt": True,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/slots/1?action=save", data={
        "filename": SLOT_FILE,
    })
    assert res.status_code == 200

    # remove the checkpoint sidecar, leaving a legacy slot file
    sidecar = os.path.join("tmp", SLOT_FILE + ".ckpt")
    if os.path.exists(sidecar):
        os.remove(sidecar)

    res = server.make_request("POST", "/slots/0?action=restore", data={
        "filename": SLOT_FILE,
    })
    assert res.status_code == 200

    res = server.make_request("POST", "/completion", data={
        "prompt": DIVERGENT_TEXT,
        "id_slot": 0,
        "cache_prompt": True,
    })
    assert res.status_code == 200
    assert res.body["timings"]["prompt_n"] == n_divergent  # full prefill
