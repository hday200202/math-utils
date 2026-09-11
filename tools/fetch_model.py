"""
Download a model from the HuggingFace Hub into models/<name>/.

Only what the loader actually needs: config.json, a tokenizer, and the
weight shards named by model.safetensors.index.json. Repos often ship
extras that would double the download for nothing - Mistral, for one,
carries a consolidated.safetensors alongside its shards, which is the same
13.5GB a second time.

Refuses architectures the engine cannot run rather than downloading tens of
gigabytes and failing at load, and says why.

    py tools/fetch_model.py Qwen/Qwen2.5-7B-Instruct
    py tools/fetch_model.py openbmb/MiniCPM5-2B --name minicpm5-2b
    py tools/fetch_model.py Qwen/Qwen3-32B --check       (inspect, download nothing)

A full browser URL works too - the bits around the org/model are stripped.
"""

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

BASE = "https://huggingface.co"

# must match the registry in include/ai/LocalModel.hpp
SUPPORTED = {"gpt2", "llama", "mistral", "qwen2", "qwen3", "phi3"}

MOE_KEYS = ("num_experts", "n_routed_experts", "num_local_experts", "moe_intermediate_size")


def normalize_repo(text):
    """
    Accepts what a person actually has in their clipboard: org/model, or any
    of the URLs the Hub shows. Everything around the org/model pair is
    stripped, because pasting the browser URL is the obvious thing to do and
    failing on it with a message about licences is worse than useless.
    """
    repo = text.strip().rstrip("/")

    for prefix in ("https://huggingface.co/", "http://huggingface.co/",
                   "https://hf.co/", "http://hf.co/", "huggingface.co/", "hf.co/"):
        if repo.startswith(prefix):
            repo = repo[len(prefix):]
            break

    # .../tree/main, .../blob/main/config.json, .../resolve/main/...
    for marker in ("/tree/", "/blob/", "/resolve/"):
        at = repo.find(marker)
        if at != -1:
            repo = repo[:at]

    return repo.strip("/")


def fetch_json(url, timeout=30):
    """Returns (parsed, http_status). status is None when the request never landed."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "util-libraries"})
        return json.load(urllib.request.urlopen(req, timeout=timeout)), 200
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception:
        return None, None


def human(n):
    return "%.2f GB" % (n / 2 ** 30) if n >= 2 ** 30 else "%.0f MB" % (n / 2 ** 20)


def download_many(repo, directory, files, sizes, jobs):
    """
    Several shards at once.

    One connection to the Hub tops out well below what the link can carry -
    measured at 18 MB/s on its own and 52 MB/s across four - so a 150 GB
    checkpoint that would take three hours in sequence takes under one. The
    files are independent and each is written whole, so the only coordination
    needed is handing out the next name.
    """
    pending = list(files)
    lock = threading.Lock()
    failed = []
    total = sum(sizes.get(f, 0) for f in files)
    got = [0]
    started = time.time()

    def progress(added):
        with lock:
            got[0] += added
            done = got[0]
        if not total:
            return
        filled = int(28 * done / total)
        rate = done / max(time.time() - started, 0.001) / 2 ** 20
        sys.stdout.write("\r  [%s%s] %3.0f%%  %s of %s  %6.1f MB/s"
                         % ("#" * filled, " " * (28 - filled), 100.0 * done / total,
                            human(done), human(total), rate))
        sys.stdout.flush()

    def worker():
        while True:
            with lock:
                if not pending or failed:
                    return
                name = pending.pop(0)

            path = os.path.join(directory, name)
            expected = sizes.get(name, 0)

            if expected and os.path.exists(path) and os.path.getsize(path) == expected:
                progress(expected)
                continue

            url = BASE + "/" + repo + "/resolve/main/" + name
            req = urllib.request.Request(url, headers={"User-Agent": "util-libraries"})
            try:
                response = urllib.request.urlopen(req, timeout=60)
                with open(path, "wb") as f:
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        progress(len(chunk))
            except Exception as e:
                with lock:
                    failed.append(name + ": " + str(e))
                return

    threads = [threading.Thread(target=worker) for _ in range(max(1, min(jobs, len(pending))))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print("")
    for f in failed:
        print("  FAILED " + f)
    return not failed


def download(url, path, expected=0):
    """Streams to disk with a progress bar, skipping a file already the right size."""
    if expected and os.path.exists(path) and os.path.getsize(path) == expected:
        print("  %-44s already have it" % os.path.basename(path))
        return True

    req = urllib.request.Request(url, headers={"User-Agent": "util-libraries"})
    try:
        response = urllib.request.urlopen(req, timeout=60)
    except Exception as e:
        print("  %-44s FAILED: %s" % (os.path.basename(path), e))
        return False

    total = int(response.headers.get("Content-Length") or expected or 0)
    name = os.path.basename(path)
    done = 0
    started = time.time()
    last = -1

    with open(path, "wb") as f:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)

            if total:
                filled = int(28 * done / total)
                if filled != last:
                    last = filled
                    rate = done / max(time.time() - started, 0.001) / 2 ** 20
                    sys.stdout.write("\r  %-30s [%s%s] %3.0f%%  %6.1f MB/s"
                                     % (name[:30], "#" * filled, " " * (28 - filled),
                                        100.0 * done / total, rate))
                    sys.stdout.flush()

    if total:
        sys.stdout.write("\r  %-30s [%s] 100%%  %s%s\n"
                         % (name[:30], "#" * 28, human(done), " " * 12))
    else:
        print("  %-44s %s" % (name, human(done)))
    return True


def inspect(repo):
    """Returns (config, reason_it_cannot_run_or_None)."""
    cfg, status = fetch_json(BASE + "/" + repo + "/resolve/main/config.json")
    if cfg is None:
        if status not in (401, 403, 404):
            return None, "could not reach the Hub (status " + str(status) + ")"

        """
        The Hub answers 401 for a repo that does not exist just as it does
        for one that merely needs a licence accepted, so that the existence
        of a private repo is not leaked. Guessing between them produces the
        worst possible message: telling someone to go accept a licence for a
        model they mistyped. The API endpoint does distinguish - it answers
        200 for a gated-but-real repo and 401 for one that is not there.
        """
        _, apiStatus = fetch_json(BASE + "/api/models/" + repo, 15)

        if apiStatus == 200:
            return None, ("gated: open " + BASE + "/" + repo
                          + " and accept the licence, then retry")
        if "/" not in repo:
            return None, ("expected org/model, got \"" + repo + "\"")
        return None, "no such model (or it is private): " + BASE + "/" + repo

    text = cfg.get("text_config", cfg)
    kind = cfg.get("model_type") or text.get("model_type")

    if any(k in text for k in MOE_KEYS):
        return cfg, "mixture-of-experts (" + str(kind) + "): needs expert routing"

    layer_types = text.get("layer_types")
    if layer_types and len(set(layer_types)) > 1:
        kinds = sorted(set(layer_types))
        return cfg, "hybrid attention (" + str(kind) + "): layer types " + str(kinds)

    if text.get("num_loops", 1) not in (1, None):
        return cfg, ("looped depth (num_loops=" + str(text["num_loops"])
                     + "): the layers are reused, so a plain forward pass is wrong")

    if "vision_config" in cfg:
        return cfg, "multimodal: has a vision tower"

    if kind not in SUPPORTED:
        return cfg, ("unsupported model_type \"" + str(kind)
                     + "\" (known: " + ", ".join(sorted(SUPPORTED)) + ")")

    return cfg, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="org/model on huggingface.co")
    ap.add_argument("--name", default=None, help="folder under models/ (default: the model name)")
    ap.add_argument("--check", action="store_true", help="inspect only, download nothing")
    ap.add_argument("--force", action="store_true", help="download even if unsupported")
    ap.add_argument("--jobs", type=int, default=4,
                    help="shards to fetch at once (one connection does not saturate the link)")
    args = ap.parse_args()

    args.repo = normalize_repo(args.repo)

    cfg, reason = inspect(args.repo)
    if cfg is None:
        print(args.repo + ": " + reason)
        return 1

    text = cfg.get("text_config", cfg)
    print(args.repo)
    print("  type       %s" % (cfg.get("model_type") or text.get("model_type")))
    print("  layers     %s" % text.get("num_hidden_layers"))
    print("  hidden     %s" % text.get("hidden_size"))
    print("  heads      %s query / %s kv" % (text.get("num_attention_heads"),
                                             text.get("num_key_value_heads")))

    info, _ = fetch_json(BASE + "/api/models/" + args.repo + "?blobs=true")
    if not info:
        print("  could not list the repo")
        return 1

    sizes = {s["rfilename"]: (s.get("size") or 0) for s in info.get("siblings", [])}

    # only the shards the index names, so duplicates like consolidated.safetensors
    # are never fetched
    wanted = []
    index, _ = fetch_json(BASE + "/" + args.repo + "/resolve/main/model.safetensors.index.json")
    if index and "weight_map" in index:
        wanted.append("model.safetensors.index.json")
        for shard in sorted(set(index["weight_map"].values())):
            wanted.append(shard)
    elif "model.safetensors" in sizes:
        wanted.append("model.safetensors")
    else:
        print("  no safetensors weights in this repo")
        return 1

    for extra in ("config.json", "generation_config.json",
                  "tokenizer.json", "vocab.json", "merges.txt"):
        if extra in sizes:
            wanted.append(extra)

    total = sum(sizes.get(f, 0) for f in wanted)
    skipped = sum(v for k, v in sizes.items() if k.endswith(".safetensors") and k not in wanted)

    print("  download   %s across %d file(s)"
          % (human(total), len([f for f in wanted if f.endswith('.safetensors')])))
    if skipped:
        print("  skipping   %s of duplicate weights in the repo" % human(skipped))
    print("  est speed  ~%.1f tok/s on a 51 GB/s machine" % (51.0 / max(total / 2 ** 30, 0.01)))

    if reason:
        print("\n  CANNOT RUN: " + reason)
        if not args.force:
            print("  pass --force to download anyway")
            return 1
        print("  --force given, downloading regardless")

    if args.check:
        return 0

    name = args.name or args.repo.split("/")[-1].lower()
    directory = os.path.join("models", name)
    os.makedirs(directory, exist_ok=True)
    print("\ninto " + directory)

    if not download_many(args.repo, directory, wanted, sizes, args.jobs):
        return 1

    print("\ndone.  ./bin/release/llm " + directory.replace("\\", "/"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
