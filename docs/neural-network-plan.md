# Neural Network Implementation Plan

Status as of this writing: all activation functions (`sigmoid`, `tanh`, `relu`,
`leakyRelu`, `softmax`, and their `*Prime` derivatives) are implemented,
verified, and correct, in both scalar and `Vec<N,T>` form. Everything else in
`NeuralNetwork.hpp` is a declaration with no body yet.

This doc covers: what's left, how `Layer`/`NeuralNetwork`/`NNConfig` are meant
to fit together, the formulas/algorithms for each remaining function, and the
one open design decision that needs deciding deliberately before `NNConfig`
can be finalized.


## 1. Design decision to make first: how are layer sizes specified?

This has to be resolved before `NNConfig` can be written, because it
determines whether layer sizes are a **compile-time** or **runtime** concept
— and everything built so far (`Vec<N,T>`, `Mat<N,M,T>`, `Layer<InN,OutN,T>`)
is compile-time-sized.

**The tension:** "pass layer sizes as plain numbers into a runtime config
struct" conflicts with `Layer<InN,OutN,T>` needing `InN`/`OutN` as template
parameters, which must be known at compile time. A `size_t` living inside a
struct field, read at runtime, can never become a template argument.

**Two real options:**

- **(Recommended) Compile-time sizes via a variadic template on `NeuralNetwork`
  itself** — `template <typename T, size_t... LayerSizes> class NeuralNetwork`.
  You'd write `NeuralNetwork<float, 784, 128, 64, 10> net;` — the sizes are
  still "just numbers" from your point of view, they just appear as template
  arguments on the type instead of as data inside a config object. Internally,
  the class builds one `Layer<LayerSizes[i], LayerSizes[i+1], T>` per
  consecutive pair, held in a `std::tuple`. `NNConfig` then holds only the
  things that are *genuinely* runtime-tunable (learning rate, epochs, etc.) —
  see section 2. This keeps everything consistent with what's already built:
  no heap allocation, no virtual dispatch, mismatched-size bugs still caught
  at compile time.

- **Fully runtime-configurable architecture** — sizes read from a config
  object (or a file) at runtime, meaning `Layer` would need to become
  dynamically-sized (heap-backed `Mat`/`Vec`, or a type-erased
  `LayerBase<T>` interface behind `unique_ptr`s). This is a real, much bigger
  rewrite — it's the same "compile-time N vs. runtime N" fork from the
  original `VecN` design conversation, just resurfacing one level up. Only
  worth it if you specifically expect to need architectures decided at
  runtime (e.g. loaded from a saved model file) rather than known when you
  write the training code.

**Recommendation: go with the compile-time/variadic option.** It's consistent
with everything else in this library, and nothing about "just numbers" in the
config requires those numbers to be *runtime* numbers — template arguments
are numbers too.

The rest of this doc assumes that choice.


## 2. `NNConfig<T>`

Holds the things that are legitimately runtime-tunable — not layer sizes
(those live in `NeuralNetwork`'s own template parameters per section 1).

```cpp
enum class Activation { Sigmoid, Tanh, ReLU, LeakyReLU };
enum class Loss { MeanSquaredError, CrossEntropy };

template <typename T>
struct NNConfig {
    Activation hiddenActivation = Activation::ReLU;
    Activation outputActivation = Activation::Sigmoid;
    Loss loss = Loss::MeanSquaredError;

    T learningRate = T(0.01);
    size_t epochs = 100;
    size_t batchSize = 1;          // 1 = plain SGD; >1 = mini-batch

    T leakyReluAlpha = T(0.01);    // only used if an Activation::LeakyReLU is selected

    unsigned int seed = 0;         // RNG seed for weight init and (optionally) data shuffling
    T initMean = T(0);
    T initStddev = T(1);           // consider scaling this by layer size (Xavier/He) rather
                                    // than using one fixed value for every layer — see section 5

    bool shuffleEachEpoch = true;
    size_t logInterval = 0;        // 0 = silent; otherwise print loss every N epochs
};
```

Notes on choices made here:
- **`hiddenActivation`/`outputActivation` are two fields, not one, and not a
  per-layer list.** In practice the output layer's activation is usually
  different from every hidden layer's (e.g. `softmax` or `sigmoid` on the
  output, `relu` everywhere else), and per-layer activation flexibility isn't
  something you've asked for — this can generalize to a full per-layer list
  later if you actually need it, but starting with two fields is simpler and
  covers the overwhelmingly common case.
- **`loss` and the activation enums are runtime choices, not template
  parameters** — unlike layer sizes, there's no structural reason they need
  to be compile-time; an `enum` selected at runtime is the natural fit.
- **`batchSize` defaults to `1`** (pure per-example SGD) — mini-batch
  averaging is a real feature worth having eventually, but it's an addition
  to the training loop, not something that blocks a first working version.


## 3. Division of responsibility: `Layer` vs `NeuralNetwork`

This is the thing that resolves an otherwise-ambiguous question: **does a
`Layer` know its own activation function?**

**No — `Layer` stays activation-agnostic, purely linear algebra.** Its
`forward()` computes only `z = W*x + b` (matches the doc comment already on
it), never applying an activation. `NeuralNetwork` is what calls a layer's
`forward()`, *then* applies that layer's configured activation to get the
value that feeds into the next layer.

Why this split, not the alternative: it keeps `Layer` simple and reusable
(one weight matrix + one bias vector + two methods, nothing else), and it's
`NeuralNetwork` — which already knows the `NNConfig`, and therefore which
activation applies where — that's the natural place for activation dispatch
to live. Baking a specific activation into `Layer` itself would mean either
templating `Layer` on its activation (awkward, and doesn't match how
`NNConfig` selects activation by enum at runtime) or storing a
`std::function` per layer (unnecessary overhead for something `NeuralNetwork`
already knows).

**Concrete implication for `Layer::backward`'s `outputGradient` parameter:**
it must already be `dL/dz` (the gradient *after* accounting for this layer's
activation derivative), not `dL/da`. `NeuralNetwork` is responsible for
computing `dL/dz = hadamard(dL/da, activationPrime(z))` *before* calling
`layer.backward(...)` — see the backward-pass walkthrough in section 6.

**Concrete implication for `NeuralNetwork::forward`:** it must cache both the
pre-activation (`z`) and post-activation (`a`) output of every layer during
the forward pass, because the backward pass needs both — `z` to compute
`activationPrime(z)`, and the previous layer's `a` as the `input` argument to
`layer.backward(...)`. This is exactly the reason activation functions were
made pure/non-mutating rather than in-place, several turns back.


## 4. What's left, and how to implement each piece

### `hadamard(a, b)`
Element-wise product. One loop:
```
result[i] = a[i] * b[i]
```

### `outerProduct(a, b)`
`a` has length `N`, `b` has length `M`, result is `N×M`:
```
result[i][j] = a[i] * b[j]
```

### `Layer::forward(input)`
Already just:
```
return weights * input + bias;
```
using `Mat*Vec` and `Vec+Vec`, both already implemented and tested.

### `Layer::backward(input, dLdz, weightGradient, biasGradient)`
Given `dLdz` (see section 3 for why it's already `dL/dz`, not `dL/da`):
```
weightGradient = outerProduct(dLdz, input)      // shape OutN x InN, matches weights
biasGradient   = dLdz                            // bias contributes additively; its
                                                  // gradient is just the incoming one
return weights.transpose() * dLdz                // dL/dx — gradient to hand to the
                                                  // previous layer's backward call
```

### `meanSquaredError` / `meanSquaredErrorPrime`
```
MSE(predicted, actual)      = (1/N) * sum_i( (predicted[i] - actual[i])^2 )
MSEPrime(predicted, actual)[i] = (2/N) * (predicted[i] - actual[i])
```

### `crossEntropy` / `crossEntropyPrime`
```
CE(predicted, actual)      = -sum_i( actual[i] * log(predicted[i]) )
CEPrime(predicted, actual)[i] = -actual[i] / predicted[i]
```
**Careful with this one:** `crossEntropyPrime` as declared is the *general*
derivative of cross-entropy alone, with no knowledge of what produced
`predicted`. If your output layer specifically uses `softmax` followed by
`crossEntropy` (the standard classifier setup — see the earlier discussion
on softmax), the *combined* gradient of loss w.r.t. the pre-softmax logits
simplifies beautifully to `predicted - actual`, which is a different,
simpler thing than calling `crossEntropyPrime` on its own and separately
differentiating through `softmax`. Don't implement `crossEntropyPrime` *as*
that shortcut — implement it as the general formula above, and apply the
`predicted - actual` shortcut as a special case specifically when you know
the pairing is softmax+cross-entropy (which is where `NeuralNetwork`'s
backward pass will actually want to use it, for the output layer only).

### `randomMat` / `randomVec`
```
std::normal_distribution<T> dist(mean, stddev);
// fill every entry via dist(rng)
```
Straightforward given `std::mt19937& rng` is already threaded through.


## 5. Weight initialization — one thing to decide when you get there

Using one fixed `stddev` for every layer regardless of size is a real, if
subtle, problem: a layer with many inputs sums many terms in `Wx`, so its
pre-activation variance grows with input size unless the weight scale shrinks
to compensate. The standard fix (Xavier/Glorot, or He for ReLU-family
activations) scales `stddev` by something like `1/sqrt(InN)`. Not required
for a first working version, but worth knowing `initMean`/`initStddev` in
`NNConfig` as flat constants is a simplification you're deliberately making,
not the "correct" long-term answer.


## 6. Putting it together: forward and backward pass walkthrough

**Forward**, for a `NeuralNetwork` with `k` layers:
```
a[0] = input
for layer i = 1..k:
    z[i] = layers[i].forward(a[i-1])
    a[i] = activation_i(z[i])          // hiddenActivation for i<k, outputActivation for i==k
return a[k]                             // this is "predicted"
```
Every `z[i]` and `a[i]` must be cached (per section 3) — they're needed again
in the backward pass.

**Backward**, given `actual` (the training label) and the cached `z`/`a`:
```
dLda[k] = lossPrime(a[k], actual)                     // e.g. meanSquaredErrorPrime, or the
                                                        // softmax+crossEntropy shortcut if applicable

for layer i = k..1 (in reverse):
    dLdz[i]        = hadamard(dLda[i], activationPrime_i(z[i]))
    dLda[i-1]      = layers[i].backward(a[i-1], dLdz[i], weightGrad[i], biasGrad[i])
    // dLda[i-1] becomes "dLda" for the next iteration down

// after this loop, weightGrad[i]/biasGrad[i] are populated for every layer
```

**Gradient descent update**, per layer, once gradients are computed — needs
no new primitives at all, `Mat - Mat*scalar` already exists:
```
layers[i].weights -= learningRate * weightGrad[i]
layers[i].bias    -= learningRate * biasGrad[i]
```

**The training loop** (not yet started, no stub exists for it either):
```
for epoch in 1..config.epochs:
    if config.shuffleEachEpoch: shuffle the dataset
    for each (input, actual) in dataset (or each mini-batch, if batchSize > 1):
        predicted = forward(input)
        compute/accumulate gradients via backward pass above
        apply the gradient descent update
    if config.logInterval > 0 and epoch % config.logInterval == 0:
        print current loss
```


## 7. `NeuralNetwork` class shape

```cpp
template <typename T, size_t... LayerSizes>
class NeuralNetwork {
    // internally: a std::tuple<Layer<LayerSizes[0],LayerSizes[1],T>,
    //                          Layer<LayerSizes[1],LayerSizes[2],T>, ...>
    // built from the LayerSizes pack — this is the one piece of real template
    // metaprogramming here (turning a size_t... pack into consecutive-pair
    // Layer types); worth researching std::index_sequence-based techniques
    // for this specifically rather than being handed the answer, since it's
    // a genuinely useful pattern to understand on your own.

public:
    void init(const NNConfig<T>& cfg);   // seeds the rng from cfg.seed, randomizes every
                                          // layer's weights/bias via randomMat/randomVec

    Vec<LastSize, T> forward(const Vec<FirstSize, T>& input) const;

    // one gradient step for a single example
    void trainStep(const Vec<FirstSize, T>& input, const Vec<LastSize, T>& actual, const NNConfig<T>& cfg);

    // the full loop from section 6, over a dataset
    void train(const std::vector<std::pair<Vec<FirstSize,T>, Vec<LastSize,T>>>& dataset, const NNConfig<T>& cfg);
};
```

`FirstSize`/`LastSize` above are the first and last values in the
`LayerSizes` pack — extracting those from a parameter pack is part of the
same metaprogramming task as building the layer tuple.


## 8. Checklist — everything before you can call `train()`

- [ ] `hadamard`
- [ ] `outerProduct`
- [ ] `Layer::forward`
- [ ] `Layer::backward`
- [ ] At least one loss + derivative pair (`meanSquaredError`/`Prime`, or `crossEntropy`/`Prime`)
- [ ] `randomMat` / `randomVec`
- [ ] Decide the layer-sizing question in section 1 (recommended: compile-time/variadic)
- [ ] `NNConfig<T>` (section 2)
- [ ] `NeuralNetwork<T, LayerSizes...>` — the tuple-of-layers construction, `init`, `forward`
- [ ] `NeuralNetwork::trainStep` / `train` — the backward pass + gradient descent update (section 6)

Everything above the `NeuralNetwork` class itself (activations, and the first
six items in this checklist) can be sanity-checked in isolation, one `Layer`
at a time, before the full class exists at all.
