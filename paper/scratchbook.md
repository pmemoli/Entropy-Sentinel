
## Related work thats relevant

### Estimating OOD performance

The problem we are fundamentally solving is one of estimating OOD performance based on a small set of target data, which is strongly correlated (miller et al). There is rich literature on OOD performance estimation on neural networks (Garg et. al, 2022; Deng & Zheng, 2021; Jiang et al., 2021; Deng et al., 2021; Chen et al., 2021a, agreement on the line papers), where the established baseline is the *Average Confidence Threshold (ATC)* (Garg et al, 2022). While ATC is simple, cheap and directly applicable to our problem, it is not adapted specifically to LLMs and is just based on a single statistic, rather than on the fine grained token sequences produced by LLMs and UQ metrics.

OOD performance estimation was studied specifically for foundational models by Saxena et. al (2024), based on the observation that ensembles of neural networks present "agreement on the line" (baek. et al, 2022). The problem with this approach is that it requires multiple inferences, making it unsuitable to our monitoring use case.

SURVEY: https://arxiv.org/pdf/2201.04234 

### Uncertainty Quantification

Following recent uncertainty quantification surveys (shorinwa, bouchard, xiaoou, huang, tonmoy), methods are typically classified according to their objective and access to the models internals.

At the model internal level, methods are classified into "black box" methods, where only the model token output response is available. And "white box" methods, where there is partial or total access to the model.

At the objective level, these consist of semantic similarity, where multiple answers are generated and the semantic differences are computed, token level-uq, where uncertainties from each token are aggregated somehow, and self consistency or LLM-as-a-judge methods. Plus the possibility of ensembles.

In our case, we require signals that are cheap to acquire (i), and available to open and closed source models (ii). From the family of UQ methods, semantic uncerainty and llm-as-a-judge are discarded due to them being expensive at scale. This leaves token-level uq, which invariably relies on logit signals from apis. Since many comerical apis make the top-k logits available, we explore which white box token-level uq signals provide discriminatory potential.

### Calibration

The field of calibration is parallel to UQ and has been extensively studied (geng, etc. etc.). It concerns how calibrated the models logprobs are compared to the true probabilities of the next token distributions. 

Recent research (plaut et al, 2025) has shown that while LLMs tend to be miscalibrated (He et al., 2023; OpenAI, 2023; Zhu et al., 2023), the probabilities still align with correctness on multiple choice QA tasks. Our experiments essentially measure how calibrated the entire entropy distribution and UQ metrics are to actual slice level correctness for OOD slices, which is different from traditional calibration for the language modeling task.

### Lenses & mechanistic interpretability

We also include entrpy statistics based on the results from entropy lens (ali et al), where the shannon entropy distribution is found to provide a model signature that distinguishes **correctness** and the **model type**. This is done in the wider context of a *lens* in the framework of mechanistic interpretability (Bereska & Gavves, 2024), where a probe is introduced within the transformer to characterize LLM computations.

### LLM as a judge for monitoring

A Survey on LLM-as-a-Judge" (Gu et al. 2024): This is a very long survey. Nothing much relevant here, but we can cite it.

LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods" (Li et al, 2024): Same as above. No mention of monitoring. 

From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge" (Dawei Li et al, 2024): "" "" 

Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena: Sooo cool.

For frameworks:

Langfuse, LangSmith, Arize Phoenix and Comet Opik.

### RQ

Can entropy signals through a simple probe measure response quality for verifiably correct LLM responses and subjective ones?

### Intro

Deployed LLMs serve heterogeneous traffic that shifts over time. Yet practitioners still lack scalable answers to two tightly coupled questions: \emph{Where is the model underperforming on current usage?} and \emph{Where should we intervene to close those gaps?} In practice, these questions are addressed either with manually curated benchmarks—expensive, slow to update, and vulnerable to contamination—or with LLM-as-a-judge evaluation over production traces, which showcase remarkable agreement with human judgement (mt bench source). While judge-based monitoring has become the de facto standard with frameworks like Langfuse (source) and Langsmith (source), its cost scales linearly with traffic volume, restricting evaluation in practice to just a small portion of production data and leaving most traffic unmonitored (source).

\begin{figure}[h]
    \centering
    \includegraphics[width=0.8\linewidth]{phi3_example.png}
    \caption{Entropy-based accuracy estimation for
    PHI-3.5-MINI-3.6B. Trained on two benchmarks (orange),
    the estimator generalizes to eight unseen STEM benchmarks
    (blue).}
    \label{fig:phi3example}
\end{figure}

We explore the alternative of creating a probe for some measure of \emph{correctness} based on signals naturally produced during inference and available to all practitioners at minimal cost. If an inexpensive uncertainty trace could \emph{robustly} predict correctness, then we could score slices of production traffic without repeated labeling or external large models. Once such estimates are available, they immediately induce a ranking of slices by predicted correctness, which could guide downstream decisions such as domain intervention prioritization. For the underlying signal to be useful for monitoring at scale, the probe must be (i) based on signals that are cheap to extract and (ii) available for both open and closed models (e.g., via top-$k$ log-probabilities), (iii) robust to domain shift, and (iv) capable of aligning with judge scores in a realistic monitoring context.

For this task we propose \textbf{Entropy Sentinel} (ES), which satisfies (i) and (ii) by construction. For each response, ES summarizes the output-entropy trajectory computed from top-$k$ next-token log-probabilities, which is a signal exposed by most model-serving APIs, into a compact feature vector, and trains a lightweight probabilistic classifier to predict instance-level correctness against any per-instance quality label—verifiable answers in a controlled setting, or judge scores for monitoring. Averaging predicted probabilities over a slice then yields a slice-level performance estimate. The entire pipeline requires a single standard inference pass, no hidden states, and no auxiliary large models, so its marginal cost over serving is negligible.

We evaluate Entropy Sentinel's capability for (iii) by framing robustness to domain shift as \emph{estimating out-of-distribution (OOD) performance on unlabeled slices} (source). We test this on ten STEM QA benchmarks, whose verifiable correctness enables an exhaustive study of ES's configuration space: for each $k\in\{1,2,3,4\}$ we train on all $\binom{10}{k}$ benchmark subsets and estimate accuracy on the remaining $10-k$, across nine LLMs (3B--20B, six families) and multiple estimator variants, totaling $>160{,}000$ configurations. Estimates often track held-out accuracy closely, with some LLMs presenting almost perfect calibration (Figure~\ref{fig:phi3example}). Across this sweep, the dominant design factor is the supervision composition, where difficulty-spanning training sets generalize substantially better than homogeneous ones.

To evaluate the capability of Entropy Sentinel of aligning with judge scores on OOD slices in a realistic monitoring situation (iv), we train ES on judge scores over multi-turn chat benchmarks grouped by category (emulating production traces clustered post hoc), and evaluate whether it recovers the judge's category-level scores and rankings on held-out sources. [Results]

[Main Takeaways]


## Experiments

We perform two experiments evaluating entropy sentinel for different tasks: 

(1) Is ES suitable for estimation of OOD accuracy in objective verifiable tasks?

This in turns provides us with substantive analysis and ablations a controlled environment for the design choices in Entropy Sentinel.

(2) Can ES be used for large scale domain monitoring by learning to score from an external judge?

### Evaluation on estimating OOD performance for verifiable STEM tasks.

We want to explore how capable entropy sentinel is at detecting **correctness** for OOD slices in black and white STEM QA tasks. This evaluates whether ES is suitable for the traditional estimating OOD performance task.

#### Models

- The 9 models balblaba

#### Benchmarks

- The 9 benchmarks

#### RQs

1. Does entropy sentinel support OOD accuracy estimation under plausible defaults? 
2. How does it compare with the ATC baseline?
3. What are the most impactful estimator design choices?

#### Results

[Aca CLAVE pasar las tablas con los resultados claves que van a quedar]

- Tabla 2
- Tabla 4
- Ablacion de features
- Figura U
- Decoupling slice MAE vs AUROC
- Estimaciones accuracy (puede abreviarse)

- Some models have their entropy distribution remarkably calibrated with correctness on STEM QA tasks at slice level.
- MAE of accuracy quality at the benchmark level is decoupled from single answer accuracy R < 0.6.
- The primary design choice --above features, model type and calibration-- is a representative difficulty spanning dataset.

### Evaluation for monitoring (distilling llm as a judge).

We want to study the capability of entropy sentinel to emulate LLM-as-a-judge scores and rankings on production traces grouped by category. Emulating the setup of the previous experiment, we select X benchmarks that measure the capabilities of language models for multi-turn chats, already grouped by category (math, reasoning, debugging, etc.). This situation emulates real monitoring for chatbots, where conversations are grouped by category post-hoc and monitored with some framework such as langfuse.

The RQ is: "Can entropy sentinel learn to rank LLM-as-a-judge scores for OOD domains?"

#### Models

Same 9 models, as before.

#### Benchmarks

We are evaluating the model on WildBench, a dataset of ~1000 curated samples from chatbot arena already clustered by 11 categories.

Original idea:

1. WildBench (1k)
2. MT-Bench (80x2)
3. Arena-Hard-Auto (500)

And maybe (probably I really should...):

4. BIGGEN-Bench (765)
5. No Robots (10k) (sample just 1000 from each category)

So about 2k samples per model... 

#### Evaluation protocol

Considering classifier choice was a second order decision compared to training composition, we fix the best performing model (RandomForest) from the previous section and sweep all **categories** 1C11...7C11 for training, and evaluate on the remainder ones. This tests the capability of ES of generalizing to OOD domains, and lets us study what training composition performs best.

The only difference with the STEM QA protocol is that we dont evaluate on a separate test split for the training categories due to limited category size.

##### Can it agree with the judge on OOD domains? 

We show spearman (and aee?) median-iqr for k = 1...7 and have a final LOCO (leave one category out) row. 

[TABLE]

[LOCO GRAPH]

##### Takeaways

[Once I have results]

#### Prompts

We'll run each benchmark with `single-v1`:

https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/data/judge_prompts.jsonl

For single turn evaluation:

```
[System] Please act as an impartial judge and evaluate the quality of
the response provided by an AI assistant to the user question below.
Your evaluation should consider factors such as helpfulness, relevance,
accuracy, depth, creativity, and level of detail. Begin your evaluation
by providing a short explanation. Be as objective as possible. After
your explanation, rate the response on a scale of 1 to 10 by strictly
following this format: "Rating: \[\[5\]\]"

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]
```

For two-turn evaluation:

```
[System] Please act as an impartial judge and evaluate the quality of
the response provided by an AI assistant to the user question
displayed below. Your evaluation should consider factors such as the
helpfulness, relevance, accuracy, depth, creativity, and level of
detail of the response. Your evaluation should focus on the
assistant's answer to the second user question. Begin your evaluation
by providing a short explanation. Be as objective as possible. After
providing your explanation, you must rate the response on a scale of
1 to 10 by strictly following this format: "Rating: [[5]]".

<|The Start of Assistant A's Conversation with User|>

### User:
{question_1}

### Assistant A:
{answer_1}

### User:
{question_2}

### Assistant A:
{answer_2}

<|The End of Assistant A's Conversation with User|>
```

#### Appendix ablations

Same thing but with raw length. Ideally they differ considerably.

## TODO

HOY (noche):

. Dejar corriendo la generacion de modelos para la parte 1.
. Dejar corriendo WILDBENCH a la noche para la parte 2.

TOMORROW (martes):

1. Reproducir experimentos de la parte 1 y retocar la parte 1 del paper.
(2). Length ablation simil wildbench.
3. Dejar corriendo WILDBENCH para la parte 2.

PASADO (miercoles):

1. Correr experimentos de la parte 2 y guardar datos.
2. Empezar a escribir esa parte.

Para el Jueves:

1. Terminar de escribir exp 2. 
2. Escribir intro, abstract y que el hilo sea coherente.

Para el Viernes:

1. Terminar de escribir el paper.
2. Pensar y correr ablaciones de length & temp para stem qa. 

Sabado:
1. Leer y releer el paper. Mandar al prox ciclo de ACL.
