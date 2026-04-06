---
tags:
- setfit
- sentence-transformers
- text-classification
- generated_from_setfit_trainer
widget:
- text: Their aim may be to gain recognition or to ease the boredom of tedious work,
    but the benefits of their efforts accrue mostly to their employer. Still others
    may be motivated by benefits to society; they may volunteer to help with health
    care or environmental research by using tools and procedures to collect, analyze,
    and document informa- tion.
- text: We will see a whole new era of "lean." Data flowing to and from products will
    allow product use and activities across the value chain to be streamlined in countless
    new ways.
- text: A sizeable fraction of those replaced jobs will be made up by new ones in
    the Second Economy. But not all of them.
- text: 'You have to provide support and resources for those who are strug ­technologies
    fully: to really develop a minimum level of fluency around what the technology
    is, what it isn't, what are the limitations, what are the risks, and what are
    the opportunities. So, everyone needs to start experimenting with it, but it's
    really important to do it very carefully.'
- text: We label them train- ers, explainers, and sustainers. Humans in these roles
    will complement the tasks per - formed by cognitive technology, ensuring that
    the work of machines is both effective and responsible — that it is fair, transpar
    - ent, and auditable.
metrics:
- accuracy
pipeline_tag: text-classification
library_name: setfit
inference: true
base_model: sentence-transformers/paraphrase-mpnet-base-v2
---

# SetFit with sentence-transformers/paraphrase-mpnet-base-v2

This is a [SetFit](https://github.com/huggingface/setfit) model that can be used for Text Classification. This SetFit model uses [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2) as the Sentence Transformer embedding model. A [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance is used for classification.

The model has been trained using an efficient few-shot learning technique that involves:

1. Fine-tuning a [Sentence Transformer](https://www.sbert.net) with contrastive learning.
2. Training a classification head with features from the fine-tuned Sentence Transformer.

## Model Details

### Model Description
- **Model Type:** SetFit
- **Sentence Transformer body:** [sentence-transformers/paraphrase-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-mpnet-base-v2)
- **Classification head:** a [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) instance
- **Maximum Sequence Length:** 512 tokens
- **Number of Classes:** 5 classes
<!-- - **Training Dataset:** [Unknown](https://huggingface.co/datasets/unknown) -->
<!-- - **Language:** Unknown -->
<!-- - **License:** Unknown -->

### Model Sources

- **Repository:** [SetFit on GitHub](https://github.com/huggingface/setfit)
- **Paper:** [Efficient Few-Shot Learning Without Prompts](https://arxiv.org/abs/2209.11055)
- **Blogpost:** [SetFit: Efficient Few-Shot Learning Without Prompts](https://huggingface.co/blog/setfit)

### Model Labels
| Label | Examples                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|:------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4     | <ul><li>'The Week recently cited A.I. experts who say there's a 50% chance of a computer with true human intelligence by 2050, with unknowable — and perhaps dire — consequences for us mortals. As technology continues to advance, will hard-earned, experience- based knowledge become obsolete? At least for now, a lot of knowledgeable people are pretty skeptical that A.I. will fully replace human "wetware" in the near future.'</li><li>'These rates of progress are embedded in the creation of intelligent machines, from robots to automobiles to drones, that will soon dominate the global economy – and in the process drive down the value of human labor with astonishing speed. This is why we will soon be looking at hordes of citizens of zero economic value.'</li><li>'Ultimately, we need a new, individualized, cultural , approach to the meaning of work and the purpose of life. Otherwise, people will find a solution – human beings always do – but it may not be the one for which we began this technological revolution.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| 0     | <ul><li>'How many oncologists could hope to read, much less remember, the 600,000 pieces of medical evidence and two million journal pages describing research and trials on lung cancer that have been fed to Watson? With all this information, Watson lists possible courses of treatment, assigns levels of confidence to each, and provides evidence for those recommendations.'</li><li>'But for the foreseeable future it remains necessary for human experts to weigh the recommendations, recognize patterns from past experience, and make the final decision. We are beginning to figure out an optimal partnership with computerized intelligence.'</li><li>'Even advocates of Big Data point out that analysis still requires human expertise to derive meaning, understand context, correctly distinguish correlation from causation, and make nuanced decisions. Summarizing the work of Harvard computer science experts , the goal is "to create systems that let humans combine what they are good at — asking the right questions and interpreting the results — with what machines are good at: computation, analysis, and statistics using large datasets." So until the androids take over, smart software and big data are merely very useful tools to help us work.'</li></ul>                                                                                                                                                                                                                                                 |
| 1     | <ul><li>'Machines replace many kinds of repetitive work, from flying airplanes to sorting through medical symptoms. And to the extent that deeply smart humans can program potential problems into the software — even relatively rare ones — the system can react faster than a human.'</li><li>'Those of the future, by substituting for man's senses and brain, will accelerate that process – but at the risk of creating millions of citizens who are simply unable to contribute economically, and with greater damage to an already declining middle class. 2 COPYRIGHT © 2014 HARVARD BUSINESS SCHOOL PUBLISHING CORPORATION.'</li><li>'Figuring out how to deal with the impacts of this development will be the greatest challenge facing free market economies in this century. If you doubt the march of worker-replacing technology, look at Foxconn , the world's largest contract manufacturer.'</li></ul>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2     | <ul><li>'Meanwhile, Brooks's solutions will lead only to bigger government and greater command and control. And it is difficult to imagine how such a sluggish government system could keep up with such a rapid rate of change when it can barely do so now.'</li><li>'To combat bias in AI, companies need more diverse AI talent. Sophisticated, innovative companies are increasingly abandoning prejudicially-fraught resume screening for project-based assessment.'</li><li>'ETHICS AI's Real Risk by Michael Schrage DECEMBER 16, 2015 The billion-dollar OpenAI initiative announced last week by Elon Musk and company recalls DARPA's Grand Challenge, the X-Prize, and MIT Media Lab's One Laptop Per Child initiative – innovative institutional mechanisms explicitly designed to attract top talent and worldwide attention to worthy problems. They can work well, and my bet is that the OpenAI will goad a similar competitive response — smart companies such as Google, Facebook, Apple, Amazon, Baidu, and Alibaba will immediately recognize that the so-called "not for profit" issues the researchers identify need to be incorporated in their own AI/ML technology roadmaps.'</li></ul>                                                                                                                                                                                                                                                                                                                                      |
| 3     | <ul><li>'The New Product Capabilities To fully grasp how smart, connected products are changing how companies work, we must first under - stand their inherent components, technology, and capabilities—something that our previous article examined. To recap: All smart, connected products, from home ap - pliances to industrial equipment, share three core elements: physical components (such as mechani - cal and electrical parts); smart components (sensors, microprocessors, data storage, controls, software, an embedded operating system, and a digital user inter - face); and connectivity components (ports, antennae, protocols, and networks that enable communication between the product and the product cloud, which runs on remote servers and contains the product's external operating system).'</li><li>'In fleets of vehicles, information about the pending service needs of each car or truck, and its location, allows service departments to stage parts, sched - ule maintenance, and increase the efficiency of repairs. Data on warranty status becomes more valuable when combined with data on product use Smart, connected products require a rethinking of design.'</li><li>'Product designs now need to incorporate additional instrumentation, data collection capability, and diagnostic software fea - tures that monitor product health and performance and warn service personnel of failures. And as software increases functionality, products can be designed to allow more remote service.'</li></ul> |

## Uses

### Direct Use for Inference

First install the SetFit library:

```bash
pip install setfit
```

Then you can load this model and run inference.

```python
from setfit import SetFitModel

# Download from the 🤗 Hub
model = SetFitModel.from_pretrained("setfit_model_id")
# Run inference
preds = model("A sizeable fraction of those replaced jobs will be made up by new ones in the Second Economy. But not all of them.")
```

<!--
### Downstream Use

*List how someone could finetune this model on their own dataset.*
-->

<!--
### Out-of-Scope Use

*List how the model may foreseeably be misused and address what users ought not to do with the model.*
-->

<!--
## Bias, Risks and Limitations

*What are the known or foreseeable issues stemming from this model? You could also flag here known failure cases or weaknesses of the model.*
-->

<!--
### Recommendations

*What are recommendations with respect to the foreseeable issues? For example, filtering explicit content.*
-->

## Training Details

### Training Set Metrics
| Training set | Min | Median  | Max |
|:-------------|:----|:--------|:----|
| Word count   | 12  | 49.8139 | 176 |

| Label | Training Sample Count |
|:------|:----------------------|
| 0     | 120                   |
| 1     | 121                   |
| 2     | 54                    |
| 3     | 73                    |
| 4     | 35                    |

### Training Hyperparameters
- batch_size: (4, 4)
- num_epochs: (1, 16)
- max_steps: -1
- sampling_strategy: oversampling
- num_iterations: 10
- body_learning_rate: (2e-05, 1e-05)
- head_learning_rate: 0.01
- loss: CosineSimilarityLoss
- distance_metric: cosine_distance
- margin: 0.25
- end_to_end: False
- use_amp: False
- warmup_proportion: 0.1
- l2_weight: 0.01
- seed: 42
- eval_max_steps: -1
- load_best_model_at_end: False

### Training Results
| Epoch  | Step | Training Loss | Validation Loss |
|:------:|:----:|:-------------:|:---------------:|
| 0.0005 | 1    | 0.1993        | -               |
| 0.0248 | 50   | 0.2525        | -               |
| 0.0496 | 100  | 0.2207        | -               |
| 0.0744 | 150  | 0.2251        | -               |
| 0.0993 | 200  | 0.2364        | -               |
| 0.1241 | 250  | 0.1922        | -               |
| 0.1489 | 300  | 0.2176        | -               |
| 0.1737 | 350  | 0.2065        | -               |
| 0.1985 | 400  | 0.1667        | -               |
| 0.2233 | 450  | 0.1752        | -               |
| 0.2481 | 500  | 0.1407        | -               |
| 0.2730 | 550  | 0.1596        | -               |
| 0.2978 | 600  | 0.1099        | -               |
| 0.3226 | 650  | 0.1567        | -               |
| 0.3474 | 700  | 0.1214        | -               |
| 0.3722 | 750  | 0.1093        | -               |
| 0.3970 | 800  | 0.0947        | -               |
| 0.4218 | 850  | 0.0744        | -               |
| 0.4467 | 900  | 0.0706        | -               |
| 0.4715 | 950  | 0.0686        | -               |
| 0.4963 | 1000 | 0.0522        | -               |
| 0.5211 | 1050 | 0.0334        | -               |
| 0.5459 | 1100 | 0.0514        | -               |
| 0.5707 | 1150 | 0.0206        | -               |
| 0.5955 | 1200 | 0.0259        | -               |
| 0.6203 | 1250 | 0.0446        | -               |
| 0.6452 | 1300 | 0.0216        | -               |
| 0.6700 | 1350 | 0.0126        | -               |
| 0.6948 | 1400 | 0.0188        | -               |
| 0.7196 | 1450 | 0.0106        | -               |
| 0.7444 | 1500 | 0.0055        | -               |
| 0.7692 | 1550 | 0.0233        | -               |
| 0.7940 | 1600 | 0.009         | -               |
| 0.8189 | 1650 | 0.0054        | -               |
| 0.8437 | 1700 | 0.0093        | -               |
| 0.8685 | 1750 | 0.0073        | -               |
| 0.8933 | 1800 | 0.011         | -               |
| 0.9181 | 1850 | 0.0051        | -               |
| 0.9429 | 1900 | 0.0131        | -               |
| 0.9677 | 1950 | 0.0008        | -               |
| 0.9926 | 2000 | 0.0137        | -               |

### Framework Versions
- Python: 3.9.6
- SetFit: 1.1.2
- Sentence Transformers: 4.1.0
- Transformers: 4.52.4
- PyTorch: 2.7.0
- Datasets: 3.6.0
- Tokenizers: 0.21.1

## Citation

### BibTeX
```bibtex
@article{https://doi.org/10.48550/arxiv.2209.11055,
    doi = {10.48550/ARXIV.2209.11055},
    url = {https://arxiv.org/abs/2209.11055},
    author = {Tunstall, Lewis and Reimers, Nils and Jo, Unso Eun Seo and Bates, Luke and Korat, Daniel and Wasserblat, Moshe and Pereg, Oren},
    keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {Efficient Few-Shot Learning Without Prompts},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

<!--
## Glossary

*Clearly define terms in order to be accessible across audiences.*
-->

<!--
## Model Card Authors

*Lists the people who create the model card, providing recognition and accountability for the detailed work that goes into its construction.*
-->

<!--
## Model Card Contact

*Provides a way for people who have updates to the Model Card, suggestions, or questions, to contact the Model Card authors.*
-->

# Vision Dynamics Analysis Project

This project analyzes the dynamics of different vision archetypes and their rhetorical stances over time.

## Project Structure

```
.
├── data/
│   ├── raw/                  # Original, immutable data
│   │   ├── future_paragraphs_with_type.csv
│   │   └── all_para_with_future_types.csv
│   └── processed/            # Cleaned and processed data
│       ├── merged_analysis.csv
│       ├── rhetorical_stance_analysis.csv
│       └── archetype_stance_timeseries.csv
│
├── src/
│   ├── analysis/            # Analysis scripts
│   │   ├── vision_dynamics_analysis_interactive.py
│   │   ├── vision_dynamics_analysis_absolute.py
│   │   ├── archetype_dynamics_analysis.py
│   │   ├── merge_and_analyze.py
│   │   ├── analyze_rhetorical_stance.py
│   │   └── analyze_inter_coder_reliability.py
│   │
│   └── visualization/       # Visualization scripts
│       ├── faceted_stacked_area.py
│       └── alluvial_plot.py
│
├── results/
│   ├── figures/            # Generated plots and visualizations
│   │   ├── archetype_stance_timeseries.png
│   │   ├── correlation_matrix.png
│   │   ├── impulse_responses.png
│   │   └── ...
│   │
│   └── tables/            # Generated tables and statistics
│
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Analysis Scripts

### Main Analysis Scripts
- `vision_dynamics_analysis_interactive.py`: Interactive analysis of vision dynamics with user-specified parameters
- `vision_dynamics_analysis_absolute.py`: Analysis using absolute counts
- `archetype_dynamics_analysis.py`: Analysis focusing on archetypes only

### Supporting Scripts
- `merge_and_analyze.py`: Merges and processes the initial data
- `analyze_rhetorical_stance.py`: Analyzes rhetorical stances
- `analyze_inter_coder_reliability.py`: Analyzes reliability between different coders

### Visualization Scripts
- `faceted_stacked_area.py`: Creates faceted stacked area plots
- `alluvial_plot.py`: Generates alluvial plots for flow analysis

## Usage

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the interactive analysis:
```bash
python src/analysis/vision_dynamics_analysis_interactive.py
```

3. View results in the `results/figures/` directory

## Data

- Raw data is stored in `data/raw/`
- Processed data is stored in `data/processed/`
- Generated figures are stored in `results/figures/`
- Generated tables are stored in `results/tables/` 