# arXiv AI Papers - 2025-09-05

Total papers: 357

## 1. Foundations and Models in Modern Computer Vision: Key Building Blocks in   Landmark Architectures

**Authors:** Radu-Andrei Bourceanu, Neil De La Fuente, Jan Grimm, Andrei Jardan, Andriy Manucharyan, Cornelius Weiss, Daniel Cremers, Roman Pflugfelder

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2507.23357v2) | [PDF](http://arxiv.org/pdf/2507.23357v2)

**Abstract:** This report analyzes the evolution of key design patterns in computer vision
by examining six influential papers. The analysis begins with foundational
architectures for image recognition. We review ResNet, which introduced
residual connections to overcome the vanishing gradient problem and enable
effective training of significantly deeper convolutional networks.
Subsequently, we examine the Vision Transformer (ViT), which established a new
paradigm by applying the Transformer architecture to sequences of image
patches, demonstrating the efficacy of attention-based models for large-scale
image recognition. Building on these visual representation backbones, we
investigate generative models. Generative Adversarial Networks (GANs) are
analyzed for their novel adversarial training process, which challenges a
generator against a discriminator to learn complex data distributions. Then,
Latent Diffusion Models (LDMs) are covered, which improve upon prior generative
methods by performing a sequential denoising process in a perceptually
compressed latent space. LDMs achieve high-fidelity synthesis with greater
computational efficiency, representing the current state-of-the-art for image
generation. Finally, we explore self-supervised learning techniques that reduce
dependency on labeled data. DINO is a self-distillation framework in which a
student network learns to match the output of a momentum-updated teacher,
yielding features with strong k-NN classification performance. We conclude with
Masked Autoencoders (MAE), which utilize an asymmetric encoder-decoder design
to reconstruct heavily masked inputs, providing a highly scalable and effective
method for pre-training large-scale vision models.

**Relevance Score:** 8.50

---

## 2. Attn-Adapter: Attention Is All You Need for Online Few-shot Learner of   Vision-Language Model

**Authors:** Phuoc-Nguyen Bui, Khanh-Binh Nguyen, Hyunseung Choo

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03895v1) | [PDF](http://arxiv.org/pdf/2509.03895v1)

**Abstract:** Contrastive vision-language models excel in zero-shot image recognition but
face challenges in few-shot scenarios due to computationally intensive offline
fine-tuning using prompt learning, which risks overfitting. To overcome these
limitations, we propose Attn-Adapter, a novel online few-shot learning
framework that enhances CLIP's adaptability via a dual attention mechanism. Our
design incorporates dataset-specific information through two components: the
Memory Attn-Adapter, which refines category embeddings using support examples,
and the Local-Global Attn-Adapter, which enriches image embeddings by
integrating local and global features. This architecture enables dynamic
adaptation from a few labeled samples without retraining the base model.
Attn-Adapter outperforms state-of-the-art methods in cross-category and
cross-dataset generalization, maintaining efficient inference and scaling
across CLIP backbones.

**Comment:** ICCV 2025 - LIMIT Workshop

**Relevance Score:** 8.50

---

## 3. Arabic Chatbot Technologies in Education: An Overview

**Authors:** Hicham Bourhil, Yacine El Younoussi

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04066v1) | [PDF](http://arxiv.org/pdf/2509.04066v1)

**Abstract:** The recent advancements in Artificial Intelligence (AI) in general, and in
Natural Language Processing (NLP) in particular, and some of its applications
such as chatbots, have led to their implementation in different domains like
education, healthcare, tourism, and customer service. Since the COVID-19
pandemic, there has been an increasing interest in these digital technologies
to allow and enhance remote access. In education, e-learning systems have been
massively adopted worldwide. The emergence of Large Language Models (LLM) such
as BERT (Bidirectional Encoder Representations from Transformers) and GPT
(Generative Pre-trained Transformers) made chatbots even more popular. In this
study, we present a survey on existing Arabic chatbots in education and their
different characteristics such as the adopted approaches, language variety, and
metrics used to measure their performance. We were able to identified some
research gaps when we discovered that, despite the success of chatbots in other
languages such as English, only a few educational Arabic chatbots used modern
techniques. Finally, we discuss future directions of research in this field.

**Comment:** Published as a book chapter in: Transformaci\'on Digital en la
  Educaci\'on: Innovaciones y Desaf\'ios desde los Campus Virtuales (UA
  Journals, 2024), pp. 11-14

**Relevance Score:** 8.50

---

## 4. Plan Verification for LLM-Based Embodied Task Completion Agents

**Authors:** Ananth Hariharan, Vardhan Dongre, Dilek Hakkani-Tür, Gokhan Tur

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.02761v2) | [PDF](http://arxiv.org/pdf/2509.02761v2)

**Abstract:** Large language model (LLM) based task plans and corresponding human
demonstrations for embodied AI may be noisy, with unnecessary actions,
redundant navigation, and logical errors that reduce policy quality. We propose
an iterative verification framework in which a Judge LLM critiques action
sequences and a Planner LLM applies the revisions, yielding progressively
cleaner and more spatially coherent trajectories. Unlike rule-based approaches,
our method relies on natural language prompting, enabling broad generalization
across error types including irrelevant actions, contradictions, and missing
steps. On a set of manually annotated actions from the TEACh embodied AI
dataset, our framework achieves up to 90% recall and 100% precision across four
state-of-the-art LLMs (GPT o4-mini, DeepSeek-R1, Gemini 2.5, LLaMA 4 Scout).
The refinement loop converges quickly, with 96.5% of sequences requiring at
most three iterations, while improving both temporal efficiency and spatial
action organization. Crucially, the method preserves human error-recovery
patterns rather than collapsing them, supporting future work on robust
corrective behavior. By establishing plan verification as a reliable LLM
capability for spatial planning and action refinement, we provide a scalable
path to higher-quality training data for imitation learning in embodied AI.

**Relevance Score:** 7.50

---

## 5. PAK-UCB Contextual Bandit: An Online Learning Approach to Prompt-Aware   Selection of Generative Models and LLMs

**Authors:** Xiaoyan Hu, Ho-fung Leung, Farzan Farnia

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2410.13287v6) | [PDF](http://arxiv.org/pdf/2410.13287v6)

**Abstract:** Selecting a sample generation scheme from multiple prompt-based generative
models, including large language models (LLMs) and prompt-guided image and
video generation models, is typically addressed by choosing the model that
maximizes an averaged evaluation score. However, this score-based selection
overlooks the possibility that different models achieve the best generation
performance for different types of text prompts. An online identification of
the best generation model for various input prompts can reduce the costs
associated with querying sub-optimal models. In this work, we explore the
possibility of varying rankings of text-based generative models for different
text prompts and propose an online learning framework to predict the best data
generation model for a given input prompt. The proposed PAK-UCB algorithm
addresses a contextual bandit (CB) setting with shared context variables across
the arms, utilizing the generated data to update kernel-based functions that
predict the score of each model available for unseen text prompts.
Additionally, we leverage random Fourier features (RFF) to accelerate the
online learning process of PAK-UCB. Our numerical experiments on real and
simulated text-to-image and image-to-text generative models show that RFF-UCB
performs successfully in identifying the best generation model across different
sample types. The code is available at:
github.com/yannxiaoyanhu/dgm-online-select.

**Comment:** accepted to ICML 2025

**Relevance Score:** 7.50

---

## 6. SMooGPT: Stylized Motion Generation using Large Language Models

**Authors:** Lei Zhong, Yi Yang, Changjian Li

**Categories:** cs.GR, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04058v1) | [PDF](http://arxiv.org/pdf/2509.04058v1)

**Abstract:** Stylized motion generation is actively studied in computer graphics,
especially benefiting from the rapid advances in diffusion models. The goal of
this task is to produce a novel motion respecting both the motion content and
the desired motion style, e.g., ``walking in a loop like a Monkey''. Existing
research attempts to address this problem via motion style transfer or
conditional motion generation. They typically embed the motion style into a
latent space and guide the motion implicitly in a latent space as well. Despite
the progress, their methods suffer from low interpretability and control,
limited generalization to new styles, and fail to produce motions other than
``walking'' due to the strong bias in the public stylization dataset. In this
paper, we propose to solve the stylized motion generation problem from a new
perspective of reasoning-composition-generation, based on our observations: i)
human motion can often be effectively described using natural language in a
body-part centric manner, ii) LLMs exhibit a strong ability to understand and
reason about human motion, and iii) human motion has an inherently
compositional nature, facilitating the new motion content or style generation
via effective recomposing. We thus propose utilizing body-part text space as an
intermediate representation, and present SMooGPT, a fine-tuned LLM, acting as a
reasoner, composer, and generator when generating the desired stylized motion.
Our method executes in the body-part text space with much higher
interpretability, enabling fine-grained motion control, effectively resolving
potential conflicts between motion content and style, and generalizes well to
new styles thanks to the open-vocabulary ability of LLMs. Comprehensive
experiments and evaluations, and a user perceptual study, demonstrate the
effectiveness of our approach, especially under the pure text-driven stylized
motion generation.

**Relevance Score:** 7.50

---

## 7. CANDY: Benchmarking LLMs' Limitations and Assistive Potential in Chinese   Misinformation Fact-Checking

**Authors:** Ruiling Guo, Xinwei Yang, Chen Huang, Tong Zhang, Yong Hu

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03957v1) | [PDF](http://arxiv.org/pdf/2509.03957v1)

**Abstract:** The effectiveness of large language models (LLMs) to fact-check
misinformation remains uncertain, despite their growing use. To this end, we
present CANDY, a benchmark designed to systematically evaluate the capabilities
and limitations of LLMs in fact-checking Chinese misinformation. Specifically,
we curate a carefully annotated dataset of ~20k instances. Our analysis shows
that current LLMs exhibit limitations in generating accurate fact-checking
conclusions, even when enhanced with chain-of-thought reasoning and few-shot
prompting. To understand these limitations, we develop a taxonomy to categorize
flawed LLM-generated explanations for their conclusions and identify factual
fabrication as the most common failure mode. Although LLMs alone are unreliable
for fact-checking, our findings indicate their considerable potential to
augment human performance when deployed as assistive tools in scenarios. Our
dataset and code can be accessed at https://github.com/SCUNLP/CANDY

**Comment:** Findings of EMNLP 2025

**Relevance Score:** 7.50

---

## 8. MiniCPM4: Ultra-Efficient LLMs on End Devices

**Authors:**  MiniCPM Team, Chaojun Xiao, Yuxuan Li, Xu Han, Yuzhuo Bai, Jie Cai, Haotian Chen, Wentong Chen, Xin Cong, Ganqu Cui, Ning Ding, Shengda Fan, Yewei Fang, Zixuan Fu, Wenyu Guan, Yitong Guan, Junshao Guo, Yufeng Han, Bingxiang He, Yuxiang Huang, Baoxi Ji, Cunliang Kong, Qiuzuo Li, Siyuan Li, Wenhao Li, Xin Li, Yanghao Li, Yishan Li, Zhen Li, Dan Liu, Biyuan Lin, Yankai Lin, Xiang Long, Quanyu Lu, Yaxi Lu, Peiyan Luo, Hongya Lyu, Litu Ou, Yinxu Pan, Lushi Pu, Zekai Qu, Qundong Shi, Zijun Song, Jiayuan Su, Zhou Su, Ao Sun, Xianghui Sun, Peijun Tang, Fangzheng Wang, Feng Wang, Shuo Wang, Yudong Wang, Zheng Wang, Yesai Wu, Zhenyu Xiao, Jie Xie, Zihao Xie, Xiaoyue Xu, Yukun Yan, Jiarui Yuan, Jinqian Zhang, Kaihuo Zhang, Lei Zhang, Linyue Zhang, Xueren Zhang, Yudi Zhang, Hengyu Zhao, Weilin Zhao, Weilun Zhao, Yuanqian Zhao, Zhi Zheng, Chuyue Zhou, Ge Zhou, Jie Zhou, Wei Zhou, Yanghao Zhou, Zihan Zhou, Zixuan Zhou, Zhiyuan Liu, Guoyang Zeng, Chao Jia, Dahai Li, Maosong Sun

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2506.07900v2) | [PDF](http://arxiv.org/pdf/2506.07900v2)

**Abstract:** This paper introduces MiniCPM4, a highly efficient large language model (LLM)
designed explicitly for end-side devices. We achieve this efficiency through
systematic innovation in four key dimensions: model architecture, training
data, training algorithms, and inference systems. Specifically, in terms of
model architecture, we propose InfLLM v2, a trainable sparse attention
mechanism that accelerates both prefilling and decoding phases for long-context
processing. Regarding training data, we propose UltraClean, an efficient and
accurate pre-training data filtering and generation strategy, and UltraChat v2,
a comprehensive supervised fine-tuning dataset. These datasets enable
satisfactory model performance to be achieved using just 8 trillion training
tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient
pre-training strategy search, and improve existing post-training methods by
introducing chunk-wise rollout for load-balanced reinforcement learning and
data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose
CPM.cu that integrates sparse attention, model quantization, and speculative
sampling to achieve efficient prefilling and decoding. To meet diverse
on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B
parameters, respectively. Furthermore, we construct a hybrid reasoning model,
MiniCPM4.1, which can be used in both deep reasoning mode and non-reasoning
mode. Evaluation results demonstrate that MiniCPM4 and MiniCPM4.1 outperform
similar-sized open-source models across benchmarks, with the 8B variants
showing significant speed improvements on long sequence understanding and
generation.

**Comment:** MiniCPM4 Technical Report

**Relevance Score:** 6.50

---

## 9. From Editor to Dense Geometry Estimator

**Authors:** JiYuan Wang, Chunyu Lin, Lei Sun, Rongying Liu, Lang Nie, Mingxing Li, Kang Liao, Xiangxiang Chu, Yao Zhao

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04338v1) | [PDF](http://arxiv.org/pdf/2509.04338v1)

**Abstract:** Leveraging visual priors from pre-trained text-to-image (T2I) generative
models has shown success in dense prediction. However, dense prediction is
inherently an image-to-image task, suggesting that image editing models, rather
than T2I generative models, may be a more suitable foundation for fine-tuning.
  Motivated by this, we conduct a systematic analysis of the fine-tuning
behaviors of both editors and generators for dense geometry estimation. Our
findings show that editing models possess inherent structural priors, which
enable them to converge more stably by ``refining" their innate features, and
ultimately achieve higher performance than their generative counterparts.
  Based on these findings, we introduce \textbf{FE2E}, a framework that
pioneeringly adapts an advanced editing model based on Diffusion Transformer
(DiT) architecture for dense geometry prediction. Specifically, to tailor the
editor for this deterministic task, we reformulate the editor's original flow
matching loss into the ``consistent velocity" training objective. And we use
logarithmic quantization to resolve the precision conflict between the editor's
native BFloat16 format and the high precision demand of our tasks.
Additionally, we leverage the DiT's global attention for a cost-free joint
estimation of depth and normals in a single forward pass, enabling their
supervisory signals to mutually enhance each other.
  Without scaling up the training data, FE2E achieves impressive performance
improvements in zero-shot monocular depth and normal estimation across multiple
datasets. Notably, it achieves over 35\% performance gains on the ETH3D dataset
and outperforms the DepthAnything series, which is trained on 100$\times$ data.
The project page can be accessed \href{https://amap-ml.github.io/FE2E/}{here}.

**Comment:** 20pages

**Relevance Score:** 6.50

---

## 10. Rapid Word Learning Through Meta In-Context Learning

**Authors:** Wentao Wang, Guangyuan Jiang, Tal Linzen, Brenden M. Lake

**Categories:** cs.CL, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2502.14791v4) | [PDF](http://arxiv.org/pdf/2502.14791v4)

**Abstract:** Humans can quickly learn a new word from a few illustrative examples, and
then systematically and flexibly use it in novel contexts. Yet the abilities of
current language models for few-shot word learning, and methods for improving
these abilities, are underexplored. In this study, we introduce a novel method,
Meta-training for IN-context learNing Of Words (Minnow). This method trains
language models to generate new examples of a word's usage given a few
in-context examples, using a special placeholder token to represent the new
word. This training is repeated on many new words to develop a general
word-learning ability. We find that training models from scratch with Minnow on
human-scale child-directed language enables strong few-shot word learning,
comparable to a large language model (LLM) pre-trained on orders of magnitude
more data. Furthermore, through discriminative and generative evaluations, we
demonstrate that finetuning pre-trained LLMs with Minnow improves their ability
to discriminate between new words, identify syntactic categories of new words,
and generate reasonable new usages and definitions for new words, based on one
or a few in-context examples. These findings highlight the data efficiency of
Minnow and its potential to improve language model performance in word learning
tasks.

**Comment:** EMNLP 2025

**Relevance Score:** 6.50

---

## 11. DMN-Guided Prompting: A Framework for Controlling LLM Behavior

**Authors:** Shaghayegh Abedi, Amin Jalali

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2505.11701v2) | [PDF](http://arxiv.org/pdf/2505.11701v2)

**Abstract:** Large Language Models (LLMs) have shown considerable potential in automating
decision logic within knowledge-intensive processes. However, their
effectiveness largely depends on the strategy and quality of prompting. Since
decision logic is typically embedded in prompts, it becomes challenging for end
users to modify or refine it. Decision Model and Notation (DMN) offers a
standardized graphical approach for defining decision logic in a structured,
user-friendly manner. This paper introduces a DMN-guided prompting framework
that breaks down complex decision logic into smaller, manageable components,
guiding LLMs through structured decision pathways. We implemented the framework
in a graduate-level course where students submitted assignments. The
assignments and DMN models representing feedback instructions served as inputs
to our framework. The instructor evaluated the generated feedback and labeled
it for performance assessment. Our approach demonstrated promising results,
outperforming chain-of-thought (CoT) prompting in our case study. Students also
responded positively to the generated feedback, reporting high levels of
perceived usefulness in a survey based on the Technology Acceptance Model.

**Comment:** Large Language Models, Decision Model and Notation, Automated
  Feedback, Prompt Engineering

**Relevance Score:** 6.50

---

## 12. Detecting Regional Spurious Correlations in Vision Transformers via   Token Discarding

**Authors:** Solha Kang, Esla Timothy Anzaku, Wesley De Neve, Arnout Van Messem, Joris Vankerschaver, Francois Rameau, Utku Ozbulak

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04009v1) | [PDF](http://arxiv.org/pdf/2509.04009v1)

**Abstract:** Due to their powerful feature association capabilities, neural network-based
computer vision models have the ability to detect and exploit unintended
patterns within the data, potentially leading to correct predictions based on
incorrect or unintended but statistically relevant signals. These clues may
vary from simple color aberrations to small texts within the image. In
situations where these unintended signals align with the predictive task,
models can mistakenly link these features with the task and rely on them for
making predictions. This phenomenon is referred to as spurious correlations,
where patterns appear to be associated with the task but are actually
coincidental. As a result, detection and mitigation of spurious correlations
have become crucial tasks for building trustworthy, reliable, and generalizable
machine learning models. In this work, we present a novel method to detect
spurious correlations in vision transformers, a type of neural network
architecture that gained significant popularity in recent years. Using both
supervised and self-supervised trained models, we present large-scale
experiments on the ImageNet dataset demonstrating the ability of the proposed
method to identify spurious correlations. We also find that, even if the same
architecture is used, the training methodology has a significant impact on the
model's reliance on spurious correlations. Furthermore, we show that certain
classes in the ImageNet dataset contain spurious signals that are easily
detected by the models and discuss the underlying reasons for those spurious
signals. In light of our findings, we provide an exhaustive list of the
aforementioned images and call for caution in their use in future research
efforts. Lastly, we present a case study investigating spurious signals in
invasive breast mass classification, grounding our work in real-world
scenarios.

**Relevance Score:** 6.50

---

## 13. EvoCoT: Overcoming the Exploration Bottleneck in Reinforcement Learning

**Authors:** Huanyu Liu, Jia Li, Chang Yu, Taozhi Chen, Yihong Dong, Lecheng Wang, XiaoLong Hu, Ge Li

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2508.07809v2) | [PDF](http://arxiv.org/pdf/2508.07809v2)

**Abstract:** Reinforcement learning with verifiable reward (RLVR) has become a promising
paradigm for post-training large language models (LLMs) to improve their
reasoning capability. However, when the rollout accuracy is low on hard
problems, the reward becomes sparse, limiting learning efficiency and causing
exploration bottlenecks. Existing approaches either rely on stronger LLMs for
distillation or filter out difficult problems, which limits scalability or
restricts reasoning improvement through exploration.
  We propose EvoCoT, a self-evolving curriculum learning framework based on
two-stage chain-of-thought (CoT) reasoning optimization. EvoCoT constrains the
exploration space by self-generating and verifying CoT trajectories, then
gradually shortens them to expand the space in a controlled way. This enables
LLMs to stably learn from initially unsolved hard problems under sparse
rewards. We apply EvoCoT to multiple LLM families, including Qwen, DeepSeek,
and Llama. Experiments show that EvoCoT enables LLMs to solve previously
unsolved problems, improves reasoning capability without external CoT
supervision, and is compatible with various RL fine-tuning methods. We release
the source code to support future research.

**Relevance Score:** 6.50

---

## 14. Res-MoCoDiff: Residual-guided diffusion models for motion artifact   correction in brain MRI

**Authors:** Mojtaba Safari, Shansong Wang, Qiang Li, Zach Eidex, Richard L. J. Qiu, Chih-Wei Chang, Hui Mao, Xiaofeng Yang

**Categories:** cs.CV, physics.med-ph

**Links:** [arXiv](http://arxiv.org/abs/2505.03498v2) | [PDF](http://arxiv.org/pdf/2505.03498v2)

**Abstract:** Objective. Motion artifacts in brain MRI, mainly from rigid head motion,
degrade image quality and hinder downstream applications. Conventional methods
to mitigate these artifacts, including repeated acquisitions or motion
tracking, impose workflow burdens. This study introduces Res-MoCoDiff, an
efficient denoising diffusion probabilistic model specifically designed for MRI
motion artifact correction.Approach.Res-MoCoDiff exploits a novel residual
error shifting mechanism during the forward diffusion process to incorporate
information from motion-corrupted images. This mechanism allows the model to
simulate the evolution of noise with a probability distribution closely
matching that of the corrupted data, enabling a reverse diffusion process that
requires only four steps. The model employs a U-net backbone, with attention
layers replaced by Swin Transformer blocks, to enhance robustness across
resolutions. Furthermore, the training process integrates a combined l1+l2 loss
function, which promotes image sharpness and reduces pixel-level errors.
Res-MoCoDiff was evaluated on both an in-silico dataset generated using a
realistic motion simulation framework and an in-vivo MR-ART dataset.
Comparative analyses were conducted against established methods, including
CycleGAN, Pix2pix, and a diffusion model with a vision transformer backbone,
using quantitative metrics such as PSNR, SSIM, and NMSE.Main results. The
proposed method demonstrated superior performance in removing motion artifacts
across minor, moderate, and heavy distortion levels. Res-MoCoDiff consistently
achieved the highest SSIM and the lowest NMSE values, with a PSNR of up to
41.91+-2.94 dB for minor distortions. Notably, the average sampling time was
reduced to 0.37 seconds per batch of two image slices, compared with 101.74
seconds for conventional approaches.

**Relevance Score:** 6.50

---

## 15. Aesthetic Image Captioning with Saliency Enhanced MLLMs

**Authors:** Yilin Tao, Jiashui Huang, Huaze Xu, Ling Shao

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04378v1) | [PDF](http://arxiv.org/pdf/2509.04378v1)

**Abstract:** Aesthetic Image Captioning (AIC) aims to generate textual descriptions of
image aesthetics, becoming a key research direction in the field of
computational aesthetics. In recent years, pretrained Multimodal Large Language
Models (MLLMs) have advanced rapidly, leading to a significant increase in
image aesthetics research that integrates both visual and textual modalities.
However, most existing studies on image aesthetics primarily focus on
predicting aesthetic ratings and have shown limited application in AIC.
Existing AIC works leveraging MLLMs predominantly rely on fine-tuning methods
without specifically adapting MLLMs to focus on target aesthetic content. To
address this limitation, we propose the Aesthetic Saliency Enhanced Multimodal
Large Language Model (ASE-MLLM), an end-to-end framework that explicitly
incorporates aesthetic saliency into MLLMs. Within this framework, we introduce
the Image Aesthetic Saliency Module (IASM), which efficiently and effectively
extracts aesthetic saliency features from images. Additionally, we design
IAS-ViT as the image encoder for MLLMs, this module fuses aesthetic saliency
features with original image features via a cross-attention mechanism. To the
best of our knowledge, ASE-MLLM is the first framework to integrate image
aesthetic saliency into MLLMs specifically for AIC tasks. Extensive experiments
demonstrated that our approach significantly outperformed traditional methods
and generic MLLMs on current mainstream AIC benchmarks, achieving
state-of-the-art (SOTA) performance.

**Relevance Score:** 6.50

---

## 16. Imitating Radiological Scrolling: A Global-Local Attention Model for 3D   Chest CT Volumes Multi-Label Anomaly Classification

**Authors:** Theo Di Piazza, Carole Lazarus, Olivier Nempont, Loic Boussel

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2503.20652v5) | [PDF](http://arxiv.org/pdf/2503.20652v5)

**Abstract:** The rapid increase in the number of Computed Tomography (CT) scan
examinations has created an urgent need for automated tools, such as organ
segmentation, anomaly classification, and report generation, to assist
radiologists with their growing workload. Multi-label classification of
Three-Dimensional (3D) CT scans is a challenging task due to the volumetric
nature of the data and the variety of anomalies to be detected. Existing deep
learning methods based on Convolutional Neural Networks (CNNs) struggle to
capture long-range dependencies effectively, while Vision Transformers require
extensive pre-training, posing challenges for practical use. Additionally,
these existing methods do not explicitly model the radiologist's navigational
behavior while scrolling through CT scan slices, which requires both global
context understanding and local detail awareness. In this study, we present
CT-Scroll, a novel global-local attention model specifically designed to
emulate the scrolling behavior of radiologists during the analysis of 3D CT
scans. Our approach is evaluated on two public datasets, demonstrating its
efficacy through comprehensive experiments and an ablation study that
highlights the contribution of each model component.

**Comment:** 13 pages, 4 figures. Accepted for publication at MIDL 2025

**Relevance Score:** 6.50

---

## 17. TaleDiffusion: Multi-Character Story Generation with Dialogue Rendering

**Authors:** Ayan Banerjee, Josep Lladós, Umapada Pal, Anjan Dutta

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04123v1) | [PDF](http://arxiv.org/pdf/2509.04123v1)

**Abstract:** Text-to-story visualization is challenging due to the need for consistent
interaction among multiple characters across frames. Existing methods struggle
with character consistency, leading to artifact generation and inaccurate
dialogue rendering, which results in disjointed storytelling. In response, we
introduce TaleDiffusion, a novel framework for generating multi-character
stories with an iterative process, maintaining character consistency, and
accurate dialogue assignment via postprocessing. Given a story, we use a
pre-trained LLM to generate per-frame descriptions, character details, and
dialogues via in-context learning, followed by a bounded attention-based
per-box mask technique to control character interactions and minimize
artifacts. We then apply an identity-consistent self-attention mechanism to
ensure character consistency across frames and region-aware cross-attention for
precise object placement. Dialogues are also rendered as bubbles and assigned
to characters via CLIPSeg. Experimental results demonstrate that TaleDiffusion
outperforms existing methods in consistency, noise reduction, and dialogue
rendering.

**Relevance Score:** 6.50

---

## 18. ANTS: Shaping the Adaptive Negative Textual Space by MLLM for OOD   Detection

**Authors:** Zhu Wenjie, Zhang Yabin, Xin Jin, Wenjun Zeng, Lei Zhang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03951v1) | [PDF](http://arxiv.org/pdf/2509.03951v1)

**Abstract:** The introduction of negative labels (NLs) has proven effective in enhancing
Out-of-Distribution (OOD) detection. However, existing methods often lack an
understanding of OOD images, making it difficult to construct an accurate
negative space. In addition, the presence of false negative labels
significantly degrades their near-OOD performance. To address these issues, we
propose shaping an Adaptive Negative Textual Space (ANTS) by leveraging the
understanding and reasoning capabilities of multimodal large language models
(MLLMs). Specifically, we identify images likely to be OOD samples as negative
images and prompt the MLLM to describe these images, generating expressive
negative sentences that precisely characterize the OOD distribution and enhance
far-OOD detection. For the near-OOD setting, where OOD samples resemble the
in-distribution (ID) subset, we first identify the subset of ID classes that
are visually similar to negative images and then leverage the reasoning
capability of MLLMs to generate visually similar negative labels tailored to
this subset, effectively reducing false negatives and improving near-OOD
detection. To balance these two types of negative textual spaces, we design an
adaptive weighted score that enables the method to handle different OOD task
settings (near-OOD and far-OOD) without relying on task-specific prior
knowledge, making it highly adaptable in open environments. On the ImageNet
benchmark, our ANTS significantly reduces the FPR95 by 4.2\%, establishing a
new state-of-the-art. Furthermore, our method is training-free and zero-shot,
enabling high scalability.

**Relevance Score:** 6.50

---

## 19. A Generative Foundation Model for Chest Radiography

**Authors:** Yuanfeng Ji, Dan Lin, Xiyue Wang, Lu Zhang, Wenhui Zhou, Chongjian Ge, Ruihang Chu, Xiaoli Yang, Junhan Zhao, Junsong Chen, Xiangde Luo, Sen Yang, Jin Fang, Ping Luo, Ruijiang Li

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03903v1) | [PDF](http://arxiv.org/pdf/2509.03903v1)

**Abstract:** The scarcity of well-annotated diverse medical images is a major hurdle for
developing reliable AI models in healthcare. Substantial technical advances
have been made in generative foundation models for natural images. Here we
develop `ChexGen', a generative vision-language foundation model that
introduces a unified framework for text-, mask-, and bounding box-guided
synthesis of chest radiographs. Built upon the latent diffusion transformer
architecture, ChexGen was pretrained on the largest curated chest X-ray dataset
to date, consisting of 960,000 radiograph-report pairs. ChexGen achieves
accurate synthesis of radiographs through expert evaluations and quantitative
metrics. We demonstrate the utility of ChexGen for training data augmentation
and supervised pretraining, which led to performance improvements across
disease classification, detection, and segmentation tasks using a small
fraction of training data. Further, our model enables the creation of diverse
patient cohorts that enhance model fairness by detecting and mitigating
demographic biases. Our study supports the transformative role of generative
foundation models in building more accurate, data-efficient, and equitable
medical AI systems.

**Relevance Score:** 6.50

---

## 20. Synthesizing Sheet Music Problems for Evaluation and Reinforcement   Learning

**Authors:** Zhilin Wang, Zhe Yang, Yun Luo, Yafu Li, Haoran Zhang, Runzhe Zhan, Derek F. Wong, Jizhe Zhou, Yu Cheng

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04059v1) | [PDF](http://arxiv.org/pdf/2509.04059v1)

**Abstract:** Enhancing the ability of Large Language Models (LLMs) and Multimodal Large
Language Models (MLLMs) to interpret sheet music is a crucial step toward
building AI musicians. However, current research lacks both evaluation
benchmarks and training data for sheet music reasoning. To address this, we
propose the idea of synthesizing sheet music problems grounded in music theory,
which can serve both as evaluation benchmarks and as training data for
reinforcement learning with verifiable rewards (RLVR). We introduce a data
synthesis framework that generates verifiable sheet music questions in both
textual and visual modalities, leading to the Synthetic Sheet Music Reasoning
Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on
SSMR-Bench show the importance of models' reasoning abilities in interpreting
sheet music. At the same time, the poor performance of Gemini 2.5-Pro
highlights the challenges that MLLMs still face in interpreting sheet music in
a visual format. By leveraging synthetic data for RLVR, Qwen3-8B-Base and
Qwen2.5-VL-Instruct achieve improvements on the SSMR-Bench. Besides, the
trained Qwen3-8B-Base surpasses GPT-4 in overall performance on
MusicTheoryBench and achieves reasoning performance comparable to GPT-4 with
the strategies of Role play and Chain-of-Thought. Notably, its performance on
math problems also improves relative to the original Qwen3-8B-Base.
Furthermore, our results show that the enhanced reasoning ability can also
facilitate music composition. In conclusion, we are the first to propose the
idea of synthesizing sheet music problems based on music theory rules, and
demonstrate its effectiveness not only in advancing model reasoning for sheet
music understanding but also in unlocking new possibilities for AI-assisted
music creation.

**Comment:** 11 pages

**Relevance Score:** 6.50

---

## 21. ACING: Actor-Critic for Instruction Learning in Black-Box LLMs

**Authors:** Salma Kharrat, Fares Fourati, Marco Canini

**Categories:** cs.CL, cs.SY, math.OC, eess.SY, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.12736v2) | [PDF](http://arxiv.org/pdf/2411.12736v2)

**Abstract:** The effectiveness of Large Language Models (LLMs) in solving tasks depends
significantly on the quality of their instructions, which often require
substantial human effort to craft. This underscores the need for automated
instruction optimization. However, optimizing instructions is particularly
challenging when working with black-box LLMs, where model parameters and
gradients are inaccessible. We introduce ACING, an actor-critic reinforcement
learning framework that formulates instruction optimization as a stateless,
continuous-action problem, enabling exploration of infinite instruction spaces
using only black-box feedback. ACING automatically discovers prompts that
outperform human-written prompts in 76% of instruction-induction tasks, with
gains of up to 33 points and a 10-point median improvement over the best
automatic baseline in 33 tasks spanning instruction-induction, summarization,
and chain-of-thought reasoning. Extensive ablations highlight its robustness
and efficiency. An implementation of ACING is available at
https://github.com/salmakh1/ACING.

**Comment:** Accepted at EMNLP 2025

**Relevance Score:** 5.50

---

## 22. EvoEmo: Towards Evolved Emotional Policies for LLM Agents in Multi-Turn   Negotiation

**Authors:** Yunbo Long, Liming Xu, Lukas Beckenbauer, Yuhan Liu, Alexandra Brintrup

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04310v1) | [PDF](http://arxiv.org/pdf/2509.04310v1)

**Abstract:** Recent research on Chain-of-Thought (CoT) reasoning in Large Language Models
(LLMs) has demonstrated that agents can engage in \textit{complex},
\textit{multi-turn} negotiations, opening new avenues for agentic AI. However,
existing LLM agents largely overlook the functional role of emotions in such
negotiations, instead generating passive, preference-driven emotional responses
that make them vulnerable to manipulation and strategic exploitation by
adversarial counterparts. To address this gap, we present EvoEmo, an
evolutionary reinforcement learning framework that optimizes dynamic emotional
expression in negotiations. EvoEmo models emotional state transitions as a
Markov Decision Process and employs population-based genetic optimization to
evolve high-reward emotion policies across diverse negotiation scenarios. We
further propose an evaluation framework with two baselines -- vanilla
strategies and fixed-emotion strategies -- for benchmarking emotion-aware
negotiation. Extensive experiments and ablation studies show that EvoEmo
consistently outperforms both baselines, achieving higher success rates, higher
efficiency, and increased buyer savings. This findings highlight the importance
of adaptive emotional expression in enabling more effective LLM agents for
multi-turn negotiation.

**Relevance Score:** 5.50

---

## 23. Style Transfer to Calvin and Hobbes comics using Stable Diffusion

**Authors:** Asvin Kumar Venkataramanan, Sloke Shrestha, Sundar Sripada Venugopalaswamy Sriraman

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2312.03993v2) | [PDF](http://arxiv.org/pdf/2312.03993v2)

**Abstract:** This project report summarizes our journey to perform stable diffusion
fine-tuning on a dataset containing Calvin and Hobbes comics. The purpose is to
convert any given input image into the comic style of Calvin and Hobbes,
essentially performing style transfer. We train stable-diffusion-v1.5 using Low
Rank Adaptation (LoRA) to efficiently speed up the fine-tuning process. The
diffusion itself is handled by a Variational Autoencoder (VAE), which is a
U-net. Our results were visually appealing for the amount of training time and
the quality of input data that went into training.

**Comment:** Updated authorship

**Relevance Score:** 5.50

---

## 24. Enhancing Technical Documents Retrieval for RAG

**Authors:** Songjiang Lai, Tsun-Hin Cheung, Ka-Chun Fung, Kaiwen Xue, Kwan-Ho Lin, Yan-Ming Choi, Vincent Ng, Kin-Man Lam

**Categories:** cs.IR, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04139v1) | [PDF](http://arxiv.org/pdf/2509.04139v1)

**Abstract:** In this paper, we introduce Technical-Embeddings, a novel framework designed
to optimize semantic retrieval in technical documentation, with applications in
both hardware and software development. Our approach addresses the challenges
of understanding and retrieving complex technical content by leveraging the
capabilities of Large Language Models (LLMs). First, we enhance user queries by
generating expanded representations that better capture user intent and improve
dataset diversity, thereby enriching the fine-tuning process for embedding
models. Second, we apply summary extraction techniques to encode essential
contextual information, refining the representation of technical documents. To
further enhance retrieval performance, we fine-tune a bi-encoder BERT model
using soft prompting, incorporating separate learning parameters for queries
and document context to capture fine-grained semantic nuances. We evaluate our
approach on two public datasets, RAG-EDA and Rust-Docs-QA, demonstrating that
Technical-Embeddings significantly outperforms baseline models in both
precision and recall. Our findings highlight the effectiveness of integrating
query expansion and contextual summarization to enhance information access and
comprehension in technical domains. This work advances the state of
Retrieval-Augmented Generation (RAG) systems, offering new avenues for
efficient and accurate technical document retrieval in engineering and product
development workflows.

**Relevance Score:** 5.50

---

## 25. MEPG:Multi-Expert Planning and Generation for Compositionally-Rich Image   Generation

**Authors:** Yuan Zhao, Liu Lin

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04126v1) | [PDF](http://arxiv.org/pdf/2509.04126v1)

**Abstract:** Text-to-image diffusion models have achieved remarkable image quality, but
they still struggle with complex, multiele ment prompts, and limited stylistic
diversity. To address these limitations, we propose a Multi-Expert Planning and
Gen eration Framework (MEPG) that synergistically integrates position- and
style-aware large language models (LLMs) with spatial-semantic expert modules.
The framework comprises two core components: (1) a Position-Style-Aware (PSA)
module that utilizes a supervised fine-tuned LLM to decom pose input prompts
into precise spatial coordinates and style encoded semantic instructions; and
(2) a Multi-Expert Dif fusion (MED) module that implements cross-region genera
tion through dynamic expert routing across both local regions and global areas.
During the generation process for each lo cal region, specialized models (e.g.,
realism experts, styliza tion specialists) are selectively activated for each
spatial par tition via attention-based gating mechanisms. The architec ture
supports lightweight integration and replacement of ex pert models, providing
strong extensibility. Additionally, an interactive interface enables real-time
spatial layout editing and per-region style selection from a portfolio of
experts. Ex periments show that MEPG significantly outperforms base line models
with the same backbone in both image quality
  and style diversity.

**Relevance Score:** 5.50

---

## 26. Transferable Mask Transformer: Cross-domain Semantic Segmentation with   Region-adaptive Transferability Estimation

**Authors:** Jianhua Liu, Zhengyu Li, Yanru Wu, Jingge Wang, Yang Tan, Ruizhe Zhao, Guan Wang, Yang Li

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2504.05774v2) | [PDF](http://arxiv.org/pdf/2504.05774v2)

**Abstract:** Recent advances in Vision Transformers (ViTs) have set new benchmarks in
semantic segmentation. However, when adapting pretrained ViTs to new target
domains, significant performance degradation often occurs due to distribution
shifts, resulting in suboptimal global attention. Since self-attention
mechanisms are inherently data-driven, they may fail to effectively attend to
key objects when source and target domains exhibit differences in texture,
scale, or object co-occurrence patterns. While global and patch-level domain
adaptation methods provide partial solutions, region-level adaptation with
dynamically shaped regions is crucial due to spatial heterogeneity in
transferability across different image areas. We present Transferable Mask
Transformer (TMT), a novel region-level adaptation framework for semantic
segmentation that aligns cross-domain representations through spatial
transferability analysis. TMT consists of two key components: (1) An Adaptive
Cluster-based Transferability Estimator (ACTE) that dynamically segments images
into structurally and semantically coherent regions for localized
transferability assessment, and (2) A Transferable Masked Attention (TMA)
module that integrates region-specific transferability maps into ViTs'
attention mechanisms, prioritizing adaptation in regions with low
transferability and high semantic uncertainty. Comprehensive evaluations across
20 cross-domain pairs demonstrate TMT's superiority, achieving an average 2%
MIoU improvement over vanilla fine-tuning and a 1.28% increase compared to
state-of-the-art baselines. The source code will be publicly available.

**Relevance Score:** 5.50

---

## 27. DaMoC: Efficiently Selecting the Optimal Large Language Model for   Fine-tuning Domain Tasks Based on Data and Model Compression

**Authors:** Wei Huang, Huang Wei, Yinggui Wang

**Categories:** cs.CL, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.01221v2) | [PDF](http://arxiv.org/pdf/2509.01221v2)

**Abstract:** Large language models (LLMs) excel in general tasks but struggle with
domain-specific ones, requiring fine-tuning with specific data. With many
open-source LLMs available, selecting the best model for fine-tuning downstream
tasks is challenging, primarily focusing on how to quickly identify the optimal
LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses
this challenge by: 1) Data Level: A systematic categorization of data filtering
methodologies for LLMs is first established, classifying them into three
distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods,
and (3) hybrid approaches considering both dimensions. Further, we enhance the
density of key tokens in the text achieving token compression. Subsequently, we
use an LLM to iterative rewrite the text to optimize its expression. 2) Model
Level: We use layer similarity scores to assess each layer's importance and
remove those with lower importance. Then, we introduce a sparse merging
paradigm to preserve as much of the original model's capability as possible.
Extensive experiments on four datasets, medical Q&A, financial Q&A, general
Q&A, and reading comprehension, show that we can select the optimal LLM while
saving approximately 20-fold in training time.

**Comment:** Accepted by EMNLP 2025

**Relevance Score:** 5.50

---

## 28. CoT-Space: A Theoretical Framework for Internal Slow-Thinking via   Reinforcement Learning

**Authors:** Zeyu Gan, Hao Yi, Yong Liu

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04027v1) | [PDF](http://arxiv.org/pdf/2509.04027v1)

**Abstract:** Reinforcement Learning (RL) has become a pivotal approach for enhancing the
reasoning capabilities of Large Language Models (LLMs). However, a significant
theoretical gap persists, as traditional token-level RL frameworks fail to
align with the reasoning-level nature of complex, multi-step thought processes
like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space,
a novel theoretical framework that recasts LLM reasoning from a discrete
token-prediction task to an optimization process within a continuous,
reasoning-level semantic space. By analyzing this process from both a noise
perspective and a risk perspective, we demonstrate that the convergence to an
optimal CoT length is a natural consequence of the fundamental trade-off
between underfitting and overfitting. Furthermore, extensive experiments
provide strong empirical validation for our theoretical findings. Our framework
not only provides a coherent explanation for empirical phenomena such as
overthinking but also offers a solid theoretical foundation to guide the future
development of more effective and generalizable reasoning agents.

**Comment:** Preprint Edition

**Relevance Score:** 5.50

---

## 29. NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware   Embeddings

**Authors:** Or Shachar, Uri Katz, Yoav Goldberg, Oren Glickman

**Categories:** cs.AI, cs.IR, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04011v1) | [PDF](http://arxiv.org/pdf/2509.04011v1)

**Abstract:** We present NER Retriever, a zero-shot retrieval framework for ad-hoc Named
Entity Retrieval, a variant of Named Entity Recognition (NER), where the types
of interest are not provided in advance, and a user-defined type description is
used to retrieve documents mentioning entities of that type. Instead of relying
on fixed schemas or fine-tuned models, our method builds on internal
representations of large language models (LLMs) to embed both entity mentions
and user-provided open-ended type descriptions into a shared semantic space. We
show that internal representations, specifically the value vectors from
mid-layer transformer blocks, encode fine-grained type information more
effectively than commonly used top-layer embeddings. To refine these
representations, we train a lightweight contrastive projection network that
aligns type-compatible entities while separating unrelated types. The resulting
entity embeddings are compact, type-aware, and well-suited for nearest-neighbor
search. Evaluated on three benchmarks, NER Retriever significantly outperforms
both lexical and dense sentence-level retrieval baselines. Our findings provide
empirical support for representation selection within LLMs and demonstrate a
practical solution for scalable, schema-free entity retrieval. The NER
Retriever Codebase is publicly available at
https://github.com/ShacharOr100/ner_retriever

**Comment:** Findings of EMNLP 2025

**Relevance Score:** 5.50

---

## 30. Promptception: How Sensitive Are Large Multimodal Models to Prompts?

**Authors:** Mohamed Insaf Ismithdeen, Muhammad Uzair Khattak, Salman Khan

**Categories:** cs.AI, cs.CL, cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03986v1) | [PDF](http://arxiv.org/pdf/2509.03986v1)

**Abstract:** Despite the success of Large Multimodal Models (LMMs) in recent years, prompt
design for LMMs in Multiple-Choice Question Answering (MCQA) remains poorly
understood. We show that even minor variations in prompt phrasing and structure
can lead to accuracy deviations of up to 15% for certain prompts and models.
This variability poses a challenge for transparent and fair LMM evaluation, as
models often report their best-case performance using carefully selected
prompts. To address this, we introduce Promptception, a systematic framework
for evaluating prompt sensitivity in LMMs. It consists of 61 prompt types,
spanning 15 categories and 6 supercategories, each targeting specific aspects
of prompt formulation, and is used to evaluate 10 LMMs ranging from lightweight
open-source models to GPT-4o and Gemini 1.5 Pro, across 3 MCQA benchmarks:
MMStar, MMMU-Pro, MVBench. Our findings reveal that proprietary models exhibit
greater sensitivity to prompt phrasing, reflecting tighter alignment with
instruction semantics, while open-source models are steadier but struggle with
nuanced and complex phrasing. Based on this analysis, we propose Prompting
Principles tailored to proprietary and open-source LMMs, enabling more robust
and fair model evaluation.

**Comment:** Accepted to EMNLP 2025

**Relevance Score:** 5.50

---

## 31. RL's Razor: Why Online Reinforcement Learning Forgets Less

**Authors:** Idan Shenfeld, Jyothish Pari, Pulkit Agrawal

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04259v1) | [PDF](http://arxiv.org/pdf/2509.04259v1)

**Abstract:** Comparison of fine-tuning models with reinforcement learning (RL) and
supervised fine-tuning (SFT) reveals that, despite similar performance at a new
task, RL preserves prior knowledge and capabilities significantly better. We
find that the degree of forgetting is determined by the distributional shift,
measured as the KL-divergence between the fine-tuned and base policy evaluated
on the new task. Our analysis reveals that on-policy RL is implicitly biased
towards KL-minimal solutions among the many that solve the new task, whereas
SFT can converge to distributions arbitrarily far from the base model. We
validate these findings through experiments with large language models and
robotic foundation models and further provide theoretical justification for why
on-policy RL updates lead to a smaller KL change. We term this principle
$\textit{RL's Razor}$: among all ways to solve a new task, RL prefers those
closest in KL to the original model.

**Relevance Score:** 5.50

---

## 32. Synthetic Survival Data Generation for Heart Failure Prognosis Using   Deep Generative Models

**Authors:** Chanon Puttanawarut, Natcha Fongsrisin, Porntep Amornritvanich, Cholatid Ratanatharathorn, Panu Looareesuwan

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04245v1) | [PDF](http://arxiv.org/pdf/2509.04245v1)

**Abstract:** Background: Heart failure (HF) research is constrained by limited access to
large, shareable datasets due to privacy regulations and institutional
barriers. Synthetic data generation offers a promising solution to overcome
these challenges while preserving patient confidentiality. Methods: We
generated synthetic HF datasets from institutional data comprising 12,552
unique patients using five deep learning models: tabular variational
autoencoder (TVAE), normalizing flow, ADSGAN, SurvivalGAN, and tabular
denoising diffusion probabilistic models (TabDDPM). We comprehensively
evaluated synthetic data utility through statistical similarity metrics,
survival prediction using machine learning and privacy assessments. Results:
SurvivalGAN and TabDDPM demonstrated high fidelity to the original dataset,
exhibiting similar variable distributions and survival curves after applying
histogram equalization. SurvivalGAN (C-indices: 0.71-0.76) and TVAE (C-indices:
0.73-0.76) achieved the strongest performance in survival prediction
evaluation, closely matched real data performance (C-indices: 0.73-0.76).
Privacy evaluation confirmed protection against re-identification attacks.
Conclusions: Deep learning-based synthetic data generation can produce
high-fidelity, privacy-preserving HF datasets suitable for research
applications. This publicly available synthetic dataset addresses critical data
sharing barriers and provides a valuable resource for advancing HF research and
predictive modeling.

**Relevance Score:** 5.50

---

## 33. Sailing Towards Zero-Shot State Estimation using Foundation Models   Combined with a UKF

**Authors:** Tobin Holtmann, David Stenger, Andres Posada-Moreno, Friedrich Solowjow, Sebastian Trimpe

**Categories:** eess.SY, cs.LG, cs.SY

**Links:** [arXiv](http://arxiv.org/abs/2509.04213v1) | [PDF](http://arxiv.org/pdf/2509.04213v1)

**Abstract:** State estimation in control and systems engineering traditionally requires
extensive manual system identification or data-collection effort. However,
transformer-based foundation models in other domains have reduced data
requirements by leveraging pre-trained generalist models. Ultimately,
developing zero-shot foundation models of system dynamics could drastically
reduce manual deployment effort. While recent work shows that transformer-based
end-to-end approaches can achieve zero-shot performance on unseen systems, they
are limited to sensor models seen during training. We introduce the foundation
model unscented Kalman filter (FM-UKF), which combines a transformer-based
model of system dynamics with analytically known sensor models via an UKF,
enabling generalization across varying dynamics without retraining for new
sensor configurations. We evaluate FM-UKF on a new benchmark of container ship
models with complex dynamics, demonstrating a competitive accuracy, effort, and
robustness trade-off compared to classical methods with approximate system
knowledge and to an end-to-end approach. The benchmark and dataset are open
sourced to further support future research in zero-shot state estimation via
foundation models.

**Comment:** Accepted for publication at CDC2025

**Relevance Score:** 5.50

---

## 34. COBRA: Multimodal Sensing Deep Learning Framework for Remote Chronic   Obesity Management via Wrist-Worn Activity Monitoring

**Authors:** Zhengyang Shen, Bo Gao, Mayue Shi

**Categories:** cs.CE, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04210v1) | [PDF](http://arxiv.org/pdf/2509.04210v1)

**Abstract:** Chronic obesity management requires continuous monitoring of energy balance
behaviors, yet traditional self-reported methods suffer from significant
underreporting and recall bias, and difficulty in integration with modern
digital health systems. This study presents COBRA (Chronic Obesity Behavioral
Recognition Architecture), a novel deep learning framework for objective
behavioral monitoring using wrist-worn multimodal sensors. COBRA integrates a
hybrid D-Net architecture combining U-Net spatial modeling, multi-head
self-attention mechanisms, and BiLSTM temporal processing to classify daily
activities into four obesity-relevant categories: Food Intake, Physical
Activity, Sedentary Behavior, and Daily Living. Validated on the WISDM-Smart
dataset with 51 subjects performing 18 activities, COBRA's optimal
preprocessing strategy combines spectral-temporal feature extraction, achieving
high performance across multiple architectures. D-Net demonstrates 96.86%
overall accuracy with category-specific F1-scores of 98.55% (Physical
Activity), 95.53% (Food Intake), 94.63% (Sedentary Behavior), and 98.68% (Daily
Living), outperforming state-of-the-art baselines by 1.18% in accuracy. The
framework shows robust generalizability with low demographic variance (<3%),
enabling scalable deployment for personalized obesity interventions and
continuous lifestyle monitoring.

**Comment:** 19 pages, 4 figures. *Correspondence: m.shi16@imperial.ac.uk.
  Accepted by the IUPESM World Congress on Medical Physics and Biomedical
  Engineering 2025

**Relevance Score:** 5.50

---

## 35. Self-adaptive Dataset Construction for Real-World Multimodal Safety   Scenarios

**Authors:** Jingen Qu, Lijun Li, Bo Zhang, Yichen Yan, Jing Shao

**Categories:** cs.CR, cs.CL, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04403v1) | [PDF](http://arxiv.org/pdf/2509.04403v1)

**Abstract:** Multimodal large language models (MLLMs) are rapidly evolving, presenting
increasingly complex safety challenges. However, current dataset construction
methods, which are risk-oriented, fail to cover the growing complexity of
real-world multimodal safety scenarios (RMS). And due to the lack of a unified
evaluation metric, their overall effectiveness remains unproven. This paper
introduces a novel image-oriented self-adaptive dataset construction method for
RMS, which starts with images and end constructing paired text and guidance
responses. Using the image-oriented method, we automatically generate an RMS
dataset comprising 35k image-text pairs with guidance responses. Additionally,
we introduce a standardized safety dataset evaluation metric: fine-tuning a
safety judge model and evaluating its capabilities on other safety
datasets.Extensive experiments on various tasks demonstrate the effectiveness
of the proposed image-oriented pipeline. The results confirm the scalability
and effectiveness of the image-oriented approach, offering a new perspective
for the construction of real-world multimodal safety datasets.

**Comment:** Accepted at EMNLP 2025 Findings

**Relevance Score:** 5.50

---

## 36. AnomalyLMM: Bridging Generative Knowledge and Discriminative Retrieval   for Text-Based Person Anomaly Search

**Authors:** Hao Ju, Hu Zhang, Zhedong Zheng

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04376v1) | [PDF](http://arxiv.org/pdf/2509.04376v1)

**Abstract:** With growing public safety demands, text-based person anomaly search has
emerged as a critical task, aiming to retrieve individuals with abnormal
behaviors via natural language descriptions. Unlike conventional person search,
this task presents two unique challenges: (1) fine-grained cross-modal
alignment between textual anomalies and visual behaviors, and (2) anomaly
recognition under sparse real-world samples. While Large Multi-modal Models
(LMMs) excel in multi-modal understanding, their potential for fine-grained
anomaly retrieval remains underexplored, hindered by: (1) a domain gap between
generative knowledge and discriminative retrieval, and (2) the absence of
efficient adaptation strategies for deployment. In this work, we propose
AnomalyLMM, the first framework that harnesses LMMs for text-based person
anomaly search. Our key contributions are: (1) A novel coarse-to-fine pipeline
integrating LMMs to bridge generative world knowledge with retrieval-centric
anomaly detection; (2) A training-free adaptation cookbook featuring masked
cross-modal prompting, behavioral saliency prediction, and knowledge-aware
re-ranking, enabling zero-shot focus on subtle anomaly cues. As the first study
to explore LMMs for this task, we conduct a rigorous evaluation on the PAB
dataset, the only publicly available benchmark for text-based person anomaly
search, with its curated real-world anomalies covering diverse scenarios (e.g.,
falling, collision, and being hit). Experiments show the effectiveness of the
proposed method, surpassing the competitive baseline by +0.96% Recall@1
accuracy. Notably, our method reveals interpretable alignment between textual
anomalies and visual behaviors, validated via qualitative analysis. Our code
and models will be released for future research.

**Relevance Score:** 5.50

---

## 37. OVGrasp: Open-Vocabulary Grasping Assistance via Multimodal Intent   Detection

**Authors:** Chen Hu, Shan Luo, Letizia Gionfrida

**Categories:** cs.CV, cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04324v1) | [PDF](http://arxiv.org/pdf/2509.04324v1)

**Abstract:** Grasping assistance is essential for restoring autonomy in individuals with
motor impairments, particularly in unstructured environments where object
categories and user intentions are diverse and unpredictable. We present
OVGrasp, a hierarchical control framework for soft exoskeleton-based grasp
assistance that integrates RGB-D vision, open-vocabulary prompts, and voice
commands to enable robust multimodal interaction. To enhance generalization in
open environments, OVGrasp incorporates a vision-language foundation model with
an open-vocabulary mechanism, allowing zero-shot detection of previously unseen
objects without retraining. A multimodal decision-maker further fuses spatial
and linguistic cues to infer user intent, such as grasp or release, in
multi-object scenarios. We deploy the complete framework on a custom
egocentric-view wearable exoskeleton and conduct systematic evaluations on 15
objects across three grasp types. Experimental results with ten participants
demonstrate that OVGrasp achieves a grasping ability score (GAS) of 87.00%,
outperforming state-of-the-art baselines and achieving improved kinematic
alignment with natural hand motion.

**Relevance Score:** 5.50

---

## 38. Enhanced Generative Data Augmentation for Semantic Segmentation via   Stronger Guidance

**Authors:** Quang-Huy Che, Duc-Tri Le, Bich-Nga Pham, Duc-Khai Lam, Vinh-Tiep Nguyen

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2409.06002v5) | [PDF](http://arxiv.org/pdf/2409.06002v5)

**Abstract:** Data augmentation is crucial for pixel-wise annotation tasks like semantic
segmentation, where labeling requires significant effort and intensive labor.
Traditional methods, involving simple transformations such as rotations and
flips, create new images but often lack diversity along key semantic dimensions
and fail to alter high-level semantic properties. To address this issue,
generative models have emerged as an effective solution for augmenting data by
generating synthetic images. Controllable Generative models offer data
augmentation methods for semantic segmentation tasks by using prompts and
visual references from the original image. However, these models face
challenges in generating synthetic images that accurately reflect the content
and structure of the original image due to difficulties in creating effective
prompts and visual references. In this work, we introduce an effective data
augmentation pipeline for semantic segmentation using Controllable Diffusion
model. Our proposed method includes efficient prompt generation using
Class-Prompt Appending and Visual Prior Blending to enhance attention to
labeled classes in real images, allowing the pipeline to generate a precise
number of augmented images while preserving the structure of
segmentation-labeled classes. In addition, we implement a class balancing
algorithm to ensure a balanced training dataset when merging the synthetic and
original images. Evaluation on PASCAL VOC datasets, our pipeline demonstrates
its effectiveness in generating high-quality synthetic images for semantic
segmentation. Our code is available at
https://github.com/chequanghuy/Enhanced-Generative-Data-Augmentation-for-Semantic-Segmentation-via-Stronger-Guidance.

**Comment:** Published in ICPRAM 2025, ISBN 978-989-758-730-6, ISSN 2184-4313

**Relevance Score:** 5.50

---

## 39. SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluation

**Authors:** Xiaofu Chen, Israfel Salazar, Yova Kementchedjhieva

**Categories:** cs.CL, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03897v1) | [PDF](http://arxiv.org/pdf/2509.03897v1)

**Abstract:** As interest grows in generating long, detailed image captions, standard
evaluation metrics become increasingly unreliable. N-gram-based metrics though
efficient, fail to capture semantic correctness. Representational Similarity
(RS) metrics, designed to address this, initially saw limited use due to high
computational costs, while today, despite advances in hardware, they remain
unpopular due to low correlation to human judgments. Meanwhile, metrics based
on large language models (LLMs) show strong correlation with human judgments,
but remain too expensive for iterative use during model development.
  We introduce SPECS (Specificity-Enhanced CLIPScore), a reference-free RS
metric tailored to long image captioning. SPECS modifies CLIP with a new
objective that emphasizes specificity: rewarding correct details and penalizing
incorrect ones. We show that SPECS matches the performance of open-source
LLM-based metrics in correlation to human judgments, while being far more
efficient. This makes it a practical alternative for iterative checkpoint
evaluation during image captioning model development.Our code can be found at
https://github.com/mbzuai-nlp/SPECS.

**Relevance Score:** 5.50

---

## 40. AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?

**Authors:** Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, Shuicheng Yan

**Categories:** cs.CL, cs.MA

**Links:** [arXiv](http://arxiv.org/abs/2509.03312v2) | [PDF](http://arxiv.org/pdf/2509.03312v2)

**Abstract:** Large Language Model (LLM)-based agentic systems, often comprising multiple
models, complex tool invocations, and orchestration protocols, substantially
outperform monolithic agents. Yet this very sophistication amplifies their
fragility, making them more prone to system failure. Pinpointing the specific
agent or step responsible for an error within long execution traces defines the
task of agentic system failure attribution. Current state-of-the-art reasoning
LLMs, however, remain strikingly inadequate for this challenge, with accuracy
generally below 10%. To address this gap, we propose AgenTracer, the first
automated framework for annotating failed multi-agent trajectories via
counterfactual replay and programmed fault injection, producing the curated
dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a
lightweight failure tracer trained with multi-granular reinforcement learning,
capable of efficiently diagnosing errors in verbose multi-agent interactions.
On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs
like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard
in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers
actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS
with 4.8-14.2% performance gains, empowering self-correcting and self-evolving
agentic AI.

**Relevance Score:** 5.50

---

## 41. Measuring Bias or Measuring the Task: Understanding the Brittle Nature   of LLM Gender Biases

**Authors:** Bufan Gao, Elisa Kreiss

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04373v1) | [PDF](http://arxiv.org/pdf/2509.04373v1)

**Abstract:** As LLMs are increasingly applied in socially impactful settings, concerns
about gender bias have prompted growing efforts both to measure and mitigate
such bias. These efforts often rely on evaluation tasks that differ from
natural language distributions, as they typically involve carefully constructed
task prompts that overtly or covertly signal the presence of gender
bias-related content. In this paper, we examine how signaling the evaluative
purpose of a task impacts measured gender bias in LLMs. Concretely, we test
models under prompt conditions that (1) make the testing context salient, and
(2) make gender-focused content salient. We then assess prompt sensitivity
across four task formats with both token-probability and discrete-choice
metrics. We find that even minor prompt changes can substantially alter bias
outcomes, sometimes reversing their direction entirely. Discrete-choice metrics
further tend to amplify bias relative to probabilistic measures. These findings
do not only highlight the brittleness of LLM gender bias evaluations but open a
new puzzle for the NLP benchmarking and development community: To what extent
can well-controlled testing designs trigger LLM ``testing mode'' performance,
and what does this mean for the ecological validity of future benchmarks.

**Relevance Score:** 5.50

---

## 42. Improving Narrative Classification and Explanation via Fine Tuned   Language Models

**Authors:** Rishit Tyagi, Rahul Bouri, Mohit Gupta

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04077v1) | [PDF](http://arxiv.org/pdf/2509.04077v1)

**Abstract:** Understanding covert narratives and implicit messaging is essential for
analyzing bias and sentiment. Traditional NLP methods struggle with detecting
subtle phrasing and hidden agendas. This study tackles two key challenges: (1)
multi-label classification of narratives and sub-narratives in news articles,
and (2) generating concise, evidence-based explanations for dominant
narratives. We fine-tune a BERT model with a recall-oriented approach for
comprehensive narrative detection, refining predictions using a GPT-4o pipeline
for consistency. For narrative explanation, we propose a ReACT (Reasoning +
Acting) framework with semantic retrieval-based few-shot prompting, ensuring
grounded and relevant justifications. To enhance factual accuracy and reduce
hallucinations, we incorporate a structured taxonomy table as an auxiliary
knowledge base. Our results show that integrating auxiliary knowledge in
prompts improves classification accuracy and justification reliability, with
applications in media analysis, education, and intelligence gathering.

**Relevance Score:** 5.50

---

## 43. Spotlight Attention: Towards Efficient LLM Generation via Non-linear   Hashing-based KV Cache Retrieval

**Authors:** Wenhao Li, Yuxin Zhang, Gen Luo, Haiyuan Wan, Ziyang Gong, Fei Chao, Rongrong Ji

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.19740v2) | [PDF](http://arxiv.org/pdf/2508.19740v2)

**Abstract:** Reducing the key-value (KV) cache burden in Large Language Models (LLMs)
significantly accelerates inference. Dynamically selecting critical KV caches
during decoding helps maintain performance. Existing methods use random linear
hashing to identify important tokens, but this approach is inefficient due to
the orthogonal distribution of queries and keys within two narrow cones in
LLMs. We introduce Spotlight Attention, a novel method that employs non-linear
hashing functions to optimize the embedding distribution of queries and keys,
enhancing coding efficiency and robustness. We also developed a lightweight,
stable training framework using a Bradley-Terry ranking-based loss, enabling
optimization of the non-linear hashing module on GPUs with 16GB memory in 8
hours. Experimental results show that Spotlight Attention drastically improves
retrieval precision while shortening the length of the hash code at least
5$\times$ compared to traditional linear hashing. Finally, we exploit the
computational advantages of bitwise operations by implementing specialized CUDA
kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a
single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla
decoding.

**Relevance Score:** 5.50

---

## 44. Exploring NLP Benchmarks in an Extremely Low-Resource Setting

**Authors:** Ulin Nuha, Adam Jatowt

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03962v1) | [PDF](http://arxiv.org/pdf/2509.03962v1)

**Abstract:** The effectiveness of Large Language Models (LLMs) diminishes for extremely
low-resource languages, such as indigenous languages, primarily due to the lack
of labeled data. Despite growing interest, the availability of high-quality
natural language processing (NLP) datasets for these languages remains limited,
making it difficult to develop robust language technologies. This paper
addresses such gap by focusing on Ladin, an endangered Romance language,
specifically targeting the Val Badia variant. Leveraging a small set of
parallel Ladin-Italian sentence pairs, we create synthetic datasets for
sentiment analysis and multiple-choice question answering (MCQA) by translating
monolingual Italian data. To ensure linguistic quality and reliability, we
apply rigorous filtering and back-translation procedures in our method. We
further demonstrate that incorporating these synthetic datasets into machine
translation training leads to substantial improvements over existing
Italian-Ladin translation baselines. Our contributions include the first
publicly available sentiment analysis and MCQA datasets for Ladin, establishing
foundational resources that can support broader NLP research and downstream
applications for this underrepresented language.

**Relevance Score:** 5.50

---

## 45. SPFT-SQL: Enhancing Large Language Model for Text-to-SQL Parsing by   Self-Play Fine-Tuning

**Authors:** Yuhao Zhang, Shaoming Duan, Jinhang Su, Chuanyi Liu, Peiyi Han

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03937v1) | [PDF](http://arxiv.org/pdf/2509.03937v1)

**Abstract:** Despite the significant advancements of self-play fine-tuning (SPIN), which
can transform a weak large language model (LLM) into a strong one through
competitive interactions between models of varying capabilities, it still faces
challenges in the Text-to-SQL task. SPIN does not generate new information, and
the large number of correct SQL queries produced by the opponent model during
self-play reduces the main model's ability to generate accurate SQL queries. To
address this challenge, we propose a new self-play fine-tuning method tailored
for the Text-to-SQL task, called SPFT-SQL. Prior to self-play, we introduce a
verification-based iterative fine-tuning approach, which synthesizes
high-quality fine-tuning data iteratively based on the database schema and
validation feedback to enhance model performance, while building a model base
with varying capabilities. During the self-play fine-tuning phase, we propose
an error-driven loss method that incentivizes incorrect outputs from the
opponent model, enabling the main model to distinguish between correct SQL and
erroneous SQL generated by the opponent model, thereby improving its ability to
generate correct SQL. Extensive experiments and in-depth analyses on six
open-source LLMs and five widely used benchmarks demonstrate that our approach
outperforms existing state-of-the-art (SOTA) methods.

**Comment:** EMNLP 2025 Findings

**Relevance Score:** 5.50

---

## 46. Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth

**Authors:** Yang Wang, Chenghao Xiao, Chia-Yi Hsiao, Zi Yan Chang, Chi-Li Chen, Tyler Loakman, Chenghua Lin

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03867v1) | [PDF](http://arxiv.org/pdf/2509.03867v1)

**Abstract:** We introduce Drivelology, a unique linguistic phenomenon characterised as
"nonsense with depth", utterances that are syntactically coherent yet
pragmatically paradoxical, emotionally loaded, or rhetorically subversive.
While such expressions may resemble surface-level nonsense, they encode
implicit meaning requiring contextual inference, moral reasoning, or emotional
interpretation. We find that current large language models (LLMs), despite
excelling at many natural language processing (NLP) tasks, consistently fail to
grasp the layered semantics of Drivelological text. To investigate this, we
construct a small but diverse benchmark dataset of over 1,200 meticulously
curated examples, with select instances in English, Mandarin, Spanish, French,
Japanese, and Korean. Annotation was especially challenging: each of the
examples required careful expert review to verify that it truly reflected
Drivelological characteristics. The process involved multiple rounds of
discussion and adjudication to address disagreements, highlighting the subtle
and subjective nature of the Drivelology. We evaluate a range of LLMs on
classification, generation, and reasoning tasks. Our results reveal clear
limitations of LLMs: models often confuse Drivelology with shallow nonsense,
produce incoherent justifications, or miss the implied rhetorical function
altogether. These findings highlight a deeper representational gap in LLMs'
pragmatic understanding and challenge the assumption that statistical fluency
implies cognitive comprehension. We release our dataset and code to facilitate
further research in modelling linguistic depth beyond surface-level coherence.

**Comment:** Accepted for oral presentation at the EMNLP 2025 Main Conference

**Relevance Score:** 5.50

---

## 47. Spatially-Enhanced Recurrent Memory for Long-Range Mapless Navigation   via End-to-End Reinforcement Learning

**Authors:** Fan Yang, Per Frivik, David Hoeller, Chen Wang, Cesar Cadena, Marco Hutter

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2506.05997v2) | [PDF](http://arxiv.org/pdf/2506.05997v2)

**Abstract:** Recent advancements in robot navigation, particularly with end-to-end
learning approaches such as reinforcement learning (RL), have demonstrated
strong performance. However, successful navigation still depends on two key
capabilities: mapping and planning (explicitly or implicitly). Classical
approaches rely on explicit mapping pipelines to register egocentric
observations into a coherent map. In contrast, end-to-end learning often
achieves this implicitly -- through recurrent neural networks (RNNs) that fuse
current and historical observations into a latent space for planning. While
existing architectures, such as LSTM and GRU, can capture temporal
dependencies, our findings reveal a critical limitation: their inability to
effectively perform spatial memorization. This capability is essential for
integrating sequential observations from varying perspectives to build spatial
representations that support planning. To address this, we propose
Spatially-Enhanced Recurrent Units (SRUs) -- a simple yet effective
modification to existing RNNs -- that enhance spatial memorization. We further
introduce an attention-based network architecture integrated with SRUs,
enabling long-range mapless navigation using a single forward-facing stereo
camera. We also employ regularization techniques to facilitate robust
end-to-end recurrent training via RL. Experimental results show 23.5% overall
improvement in long-range navigation compared to existing RNNs. With SRU
memory, our method outperforms RL baselines -- one relying on explicit mapping
and the other on stacked historical observations -- by 29.6% and 105.0%,
respectively, across diverse environments requiring long-horizon mapping and
memorization. Finally, we address the sim-to-real gap by leveraging large-scale
pretraining on synthetic depth data, enabling zero-shot transfer for deployment
across diverse and complex real-world environments.

**Comment:** 22 pages

**Relevance Score:** 5.00

---

## 48. Delta Activations: A Representation for Finetuned Large Language Models

**Authors:** Zhiqiu Xu, Amish Sethi, Mayur Naik, Ser-Nam Lim

**Categories:** cs.CL, cs.IR, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04442v1) | [PDF](http://arxiv.org/pdf/2509.04442v1)

**Abstract:** The success of powerful open source Large Language Models (LLMs) has enabled
the community to create a vast collection of post-trained models adapted to
specific tasks and domains. However, navigating and understanding these models
remains challenging due to inconsistent metadata and unstructured repositories.
We introduce Delta Activations, a method to represent finetuned models as
vector embeddings by measuring shifts in their internal activations relative to
a base model. This representation allows for effective clustering by domain and
task, revealing structure in the model landscape. Delta Activations also
demonstrate desirable properties: it is robust across finetuning settings and
exhibits an additive property when finetuning datasets are mixed. In addition,
we show that Delta Activations can embed tasks via few-shot finetuning, and
further explore its use for model selection and merging. We hope Delta
Activations can facilitate the practice of reusing publicly available models.
Code is available at https://github.com/OscarXZQ/delta_activations.

**Relevance Score:** 4.50

---

## 49. ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory

**Authors:** Matthew Ho, Chen Si, Zhaoxiang Feng, Fangxu Yu, Zhijian Liu, Zhiting Hu, Lianhui Qin

**Categories:** cs.AI, cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04439v1) | [PDF](http://arxiv.org/pdf/2509.04439v1)

**Abstract:** While inference-time scaling enables LLMs to carry out increasingly long and
capable reasoning traces, the patterns and insights uncovered during these
traces are immediately discarded once the context window is reset for a new
query. External memory is a natural way to persist these discoveries, and
recent work has shown clear benefits for reasoning-intensive tasks. We see an
opportunity to make such memories more broadly reusable and scalable by moving
beyond instance-based memory entries (e.g. exact query/response pairs, or
summaries tightly coupled with the original problem context) toward
concept-level memory: reusable, modular abstractions distilled from solution
traces and stored in natural language. For future queries, relevant concepts
are selectively retrieved and integrated into the prompt, enabling test-time
continual learning without weight updates. Our design introduces new strategies
for abstracting takeaways from rollouts and retrieving entries for new queries,
promoting reuse and allowing memory to expand with additional experiences. On
the challenging ARC-AGI benchmark, our method yields a 7.5% relative gain over
a strong no-memory baseline with performance continuing to scale with inference
compute. We find abstract concepts to be the most consistent memory design,
outscoring the baseline at all tested inference compute scales. Moreover, we
confirm that dynamically updating memory during test-time outperforms an
otherwise identical fixed memory setting with additional attempts, supporting
the hypothesis that solving more problems and abstracting more patterns to
memory enables further solutions in a form of self-improvement. Code available
at https://github.com/matt-seb-ho/arc_memo.

**Relevance Score:** 4.50

---

## 50. Towards a Unified View of Large Language Model Post-Training

**Authors:** Xingtai Lv, Yuxin Zuo, Youbang Sun, Hongyi Liu, Yuntian Wei, Zhekai Chen, Lixuan He, Xuekai Zhu, Kaiyan Zhang, Bingning Wang, Ning Ding, Bowen Zhou

**Categories:** cs.AI, cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04419v1) | [PDF](http://arxiv.org/pdf/2509.04419v1)

**Abstract:** Two major sources of training data exist for post-training modern language
models: online (model-generated rollouts) data, and offline (human or
other-model demonstrations) data. These two types of data are typically used by
approaches like Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT),
respectively. In this paper, we show that these approaches are not in
contradiction, but are instances of a single optimization process. We derive a
Unified Policy Gradient Estimator, and present the calculations of a wide
spectrum of post-training approaches as the gradient of a common objective
under different data distribution assumptions and various bias-variance
tradeoffs. The gradient estimator is constructed with four interchangeable
parts: stabilization mask, reference policy denominator, advantage estimate,
and likelihood gradient. Motivated by our theoretical findings, we propose
Hybrid Post-Training (HPT), an algorithm that dynamically selects different
training signals. HPT is designed to yield both effective exploitation of
demonstration and stable exploration without sacrificing learned reasoning
patterns. We provide extensive experiments and ablation studies to verify the
effectiveness of our unified theoretical framework and HPT. Across six
mathematical reasoning benchmarks and two out-of-distribution suites, HPT
consistently surpasses strong baselines across models of varying scales and
families.

**Relevance Score:** 4.50

---

## 51. Modular Techniques for Synthetic Long-Context Data Generation in   Language Model Training and Evaluation

**Authors:** Seganrasan Subramanian, Abhigya Verma

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.01185v2) | [PDF](http://arxiv.org/pdf/2509.01185v2)

**Abstract:** The ability of large language models (LLMs) to process and reason over long
textual inputs is critical for a wide range of real-world applications.
However, progress in this area is significantly constrained by the absence of
high-quality, diverse, and verifiable long-context datasets suitable for both
training and evaluation. This work introduces a modular, extensible framework
for synthetic long-context data generation via prompt-based interaction with
LLMs. The framework supports multiple training and alignment objectives,
including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO),
and Group Relative Policy Optimization (GRPO). It encompasses four core
generation paradigms: multi-turn conversational dialogues, document-grounded
input-output pairs, verifiable instruction-response tasks, and long-context
reasoning examples. Through templated prompting, a model-agnostic architecture,
and metadata-enriched outputs, the proposed approach facilitates scalable,
controllable, and purpose-aligned dataset creation for advancing long-context
capabilities in LLMs.

**Comment:** 26 pages, 4 figures

**Relevance Score:** 4.50

---

## 52. Psychologically Enhanced AI Agents

**Authors:** Maciej Besta, Shriram Chandran, Robert Gerstenberger, Mathis Lindner, Marcin Chrapek, Sebastian Hermann Martschat, Taraneh Ghandi, Patrick Iff, Hubert Niewiadomski, Piotr Nyczyk, Jürgen Müller, Torsten Hoefler

**Categories:** cs.MA, cs.CL, cs.CY, cs.AI, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04343v1) | [PDF](http://arxiv.org/pdf/2509.04343v1)

**Abstract:** We introduce MBTI-in-Thoughts, a framework for enhancing the effectiveness of
Large Language Model (LLM) agents through psychologically grounded personality
conditioning. Drawing on the Myers-Briggs Type Indicator (MBTI), our method
primes agents with distinct personality archetypes via prompt engineering,
enabling control over behavior along two foundational axes of human psychology,
cognition and affect. We show that such personality priming yields consistent,
interpretable behavioral biases across diverse tasks: emotionally expressive
agents excel in narrative generation, while analytically primed agents adopt
more stable strategies in game-theoretic settings. Our framework supports
experimenting with structured multi-agent communication protocols and reveals
that self-reflection prior to interaction improves cooperation and reasoning
quality. To ensure trait persistence, we integrate the official 16Personalities
test for automated verification. While our focus is on MBTI, we show that our
approach generalizes seamlessly to other psychological frameworks such as Big
Five, HEXACO, or Enneagram. By bridging psychological theory and LLM behavior
design, we establish a foundation for psychologically enhanced AI agents
without any fine-tuning.

**Relevance Score:** 4.50

---

## 53. AudioCodecBench: A Comprehensive Benchmark for Audio Codec Evaluation

**Authors:** Lu Wang, Hao Chen, Siyu Wu, Zhiyue Wu, Hao Zhou, Chengfeng Zhang, Ting Wang, Haodi Zhang

**Categories:** cs.SD, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.02349v2) | [PDF](http://arxiv.org/pdf/2509.02349v2)

**Abstract:** Multimodal Large Language Models (MLLMs) have been widely applied in speech
and music. This tendency has led to a focus on audio tokenization for Large
Models (LMs). Unlike semantic-only text tokens, audio tokens must both capture
global semantic content and preserve fine-grained acoustic details. Moreover,
they provide a discrete method for speech and music that can be effectively
integrated into MLLMs. However, existing research is unsuitable in the
definitions of semantic tokens and acoustic tokens. In addition, the evaluation
of different codecs typically concentrates on specific domains or tasks, such
as reconstruction or Automatic Speech Recognition (ASR) task, which prevents
fair and comprehensive comparisons. To address these problems, this paper
provides suitable definitions for semantic and acoustic tokens and introduces a
systematic evaluation framework. This framework allows for a comprehensive
assessment of codecs' capabilities which evaluate across four dimensions: audio
reconstruction metric, codebook index (ID) stability, decoder-only transformer
perplexity, and performance on downstream probe tasks. Our results show the
correctness of the provided suitable definitions and the correlation among
reconstruction metrics, codebook ID stability, downstream probe tasks and
perplexity.

**Relevance Score:** 4.50

---

## 54. DeepVIS: Bridging Natural Language and Data Visualization Through   Step-wise Reasoning

**Authors:** Zhihao Shuai, Boyan Li, Siyu Yan, Yuyu Luo, Weikai Yang

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2508.01700v2) | [PDF](http://arxiv.org/pdf/2508.01700v2)

**Abstract:** Although data visualization is powerful for revealing patterns and
communicating insights, creating effective visualizations requires familiarity
with authoring tools and often disrupts the analysis flow. While large language
models show promise for automatically converting analysis intent into
visualizations, existing methods function as black boxes without transparent
reasoning processes, which prevents users from understanding design rationales
and refining suboptimal outputs. To bridge this gap, we propose integrating
Chain-of-Thought (CoT) reasoning into the Natural Language to Visualization
(NL2VIS) pipeline. First, we design a comprehensive CoT reasoning process for
NL2VIS and develop an automatic pipeline to equip existing datasets with
structured reasoning steps. Second, we introduce nvBench-CoT, a specialized
dataset capturing detailed step-by-step reasoning from ambiguous natural
language descriptions to finalized visualizations, which enables
state-of-the-art performance when used for model fine-tuning. Third, we develop
DeepVIS, an interactive visual interface that tightly integrates with the CoT
reasoning process, allowing users to inspect reasoning steps, identify errors,
and make targeted adjustments to improve visualization outcomes. Quantitative
benchmark evaluations, two use cases, and a user study collectively demonstrate
that our CoT framework effectively enhances NL2VIS quality while providing
insightful reasoning steps to users.

**Comment:** IEEE VIS 2025 full paper

**Relevance Score:** 4.50

---

## 55. CP-Bench: Evaluating Large Language Models for Constraint Modelling

**Authors:** Kostis Michailidis, Dimos Tsouros, Tias Guns

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2506.06052v2) | [PDF](http://arxiv.org/pdf/2506.06052v2)

**Abstract:** Constraint Programming (CP) is widely used to solve combinatorial problems,
but its core process, namely constraint modelling, requires significant
expertise and is considered to be a bottleneck for wider adoption. Aiming to
alleviate this bottleneck, recent studies have explored using Large Language
Models (LLMs) to transform combinatorial problem descriptions into executable
constraint models. However, the existing evaluation datasets for constraint
modelling are often limited to small, homogeneous, or domain-specific
instances, which do not capture the diversity of real-world scenarios. This
work addresses this gap by introducing CP-Bench, a novel benchmark that
includes a diverse set of well-known combinatorial problems sourced from the CP
community, structured explicitly for evaluating LLM-driven CP modelling. With
this dataset, and given the variety of constraint modelling frameworks, we
compare and evaluate the modelling capabilities of LLMs for three distinct
constraint modelling systems, which vary in abstraction level and underlying
syntax. Notably, the results show higher performance when modelling with a
high-level Python-based framework. Additionally, we systematically evaluate the
use of prompt-based and inference-time compute methods across different LLMs,
which further increase accuracy, reaching up to 70% on this highly challenging
benchmark.

**Comment:** ECAI 25

**Relevance Score:** 4.50

---

## 56. Meta-Policy Reflexion: Reusable Reflective Memory and Rule Admissibility   for Resource-Efficient LLM Agent

**Authors:** Chunlong Wu, Zhibo Qu

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03990v1) | [PDF](http://arxiv.org/pdf/2509.03990v1)

**Abstract:** Large language model (LLM) agents achieve impressive single-task performance
but commonly exhibit repeated failures, inefficient exploration, and limited
cross-task adaptability. Existing reflective strategies (e.g., Reflexion,
ReAct) improve per-episode behavior but typically produce ephemeral,
task-specific traces that are not reused across tasks. Reinforcement-learning
based alternatives can produce transferable policies but require substantial
parameter updates and compute. In this work we introduce Meta-Policy Reflexion
(MPR): a hybrid framework that consolidates LLM-generated reflections into a
structured, predicate-like Meta-Policy Memory (MPM) and applies that memory at
inference time through two complementary mechanisms soft memory-guided decoding
and hard rule admissibility checks(HAC). MPR (i) externalizes reusable
corrective knowledge without model weight updates, (ii) enforces domain
constraints to reduce unsafe or invalid actions, and (iii) retains the
adaptability of language-based reflection. We formalize the MPM representation,
present algorithms for update and decoding, and validate the approach in a
text-based agent environment following the experimental protocol described in
the provided implementation (AlfWorld-based). Empirical results reported in the
supplied material indicate consistent gains in execution accuracy and
robustness when compared to Reflexion baselines; rule admissibility further
improves stability. We analyze mechanisms that explain these gains, discuss
scalability and failure modes, and outline future directions for multimodal and
multi?agent extensions.

**Relevance Score:** 4.50

---

## 57. Expanding Foundational Language Capabilities in Open-Source LLMs through   a Korean Case Study

**Authors:** Junghwan Lim, Gangwon Jo, Sungmin Lee, Jiyoung Park, Dongseok Kim, Jihwan Kim, Junhyeok Lee, Wai Ting Cheung, Dahye Choi, Kibong Choi, Jaeyeon Huh, Beomgyu Kim, Jangwoong Kim, Taehyun Kim, Haesol Lee, Jeesoo Lee, Dongpin Oh, Changseok Song, Daewon Suh

**Categories:** cs.AI, cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03972v1) | [PDF](http://arxiv.org/pdf/2509.03972v1)

**Abstract:** We introduce Llama-3-Motif, a language model consisting of 102 billion
parameters, specifically designed to enhance Korean capabilities while
retaining strong performance in English. Developed on the Llama 3 architecture,
Llama-3-Motif employs advanced training techniques, including LlamaPro and
Masked Structure Growth, to effectively scale the model without altering its
core Transformer architecture. Using the MoAI platform for efficient training
across hyperscale GPU clusters, we optimized Llama-3-Motif using a carefully
curated dataset that maintains a balanced ratio of Korean and English data.
Llama-3-Motif shows decent performance on Korean-specific benchmarks,
outperforming existing models and achieving results comparable to GPT-4.

**Relevance Score:** 4.50

---

## 58. PagedEviction: Structured Block-wise KV Cache Pruning for Efficient   Large Language Model Inference

**Authors:** Krishna Teja Chitty-Venkata, Jie Ye, Xian-He Sun, Anthony Kougkas, Murali Emani, Venkatram Vishwanath, Bogdan Nicolae

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04377v1) | [PDF](http://arxiv.org/pdf/2509.04377v1)

**Abstract:** KV caching significantly improves the efficiency of Large Language Model
(LLM) inference by storing attention states from previously processed tokens,
enabling faster generation of subsequent tokens. However, as sequence length
increases, the KV cache quickly becomes a major memory bottleneck. To address
this, we propose PagedEviction, a novel fine-grained, structured KV cache
pruning strategy that enhances the memory efficiency of vLLM's PagedAttention.
Unlike existing approaches that rely on attention-based token importance or
evict tokens across different vLLM pages, PagedEviction introduces an efficient
block-wise eviction algorithm tailored for paged memory layouts. Our method
integrates seamlessly with PagedAttention without requiring any modifications
to its CUDA attention kernels. We evaluate PagedEviction across
Llama-3.1-8B-Instruct, Llama-3.2-1B-Instruct, and Llama-3.2-3B-Instruct models
on the LongBench benchmark suite, demonstrating improved memory usage with
better accuracy than baselines on long context tasks.

**Comment:** Preprint

**Relevance Score:** 4.50

---

## 59. Connections between reinforcement learning with feedback,test-time   scaling, and diffusion guidance: An anthology

**Authors:** Yuchen Jiao, Yuxin Chen, Gen Li

**Categories:** cs.GL, math.ST, stat.ML, stat.TH, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04372v1) | [PDF](http://arxiv.org/pdf/2509.04372v1)

**Abstract:** In this note, we reflect on several fundamental connections among widely used
post-training techniques. We clarify some intimate connections and equivalences
between reinforcement learning with human feedback, reinforcement learning with
internal feedback, and test-time scaling (particularly soft best-of-$N$
sampling), while also illuminating intrinsic links between diffusion guidance
and test-time scaling. Additionally, we introduce a resampling approach for
alignment and reward-directed diffusion models, sidestepping the need for
explicit reinforcement learning techniques.

**Relevance Score:** 4.50

---

## 60. Towards Reasoning for PDE Foundation Models: A Reward-Model-Driven   Inference-Time-Scaling Algorithm

**Authors:** Siddharth Mansingh, James Amarel, Ragib Arnab, Arvind Mohan, Kamaljeet Singh, Gerd J. Kunde, Nicolas Hengartner, Benjamin Migliori, Emily Casleton, Nathan A. Debardeleben, Ayan Biswas, Diane Oyen, Earl Lawrence

**Categories:** cs.LG, physics.comp-ph

**Links:** [arXiv](http://arxiv.org/abs/2509.02846v2) | [PDF](http://arxiv.org/pdf/2509.02846v2)

**Abstract:** Partial Differential Equations (PDEs) are the bedrock for modern
computational sciences and engineering, and inherently computationally
expensive. While PDE foundation models have shown much promise for simulating
such complex spatio-temporal phenomena, existing models remain constrained by
the pretraining datasets and struggle with auto-regressive rollout performance,
especially in out-of-distribution (OOD) cases. Furthermore, they have
significant compute and training data requirements which hamper their use in
many critical applications. Inspired by recent advances in ``thinking"
strategies used in large language models (LLMs), we introduce the first
test-time computing (TTC) strategy for PDEs that utilizes computational
resources during inference to achieve more accurate predictions with fewer
training samples and smaller models. We accomplish this with two types of
reward models that evaluate predictions of a stochastic based model for
spatio-temporal consistency. We demonstrate this method on compressible
Euler-equation simulations from the PDEGym benchmark and show that TTC captures
improved predictions relative to standard non-adaptive auto-regressive
inference. This TTC framework marks a foundational step towards more advanced
reasoning algorithms or PDE modeling, inluding building
reinforcement-learning-based approaches, potentially transforming computational
workflows in physics and engineering.

**Relevance Score:** 4.50

---

## 61. An Unsupervised Natural Language Processing Pipeline for Assessing   Referral Appropriateness

**Authors:** Vittorio Torri, Annamaria Bottelli, Michele Ercolanoni, Olivia Leoni, Francesca Ieva

**Categories:** I.2.7; J.1; J.3, 68T50, cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2501.14701v2) | [PDF](http://arxiv.org/pdf/2501.14701v2)

**Abstract:** Objective: Assessing the appropriateness of diagnostic referrals is critical
for improving healthcare efficiency and reducing unnecessary procedures.
However, this task becomes challenging when referral reasons are recorded only
as free text rather than structured codes, like in the Italian NHS. To address
this gap, we propose a fully unsupervised Natural Language Processing (NLP)
pipeline capable of extracting and evaluating referral reasons without relying
on labelled datasets.
  Methods: Our pipeline leverages Transformer-based embeddings pre-trained on
Italian medical texts to cluster referral reasons and assess their alignment
with appropriateness guidelines. It operates in an unsupervised setting and is
designed to generalize across different examination types. We analyzed two
complete regional datasets from the Lombardy Region (Italy), covering all
referrals between 2019 and 2021 for venous echocolordoppler of the lower limbs
(ECD;n=496,971; development) and flexible endoscope colonoscopy (FEC;
n=407,949; testing only). For both, a random sample of 1,000 referrals was
manually annotated to measure performance.
  Results: The pipeline achieved high performance in identifying referral
reasons (Prec=92.43% (ECD), 93.59% (FEC); Rec=83.28% (ECD), 92.70% (FEC)) and
appropriateness (Prec=93.58% (ECD), 94.66% (FEC); Rec=91.52% (ECD), 93.96%
(FEC)). At the regional level, the analysis identified relevant inappropriate
referral groups and variation across contexts, findings that informed a new
Lombardy Region resolution to reinforce guideline adherence.
  Conclusions: This study presents a robust, scalable, unsupervised NLP
pipeline for assessing referral appropriateness in large, real-world datasets.
It demonstrates how such data can be effectively leveraged, providing public
health authorities with a deployable AI tool to monitor practices and support
evidence-based policy.

**Comment:** 49 pages, 10 figures

**Relevance Score:** 4.50

---

## 62. Straighter Flow Matching via a Diffusion-Based Coupling Prior

**Authors:** Siyu Xing, Jie Cao, Huaibo Huang, Haichao Shi, Xiao-Yu Zhang

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2311.16507v2) | [PDF](http://arxiv.org/pdf/2311.16507v2)

**Abstract:** Flow matching as a paradigm of generative model achieves notable success
across various domains. However, existing methods use either multi-round
training or knowledge within minibatches, posing challenges in finding a
favorable coupling strategy for straightening trajectories to few-step
generation. To address this issue, we propose a novel approach, Straighter
trajectories of Flow Matching (StraightFM). It straightens trajectories with
the coupling strategy from the entire distribution level. More specifically,
during training, StraightFM creates couplings of images and noise via one
diffusion model as a coupling prior to straighten trajectories for few-step
generation. Our coupling strategy can also integrate with the existing coupling
direction from real data to noise, improving image quality in few-step
generation. Experimental results on pixel space and latent space show that
StraightFM yields attractive samples within 5 steps. Moreover, our
unconditional StraightFM is seamlessly compatible with training-free multimodal
conditional generation, maintaining high-quality image generation in few steps.

**Relevance Score:** 4.50

---

## 63. One-Embedding-Fits-All: Efficient Zero-Shot Time Series Forecasting by a   Model Zoo

**Authors:** Hao-Nan Shi, Ting-Ji Huang, Lu Han, De-Chuan Zhan, Han-Jia Ye

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04208v1) | [PDF](http://arxiv.org/pdf/2509.04208v1)

**Abstract:** The proliferation of Time Series Foundation Models (TSFMs) has significantly
advanced zero-shot forecasting, enabling predictions for unseen time series
without task-specific fine-tuning. Extensive research has confirmed that no
single TSFM excels universally, as different models exhibit preferences for
distinct temporal patterns. This diversity suggests an opportunity: how to take
advantage of the complementary abilities of TSFMs. To this end, we propose
ZooCast, which characterizes each model's distinct forecasting strengths.
ZooCast can intelligently assemble current TSFMs into a model zoo that
dynamically selects optimal models for different forecasting tasks. Our key
innovation lies in the One-Embedding-Fits-All paradigm that constructs a
unified representation space where each model in the zoo is represented by a
single embedding, enabling efficient similarity matching for all tasks.
Experiments demonstrate ZooCast's strong performance on the GIFT-Eval zero-shot
forecasting benchmark while maintaining the efficiency of a single TSFM. In
real-world scenarios with sequential model releases, the framework seamlessly
adds new models for progressive accuracy gains with negligible overhead.

**Relevance Score:** 4.50

---

## 64. KubeGuard: LLM-Assisted Kubernetes Hardening via Configuration Files and   Runtime Logs Analysis

**Authors:** Omri Sgan Cohen, Ehud Malul, Yair Meidan, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Categories:** cs.CR, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04191v1) | [PDF](http://arxiv.org/pdf/2509.04191v1)

**Abstract:** The widespread adoption of Kubernetes (K8s) for orchestrating cloud-native
applications has introduced significant security challenges, such as
misconfigured resources and overly permissive configurations. Failing to
address these issues can result in unauthorized access, privilege escalation,
and lateral movement within clusters. Most existing K8s security solutions
focus on detecting misconfigurations, typically through static analysis or
anomaly detection. In contrast, this paper presents KubeGuard, a novel runtime
log-driven recommender framework aimed at mitigating risks by addressing overly
permissive configurations. KubeGuard is designed to harden K8s environments
through two complementary tasks: Resource Creation and Resource Refinement. It
leverages large language models (LLMs) to analyze manifests and runtime logs
reflecting actual system behavior, using modular prompt-chaining workflows.
This approach enables KubeGuard to create least-privilege configurations for
new resources and refine existing manifests to reduce the attack surface.
KubeGuard's output manifests are presented as recommendations that users (e.g.,
developers and operators) can review and adopt to enhance cluster security. Our
evaluation demonstrates that KubeGuard effectively generates and refines K8s
manifests for Roles, NetworkPolicies, and Deployments, leveraging both
proprietary and open-source LLMs. The high precision, recall, and F1-scores
affirm KubeGuard's practicality as a framework that translates runtime
observability into actionable, least-privilege configuration guidance.

**Relevance Score:** 4.50

---

## 65. TRUST-VL: An Explainable News Assistant for General Multimodal   Misinformation Detection

**Authors:** Zehong Yan, Peng Qi, Wynne Hsu, Mong Li Lee

**Categories:** cs.CV, cs.MM

**Links:** [arXiv](http://arxiv.org/abs/2509.04448v1) | [PDF](http://arxiv.org/pdf/2509.04448v1)

**Abstract:** Multimodal misinformation, encompassing textual, visual, and cross-modal
distortions, poses an increasing societal threat that is amplified by
generative AI. Existing methods typically focus on a single type of distortion
and struggle to generalize to unseen scenarios. In this work, we observe that
different distortion types share common reasoning capabilities while also
requiring task-specific skills. We hypothesize that joint training across
distortion types facilitates knowledge sharing and enhances the model's ability
to generalize. To this end, we introduce TRUST-VL, a unified and explainable
vision-language model for general multimodal misinformation detection. TRUST-VL
incorporates a novel Question-Aware Visual Amplifier module, designed to
extract task-specific visual features. To support training, we also construct
TRUST-Instruct, a large-scale instruction dataset containing 198K samples
featuring structured reasoning chains aligned with human fact-checking
workflows. Extensive experiments on both in-domain and zero-shot benchmarks
demonstrate that TRUST-VL achieves state-of-the-art performance, while also
offering strong generalization and interpretability.

**Comment:** EMNLP 2025; Project Homepage: https://yanzehong.github.io/trust-vl/

**Relevance Score:** 4.50

---

## 66. Plot'n Polish: Zero-shot Story Visualization and Disentangled Editing   with Text-to-Image Diffusion Models

**Authors:** Kiymet Akdemir, Jing Shi, Kushal Kafle, Brian Price, Pinar Yanardag

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04446v1) | [PDF](http://arxiv.org/pdf/2509.04446v1)

**Abstract:** Text-to-image diffusion models have demonstrated significant capabilities to
generate diverse and detailed visuals in various domains, and story
visualization is emerging as a particularly promising application. However, as
their use in real-world creative domains increases, the need for providing
enhanced control, refinement, and the ability to modify images post-generation
in a consistent manner becomes an important challenge. Existing methods often
lack the flexibility to apply fine or coarse edits while maintaining visual and
narrative consistency across multiple frames, preventing creators from
seamlessly crafting and refining their visual stories. To address these
challenges, we introduce Plot'n Polish, a zero-shot framework that enables
consistent story generation and provides fine-grained control over story
visualizations at various levels of detail.

**Relevance Score:** 4.50

---

## 67. TauGenNet: Plasma-Driven Tau PET Image Synthesis via Text-Guided 3D   Diffusion Models

**Authors:** Yuxin Gong, Se-in Jang, Wei Shao, Yi Su, Kuang Gong

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04269v1) | [PDF](http://arxiv.org/pdf/2509.04269v1)

**Abstract:** Accurate quantification of tau pathology via tau positron emission tomography
(PET) scan is crucial for diagnosing and monitoring Alzheimer's disease (AD).
However, the high cost and limited availability of tau PET restrict its
widespread use. In contrast, structural magnetic resonance imaging (MRI) and
plasma-based biomarkers provide non-invasive and widely available complementary
information related to brain anatomy and disease progression. In this work, we
propose a text-guided 3D diffusion model for 3D tau PET image synthesis,
leveraging multimodal conditions from both structural MRI and plasma
measurement. Specifically, the textual prompt is from the plasma p-tau217
measurement, which is a key indicator of AD progression, while MRI provides
anatomical structure constraints. The proposed framework is trained and
evaluated using clinical AV1451 tau PET data from the Alzheimer's Disease
Neuroimaging Initiative (ADNI) database. Experimental results demonstrate that
our approach can generate realistic, clinically meaningful 3D tau PET across a
range of disease stages. The proposed framework can help perform tau PET data
augmentation under different settings, provide a non-invasive, cost-effective
alternative for visualizing tau pathology, and support the simulation of
disease progression under varying plasma biomarker levels and cognitive
conditions.

**Comment:** 9 pages, 4 figures, submitted to IEEE Transactions on Radiation and
  Plasma Medical Sciences

**Relevance Score:** 4.50

---

## 68. Spatial-aware Transformer-GRU Framework for Enhanced Glaucoma Diagnosis   from 3D OCT Imaging

**Authors:** Mona Ashtari-Majlan, David Masip

**Categories:** eess.IV, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2403.05702v2) | [PDF](http://arxiv.org/pdf/2403.05702v2)

**Abstract:** Glaucoma, a leading cause of irreversible blindness, necessitates early
detection for accurate and timely intervention to prevent irreversible vision
loss. In this study, we present a novel deep learning framework that leverages
the diagnostic value of 3D Optical Coherence Tomography (OCT) imaging for
automated glaucoma detection. In this framework, we integrate a pre-trained
Vision Transformer on retinal data for rich slice-wise feature extraction and a
bidirectional Gated Recurrent Unit for capturing inter-slice spatial
dependencies. This dual-component approach enables comprehensive analysis of
local nuances and global structural integrity, crucial for accurate glaucoma
diagnosis. Experimental results on a large dataset demonstrate the superior
performance of the proposed method over state-of-the-art ones, achieving an
F1-score of 93.01%, Matthews Correlation Coefficient (MCC) of 69.33%, and AUC
of 94.20%. The framework's ability to leverage the valuable information in 3D
OCT data holds significant potential for enhancing clinical decision support
systems and improving patient outcomes in glaucoma management.

**Relevance Score:** 4.50

---

## 69. Hardware-Friendly Diffusion Models with Fixed-Size Reusable Structures   for On-Device Image Generation

**Authors:** Sanchar Palit, Sathya Veera Reddy Dendi, Mallikarjuna Talluri, Raj Narayana Gadde

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.06119v2) | [PDF](http://arxiv.org/pdf/2411.06119v2)

**Abstract:** Vision Transformers and U-Net architectures have been widely adopted in the
implementation of Diffusion Models. However, each architecture presents
specific challenges while realizing them on-device. Vision Transformers require
positional embedding to maintain correspondence between the tokens processed by
the transformer, although they offer the advantage of using fixed-size,
reusable repetitive blocks following tokenization. The U-Net architecture lacks
these attributes, as it utilizes variable-sized intermediate blocks for
down-convolution and up-convolution in the noise estimation backbone for the
diffusion process. To address these issues, we propose an architecture that
utilizes a fixed-size, reusable transformer block as a core structure, making
it more suitable for hardware implementation. Our architecture is characterized
by low complexity, token-free design, absence of positional embeddings,
uniformity, and scalability, making it highly suitable for deployment on mobile
and resource-constrained devices. The proposed model exhibit competitive and
consistent performance across both unconditional and conditional image
generation tasks. The model achieved a state-of-the-art FID score of 1.6 on
unconditional image generation with the CelebA.

**Comment:** presented at IJCNN 2025 poster track

**Relevance Score:** 4.50

---

## 70. R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code   Completion Abilities of Code Large Language Models

**Authors:** Ken Deng, Jiaheng Liu, He Zhu, Congnan Liu, Jingxin Li, Jiakai Wang, Peng Zhao, Chenchen Zhang, Yanan Wu, Xueqiao Yin, Yuanxing Zhang, Zizheng Zhan, Wenbo Su, Bangyu Xiang, Tiezheng Ge, Bo Zheng

**Categories:** cs.CL, cs.SE

**Links:** [arXiv](http://arxiv.org/abs/2406.01359v3) | [PDF](http://arxiv.org/pdf/2406.01359v3)

**Abstract:** Code completion models have made significant progress in recent years.
Recently, repository-level code completion has drawn more attention in modern
software development, and several baseline methods and benchmarks have been
proposed. However, existing repository-level code completion methods often fall
short of fully using the extensive context of a project repository, such as the
intricacies of relevant files and class hierarchies. Besides, the existing
benchmarks usually focus on limited code completion scenarios, which cannot
reflect the repository-level code completion abilities well of existing
methods. To address these limitations, we propose the R2C2-Coder to enhance and
benchmark the real-world repository-level code completion abilities of code
Large Language Models, where the R2C2-Coder includes a code prompt construction
method R2C2-Enhance and a well-designed benchmark R2C2-Bench. Specifically,
first, in R2C2-Enhance, we first construct the candidate retrieval pool and
then assemble the completion prompt by retrieving from the retrieval pool for
each completion cursor position. Second, based on R2C2 -Enhance, we can
construct a more challenging and diverse R2C2-Bench with training, validation
and test splits, where a context perturbation strategy is proposed to simulate
the real-world repository-level code completion well. Extensive results on
multiple benchmarks demonstrate the effectiveness of our R2C2-Coder.

**Relevance Score:** 4.50

---

## 71. Small Changes, Large Consequences: Analyzing the Allocational Fairness   of LLMs in Hiring Contexts

**Authors:** Preethi Seshadri, Hongyu Chen, Sameer Singh, Seraphina Goldfarb-Tarrant

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2501.04316v2) | [PDF](http://arxiv.org/pdf/2501.04316v2)

**Abstract:** Large language models (LLMs) are increasingly being deployed in high-stakes
applications like hiring, yet their potential for unfair decision-making
remains understudied in generative and retrieval settings. In this work, we
examine the allocational fairness of LLM-based hiring systems through two tasks
that reflect actual HR usage: resume summarization and applicant ranking. By
constructing a synthetic resume dataset with controlled perturbations and
curating job postings, we investigate whether model behavior differs across
demographic groups. Our findings reveal that generated summaries exhibit
meaningful differences more frequently for race than for gender perturbations.
Models also display non-uniform retrieval selection patterns across demographic
groups and exhibit high ranking sensitivity to both gender and race
perturbations. Surprisingly, retrieval models can show comparable sensitivity
to both demographic and non-demographic changes, suggesting that fairness
issues may stem from broader model brittleness. Overall, our results indicate
that LLM-based hiring systems, especially in the retrieval stage, can exhibit
notable biases that lead to discriminatory outcomes in real-world contexts.

**Relevance Score:** 4.50

---

## 72. Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow   Real Instructions?

**Authors:** Qinyan Zhang, Xinping Lei, Ruijie Miao, Yu Fu, Haojie Fan, Le Chang, Jiafan Hou, Dingling Zhang, Zhongfei Hou, Ziqiang Yang, Changxin Pu, Fei Hu, Jingkai Liu, Mengyun Liu, Yang Liu, Xiang Gao, Jiaheng Liu, Tong Yang, Zaiyuan Wang, Ge Zhang, Wenhao Huang

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04292v1) | [PDF](http://arxiv.org/pdf/2509.04292v1)

**Abstract:** Large Language Models (LLMs) achieve strong performance on diverse tasks but
often exhibit cognitive inertia, struggling to follow instructions that
conflict with the standardized patterns learned during supervised fine-tuning
(SFT). To evaluate this limitation, we propose Inverse IFEval, a benchmark that
measures models Counter-intuitive Abilitytheir capacity to override
training-induced biases and comply with adversarial instructions. Inverse
IFEval introduces eight types of such challenges, including Question
Correction, Intentional Textual Flaws, Code without Comments, and
Counterfactual Answering. Using a human-in-the-loop pipeline, we construct a
dataset of 1012 high-quality Chinese and English questions across 23 domains,
evaluated under an optimized LLM-as-a-Judge framework. Experiments on existing
leading LLMs demonstrate the necessity of our proposed Inverse IFEval
benchmark. Our findings emphasize that future alignment efforts should not only
pursue fluency and factual correctness but also account for adaptability under
unconventional contexts. We hope that Inverse IFEval serves as both a
diagnostic tool and a foundation for developing methods that mitigate cognitive
inertia, reduce overfitting to narrow patterns, and ultimately enhance the
instruction-following reliability of LLMs in diverse and unpredictable
real-world scenarios.

**Relevance Score:** 4.50

---

## 73. Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large   Language Models via a Multi-Paradigm Perspective

**Authors:** Yiyao Yu, Yuxiang Zhang, Dongdong Zhang, Xiao Liang, Hengyuan Zhang, Xingxing Zhang, Ziyi Yang, Mahmoud Khademi, Hany Awadalla, Junjie Wang, Yujiu Yang, Furu Wei

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2501.11110v4) | [PDF](http://arxiv.org/pdf/2501.11110v4)

**Abstract:** Large Language Models (LLMs) have made notable progress in mathematical
reasoning, yet often rely on single-paradigm reasoning, limiting their
effectiveness across diverse tasks. We introduce Chain-of-Reasoning (CoR), a
novel unified framework integrating multiple reasoning paradigms--Natural
Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning
(SR)--to enable synergistic collaboration. CoR generates multiple potential
answers via different reasoning paradigms and synthesizes them into a coherent
final solution. We propose a Progressive Paradigm Training (PPT) strategy for
models to progressively master these paradigms, leading to CoR-Math-7B.
Experimental results demonstrate that CoR-Math-7B significantly outperforms
current SOTA models, achieving up to a 41.0% absolute improvement over GPT-4o
in theorem proving and a 15.0% improvement over RL-based methods on the MATH
benchmark in arithmetic tasks. These results show the enhanced mathematical
comprehension ability of our model, enabling zero-shot generalization across
tasks.

**Comment:** Accepted to ACL 2025 (Main)

**Relevance Score:** 4.50

---

## 74. Context Reasoner: Incentivizing Reasoning Capability for Contextualized   Privacy and Safety Compliance via Reinforcement Learning

**Authors:** Wenbin Hu, Haoran Li, Huihao Jing, Qi Hu, Ziqian Zeng, Sirui Han, Heli Xu, Tianshu Chu, Peizhao Hu, Yangqiu Song

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2505.14585v2) | [PDF](http://arxiv.org/pdf/2505.14585v2)

**Abstract:** While Large Language Models (LLMs) exhibit remarkable capabilities, they also
introduce significant safety and privacy risks. Current mitigation strategies
often fail to preserve contextual reasoning capabilities in risky scenarios.
Instead, they rely heavily on sensitive pattern matching to protect LLMs, which
limits the scope. Furthermore, they overlook established safety and privacy
standards, leading to systemic risks for legal compliance. To address these
gaps, we formulate safety and privacy issues into contextualized compliance
problems following the Contextual Integrity (CI) theory. Under the CI
framework, we align our model with three critical regulatory standards: GDPR,
EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with
a rule-based reward to incentivize contextual reasoning capabilities while
enhancing compliance with safety and privacy norms. Through extensive
experiments, we demonstrate that our method not only significantly enhances
legal compliance (achieving a +8.58% accuracy improvement in safety/privacy
benchmarks) but also further improves general reasoning capability. For
OpenThinker-7B, a strong reasoning model that significantly outperforms its
base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its
general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on
the MMLU and LegalBench benchmark, respectively.

**Comment:** Accepted to EMNLP 2025 Main

**Relevance Score:** 4.50

---

## 75. A RoBERTa-Based Functional Syntax Annotation Model for Chinese Texts

**Authors:** Han Xiaohui, Zhang Yunlong, Guo Yuxi

**Categories:** cs.CL, I.2.7

**Links:** [arXiv](http://arxiv.org/abs/2509.04046v1) | [PDF](http://arxiv.org/pdf/2509.04046v1)

**Abstract:** Systemic Functional Grammar and its branch, Cardiff Grammar, have been widely
applied to discourse analysis, semantic function research, and other tasks
across various languages and texts. However, an automatic annotation system
based on this theory for Chinese texts has not yet been developed, which
significantly constrains the application and promotion of relevant theories. To
fill this gap, this research introduces a functional syntax annotation model
for Chinese based on RoBERTa (Robustly Optimized BERT Pretraining Approach).
The study randomly selected 4,100 sentences from the People's Daily 2014 corpus
and annotated them according to functional syntax theory to establish a dataset
for training. The study then fine-tuned the RoBERTa-Chinese wwm-ext model based
on the dataset to implement the named entity recognition task, achieving an F1
score of 0.852 on the test set that significantly outperforms other comparative
models. The model demonstrated excellent performance in identifying core
syntactic elements such as Subject (S), Main Verb (M), and Complement (C).
Nevertheless, there remains room for improvement in recognizing entities with
imbalanced label samples. As the first integration of functional syntax with
attention-based NLP models, this research provides a new method for automated
Chinese functional syntax analysis and lays a solid foundation for subsequent
studies.

**Comment:** The paper includes 10 pages, 6 tables, and 4 figures. This project is
  completed with the assistance of National Center for Language Technology and
  Digital Economy Research (No. GJLX20250002), and is funded by Heilongjiang
  Language Research Committee Project Construction of an Adaptive Intelligent
  Chinese Learning Platform for International Students in China (No. G2025Y003)

**Relevance Score:** 4.50

---

## 76. Training LLMs to be Better Text Embedders through Bidirectional   Reconstruction

**Authors:** Chang Su, Dengliang Shi, Siyuan Huang, Jintao Du, Changhua Meng, Yu Cheng, Weiqiang Wang, Zhouhan Lin

**Categories:** cs.IR, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03020v2) | [PDF](http://arxiv.org/pdf/2509.03020v2)

**Abstract:** Large language models (LLMs) have increasingly been explored as powerful text
embedders. Existing LLM-based text embedding approaches often leverage the
embedding of the final token, typically a reserved special token such as [EOS].
However, these tokens have not been intentionally trained to capture the
semantics of the whole context, limiting their capacity as text embeddings,
especially for retrieval and re-ranking tasks. We propose to add a new training
stage before contrastive learning to enrich the semantics of the final token
embedding. This stage employs bidirectional generative reconstruction tasks,
namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based
Document-to-Query), which interleave to anchor the [EOS] embedding and
reconstruct either side of Query-Document pairs. Experimental results
demonstrate that our additional training stage significantly improves LLM
performance on the Massive Text Embedding Benchmark (MTEB), achieving new
state-of-the-art results across different LLM base models and scales.

**Comment:** accepted by EMNLP 2025 Main Conference

**Relevance Score:** 4.50

---

## 77. SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented   Generation via Distribution Self-Alignment

**Authors:** Yuqing Huang, Rongyang Zhang, Qimeng Wang, Chengqiang Lu, Yan Gao, Yi Wu, Yao Hu, Xuyang Zhi, Guiquan Liu, Xin Li, Hao Wang, Enhong Chen

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03934v1) | [PDF](http://arxiv.org/pdf/2509.03934v1)

**Abstract:** Recent advancements in large language models (LLMs) have revolutionized
natural language processing through their remarkable capabilities in
understanding and executing diverse tasks. While supervised fine-tuning,
particularly in Retrieval-Augmented Generation (RAG) scenarios, effectively
enhances task-specific performance, it often leads to catastrophic forgetting,
where models lose their previously acquired knowledge and general capabilities.
Existing solutions either require access to general instruction data or face
limitations in preserving the model's original distribution. To overcome these
limitations, we propose SelfAug, a self-distribution alignment method that
aligns input sequence logits to preserve the model's semantic distribution,
thereby mitigating catastrophic forgetting and improving downstream
performance. Extensive experiments demonstrate that SelfAug achieves a superior
balance between downstream learning and general capability retention. Our
comprehensive empirical analysis reveals a direct correlation between
distribution shifts and the severity of catastrophic forgetting in RAG
scenarios, highlighting how the absence of RAG capabilities in general
instruction tuning leads to significant distribution shifts during fine-tuning.
Our findings not only advance the understanding of catastrophic forgetting in
RAG contexts but also provide a practical solution applicable across diverse
fine-tuning scenarios. Our code is publicly available at
https://github.com/USTC-StarTeam/SelfAug.

**Relevance Score:** 4.50

---

## 78. MTQA:Matrix of Thought for Enhanced Reasoning in Complex Question   Answering

**Authors:** Fengxiao Tang, Yufeng Li, Zongzong Wu, Ming Zhao

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03918v1) | [PDF](http://arxiv.org/pdf/2509.03918v1)

**Abstract:** Complex Question Answering (QA) is a fundamental and challenging task in NLP.
While large language models (LLMs) exhibit impressive performance in QA, they
suffer from significant performance degradation when facing complex and
abstract QA tasks due to insufficient reasoning capabilities. Works such as
Chain-of-Thought (CoT) and Tree-of-Thought (ToT) aim to enhance LLMs' reasoning
abilities, but they face issues such as in-layer redundancy in tree structures
and single paths in chain structures. Although some studies utilize
Retrieval-Augmented Generation (RAG) methods to assist LLMs in reasoning, the
challenge of effectively utilizing large amounts of information involving
multiple entities and hops remains critical. To address this, we propose the
Matrix of Thought (MoT), a novel and efficient LLM thought structure. MoT
explores the problem in both horizontal and vertical dimensions through the
"column-cell communication" mechanism, enabling LLMs to actively engage in
multi-strategy and deep-level thinking, reducing redundancy within the column
cells and enhancing reasoning capabilities. Furthermore, we develop a
fact-correction mechanism by constructing knowledge units from retrieved
knowledge graph triples and raw text to enhance the initial knowledge for LLM
reasoning and correct erroneous answers. This leads to the development of an
efficient and accurate QA framework (MTQA). Experimental results show that our
framework outperforms state-of-the-art methods on four widely-used datasets in
terms of F1 and EM scores, with reasoning time only 14.4\% of the baseline
methods, demonstrating both its efficiency and accuracy. The code for this
framework is available at https://github.com/lyfiter/mtqa.

**Relevance Score:** 4.50

---

## 79. How Can I Publish My LLM Benchmark Without Giving the True Answers Away?

**Authors:** Takashi Ishida, Thanawat Lodkaew, Ikko Yamane

**Categories:** cs.LG, cs.AI, cs.CL, stat.ME

**Links:** [arXiv](http://arxiv.org/abs/2505.18102v5) | [PDF](http://arxiv.org/pdf/2505.18102v5)

**Abstract:** Publishing a large language model (LLM) benchmark on the Internet risks
contaminating future LLMs: the benchmark may be unintentionally (or
intentionally) used to train or select a model. A common mitigation is to keep
the benchmark private and let participants submit their models or predictions
to the organizers. However, this strategy will require trust in a single
organization and still permits test-set overfitting through repeated queries.
To overcome this issue, we propose a way to publish benchmarks without
completely disclosing the ground-truth answers to the questions, while still
maintaining the ability to openly evaluate LLMs. Our main idea is to inject
randomness to the answers by preparing several logically correct answers, and
only include one of them as the solution in the benchmark. This reduces the
best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is
this helpful to keep us from disclosing the ground truth, but this approach
also offers a test for detecting data contamination. In principle, even fully
capable models should not surpass the Bayes accuracy. If a model surpasses this
ceiling despite this expectation, this is a strong signal of data
contamination. We present experimental evidence that our method can detect data
contamination accurately on a wide range of benchmarks, models, and training
methodologies.

**Comment:** Extended version of the paper presented as an Oral at the ICML 2025
  Workshop on the Impact of Memorization on Trustworthy Foundation Models

**Relevance Score:** 4.50

---

## 80. Diffusion Generative Models Meet Compressed Sensing, with Applications   to Image Data and Financial Time Series

**Authors:** Zhengyi Guo, Jiatu Li, Wenpin Tang, David D. Yao

**Categories:** stat.ML, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03898v1) | [PDF](http://arxiv.org/pdf/2509.03898v1)

**Abstract:** This paper develops dimension reduction techniques for accelerating diffusion
model inference in the context of synthetic data generation. The idea is to
integrate compressed sensing into diffusion models: (i) compress the data into
a latent space, (ii) train a diffusion model in the latent space, and (iii)
apply a compressed sensing algorithm to the samples generated in the latent
space, facilitating the efficiency of both model training and inference. Under
suitable sparsity assumptions on data, the proposed algorithm is proved to
enjoy faster convergence by combining diffusion model inference with sparse
recovery. As a byproduct, we obtain an optimal value for the latent space
dimension. We also conduct numerical experiments on a range of datasets,
including image data (handwritten digits, medical images, and climate data) and
financial time series for stress testing.

**Relevance Score:** 4.00

---

## 81. INGRID: Intelligent Generative Robotic Design Using Large Language   Models

**Authors:** Guanglu Jia, Ceng Zhang, Gregory S. Chirikjian

**Categories:** cs.RO, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03842v1) | [PDF](http://arxiv.org/pdf/2509.03842v1)

**Abstract:** The integration of large language models (LLMs) into robotic systems has
accelerated progress in embodied artificial intelligence, yet current
approaches remain constrained by existing robotic architectures, particularly
serial mechanisms. This hardware dependency fundamentally limits the scope of
robotic intelligence. Here, we present INGRID (Intelligent Generative Robotic
Design), a framework that enables the automated design of parallel robotic
mechanisms through deep integration with reciprocal screw theory and kinematic
synthesis methods. We decompose the design challenge into four progressive
tasks: constraint analysis, kinematic joint generation, chain construction, and
complete mechanism design. INGRID demonstrates the ability to generate novel
parallel mechanisms with both fixed and variable mobility, discovering
kinematic configurations not previously documented in the literature. We
validate our approach through three case studies demonstrating how INGRID
assists users in designing task-specific parallel robots based on desired
mobility requirements. By bridging the gap between mechanism theory and machine
learning, INGRID enables researchers without specialized robotics training to
create custom parallel mechanisms, thereby decoupling advances in robotic
intelligence from hardware constraints. This work establishes a foundation for
mechanism intelligence, where AI systems actively design robotic hardware,
potentially transforming the development of embodied AI systems.

**Comment:** 15 pages, 6 figures

**Relevance Score:** 4.00

---

## 82. Write on Paper, Wrong in Practice: Why LLMs Still Struggle with Writing   Clinical Notes

**Authors:** Kristina L. Kupferschmidt, Kieran O'Doherty, Joshua A. Skorburg

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04340v1) | [PDF](http://arxiv.org/pdf/2509.04340v1)

**Abstract:** Large Language Models (LLMs) are often proposed as tools to streamline
clinical documentation, a task viewed as both high-volume and low-risk.
However, even seemingly straightforward applications of LLMs raise complex
sociotechnical considerations to translate into practice. This case study,
conducted at KidsAbility, a pediatric rehabilitation facility in Ontario,
Canada examined the use of LLMs to support occupational therapists in reducing
documentation burden.We conducted a qualitative study involving 20 clinicians
who participated in pilot programs using two AI technologies: a general-purpose
proprietary LLM and a bespoke model fine-tuned on proprietary historical
documentation.
  Our findings reveal that documentation challenges are sociotechnical in
nature, shaped by clinical workflows, organizational policies, and system
constraints. Four key themes emerged: (1) the heterogeneity of workflows, (2)
the documentation burden is systemic and not directly linked to the creation of
any single type of documentation, (3) the need for flexible tools and clinician
autonomy, and (4) effective implementation requires mutual learning between
clinicians and AI systems.
  While LLMs show promise in easing documentation tasks, their success will
depend on flexible, adaptive integration that supports clinician autonomy.
Beyond technical performance, sustained adoption will require training programs
and implementation strategies that reflect the complexity of clinical
environments.

**Relevance Score:** 4.00

---

## 83. TimeCopilot

**Authors:** Azul Garza, Reneé Rosillo

**Categories:** cs.LG, cs.AI, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.00616v2) | [PDF](http://arxiv.org/pdf/2509.00616v2)

**Abstract:** We introduce TimeCopilot, the first open-source agentic framework for
forecasting that combines multiple Time Series Foundation Models (TSFMs) with
Large Language Models (LLMs) through a single unified API. TimeCopilot
automates the forecasting pipeline: feature analysis, model selection,
cross-validation, and forecast generation, while providing natural language
explanations and supporting direct queries about the future. The framework is
LLM-agnostic, compatible with both commercial and open-source models, and
supports ensembles across diverse forecasting families. Results on the
large-scale GIFT-Eval benchmark show that TimeCopilot achieves state-of-the-art
probabilistic forecasting performance at low cost. Our framework provides a
practical foundation for reproducible, explainable, and accessible agentic
forecasting systems.

**Relevance Score:** 4.00

---

## 84. MultiConIR: Towards multi-condition Information Retrieval

**Authors:** Xuan Lu, Sifan Liu, Bochao Yin, Yongqi Li, Xinghao Chen, Hui Su, Yaohui Jin, Wenjun Zeng, Xiaoyu Shen

**Categories:** cs.IR

**Links:** [arXiv](http://arxiv.org/abs/2503.08046v3) | [PDF](http://arxiv.org/pdf/2503.08046v3)

**Abstract:** Multi-condition information retrieval (IR) presents a significant, yet
underexplored challenge for existing systems. This paper introduces MultiConIR,
a benchmark specifically designed to evaluate retrieval and reranking models
under nuanced multi-condition query scenarios across five diverse domains. We
systematically assess model capabilities through three critical tasks:
complexity robustness, relevance monotonicity, and query format sensitivity.
Our extensive experiments on 15 models reveal a critical vulnerability: most
retrievers and rerankers exhibit severe performance degradation as query
complexity increases. Key deficiencies include widespread failure to maintain
relevance monotonicity, and high sensitivity to query style and condition
placement. The superior performance of GPT-4o reveals the performance gap
between IR systems and advanced LLM for handling sophisticated natural language
queries. Furthermore, this work delves into the factors contributing to
reranker performance deterioration and examines how condition positioning
within queries affects similarity assessment, providing crucial insights for
advancing IR systems towards complex search scenarios. The code and datasets
are available at https://github.com/EIT-NLP/MultiConIR

**Comment:** EMNLP 2025 Findings

**Relevance Score:** 4.00

---

## 85. No Thoughts Just AI: Biased LLM Recommendations Limit Human Agency in   Resume Screening

**Authors:** Kyra Wilson, Mattea Sim, Anna-Maria Gueorguieva, Aylin Caliskan

**Categories:** cs.CL, cs.CY, K.4.2, cs.AI, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04404v1) | [PDF](http://arxiv.org/pdf/2509.04404v1)

**Abstract:** In this study, we conduct a resume-screening experiment (N=528) where people
collaborate with simulated AI models exhibiting race-based preferences (bias)
to evaluate candidates for 16 high and low status occupations. Simulated AI
bias approximates factual and counterfactual estimates of racial bias in
real-world AI systems. We investigate people's preferences for White, Black,
Hispanic, and Asian candidates (represented through names and affinity groups
on quality-controlled resumes) across 1,526 scenarios and measure their
unconscious associations between race and status using implicit association
tests (IATs), which predict discriminatory hiring decisions but have not been
investigated in human-AI collaboration. When making decisions without AI or
with AI that exhibits no race-based preferences, people select all candidates
at equal rates. However, when interacting with AI favoring a particular group,
people also favor those candidates up to 90% of the time, indicating a
significant behavioral shift. The likelihood of selecting candidates whose
identities do not align with common race-status stereotypes can increase by 13%
if people complete an IAT before conducting resume screening. Finally, even if
people think AI recommendations are low quality or not important, their
decisions are still vulnerable to AI bias under certain circumstances. This
work has implications for people's autonomy in AI-HITL scenarios, AI and work,
design and evaluation of AI hiring systems, and strategies for mitigating bias
in collaborative decision-making tasks. In particular, organizational and
regulatory policy should acknowledge the complex nature of AI-HITL decision
making when implementing these systems, educating people who use them, and
determining which are subject to oversight.

**Comment:** Published in Proceedings of the 2025 AAAI/ACM Conference on AI,
  Ethics, and Society; code available at
  https://github.com/kyrawilson/No-Thoughts-Just-AI

**Relevance Score:** 3.50

---

## 86. IPA: An Information-Preserving Input Projection Framework for Efficient   Foundation Model Adaptation

**Authors:** Yuan Yin, Shashanka Venkataramanan, Tuan-Hung Vu, Andrei Bursuc, Matthieu Cord

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04398v1) | [PDF](http://arxiv.org/pdf/2509.04398v1)

**Abstract:** Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, reduce
adaptation cost by injecting low-rank updates into pretrained weights. However,
LoRA's down-projection is randomly initialized and data-agnostic, discarding
potentially useful information. Prior analyses show that this projection
changes little during training, while the up-projection carries most of the
adaptation, making the random input compression a performance bottleneck. We
propose IPA, a feature-aware projection framework that explicitly preserves
information in the reduced hidden space. In the linear case, we instantiate IPA
with algorithms approximating top principal components, enabling efficient
projector pretraining with negligible inference overhead. Across language and
vision benchmarks, IPA consistently improves over LoRA and DoRA, achieving on
average 1.5 points higher accuracy on commonsense reasoning and 2.3 points on
VTAB-1k, while matching full LoRA performance with roughly half the trainable
parameters when the projection is frozen.

**Relevance Score:** 3.50

---

## 87. Parking Availability Prediction via Fusing Multi-Source Data with A   Self-Supervised Learning Enhanced Spatio-Temporal Inverted Transformer

**Authors:** Yin Huang, Yongqi Dong, Youhua Tang, Li Li

**Categories:** stat.ML, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04362v1) | [PDF](http://arxiv.org/pdf/2509.04362v1)

**Abstract:** The rapid growth of private car ownership has worsened the urban parking
predicament, underscoring the need for accurate and effective parking
availability prediction to support urban planning and management. To address
key limitations in modeling spatio-temporal dependencies and exploiting
multi-source data for parking availability prediction, this study proposes a
novel approach with SST-iTransformer. The methodology leverages K-means
clustering to establish parking cluster zones (PCZs), extracting and
integrating traffic demand characteristics from various transportation modes
(i.e., metro, bus, online ride-hailing, and taxi) associated with the targeted
parking lots. Upgraded on vanilla iTransformer, SST-iTransformer integrates
masking-reconstruction-based pretext tasks for self-supervised spatio-temporal
representation learning, and features an innovative dual-branch attention
mechanism: Series Attention captures long-term temporal dependencies via
patching operations, while Channel Attention models cross-variate interactions
through inverted dimensions. Extensive experiments using real-world data from
Chengdu, China, demonstrate that SST-iTransformer outperforms baseline deep
learning models (including Informer, Autoformer, Crossformer, and
iTransformer), achieving state-of-the-art performance with the lowest mean
squared error (MSE) and competitive mean absolute error (MAE). Comprehensive
ablation studies quantitatively reveal the relative importance of different
data sources: incorporating ride-hailing data provides the largest performance
gains, followed by taxi, whereas fixed-route transit features (bus/metro)
contribute marginally. Spatial correlation analysis further confirms that
excluding historical data from correlated parking lots within PCZs leads to
substantial performance degradation, underscoring the importance of modeling
spatial dependencies.

**Comment:** 25 pages, 5 figures, under review for journal publication

**Relevance Score:** 3.50

---

## 88. Transplant Then Regenerate: A New Paradigm for Text Data Augmentation

**Authors:** Guangzhan Wang, Hongyu Zhang, Beijun Shen, Xiaodong Gu

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.14723v2) | [PDF](http://arxiv.org/pdf/2508.14723v2)

**Abstract:** Data augmentation is a critical technique in deep learning. Traditional
methods like Back-translation typically focus on lexical-level rephrasing,
which primarily produces variations with the same semantics. While large
language models (LLMs) have enhanced text augmentation by their "knowledge
emergence" capability, controlling the style and structure of these outputs
remains challenging and requires meticulous prompt engineering. In this paper,
we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs.
The core idea of LMTransplant is transplant-then-regenerate: incorporating seed
text into a context expanded by LLM, and asking the LLM to regenerate a variant
based on the expanded context. This strategy allows the model to create more
diverse and creative content-level variants by fully leveraging the knowledge
embedded in LLMs, while preserving the core attributes of the original text. We
evaluate LMTransplant across various text-related tasks, demonstrating its
superior performance over existing text augmentation methods. Moreover,
LMTransplant demonstrates exceptional scalability as the size of augmented data
grows.

**Comment:** Accepted by EMNLP 2025

**Relevance Score:** 3.50

---

## 89. Depth-Breadth Synergy in RLVR: Unlocking LLM Reasoning Gains with   Adaptive Exploration

**Authors:** Zhicheng Yang, Zhijiang Guo, Yinya Huang, Yongxin Wang, Dongchun Xie, Yiwei Wang, Xiaodan Liang, Jing Tang

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2508.13755v2) | [PDF](http://arxiv.org/pdf/2508.13755v2)

**Abstract:** Reinforcement Learning with Verifiable Reward (RLVR) has emerged as a
powerful paradigm for unlocking reasoning capabilities in large language
models, yet its full potential is hindered by two under-explored dimensions:
Depth-the hardest problem a model can sample; Breadth-the number of instances
consumed in a single iteration. We dissect the popular GRPO algorithm and
reveal a systematic bias: the cumulative-advantage disproportionately weights
samples with medium accuracy, while down-weighting the low-accuracy instances
that are crucial for pushing reasoning boundaries. To rectify the depth
neglect, we introduce Difficulty Adaptive Rollout Sampling (DARS), which
re-weights hard problems through targeted multi-stage rollouts, thereby
increasing the number of positive rollouts for hard problems. Empirically,
naively enlarging rollout size only accelerates convergence and even hurts
Pass@K. Our DARS, in contrast, delivers consistent Pass@K gains without extra
inference cost at convergence. Just as we adaptively expanded the depth of
exploration, we now ask whether aggressively scaling the breadth of training
data can further amplify reasoning gains. To this end, we intensely scale batch
size and replace PPO's mini-batch iterations with full-batch updates over
multiple epochs. Increasing breadth significantly enhances Pass@1 performance.
Large-breadth training sustains high token-level entropy, indicating continued
exploration and reduced gradient noise. We further present DARS-B, which
augments DARS with large breadth, and demonstrate simultaneous gains in Pass@K
and Pass@1. The results confirm that breadth and adaptive exploration across
depth operate as orthogonal dimensions in RLVR, which are key to unleashing the
reasoning power of RLVR.

**Comment:** 16 pages, 14 figures

**Relevance Score:** 3.50

---

## 90. Facts Fade Fast: Evaluating Memorization of Outdated Medical Knowledge   in Large Language Models

**Authors:** Juraj Vladika, Mahdi Dhaini, Florian Matthes

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04304v1) | [PDF](http://arxiv.org/pdf/2509.04304v1)

**Abstract:** The growing capabilities of Large Language Models (LLMs) show significant
potential to enhance healthcare by assisting medical researchers and
physicians. However, their reliance on static training data is a major risk
when medical recommendations evolve with new research and developments. When
LLMs memorize outdated medical knowledge, they can provide harmful advice or
fail at clinical reasoning tasks. To investigate this problem, we introduce two
novel question-answering (QA) datasets derived from systematic reviews:
MedRevQA (16,501 QA pairs covering general biomedical knowledge) and
MedChangeQA (a subset of 512 QA pairs where medical consensus has changed over
time). Our evaluation of eight prominent LLMs on the datasets reveals
consistent reliance on outdated knowledge across all models. We additionally
analyze the influence of obsolete pre-training data and training strategies to
explain this phenomenon and propose future directions for mitigation, laying
the groundwork for developing more current and reliable medical AI systems.

**Comment:** Accepted to Findings of EMNLP 2025

**Relevance Score:** 3.50

---

## 91. HumAIne-Chatbot: Real-Time Personalized Conversational AI via   Reinforcement Learning

**Authors:** Georgios Makridis, Georgios Fragiadakis, Jorge Oliveira, Tomaz Saraiva, Philip Mavrepis, Georgios Fatouros, Dimosthenis Kyriazis

**Categories:** cs.AI, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04303v1) | [PDF](http://arxiv.org/pdf/2509.04303v1)

**Abstract:** Current conversational AI systems often provide generic, one-size-fits-all
interactions that overlook individual user characteristics and lack adaptive
dialogue management. To address this gap, we introduce
\textbf{HumAIne-chatbot}, an AI-driven conversational agent that personalizes
responses through a novel user profiling framework. The system is pre-trained
on a diverse set of GPT-generated virtual personas to establish a broad prior
over user types. During live interactions, an online reinforcement learning
agent refines per-user models by combining implicit signals (e.g. typing speed,
sentiment, engagement duration) with explicit feedback (e.g., likes and
dislikes). This profile dynamically informs the chatbot dialogue policy,
enabling real-time adaptation of both content and style. To evaluate the
system, we performed controlled experiments with 50 synthetic personas in
multiple conversation domains. The results showed consistent improvements in
user satisfaction, personalization accuracy, and task achievement when
personalization features were enabled. Statistical analysis confirmed
significant differences between personalized and nonpersonalized conditions,
with large effect sizes across key metrics. These findings highlight the
effectiveness of AI-driven user profiling and provide a strong foundation for
future real-world validation.

**Comment:** 11 pages, 4 figures, IEEE conference format

**Relevance Score:** 3.50

---

## 92. Street-Level AI: Are Large Language Models Ready for Real-World   Judgments?

**Authors:** Gaurab Pokharel, Shafkat Farabi, Patrick J. Fowler, Sanmay Das

**Categories:** cs.CY, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2508.08193v2) | [PDF](http://arxiv.org/pdf/2508.08193v2)

**Abstract:** A surge of recent work explores the ethical and societal implications of
large-scale AI models that make "moral" judgments. Much of this literature
focuses either on alignment with human judgments through various thought
experiments or on the group fairness implications of AI judgments. However, the
most immediate and likely use of AI is to help or fully replace the so-called
street-level bureaucrats, the individuals deciding to allocate scarce social
resources or approve benefits. There is a rich history underlying how
principles of local justice determine how society decides on prioritization
mechanisms in such domains. In this paper, we examine how well LLM judgments
align with human judgments, as well as with socially and politically determined
vulnerability scoring systems currently used in the domain of homelessness
resource allocation. Crucially, we use real data on those needing services
(maintaining strict confidentiality by only using local large models) to
perform our analyses. We find that LLM prioritizations are extremely
inconsistent in several ways: internally on different runs, between different
LLMs, and between LLMs and the vulnerability scoring systems. At the same time,
LLMs demonstrate qualitative consistency with lay human judgments in pairwise
testing. Findings call into question the readiness of current generation AI
systems for naive integration in high-stakes societal decision-making.

**Comment:** This work has been accepted for publication as a full paper at the
  AAAI/ACM Conference on AI, Ethics, and Society (AIES 2025)

**Relevance Score:** 3.50

---

## 93. TriCLIP-3D: A Unified Parameter-Efficient Framework for Tri-Modal 3D   Visual Grounding based on CLIP

**Authors:** Fan Li, Zanyi Wang, Zeyi Huang, Guang Dai, Jingdong Wang, Mengmeng Wang

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2507.14904v2) | [PDF](http://arxiv.org/pdf/2507.14904v2)

**Abstract:** 3D visual grounding allows an embodied agent to understand visual information
in real-world 3D environments based on human instructions, which is crucial for
embodied intelligence. Existing 3D visual grounding methods typically rely on
separate encoders for different modalities (e.g., RGB images, text, and 3D
point clouds), resulting in large and complex models that are inefficient to
train. While some approaches use pre-trained 2D multi-modal models like CLIP
for 3D tasks, they still struggle with aligning point cloud data to 2D
encoders. As a result, these methods continue to depend on 3D encoders for
feature extraction, further increasing model complexity and training
inefficiency. In this paper, we propose a unified 2D pre-trained multi-modal
network to process all three modalities (RGB images, text, and point clouds),
significantly simplifying the architecture. By leveraging a 2D CLIP bi-modal
model with adapter-based fine-tuning, this framework effectively adapts to the
tri-modal setting, improving both adaptability and performance across
modalities. Our Geometric-Aware 2D-3D Feature Recovery and Fusion (GARF) module
is designed to fuse geometric multi-scale features from point clouds and
images. We then integrate textual features for final modality fusion and
introduce a multi-modal decoder to facilitate deep cross-modal understanding.
Together, our method achieves unified feature extraction and fusion across the
three modalities, enabling an end-to-end 3D visual grounding model. Compared to
the baseline, our method reduces the number of trainable parameters by
approximately 58\%, while achieving a 6.52\% improvement in the 3D detection
task and a 6.25\% improvement in the 3D visual grounding task.

**Relevance Score:** 3.50

---

## 94. MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn   Mental Health Counseling Sessions

**Authors:** Aishik Mandal, Tanmoy Chakraborty, Iryna Gurevych

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04183v1) | [PDF](http://arxiv.org/pdf/2509.04183v1)

**Abstract:** The growing demand for scalable psychological counseling highlights the need
for fine-tuning open-source Large Language Models (LLMs) with high-quality,
privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT,
a novel multi-agent framework for synthetic psychological counseling session
generation that decomposes counselor response generation into coordinated
sub-tasks handled by specialized LLM agents, each modeling a key psychological
technique. Unlike prior single-agent approaches, MAGneT better captures the
structure and nuance of real counseling. In addition, we address
inconsistencies in prior evaluation protocols by proposing a unified evaluation
framework integrating diverse automatic and expert metrics. Furthermore, we
expand the expert evaluations from four aspects of counseling in previous works
to nine aspects, enabling a more thorough and robust assessment of data
quality. Empirical results show that MAGneT significantly outperforms existing
methods in quality, diversity, and therapeutic alignment of the generated
counseling sessions, improving general counseling skills by 3.2% and
CBT-specific skills by 4.3% on average on cognitive therapy rating scale
(CTRS). Crucially, experts prefer MAGneT-generated sessions in 77.2% of cases
on average across all aspects. Moreover, fine-tuning an open-source model on
MAGneT-generated sessions shows better performance, with improvements of 6.3%
on general counseling skills and 7.3% on CBT-specific skills on average on CTRS
over those fine-tuned with sessions generated by baseline methods. We also make
our code and data public.

**Comment:** 25 pages, 29 figures

**Relevance Score:** 3.50

---

## 95. VisioFirm: Cross-Platform AI-assisted Annotation Tool for Computer   Vision

**Authors:** Safouane El Ghazouali, Umberto Michelucci

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04180v1) | [PDF](http://arxiv.org/pdf/2509.04180v1)

**Abstract:** AI models rely on annotated data to learn pattern and perform prediction.
Annotation is usually a labor-intensive step that require associating labels
ranging from a simple classification label to more complex tasks such as object
detection, oriented bounding box estimation, and instance segmentation.
Traditional tools often require extensive manual input, limiting scalability
for large datasets. To address this, we introduce VisioFirm, an open-source web
application designed to streamline image labeling through AI-assisted
automation. VisioFirm integrates state-of-the-art foundation models into an
interface with a filtering pipeline to reduce human-in-the-loop efforts. This
hybrid approach employs CLIP combined with pre-trained detectors like
Ultralytics models for common classes and zero-shot models such as Grounding
DINO for custom labels, generating initial annotations with low-confidence
thresholding to maximize recall. Through this framework, when tested on
COCO-type of classes, initial prediction have been proven to be mostly correct
though the users can refine these via interactive tools supporting bounding
boxes, oriented bounding boxes, and polygons. Additionally, VisioFirm has
on-the-fly segmentation powered by Segment Anything accelerated through WebGPU
for browser-side efficiency. The tool supports multiple export formats (YOLO,
COCO, Pascal VOC, CSV) and operates offline after model caching, enhancing
accessibility. VisioFirm demonstrates up to 90\% reduction in manual effort
through benchmarks on diverse datasets, while maintaining high annotation
accuracy via clustering of connected CLIP-based disambiguate components and
IoU-graph for redundant detection suppression. VisioFirm can be accessed from
\href{https://github.com/OschAI/VisioFirm}{https://github.com/OschAI/VisioFirm}.

**Relevance Score:** 3.50

---

## 96. Robust training of implicit generative models for multivariate and   heavy-tailed distributions with an invariant statistical loss

**Authors:** José Manuel de Frutos, Manuel A. Vázquez, Pablo Olmos, Joaquín Míguez

**Categories:** stat.ML, cs.AI, stat.CO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2410.22381v2) | [PDF](http://arxiv.org/pdf/2410.22381v2)

**Abstract:** Traditional implicit generative models are capable of learning highly complex
data distributions. However, their training involves distinguishing real data
from synthetically generated data using adversarial discriminators, which can
lead to unstable training dynamics and mode dropping issues. In this work, we
build on the \textit{invariant statistical loss} (ISL) method introduced in
\cite{de2024training}, and extend it to handle heavy-tailed and multivariate
data distributions.
  The data generated by many real-world phenomena can only be properly
characterised using heavy-tailed probability distributions, and traditional
implicit methods struggle to effectively capture their asymptotic behavior. To
address this problem, we introduce a generator trained with ISL, that uses
input noise from a generalised Pareto distribution (GPD). We refer to this
generative scheme as Pareto-ISL for conciseness. Our experiments demonstrate
that Pareto-ISL accurately models the tails of the distributions while still
effectively capturing their central characteristics.
  The original ISL function was conceived for 1D data sets. When the actual
data is $n$-dimensional, a straightforward extension of the method was obtained
by targeting the $n$ marginal distributions of the data. This approach is
computationally infeasible and ineffective in high-dimensional spaces. To
overcome this, we extend the 1D approach using random projections and define a
new loss function suited for multivariate data, keeping problems tractable by
adjusting the number of projections. We assess its performance in
multidimensional generative modeling and explore its potential as a pretraining
technique for generative adversarial networks (GANs) to prevent mode collapse,
reporting promising results and highlighting its robustness across various
hyperparameter settings.

**Relevance Score:** 3.50

---

## 97. Crossing the Species Divide: Transfer Learning from Speech to Animal   Sounds

**Authors:** Jules Cauzinille, Marius Miron, Olivier Pietquin, Masato Hagiwara, Ricard Marxer, Arnaud Rey, Benoit Favre

**Categories:** 68T07, cs.CL, cs.SD, cs.AI, I.5.4; I.2.6; H.5.5, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04166v1) | [PDF](http://arxiv.org/pdf/2509.04166v1)

**Abstract:** Self-supervised speech models have demonstrated impressive performance in
speech processing, but their effectiveness on non-speech data remains
underexplored. We study the transfer learning capabilities of such models on
bioacoustic detection and classification tasks. We show that models such as
HuBERT, WavLM, and XEUS can generate rich latent representations of animal
sounds across taxa. We analyze the models properties with linear probing on
time-averaged representations. We then extend the approach to account for the
effect of time-wise information with other downstream architectures. Finally,
we study the implication of frequency range and noise on performance. Notably,
our results are competitive with fine-tuned bioacoustic pre-trained models and
show the impact of noise-robust pre-training setups. These findings highlight
the potential of speech-based self-supervised learning as an efficient
framework for advancing bioacoustic research.

**Comment:** 5 pages, 3 figures, uses dcase2025.sty, submitted to DCASE 2025

**Relevance Score:** 3.50

---

## 98. TAGAL: Tabular Data Generation using Agentic LLM Methods

**Authors:** Benoît Ronval, Pierre Dupont, Siegfried Nijssen

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04152v1) | [PDF](http://arxiv.org/pdf/2509.04152v1)

**Abstract:** The generation of data is a common approach to improve the performance of
machine learning tasks, among which is the training of models for
classification. In this paper, we present TAGAL, a collection of methods able
to generate synthetic tabular data using an agentic workflow. The methods
leverage Large Language Models (LLMs) for an automatic and iterative process
that uses feedback to improve the generated data without any further LLM
training. The use of LLMs also allows for the addition of external knowledge in
the generation process. We evaluate TAGAL across diverse datasets and different
aspects of quality for the generated data. We look at the utility of downstream
ML models, both by training classifiers on synthetic data only and by combining
real and synthetic data. Moreover, we compare the similarities between the real
and the generated data. We show that TAGAL is able to perform on par with
state-of-the-art approaches that require LLM training and generally outperforms
other training-free approaches. These findings highlight the potential of
agentic workflow and open new directions for LLM-based data generation methods.

**Relevance Score:** 3.50

---

## 99. Oyster-I: Beyond Refusal -- Constructive Safety Alignment for   Responsible Language Models

**Authors:** Ranjie Duan, Jiexi Liu, Xiaojun Jia, Shiji Zhao, Ruoxi Cheng, Fengxiang Wang, Cheng Wei, Yong Xie, Chang Liu, Defeng Li, Yinpeng Dong, Yichi Zhang, Yuefeng Chen, Chongwen Wang, Xingjun Ma, Xingxing Wei, Yang Liu, Hang Su, Jun Zhu, Xinfeng Li, Yitong Sun, Jie Zhang, Jinzhao Hu, Sha Xu, Yitong Yang, Jialing Tao, Hui Xue

**Categories:** cs.SC, cs.CL, cs.CY, cs.AI, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.01909v2) | [PDF](http://arxiv.org/pdf/2509.01909v2)

**Abstract:** Large language models (LLMs) typically deploy safety mechanisms to prevent
harmful content generation. Most current approaches focus narrowly on risks
posed by malicious actors, often framing risks as adversarial events and
relying on defensive refusals. However, in real-world settings, risks also come
from non-malicious users seeking help while under psychological distress (e.g.,
self-harm intentions). In such cases, the model's response can strongly
influence the user's next actions. Simple refusals may lead them to repeat,
escalate, or move to unsafe platforms, creating worse outcomes. We introduce
Constructive Safety Alignment (CSA), a human-centric paradigm that protects
against malicious misuse while actively guiding vulnerable users toward safe
and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic
anticipation of user reactions, fine-grained risk boundary discovery, and
interpretable reasoning control, turning safety into a trust-building process.
Oy1 achieves state-of-the-art safety among open models while retaining high
general capabilities. On our Constructive Benchmark, it shows strong
constructive engagement, close to GPT-5, and unmatched robustness on the
Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from
refusal-first to guidance-first safety, CSA redefines the model-user
relationship, aiming for systems that are not just safe, but meaningfully
helpful. We release Oy1, code, and the benchmark to support responsible,
user-centered AI.

**Comment:** Technical Report Code & Model weights available:
  https://github.com/Alibaba-AAIG/Oyster

**Relevance Score:** 3.50

---

## 100. Diffusion on language model encodings for protein sequence generation

**Authors:** Viacheslav Meshchaninov, Pavel Strashnov, Andrey Shevtsov, Fedor Nikolaev, Nikita Ivanisenko, Olga Kardymon, Dmitry Vetrov

**Categories:** q-bio.BM, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2403.03726v3) | [PDF](http://arxiv.org/pdf/2403.03726v3)

**Abstract:** Protein sequence design has seen significant advances through discrete
diffusion and autoregressive approaches, yet the potential of continuous
diffusion remains underexplored. Here, we present DiMA, a latent diffusion
framework that operates on protein language model representations. Through
systematic exploration of architectural choices and diffusion components, we
develop a robust methodology that generalizes across multiple protein encoders
ranging from 8M to 3B parameters. We demonstrate that our framework achieves
consistently high performance across sequence-only (ESM-2, ESMc),
dual-decodable (CHEAP), and multimodal (SaProt) representations using the same
architecture and training approach. We extensively evaluate existing methods
alongside DiMA using multiple metrics across two protein modalities, covering
quality, diversity, novelty, and distribution matching of generated proteins.
DiMA consistently produces novel, high-quality and diverse protein sequences
and achieves strong results compared to baselines such as autoregressive,
discrete diffusion and flow matching language models. The model demonstrates
versatile functionality, supporting conditional generation tasks including
protein family-generation, motif scaffolding and infilling, and fold-specific
sequence design. This work provides a universal continuous diffusion framework
for protein sequence generation, offering both architectural insights and
practical applicability across various protein design scenarios.

**Relevance Score:** 3.50

---

## 101. Intermediate Languages Matter: Formal Languages and LLMs affect   Neurosymbolic Reasoning

**Authors:** Alexander Beiser, David Penz, Nysret Musliu

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04083v1) | [PDF](http://arxiv.org/pdf/2509.04083v1)

**Abstract:** Large language models (LLMs) achieve astonishing results on a wide range of
tasks. However, their formal reasoning ability still lags behind. A promising
approach is Neurosymbolic LLM reasoning. It works by using LLMs as translators
from natural to formal languages and symbolic solvers for deriving correct
results. Still, the contributing factors to the success of Neurosymbolic LLM
reasoning remain unclear. This paper demonstrates that one previously
overlooked factor is the choice of the formal language. We introduce the
intermediate language challenge: selecting a suitable formal language for
neurosymbolic reasoning. By comparing four formal languages across three
datasets and seven LLMs, we show that the choice of formal language affects
both syntactic and semantic reasoning capabilities. We also discuss the varying
effects across different LLMs.

**Comment:** To appear in the proceedings of The Second Workshop on Knowledge
  Graphs and Neurosymbolic AI (KG-NeSy) Co-located with SEMANTiCS 2025
  Conference, Vienna, Austria - September 3rd, 2025

**Relevance Score:** 3.50

---

## 102. RepoDebug: Repository-Level Multi-Task and Multi-Language Debugging   Evaluation of Large Language Models

**Authors:** Jingjing Liu, Zeming Liu, Zihao Cheng, Mengliang He, Xiaoming Shi, Yuhang Guo, Xiangrong Zhu, Yuanfang Guo, Yunhong Wang, Haifeng Wang

**Categories:** cs.SE, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04078v1) | [PDF](http://arxiv.org/pdf/2509.04078v1)

**Abstract:** Large Language Models (LLMs) have exhibited significant proficiency in code
debugging, especially in automatic program repair, which may substantially
reduce the time consumption of developers and enhance their efficiency.
Significant advancements in debugging datasets have been made to promote the
development of code debugging. However, these datasets primarily focus on
assessing the LLM's function-level code repair capabilities, neglecting the
more complex and realistic repository-level scenarios, which leads to an
incomplete understanding of the LLM's challenges in repository-level debugging.
While several repository-level datasets have been proposed, they often suffer
from limitations such as limited diversity of tasks, languages, and error
types. To mitigate this challenge, this paper introduces RepoDebug, a
multi-task and multi-language repository-level code debugging dataset with 22
subtypes of errors that supports 8 commonly used programming languages and 3
debugging tasks. Furthermore, we conduct evaluation experiments on 10 LLMs,
where Claude 3.5 Sonnect, the best-performing model, still cannot perform well
in repository-level debugging.

**Comment:** 30 pages, 12 figures, EMNLP 2025 Findings

**Relevance Score:** 3.50

---

## 103. Keypoint-based Diffusion for Robotic Motion Planning on the NICOL Robot

**Authors:** Lennart Clasmeier, Jan-Gerrit Habekost, Connor Gäde, Philipp Allgeuer, Stefan Wermter

**Categories:** cs.AI, cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04076v1) | [PDF](http://arxiv.org/pdf/2509.04076v1)

**Abstract:** We propose a novel diffusion-based action model for robotic motion planning.
Commonly, established numerical planning approaches are used to solve general
motion planning problems, but have significant runtime requirements. By
leveraging the power of deep learning, we are able to achieve good results in a
much smaller runtime by learning from a dataset generated by these planners.
While our initial model uses point cloud embeddings in the input to predict
keypoint-based joint sequences in its output, we observed in our ablation study
that it remained challenging to condition the network on the point cloud
embeddings. We identified some biases in our dataset and refined it, which
improved the model's performance. Our model, even without the use of the point
cloud encodings, outperforms numerical models by an order of magnitude
regarding the runtime, while reaching a success rate of up to 90% of collision
free solutions on the test set.

**Comment:** Submitted to ICANN 20255 Special Session on Neural Robotics

**Relevance Score:** 3.50

---

## 104. On Robustness and Reliability of Benchmark-Based Evaluation of LLMs

**Authors:** Riccardo Lunardi, Vincenzo Della Mea, Stefano Mizzaro, Kevin Roitero

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04013v1) | [PDF](http://arxiv.org/pdf/2509.04013v1)

**Abstract:** Large Language Models (LLMs) effectiveness is usually evaluated by means of
benchmarks such as MMLU, ARC-C, or HellaSwag, where questions are presented in
their original wording, thus in a fixed, standardized format. However,
real-world applications involve linguistic variability, requiring models to
maintain their effectiveness across diverse rewordings of the same question or
query. In this study, we systematically assess the robustness of LLMs to
paraphrased benchmark questions and investigate whether benchmark-based
evaluations provide a reliable measure of model capabilities. We systematically
generate various paraphrases of all the questions across six different common
benchmarks, and measure the resulting variations in effectiveness of 34
state-of-the-art LLMs, of different size and effectiveness. Our findings reveal
that while LLM rankings remain relatively stable across paraphrased inputs,
absolute effectiveness scores change, and decline significantly. This suggests
that LLMs struggle with linguistic variability, raising concerns about their
generalization abilities and evaluation methodologies. Furthermore, the
observed performance drop challenges the reliability of benchmark-based
evaluations, indicating that high benchmark scores may not fully capture a
model's robustness to real-world input variations. We discuss the implications
of these findings for LLM evaluation methodologies, emphasizing the need for
robustness-aware benchmarks that better reflect practical deployment scenarios.

**Comment:** Accepted at ECAI 2025

**Relevance Score:** 3.50

---

## 105. AutoPBO: LLM-powered Optimization for Local Search PBO Solvers

**Authors:** Jinyuan Li, Yi Chu, Yiwen Sun, Mengchuan Zou, Shaowei Cai

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04007v1) | [PDF](http://arxiv.org/pdf/2509.04007v1)

**Abstract:** Pseudo-Boolean Optimization (PBO) provides a powerful framework for modeling
combinatorial problems through pseudo-Boolean (PB) constraints. Local search
solvers have shown excellent performance in PBO solving, and their efficiency
is highly dependent on their internal heuristics to guide the search. Still,
their design often requires significant expert effort and manual tuning in
practice. While Large Language Models (LLMs) have demonstrated potential in
automating algorithm design, their application to optimizing PBO solvers
remains unexplored. In this work, we introduce AutoPBO, a novel LLM-powered
framework to automatically enhance PBO local search solvers. We conduct
experiments on a broad range of four public benchmarks, including one
real-world benchmark, a benchmark from PB competition, an integer linear
programming optimization benchmark, and a crafted combinatorial benchmark, to
evaluate the performance improvement achieved by AutoPBO and compare it with
six state-of-the-art competitors, including two local search PBO solvers NuPBO
and OraSLS, two complete PB solvers PBO-IHS and RoundingSat, and two mixed
integer programming (MIP) solvers Gurobi and SCIP. AutoPBO demonstrates
significant improvements over previous local search approaches, while
maintaining competitive performance compared to state-of-the-art competitors.
The results suggest that AutoPBO offers a promising approach to automating
local search solver design.

**Relevance Score:** 3.50

---

## 106. RTQA : Recursive Thinking for Complex Temporal Knowledge Graph Question   Answering with Large Language Models

**Authors:** Zhaoyan Gong, Juan Li, Zhiqiang Liu, Lei Liang, Huajun Chen, Wen Zhang

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03995v1) | [PDF](http://arxiv.org/pdf/2509.03995v1)

**Abstract:** Current temporal knowledge graph question answering (TKGQA) methods primarily
focus on implicit temporal constraints, lacking the capability of handling more
complex temporal queries, and struggle with limited reasoning abilities and
error propagation in decomposition frameworks. We propose RTQA, a novel
framework to address these challenges by enhancing reasoning over TKGs without
requiring training. Following recursive thinking, RTQA recursively decomposes
questions into sub-problems, solves them bottom-up using LLMs and TKG
knowledge, and employs multi-path answer aggregation to improve fault
tolerance. RTQA consists of three core components: the Temporal Question
Decomposer, the Recursive Solver, and the Answer Aggregator. Experiments on
MultiTQ and TimelineKGQA benchmarks demonstrate significant Hits@1 improvements
in "Multiple" and "Complex" categories, outperforming state-of-the-art methods.
Our code and data are available at https://github.com/zjukg/RTQA.

**Comment:** EMNLP 2025

**Relevance Score:** 3.50

---

## 107. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language   Models

**Authors:** Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Categories:** cs.CR, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03985v1) | [PDF](http://arxiv.org/pdf/2509.03985v1)

**Abstract:** In deployment and application, large language models (LLMs) typically undergo
safety alignment to prevent illegal and unethical outputs. However, the
continuous advancement of jailbreak attack techniques, designed to bypass
safety mechanisms with adversarial prompts, has placed increasing pressure on
the security defenses of LLMs. Strengthening resistance to jailbreak attacks
requires an in-depth understanding of the security mechanisms and
vulnerabilities of LLMs. However, the vast number of parameters and complex
structure of LLMs make analyzing security weaknesses from an internal
perspective a challenging task. This paper presents NeuroBreak, a top-down
jailbreak analysis system designed to analyze neuron-level safety mechanisms
and mitigate vulnerabilities. We carefully design system requirements through
collaboration with three experts in the field of AI security. The system
provides a comprehensive analysis of various jailbreak attack methods. By
incorporating layer-wise representation probing analysis, NeuroBreak offers a
novel perspective on the model's decision-making process throughout its
generation steps. Furthermore, the system supports the analysis of critical
neurons from both semantic and functional perspectives, facilitating a deeper
exploration of security mechanisms. We conduct quantitative evaluations and
case studies to verify the effectiveness of our system, offering mechanistic
insights for developing next-generation defense strategies against evolving
jailbreak attacks.

**Comment:** 12 pages, 9 figures

**Relevance Score:** 3.50

---

## 108. MultiGen: Child-Friendly Multilingual Speech Generator with LLMs

**Authors:** Xiaoxue Gao, Huayun Zhang, Nancy F. Chen

**Categories:** cs.CL, eess.SP, eess.AS, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2508.08715v3) | [PDF](http://arxiv.org/pdf/2508.08715v3)

**Abstract:** Generative speech models have demonstrated significant potential in improving
human-machine interactions, offering valuable real-world applications such as
language learning for children. However, achieving high-quality, child-friendly
speech generation remains challenging, particularly for low-resource languages
across diverse languages and cultural contexts. In this paper, we propose
MultiGen, a multilingual speech generation model with child-friendly
interaction, leveraging LLM architecture for speech generation tailored for
low-resource languages. We propose to integrate age-appropriate multilingual
speech generation using LLM architectures, which can be used to facilitate
young children's communication with AI systems through culturally relevant
context in three low-resource languages: Singaporean accent Mandarin, Malay,
and Tamil. Experimental results from both objective metrics and subjective
evaluations demonstrate the superior performance of the proposed MultiGen
compared to baseline methods.

**Comment:** 5 pages

**Relevance Score:** 3.50

---

## 109. Enhancing Text2Cypher with Schema Filtering

**Authors:** Makbule Gulcin Ozsoy

**Categories:** cs.DB, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2505.05118v2) | [PDF](http://arxiv.org/pdf/2505.05118v2)

**Abstract:** Knowledge graphs represent complex data using nodes, relationships, and
properties. Cypher, a powerful query language for graph databases, enables
efficient modeling and querying. Recent advancements in large language models
allow translation of natural language questions into Cypher queries -
Text2Cypher. A common approach is incorporating database schema into prompts.
However, complex schemas can introduce noise, increase hallucinations, and
raise computational costs. Schema filtering addresses these challenges by
including only relevant schema elements, improving query generation while
reducing token costs. This work explores various schema filtering methods for
Text2Cypher task and analyzes their impact on token length, performance, and
cost. Results show that schema filtering effectively optimizes Text2Cypher,
especially for smaller models. Consistent with prior research, we find that
larger models benefit less from schema filtering due to their longer context
capabilities. However, schema filtering remains valuable for both larger and
smaller models in cost reduction.

**Relevance Score:** 3.50

---

## 110. Text2Cypher: Data Pruning using Hard Example Selection

**Authors:** Makbule Gulcin Ozsoy

**Categories:** cs.DB, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2505.05122v2) | [PDF](http://arxiv.org/pdf/2505.05122v2)

**Abstract:** Database query languages such as SQL for relational databases and Cypher for
graph databases have been widely adopted. Recent advancements in large language
models (LLMs) enable natural language interactions with databases through
models like Text2SQL and Text2Cypher. Fine-tuning these models typically
requires large, diverse datasets containing non-trivial examples. However, as
dataset size increases, the cost of fine-tuning also rises. This makes smaller,
high-quality datasets essential for reducing costs for the same or better
performance. In this paper, we propose five hard-example selection techniques
for pruning the Text2Cypher dataset, aiming to preserve or improve performance
while reducing resource usage. Our results show that these hard-example
selection approaches can halve training time and costs with minimal impact on
performance, and demonstrates that hard-example selection provides a
cost-effective solution.

**Relevance Score:** 3.50

---

## 111. Transition Models: Rethinking the Generative Learning Objective

**Authors:** Zidong Wang, Yiyuan Zhang, Xiaoyu Yue, Xiangyu Yue, Yangguang Li, Wanli Ouyang, Lei Bai

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04394v1) | [PDF](http://arxiv.org/pdf/2509.04394v1)

**Abstract:** A fundamental dilemma in generative modeling persists: iterative diffusion
models achieve outstanding fidelity, but at a significant computational cost,
while efficient few-step alternatives are constrained by a hard quality
ceiling. This conflict between generation steps and output quality arises from
restrictive training objectives that focus exclusively on either infinitesimal
dynamics (PF-ODEs) or direct endpoint prediction. We address this challenge by
introducing an exact, continuous-time dynamics equation that analytically
defines state transitions across any finite time interval. This leads to a
novel generative paradigm, Transition Models (TiM), which adapt to
arbitrary-step transitions, seamlessly traversing the generative trajectory
from single leaps to fine-grained refinement with more steps. Despite having
only 865M parameters, TiM achieves state-of-the-art performance, surpassing
leading models such as SD3.5 (8B parameters) and FLUX.1 (12B parameters) across
all evaluated step counts. Importantly, unlike previous few-step generators,
TiM demonstrates monotonic quality improvement as the sampling budget
increases. Additionally, when employing our native-resolution strategy, TiM
delivers exceptional fidelity at resolutions up to 4096x4096.

**Comment:** The code is released at https://github.com/WZDTHU/TiM

**Relevance Score:** 3.50

---

## 112. Imputation-free Learning of Tabular Data with Missing Values using   Incremental Feature Partitions in Transformer

**Authors:** Manar D. Samad, Kazi Fuad B. Akhter, Shourav B. Rabbani, Ibna Kowsar

**Categories:** stat.ML, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2504.14610v4) | [PDF](http://arxiv.org/pdf/2504.14610v4)

**Abstract:** Tabular data sets with varying missing values are prepared for machine
learning using an arbitrary imputation strategy. Synthetic values generated by
imputation models often raise concerns about data quality and the reliability
of data-driven outcomes. To address these concerns, this article proposes an
imputation-free incremental attention learning (IFIAL) method for tabular data.
A pair of attention masks is derived and retrofitted to a transformer to
directly streamline tabular data without imputing or initializing missing
values. The proposed method incrementally learns partitions of overlapping and
fixed-size feature sets to enhance the efficiency and performance of the
transformer. The average classification performance rank order across 17
diverse tabular data sets highlights the superiority of IFIAL over 11
state-of-the-art learning methods with or without missing value imputations.
Further experiments substantiate the robustness of IFIAL against varying
missing value types and rates compared to methods involving missing value
imputation. Our analysis reveals that a feature partition size of half the
original feature space is, both computationally and in terms of accuracy, the
best choice for the proposed incremental learning. The proposed method is one
of the first solutions to enable deep attention learning of tabular data
without requiring missing-value imputation. The source code for this paper is
publicly available.

**Relevance Score:** 3.50

---

## 113. Rethinking the long-range dependency in Mamba/SSM and transformer models

**Authors:** Cong Ma, Kayvan Najarian

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04226v1) | [PDF](http://arxiv.org/pdf/2509.04226v1)

**Abstract:** Long-range dependency is one of the most desired properties of recent
sequence models such as state-space models (particularly Mamba) and transformer
models. New model architectures are being actively developed and benchmarked
for prediction tasks requiring long-range dependency. However, the capability
of modeling long-range dependencies of these models has not been investigated
from a theoretical perspective, which hinders a systematic improvement on this
aspect. In this work, we mathematically define long-range dependency using the
derivative of hidden states with respect to past inputs and compare the
capability of SSM and transformer models of modeling long-range dependency
based on this definition. We showed that the long-range dependency of SSM
decays exponentially with the sequence length, which aligns with the
exponential decay of memory function in RNN. But the attention mechanism used
in transformers is more flexible and is not constrained to exponential decay,
which could in theory perform better at modeling long-range dependency with
sufficient training data, computing resources, and proper training. To combine
the flexibility of long-range dependency of attention mechanism and computation
efficiency of SSM, we propose a new formulation for hidden state update in SSM
and prove its stability under a standard Gaussian distribution of the input
data.

**Relevance Score:** 3.50

---

## 114. DUDE: Diffusion-Based Unsupervised Cross-Domain Image Retrieval

**Authors:** Ruohong Yang, Peng Hu, Yunfan Li, Xi Peng

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04193v1) | [PDF](http://arxiv.org/pdf/2509.04193v1)

**Abstract:** Unsupervised cross-domain image retrieval (UCIR) aims to retrieve images of
the same category across diverse domains without relying on annotations.
Existing UCIR methods, which align cross-domain features for the entire image,
often struggle with the domain gap, as the object features critical for
retrieval are frequently entangled with domain-specific styles. To address this
challenge, we propose DUDE, a novel UCIR method building upon feature
disentanglement. In brief, DUDE leverages a text-to-image generative model to
disentangle object features from domain-specific styles, thus facilitating
semantical image retrieval. To further achieve reliable alignment of the
disentangled object features, DUDE aligns mutual neighbors from within domains
to across domains in a progressive manner. Extensive experiments demonstrate
that DUDE achieves state-of-the-art performance across three benchmark datasets
over 13 domains. The code will be released.

**Relevance Score:** 3.50

---

## 115. Plugging Attention into Power Grids: Towards Transparent Forecasting

**Authors:** Eloi Campagne, Itai Zehavi, Yvenn Amara-Ouali, Yannig Goude, Argyris Kalogeratos

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2507.03690v2) | [PDF](http://arxiv.org/pdf/2507.03690v2)

**Abstract:** Reliable prediction of electricity demand plays a key role in safeguarding
grid stability and guiding generation decisions, a need that grows with the
decentralization and complexity of modern systems. While classical approaches
such as Generalized Additive Models (GAMs) remain widely used, they often fail
to capture the spatial dependencies inherent in energy networks. Graph Neural
Networks (GNNs) offer a principled framework to incorporate this structure by
directly leveraging graph topologies. In this work, we evaluate a broad set of
GNN architectures -- including GCN, GraphSAGE, ChebConv, TAG, APPNP,
TransformerConv, and Graph Attention Networks (GAT and GATv2) -- on two
real-world electricity consumption datasets from France and the UK. Our results
show that simpler models such as GCN, SAGE, or APPNP often outperform more
complex alternatives in low-data regimes, while GAT ranks among the strongest
architectures in our benchmarks, combining high accuracy with valuable
interpretability. We perform a temporal analysis of attention weights,
revealing evolving patterns of regional interaction linked to seasonal and
meteorological variability. These results highlight that, although attention is
not universally superior, it provides valuable explanatory power when spatial
dependencies are prominent. Additionally, we demonstrate that ensemble-based
expert aggregation strategies, particularly bottom-up combinations,
significantly improve robustness and yield state-of-the-art performance across
both datasets. These findings highlight the dual promise of GNNs for accurate
and interpretable forecasting, and suggest that architectural simplicity
coupled with ensemble methods can provide a practical path forward for
transparent energy analytics.

**Comment:** 16 pages, ECML PKDD 2025 Workshop paper

**Relevance Score:** 3.50

---

## 116. UniExtreme: A Universal Foundation Model for Extreme Weather Forecasting

**Authors:** Hang Ni, Weijia Zhang, Hao Liu

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2508.01426v2) | [PDF](http://arxiv.org/pdf/2508.01426v2)

**Abstract:** Recent advancements in deep learning have led to the development of
Foundation Models (FMs) for weather forecasting, yet their ability to predict
extreme weather events remains limited. Existing approaches either focus on
general weather conditions or specialize in specific-type extremes, neglecting
the real-world atmospheric patterns of diversified extreme events. In this
work, we identify two key characteristics of extreme events: (1) the spectral
disparity against normal weather regimes, and (2) the hierarchical drivers and
geographic blending of diverse extremes. Along this line, we propose
UniExtreme, a universal extreme weather forecasting foundation model that
integrates (1) an Adaptive Frequency Modulation (AFM) module that captures
region-wise spectral differences between normal and extreme weather, through
learnable Beta-distribution filters and multi-granularity spectral aggregation,
and (2) an Event Prior Augmentation (EPA) module which incorporates
region-specific extreme event priors to resolve hierarchical extreme diversity
and composite extreme schema, via a dual-level memory fusion network. Extensive
experiments demonstrate that UniExtreme outperforms state-of-the-art baselines
in both extreme and general weather forecasting, showcasing superior
adaptability across diverse extreme scenarios.

**Comment:** 35 pages, 80 figures, submitted to ACM KDD 2026 conference

**Relevance Score:** 3.50

---

## 117. Zero-shot Generalization in Inventory Management: Train, then Estimate   and Decide

**Authors:** Tarkan Temizöz, Christina Imdahl, Remco Dijkman, Douniel Lamghari-Idrissi, Willem van Jaarsveld

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.00515v2) | [PDF](http://arxiv.org/pdf/2411.00515v2)

**Abstract:** Deploying deep reinforcement learning (DRL) in real-world inventory
management presents challenges, including dynamic environments and uncertain
problem parameters, e.g. demand and lead time distributions. These challenges
highlight a research gap, suggesting a need for a unifying framework to model
and solve sequential decision-making under parameter uncertainty. We address
this by exploring an underexplored area of DRL for inventory management:
training generally capable agents (GCAs) under zero-shot generalization (ZSG).
Here, GCAs are advanced DRL policies designed to handle a broad range of
sampled problem instances with diverse inventory challenges. ZSG refers to the
ability to successfully apply learned policies to unseen instances with unknown
parameters without retraining.
  We propose a unifying Super-Markov Decision Process formulation and the
Train, then Estimate and Decide (TED) framework to train and deploy a GCA
tailored to inventory management applications. The TED framework consists of
three phases: training a GCA on varied problem instances, continuously
estimating problem parameters during deployment, and making decisions based on
these estimates. Applied to periodic review inventory problems with lost sales,
cyclic demand patterns, and stochastic lead times, our trained agent, the
Generally Capable Lost Sales Network (GC-LSN) consistently outperforms
well-known traditional policies when problem parameters are known. Moreover,
under conditions where demand and/or lead time distributions are initially
unknown and must be estimated, we benchmark against online learning methods
that provide worst-case performance guarantees. Our GC-LSN policy, paired with
the Kaplan-Meier estimator, is demonstrated to complement these methods by
providing superior empirical performance.

**Relevance Score:** 3.50

---

## 118. Balancing Signal and Variance: Adaptive Offline RL Post-Training for VLA   Flow Models

**Authors:** Hongyin Zhang, Shiyuan Zhang, Junxi Jin, Qixin Zeng, Yifan Qiao, Hongchao Lu, Donglin Wang

**Categories:** cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04063v1) | [PDF](http://arxiv.org/pdf/2509.04063v1)

**Abstract:** Vision-Language-Action (VLA) models based on flow matching have shown
excellent performance in general-purpose robotic manipulation tasks. However,
the action accuracy of these models on complex downstream tasks is
unsatisfactory. One important reason is that these models rely solely on the
post-training paradigm of imitation learning, which makes it difficult to have
a deeper understanding of the distribution properties of data quality, which is
exactly what Reinforcement Learning (RL) excels at. In this paper, we
theoretically propose an offline RL post-training objective for VLA flow models
and induce an efficient and feasible offline RL fine-tuning algorithm --
Adaptive Reinforced Flow Matching (ARFM). By introducing an adaptively adjusted
scaling factor in the VLA flow model loss, we construct a principled
bias-variance trade-off objective function to optimally control the impact of
RL signal on flow loss. ARFM adaptively balances RL advantage preservation and
flow loss gradient variance control, resulting in a more stable and efficient
fine-tuning process. Extensive simulation and real-world experimental results
show that ARFM exhibits excellent generalization, robustness, few-shot
learning, and continuous learning performance.

**Relevance Score:** 3.50

---

## 119. MARS: Unleashing the Power of Variance Reduction for Training Large   Models

**Authors:** Huizhuo Yuan, Yifeng Liu, Shuang Wu, Xun Zhou, Quanquan Gu

**Categories:** math.OC, stat.ML, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.10438v4) | [PDF](http://arxiv.org/pdf/2411.10438v4)

**Abstract:** Training deep neural networks--and more recently, large models demands
efficient and scalable optimizers. Adaptive gradient algorithms like Adam,
AdamW, and their variants have been central to this task. Despite the
development of numerous variance reduction algorithms in the past decade aimed
at accelerating stochastic optimization in both convex and nonconvex settings,
variance reduction has not found widespread success in training deep neural
networks or large language models. Consequently, it has remained a less favored
approach in modern AI. In this paper, to unleash the power of variance
reduction for efficient training of large models, we propose a unified
optimization framework, MARS (Make vAriance Reduction Shine), which reconciles
preconditioned gradient methods with variance reduction via a scaled stochastic
recursive momentum technique. Within our framework, we introduce three
instances of MARS that leverage preconditioned gradient updates based on AdamW,
Lion, and Shampoo, respectively. We also draw a connection between our
algorithms and existing optimizers. Experimental results on training GPT-2
models indicate that MARS consistently outperforms AdamW by a large margin. The
implementation of MARS is available at https://github.com/AGI-Arena/MARS.

**Comment:** 35 pages, 19 figures, 12 tables

**Relevance Score:** 3.50

---

## 120. EvolveSignal: A Large Language Model Powered Coding Agent for   Discovering Traffic Signal Control Algorithms

**Authors:** Leizhen Wang, Peibo Duan, Hao Wang, Yue Wang, Jian Xu, Nan Zheng, Zhenliang Ma

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03335v2) | [PDF](http://arxiv.org/pdf/2509.03335v2)

**Abstract:** In traffic engineering, the fixed-time traffic signal control remains widely
used for its low cost, stability, and interpretability. However, its design
depends on hand-crafted formulas (e.g., Webster) and manual re-timing by
engineers to adapt to demand changes, which is labor-intensive and often yields
suboptimal results under heterogeneous or congested conditions. This paper
introduces the EvolveSignal, a large language models (LLMs) powered coding
agent to automatically discover new traffic signal control algorithms. We
formulate the problem as program synthesis, where candidate algorithms are
represented as Python functions with fixed input-output structures, and
iteratively optimized through external evaluations (e.g., a traffic simulator)
and evolutionary search. Experiments on a signalized intersection demonstrate
that the discovered algorithms outperform Webster's baseline, reducing average
delay by 20.1% and average stops by 47.1%. Beyond performance, ablation and
incremental analyses reveal that EvolveSignal modifications-such as adjusting
cycle length bounds, incorporating right-turn demand, and rescaling green
allocations-can offer practically meaningful insights for traffic engineers.
This work opens a new research direction by leveraging AI for algorithm design
in traffic signal control, bridging program synthesis with transportation
engineering.

**Relevance Score:** 3.50

---

## 121. RadioDiff-Loc: Diffusion Model Enhanced Scattering Congnition for NLoS   Localization with Sparse Radio Map Estimation

**Authors:** Xiucheng Wang, Qiming Zhang, Nan Cheng

**Categories:** eess.SY, cs.LG, cs.SY

**Links:** [arXiv](http://arxiv.org/abs/2509.01875v2) | [PDF](http://arxiv.org/pdf/2509.01875v2)

**Abstract:** Accurate localization of non-cooperative signal sources in non-line-of-sight
(NLoS) environments remains a critical challenge with a wide range of
applications, including autonomous navigation, industrial automation, and
emergency response. In such settings, traditional positioning techniques
relying on line-of-sight (LoS) or cooperative signaling fail due to severe
multipath propagation and unknown transmit power. This paper proposes a novel
generative inference framework for NLoS localization based on conditional
diffusion models. By leveraging the physical insight that diffracted
electromagnetic energy concentrates near building edges, we develop a sampling
strategy that collects sparse received signal strength (RSS) measurements at
the geometric vertices of obstacles--locations that maximize Fisher information
and mutual information with respect to the unknown source. To overcome the lack
of known transmission power, we normalize all sampled RSS values relative to
the maximum observed intensity, enabling the construction of a power-invariant
radio map (RM). A conditional diffusion model is trained to reconstruct the
full RM based on environmental layout and sparse RSS observations. Localization
is then achieved by identifying the brightest point on the generated RM.
Moreover, the proposed framework is compatible with existing RSS-based
localization algorithms, enabling a dual-driven paradigm that fuses physical
knowledge and data-driven inference for improved accuracy. Extensive
theoretical analysis and empirical validation demonstrate that our approach
achieves high localization accuracy with significantly reduced sampling cost,
offering a scalable and physically grounded solution for non-cooperative NLoS
emitter localization.

**Relevance Score:** 3.50

---

## 122. Efficient Odd-One-Out Anomaly Detection

**Authors:** Silvio Chito, Paolo Rabino, Tatiana Tommasi

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04326v1) | [PDF](http://arxiv.org/pdf/2509.04326v1)

**Abstract:** The recently introduced odd-one-out anomaly detection task involves
identifying the odd-looking instances within a multi-object scene. This problem
presents several challenges for modern deep learning models, demanding spatial
reasoning across multiple views and relational reasoning to understand context
and generalize across varying object categories and layouts. We argue that
these challenges must be addressed with efficiency in mind. To this end, we
propose a DINO-based model that reduces the number of parameters by one third
and shortens training time by a factor of three compared to the current
state-of-the-art, while maintaining competitive performance. Our experimental
evaluation also introduces a Multimodal Large Language Model baseline,
providing insights into its current limitations in structured visual reasoning
tasks. The project page can be found at
https://silviochito.github.io/EfficientOddOneOut/

**Comment:** Accepted at ICIAP 2025

**Relevance Score:** 3.50

---

## 123. Differential Morphological Profile Neural Networks for Semantic   Segmentation

**Authors:** David Huangal, J. Alex Hurt

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04268v1) | [PDF](http://arxiv.org/pdf/2509.04268v1)

**Abstract:** Semantic segmentation of overhead remote sensing imagery enables applications
in mapping, urban planning, and disaster response. State-of-the-art
segmentation networks are typically developed and tuned on ground-perspective
photographs and do not directly address remote sensing challenges such as
extreme scale variation, foreground-background imbalance, and large image
sizes. We explore the incorporation of the differential morphological profile
(DMP), a multi-scale shape extraction method based on grayscale morphology,
into modern segmentation networks. Prior studies have shown that the DMP can
provide critical shape information to Deep Neural Networks to enable superior
detection and classification performance in overhead imagery. In this work, we
extend prior DMPNet work beyond classification and object detection by
integrating DMP features into three state-of-the-art convolutional and
transformer semantic segmentation architectures. We utilize both direct input,
which adapts the input stem of feature extraction architectures to accept DMP
channels, and hybrid architectures, a dual-stream design that fuses RGB and DMP
encoders. Using the iSAID benchmark dataset, we evaluate a variety of DMP
differentials and structuring element shapes to more effectively provide shape
information to the model. Our results show that while non-DMP models generally
outperform the direct-input variants, hybrid DMP consistently outperforms
direct-input and is capable of surpassing a non-DMP model on mIoU, F1, and
Recall.

**Comment:** 14 pages, 7 figures

**Relevance Score:** 3.50

---

## 124. Hyper Diffusion Avatars: Dynamic Human Avatar Generation using Network   Weight Space Diffusion

**Authors:** Dongliang Cao, Guoxing Sun, Marc Habermann, Florian Bernard

**Categories:** cs.GR, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04145v1) | [PDF](http://arxiv.org/pdf/2509.04145v1)

**Abstract:** Creating human avatars is a highly desirable yet challenging task. Recent
advancements in radiance field rendering have achieved unprecedented
photorealism and real-time performance for personalized dynamic human avatars.
However, these approaches are typically limited to person-specific rendering
models trained on multi-view video data for a single individual, limiting their
ability to generalize across different identities. On the other hand,
generative approaches leveraging prior knowledge from pre-trained 2D diffusion
models can produce cartoonish, static human avatars, which are animated through
simple skeleton-based articulation. Therefore, the avatars generated by these
methods suffer from lower rendering quality compared to person-specific
rendering methods and fail to capture pose-dependent deformations such as cloth
wrinkles. In this paper, we propose a novel approach that unites the strengths
of person-specific rendering and diffusion-based generative modeling to enable
dynamic human avatar generation with both high photorealism and realistic
pose-dependent deformations. Our method follows a two-stage pipeline: first, we
optimize a set of person-specific UNets, with each network representing a
dynamic human avatar that captures intricate pose-dependent deformations. In
the second stage, we train a hyper diffusion model over the optimized network
weights. During inference, our method generates network weights for real-time,
controllable rendering of dynamic human avatars. Using a large-scale,
cross-identity, multi-view video dataset, we demonstrate that our approach
outperforms state-of-the-art human avatar generation methods.

**Relevance Score:** 3.50

---

## 125. Deep Learning Advances in Vision-Based Traffic Accident Anticipation: A   Comprehensive Review of Methods, Datasets, and Future Directions

**Authors:** Ruonan Lin, Tao Tang, Yongtai Liu, Wenye Zhou, Xin Yang, Hao Zheng, Jianpu Lin, Yi Zhang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2505.07611v2) | [PDF](http://arxiv.org/pdf/2505.07611v2)

**Abstract:** Traffic accident prediction and detection are critical for enhancing road
safety, and vision-based traffic accident anticipation (Vision-TAA) has emerged
as a promising approach in the era of deep learning. This paper reviews 147
recent studies, focusing on the application of supervised, unsupervised, and
hybrid deep learning models for accident prediction, alongside the use of
real-world and synthetic datasets. Current methodologies are categorized into
four key approaches: image and video feature-based prediction, spatio-temporal
feature-based prediction, scene understanding, and multi modal data fusion.
While these methods demonstrate significant potential, challenges such as data
scarcity, limited generalization to complex scenarios, and real-time
performance constraints remain prevalent. This review highlights opportunities
for future research, including the integration of multi modal data fusion,
self-supervised learning, and Transformer-based architectures to enhance
prediction accuracy and scalability. By synthesizing existing advancements and
identifying critical gaps, this paper provides a foundational reference for
developing robust and adaptive Vision-TAA systems, contributing to road safety
and traffic management.

**Relevance Score:** 3.50

---

## 126. SliceSemOcc: Vertical Slice Based Multimodal 3D Semantic Occupancy   Representation

**Authors:** Han Huang, Han Sun, Ningzhong Liu, Huiyu Zhou, Jiaquan Shen

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03999v1) | [PDF](http://arxiv.org/pdf/2509.03999v1)

**Abstract:** Driven by autonomous driving's demands for precise 3D perception, 3D semantic
occupancy prediction has become a pivotal research topic. Unlike
bird's-eye-view (BEV) methods, which restrict scene representation to a 2D
plane, occupancy prediction leverages a complete 3D voxel grid to model spatial
structures in all dimensions, thereby capturing semantic variations along the
vertical axis. However, most existing approaches overlook height-axis
information when processing voxel features. And conventional SENet-style
channel attention assigns uniform weight across all height layers, limiting
their ability to emphasize features at different heights. To address these
limitations, we propose SliceSemOcc, a novel vertical slice based multimodal
framework for 3D semantic occupancy representation. Specifically, we extract
voxel features along the height-axis using both global and local vertical
slices. Then, a global local fusion module adaptively reconciles fine-grained
spatial details with holistic contextual information. Furthermore, we propose
the SEAttention3D module, which preserves height-wise resolution through
average pooling and assigns dynamic channel attention weights to each height
layer. Extensive experiments on nuScenes-SurroundOcc and nuScenes-OpenOccupancy
datasets verify that our method significantly enhances mean IoU, achieving
especially pronounced gains on most small-object categories. Detailed ablation
studies further validate the effectiveness of the proposed SliceSemOcc
framework.

**Comment:** 14 pages, accepted by PRCV2025

**Relevance Score:** 3.50

---

## 127. Separate to Collaborate: Dual-Stream Diffusion Model for Coordinated   Piano Hand Motion Synthesis

**Authors:** Zihao Liu, Mingwen Ou, Zunnan Xu, Jiaqi Huang, Haonan Han, Ronghui Li, Xiu Li

**Categories:** cs.SD, cs.CV, eess.AS

**Links:** [arXiv](http://arxiv.org/abs/2504.09885v2) | [PDF](http://arxiv.org/pdf/2504.09885v2)

**Abstract:** Automating the synthesis of coordinated bimanual piano performances poses
significant challenges, particularly in capturing the intricate choreography
between the hands while preserving their distinct kinematic signatures. In this
paper, we propose a dual-stream neural framework designed to generate
synchronized hand gestures for piano playing from audio input, addressing the
critical challenge of modeling both hand independence and coordination. Our
framework introduces two key innovations: (i) a decoupled diffusion-based
generation framework that independently models each hand's motion via
dual-noise initialization, sampling distinct latent noise for each while
leveraging a shared positional condition, and (ii) a Hand-Coordinated
Asymmetric Attention (HCAA) mechanism suppresses symmetric (common-mode) noise
to highlight asymmetric hand-specific features, while adaptively enhancing
inter-hand coordination during denoising. Comprehensive evaluations demonstrate
that our framework outperforms existing state-of-the-art methods across
multiple metrics. Our project is available at
https://monkek123king.github.io/S2C_page/.

**Comment:** 15 pages, 7 figures, Accepted to ACMMM 2025

**Relevance Score:** 3.50

---

## 128. DianJin-OCR-R1: Enhancing OCR Capabilities via a Reasoning-and-Tool   Interleaved Vision-Language Model

**Authors:** Qian Chen, Xianyin Zhang, Lifan Guo, Feng Chen, Chi Zhang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.13238v2) | [PDF](http://arxiv.org/pdf/2508.13238v2)

**Abstract:** Recent advances in large vision-language models (LVLMs) have enabled a new
paradigm of end-to-end document image parsing, excelling in Optical Character
Recognition (OCR) tasks such as text, table, and formula recognition. However,
generative LVLMs, similarly to large language models (LLMs), are prone to
hallucinations--generating words that do not exist in input images.
Furthermore, LVLMs are designed for general purposes and tend to be less
effective on OCR tasks compared to expert models that are trained on
domain-specific datasets. In this paper, we propose DianJin-OCR-R1, a
reasoning-enhanced framework designed to address these limitations through
training reasoning-and-tool interleaved VLMs. Given a recognition instruction,
our DianJin-OCR-R1 model first recognizes the content in the input image by its
own OCR capabilities, and then calls other tools (i.e., other expert models) to
obtain their results as references, finally "looks again" the image and
rethinks about the reasoning process to provide the final recognized content.
Since architectures of expert models are tailored for specific OCR tasks, which
makes them less prone to hallucinations, their results can help VLMs mitigate
hallucinations. We evaluate our model on ReST and OmniDocBench, and
experimental results show that our DianJin-OCR-R1 models consistently
outperform their non-reasoning counterparts and expert OCR models, which proves
the effectiveness of our method. Additionally, the results indicate that
enhancing expert models, which are typically small and easy to iterate, enable
performance improvements for VLMs.

**Relevance Score:** 3.50

---

## 129. Multimodal Feature Fusion Network with Text Difference Enhancement for   Remote Sensing Change Detection

**Authors:** Yijun Zhou, Yikui Zhai, Zilu Ying, Tingfeng Xian, Wenlve Zhou, Zhiheng Zhou, Xiaolin Tian, Xudong Jia, Hongsheng Zhang, C. L. Philip Chen

**Categories:** cs.CV, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03961v1) | [PDF](http://arxiv.org/pdf/2509.03961v1)

**Abstract:** Although deep learning has advanced remote sensing change detection (RSCD),
most methods rely solely on image modality, limiting feature representation,
change pattern modeling, and generalization especially under illumination and
noise disturbances. To address this, we propose MMChange, a multimodal RSCD
method that combines image and text modalities to enhance accuracy and
robustness. An Image Feature Refinement (IFR) module is introduced to highlight
key regions and suppress environmental noise. To overcome the semantic
limitations of image features, we employ a vision language model (VLM) to
generate semantic descriptions of bitemporal images. A Textual Difference
Enhancement (TDE) module then captures fine grained semantic shifts, guiding
the model toward meaningful changes. To bridge the heterogeneity between
modalities, we design an Image Text Feature Fusion (ITFF) module that enables
deep cross modal integration. Extensive experiments on LEVIRCD, WHUCD, and
SYSUCD demonstrate that MMChange consistently surpasses state of the art
methods across multiple metrics, validating its effectiveness for multimodal
RSCD. Code is available at: https://github.com/yikuizhai/MMChange.

**Relevance Score:** 3.50

---

## 130. Is an Ultra Large Natural Image-Based Foundation Model Superior to a   Retina-Specific Model for Detecting Ocular and Systemic Diseases?

**Authors:** Qingshan Hou, Yukun Zhou, Jocelyn Hui Lin Goh, Ke Zou, Samantha Min Er Yew, Sahana Srinivasan, Meng Wang, Thaddaeus Lo, Xiaofeng Lei, Siegfried K. Wagner, Mark A. Chia, Dawei Yang, Hongyang Jiang, An Ran Ran, Rui Santos, Gabor Mark Somfai, Juan Helen Zhou, Haoyu Chen, Qingyu Chen, Carol Y. Cheung, Pearse A. Keane, Yih Chung Tham

**Categories:** eess.IV, cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2502.06289v2) | [PDF](http://arxiv.org/pdf/2502.06289v2)

**Abstract:** The advent of foundation models (FMs) is transforming medical domain. In
ophthalmology, RETFound, a retina-specific FM pre-trained sequentially on 1.4
million natural images and 1.6 million retinal images, has demonstrated high
adaptability across clinical applications. Conversely, DINOv2, a
general-purpose vision FM pre-trained on 142 million natural images, has shown
promise in non-medical domains. However, its applicability to clinical tasks
remains underexplored. To address this, we conducted head-to-head evaluations
by fine-tuning RETFound and three DINOv2 models (large, base, small) for ocular
disease detection and systemic disease prediction tasks, across eight
standardized open-source ocular datasets, as well as the Moorfields AlzEye and
the UK Biobank datasets. DINOv2-large model outperformed RETFound in detecting
diabetic retinopathy (AUROC=0.850-0.952 vs 0.823-0.944, across three datasets,
all P<=0.007) and multi-class eye diseases (AUROC=0.892 vs. 0.846, P<0.001). In
glaucoma, DINOv2-base model outperformed RETFound (AUROC=0.958 vs 0.940,
P<0.001). Conversely, RETFound achieved superior performance over all DINOv2
models in predicting heart failure, myocardial infarction, and ischaemic stroke
(AUROC=0.732-0.796 vs 0.663-0.771, all P<0.001). These trends persisted even
with 10% of the fine-tuning data. These findings showcase the distinct
scenarios where general-purpose and domain-specific FMs excel, highlighting the
importance of aligning FM selection with task-specific requirements to optimise
clinical performance.

**Comment:** Accepted by Ophthalmology Science and is currently in press

**Relevance Score:** 3.50

---

## 131. EQ-Knight: A Memory-Augmented LLM Agent for Strategic Affective Gaming   in Debt Recovery

**Authors:** Yunbo Long, Yuhan Liu, Liming Xu, Alexandra Brintrup

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2503.21080v4) | [PDF](http://arxiv.org/pdf/2503.21080v4)

**Abstract:** Large language model-based chatbots have enhanced engagement in financial
negotiations, but their overreliance on passive empathy introduces critical
risks in credit collection. While empathy-driven approaches preserve client
satisfaction in benign cases, they fail catastrophically against dishonest
debtors--individuals who exploit conciliatory tactics to manipulate terms or
evade repayment. Blindly prioritizing "customer experience" in such scenarios
leads to creditor vulnerabilities: revenue leakage, moral hazard, and systemic
exploitation. To address this, we propose EQ-Knight, an LLM agent that
dynamically optimizes emotional strategy to defend creditor interests. Unlike
naive empathy-centric bots, EQ-Knight integrates emotion memory and
game-theoretic reasoning, powered by a Hidden Markov Model (HMM) to track and
predict debtor emotional states. By analyzing both real-time and historical
emotional cues, EQ-Knight strategically counters negative emotions (e.g.,
aggression, feigned distress) while preserving productive debtor relationships.
Experiments demonstrate EQ-Knight's superiority over conventional LLM
negotiators: it achieves a 32\% reduction in concession losses without
compromising recovery rates, particularly in adversarial cases where debtors
weaponize negative emotions (e.g., intimidation, guilt-tripping) to coerce
concessions. For credit agencies, EQ-Knight transforms LLMs from high-risk
"people-pleasers" into strategic emotion-defenders--balancing emotional
intelligence with tactical rigor to enforce accountability and deter
exploitation.

**Relevance Score:** 3.50

---

## 132. Explicit Learning and the LLM in Machine Translation

**Authors:** Malik Marmonier, Rachel Bawden, Benoît Sagot

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2503.09454v4) | [PDF](http://arxiv.org/pdf/2503.09454v4)

**Abstract:** This study explores an LLM's ability to learn new languages using
explanations found in a grammar book, a process we term "explicit learning." To
rigorously assess this ability, we design controlled translation experiments
between English and constructed languages generated, through specific
cryptographic means, from Latin or French. Contrary to previous studies, our
results demonstrate that LLMs do possess a measurable capacity for explicit
learning. This ability, however, diminishes as the complexity of the linguistic
phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of
thought significantly enhances LLM performance but struggles to generalize to
typologically novel or more complex linguistic features. These findings point
to the need for more diverse training sets and alternative fine-tuning
strategies to further improve explicit learning by LLMs, benefiting
low-resource languages typically described in grammar books but lacking
extensive corpora.

**Relevance Score:** 3.50

---

## 133. Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges

**Authors:** Yuqi Tang, Kehua Feng, Yunfeng Wang, Zhiwen Chen, Chengfei Lv, Gang Yu, Qiang Zhang, Keyan Ding

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.00454v2) | [PDF](http://arxiv.org/pdf/2508.00454v2)

**Abstract:** Evaluating the conversational abilities of large language models (LLMs)
remains a challenging task. Current mainstream approaches primarily rely on the
"LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to
assess dialogue quality. However, such methods often suffer from various
biases, which undermine the reliability and consistency of the evaluation
results. To mitigate these biases, recent methods employ multiple LLMs as
judges and aggregate their judgments to select the optimal assessment. Although
effective, this multi-judge approach incurs significant computational overhead
during inference. In this paper, we propose an efficient multi-turn dialogue
evaluator that captures the collective wisdom of multiple LLM judges by
aggregating their preference knowledge into a single model. Our approach
preserves the advantages of diverse multi-judge feedback while drastically
reducing the evaluation cost, enabling fast and flexible dialogue quality
assessment. Extensive experiments on seven single rating and pairwise
comparison dialogue evaluation benchmarks demonstrate that our method
outperforms existing baselines across diverse scenarios, showcasing its
efficiency and robustness.

**Comment:** 15 pages, 2 pages, under review

**Relevance Score:** 3.50

---

## 134. MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for   Speech Emotion Recognition in Naturalistic Conditions

**Authors:** Georgios Chatzichristodoulou, Despoina Kosmopoulou, Antonios Kritikos, Anastasia Poulopoulou, Efthymios Georgiou, Athanasios Katsamanis, Vassilis Katsouros, Alexandros Potamianos

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2506.09556v2) | [PDF](http://arxiv.org/pdf/2506.09556v2)

**Abstract:** SER is a challenging task due to the subjective nature of human emotions and
their uneven representation under naturalistic conditions. We propose MEDUSA, a
multimodal framework with a four-stage training pipeline, which effectively
handles class imbalance and emotion ambiguity. The first two stages train an
ensemble of classifiers that utilize DeepSER, a novel extension of a deep
cross-modal transformer fusion mechanism from pretrained self-supervised
acoustic and linguistic representations. Manifold MixUp is employed for further
regularization. The last two stages optimize a trainable meta-classifier that
combines the ensemble predictions. Our training approach incorporates human
annotation scores as soft targets, coupled with balanced data sampling and
multitask learning. MEDUSA ranked 1st in Task 1: Categorical Emotion
Recognition in the Interspeech 2025: Speech Emotion Recognition in Naturalistic
Conditions Challenge.

**Comment:** Interspeech 2025

**Relevance Score:** 3.50

---

## 135. LibriQuote: A Speech Dataset of Fictional Character Utterances for   Expressive Zero-Shot Speech Synthesis

**Authors:** Gaspard Michel, Elena V. Epure, Christophe Cerisara

**Categories:** eess.AS, cs.CL, cs.SD

**Links:** [arXiv](http://arxiv.org/abs/2509.04072v1) | [PDF](http://arxiv.org/pdf/2509.04072v1)

**Abstract:** Text-to-speech (TTS) systems have recently achieved more expressive and
natural speech synthesis by scaling to large speech datasets. However, the
proportion of expressive speech in such large-scale corpora is often unclear.
Besides, existing expressive speech corpora are typically smaller in scale and
primarily used for benchmarking TTS systems. In this paper, we introduce the
LibriQuote dataset, an English corpus derived from read audiobooks, designed
for both fine-tuning and benchmarking expressive zero-shot TTS system. The
training dataset includes 12.7K hours of read, non-expressive speech and 5.3K
hours of mostly expressive speech drawn from character quotations. Each
utterance in the expressive subset is supplemented with the context in which it
was written, along with pseudo-labels of speech verbs and adverbs used to
describe the quotation (\textit{e.g. ``he whispered softly''}). Additionally,
we provide a challenging 7.5 hour test set intended for benchmarking TTS
systems: given a neutral reference speech as input, we evaluate system's
ability to synthesize an expressive utterance while preserving reference
timbre. We validate qualitatively the test set by showing that it covers a wide
range of emotions compared to non-expressive speech, along with various
accents. Extensive subjective and objective evaluations show that fine-tuning a
baseline TTS system on LibriQuote significantly improves its synthesized speech
intelligibility, and that recent systems fail to synthesize speech as
expressive and natural as the ground-truth utterances. The dataset and
evaluation code are freely available. Audio samples can be found at
https://libriquote.github.io/.

**Relevance Score:** 3.50

---

## 136. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to   Enhance LLM Safety Guardrail to Potential Attacks

**Authors:** Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.20038v3) | [PDF](http://arxiv.org/pdf/2508.20038v3)

**Abstract:** Despite advances in improving large language model (LLM) to refuse to answer
malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks
where attackers generate instructions with distributions differing from safety
alignment corpora. New attacks expose LLMs' inability to recognize unseen
malicious instructions, highlighting a critical distributional mismatch between
training data and real-world attacks that forces developers into reactive
patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis
framework that leverages embedding space distribution analysis to generate
jailbreak-like instructions. This approach effectively fills the distributional
gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE
follows an iterative optimization process that dynamically evolves text
generation distributions across iterations, thereby augmenting the coverage of
safety alignment data distributions through synthesized data examples. Based on
the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates
significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2
without compromising their utility.

**Comment:** EMNLP 2025 findings

**Relevance Score:** 3.50

---

## 137. Mitigating Bias in Text Classification via Prompt-Based Text   Transformation

**Authors:** Charmaine Barker, Dimitar Kazakov

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2305.06166v3) | [PDF](http://arxiv.org/pdf/2305.06166v3)

**Abstract:** The presence of specific linguistic signals particular to a certain sub-group
can become highly salient to language models during training. In automated
decision-making settings, this may lead to biased outcomes when models rely on
cues that correlate with protected characteristics. We investigate whether
prompting ChatGPT to rewrite text using simplification, neutralisation,
localisation, and formalisation can reduce demographic signals while preserving
meaning. Experimental results show a statistically significant drop in location
classification accuracy across multiple models after transformation, suggesting
reduced reliance on group-specific language. At the same time, sentiment
analysis and rating prediction tasks confirm that the core meaning of the
reviews remains greatly intact. These results suggest that prompt-based
rewriting offers a practical and generalisable approach for mitigating bias in
text classification.

**Comment:** This version corrects an error in the model specification

**Relevance Score:** 3.50

---

## 138. FutureGen: A RAG-based Approach to Generate the Future Work of   Scientific Article

**Authors:** Ibrahim Al Azher, Miftahul Jannat Mokarrama, Zhishuai Guo, Sagnik Ray Choudhury, Hamed Alhoori

**Categories:** cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2503.16561v3) | [PDF](http://arxiv.org/pdf/2503.16561v3)

**Abstract:** The Future Work section of a scientific article outlines potential research
directions by identifying gaps and limitations of a current study. This section
serves as a valuable resource for early-career researchers seeking unexplored
areas and experienced researchers looking for new projects or collaborations.
In this study, we generate future work suggestions from a scientific article.
To enrich the generation process with broader insights and reduce the chance of
missing important research directions, we use context from related papers using
RAG. We experimented with various Large Language Models (LLMs) integrated into
Retrieval-Augmented Generation (RAG). We incorporate an LLM feedback mechanism
to enhance the quality of the generated content and introduce an LLM-as-a-judge
framework for robust evaluation, assessing key aspects such as novelty,
hallucination, and feasibility. Our results demonstrate that the RAG-based
approach using GPT-4o mini, combined with an LLM feedback mechanism,
outperforms other methods based on both qualitative and quantitative
evaluations. Moreover, we conduct a human evaluation to assess the LLM as an
extractor, generator, and feedback provider.

**Comment:** 12 pages, 6 figures, Accepted for publication at the Workshop on AI
  Principles in Science Communication (Ai4SC'25), held in conjunction with the
  IEEE eScience Conference 2025

**Relevance Score:** 3.50

---

## 139. Short-video Propagation Influence Rating: A New Real-world Dataset and A   New Large Graph Model

**Authors:** Dizhan Xue, Shengsheng Qian, Chuanrui Hu, Changsheng Xu

**Categories:** cs.CV, cs.CL, cs.LG, cs.MM, cs.SI

**Links:** [arXiv](http://arxiv.org/abs/2503.23746v2) | [PDF](http://arxiv.org/pdf/2503.23746v2)

**Abstract:** Short-video platforms have gained immense popularity, captivating the
interest of millions, if not billions, of users globally. Recently, researchers
have highlighted the significance of analyzing the propagation of short-videos,
which typically involves discovering commercial values, public opinions, user
behaviors, etc. This paper proposes a new Short-video Propagation Influence
Rating (SPIR) task and aims to promote SPIR from both the dataset and method
perspectives. First, we propose a new Cross-platform Short-Video (XS-Video)
dataset, which aims to provide a large-scale and real-world short-video
propagation network across various platforms to facilitate the research on
short-video propagation. Our XS-Video dataset includes 117,720 videos, 381,926
samples, and 535 topics across 5 biggest Chinese platforms, annotated with the
propagation influence from level 0 to 9. To the best of our knowledge, this is
the first large-scale short-video dataset that contains cross-platform data or
provides all of the views, likes, shares, collects, fans, comments, and comment
content. Second, we propose a Large Graph Model (LGM) named NetGPT, based on a
novel three-stage training mechanism, to bridge heterogeneous graph-structured
data with the powerful reasoning ability and knowledge of Large Language Models
(LLMs). Our NetGPT can comprehend and analyze the short-video propagation
graph, enabling it to predict the long-term propagation influence of
short-videos. Comprehensive experimental results evaluated by both
classification and regression metrics on our XS-Video dataset indicate the
superiority of our method for SPIR.

**Relevance Score:** 3.50

---

## 140. Explaining Length Bias in LLM-Based Preference Evaluations

**Authors:** Zhengyu Hu, Linxin Song, Jieyu Zhang, Zheyuan Xiao, Tianfu Wang, Zhengyu Chen, Nicholas Jing Yuan, Jianxun Lian, Kaize Ding, Hui Xiong

**Categories:** cs.LG, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2407.01085v5) | [PDF](http://arxiv.org/pdf/2407.01085v5)

**Abstract:** The use of large language models (LLMs) as judges, particularly in preference
comparisons, has become widespread, but this reveals a notable bias towards
longer responses, undermining the reliability of such evaluations. To better
understand such bias, we propose to decompose the preference evaluation metric,
specifically the win rate, into two key components: desirability and
information mass, where the former is length-independent and related to
trustworthiness such as correctness, toxicity, and consistency, and the latter
is length-dependent and represents the amount of information in the response.
We empirically demonstrated the decomposition through controlled experiments
and found that response length impacts evaluations by influencing information
mass. To derive a reliable evaluation metric that assesses content quality
without being confounded by response length, we propose AdapAlpaca, a simple
yet effective adjustment to win rate measurement. Specifically, AdapAlpaca
ensures a fair comparison of response quality by aligning the lengths of
reference and test model responses under equivalent length intervals.

**Relevance Score:** 3.50

---

## 141. Science Across Languages: Assessing LLM Multilingual Translation of   Scientific Papers

**Authors:** Hannah Calzi Kleidermacher, James Zou

**Categories:** cs.AI, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2502.17882v2) | [PDF](http://arxiv.org/pdf/2502.17882v2)

**Abstract:** Scientific research is inherently global. However, the vast majority of
academic journals are published exclusively in English, creating barriers for
non-native-English-speaking researchers. In this study, we leverage large
language models (LLMs) to translate published scientific articles while
preserving their native JATS XML formatting, thereby developing a practical,
automated approach for implementation by academic journals. Using our approach,
we translate articles across multiple scientific disciplines into 28 languages.
To evaluate translation accuracy, we introduce a novel question-and-answer (QA)
benchmarking method, in which an LLM generates comprehension-based questions
from the original text and then answers them based on the translated text. Our
benchmark results show an average performance of 95.9%, showing that the key
scientific details are accurately conveyed. In a user study, we translate the
scientific papers of 15 researchers into their native languages, finding that
the authors consistently found the translations to accurately capture the
original information in their articles. Interestingly, a third of the authors
found many technical terms "overtranslated," expressing a preference to keep
terminology more familiar in English untranslated. Finally, we demonstrate how
in-context learning techniques can be used to align translations with
domain-specific preferences such as mitigating overtranslation, highlighting
the adaptability and utility of LLM-driven scientific translation. The code and
translated articles are available at https://hankleid.github.io/ProjectMundo.

**Relevance Score:** 3.50

---

## 142. SRWToolkit: An Open Source Wizard of Oz Toolkit to Create Social Robotic   Avatars

**Authors:** Atikkhan Faridkhan Nilgar, Kristof Van Laerhoven, Ayub Kinoti

**Categories:** cs.HC, cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04356v1) | [PDF](http://arxiv.org/pdf/2509.04356v1)

**Abstract:** We present SRWToolkit, an open-source Wizard of Oz toolkit designed to
facilitate the rapid prototyping of social robotic avatars powered by local
large language models (LLMs). Our web-based toolkit enables multimodal
interaction through text input, button-activated speech, and wake-word command.
The toolkit offers real-time configuration of avatar appearance, behavior,
language, and voice via an intuitive control panel. In contrast to prior works
that rely on cloud-based LLM services, SRWToolkit emphasizes modularity and
ensures on-device functionality through local LLM inference. In our small-scale
user study ($n=11$), participants created and interacted with diverse robotic
roles (hospital receptionist, mathematics teacher, and driving assistant),
which demonstrated positive outcomes in the toolkit's usability, trust, and
user experience. The toolkit enables rapid and efficient development of robot
characters customized to researchers' needs, supporting scalable research in
human-robot interaction.

**Relevance Score:** 3.00

---

## 143. MuMTAffect: A Multimodal Multitask Affective Framework for Personality   and Emotion Recognition from Physiological Signals

**Authors:** Meisam Jamshidi Seikavandi, Fabricio Batista Narcizo, Ted Vucurevich, Andrew Burke Dittberner, Paolo Burelli

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04254v1) | [PDF](http://arxiv.org/pdf/2509.04254v1)

**Abstract:** We present MuMTAffect, a novel Multimodal Multitask Affective Embedding
Network designed for joint emotion classification and personality prediction
(re-identification) from short physiological signal segments. MuMTAffect
integrates multiple physiological modalities pupil dilation, eye gaze, facial
action units, and galvanic skin response using dedicated, transformer-based
encoders for each modality and a fusion transformer to model cross-modal
interactions. Inspired by the Theory of Constructed Emotion, the architecture
explicitly separates core affect encoding (valence/arousal) from higher-level
conceptualization, thereby grounding predictions in contemporary affective
neuroscience. Personality trait prediction is leveraged as an auxiliary task to
generate robust, user-specific affective embeddings, significantly enhancing
emotion recognition performance. We evaluate MuMTAffect on the AFFEC dataset,
demonstrating that stimulus-level emotional cues (Stim Emo) and galvanic skin
response substantially improve arousal classification, while pupil and gaze
data enhance valence discrimination. The inherent modularity of MuMTAffect
allows effortless integration of additional modalities, ensuring scalability
and adaptability. Extensive experiments and ablation studies underscore the
efficacy of our multimodal multitask approach in creating personalized,
context-aware affective computing systems, highlighting pathways for further
advancements in cross-subject generalisation.

**Relevance Score:** 3.00

---

## 144. POET: Supporting Prompting Creativity and Personalization with Automated   Expansion of Text-to-Image Generation

**Authors:** Evans Xu Han, Alice Qian Zhang, Haiyi Zhu, Hong Shen, Paul Pu Liang, Jane Hsieh

**Categories:** cs.CV, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2504.13392v2) | [PDF](http://arxiv.org/pdf/2504.13392v2)

**Abstract:** State-of-the-art visual generative AI tools hold immense potential to assist
users in the early ideation stages of creative tasks -- offering the ability to
generate (rather than search for) novel and unprecedented (instead of existing)
images of considerable quality that also adhere to boundless combinations of
user specifications. However, many large-scale text-to-image systems are
designed for broad applicability, yielding conventional output that may limit
creative exploration. They also employ interaction methods that may be
difficult for beginners. Given that creative end users often operate in
diverse, context-specific ways that are often unpredictable, more variation and
personalization are necessary. We introduce POET, a real-time interactive tool
that (1) automatically discovers dimensions of homogeneity in text-to-image
generative models, (2) expands these dimensions to diversify the output space
of generated images, and (3) learns from user feedback to personalize
expansions. An evaluation with 28 users spanning four creative task domains
demonstrated POET's ability to generate results with higher perceived diversity
and help users reach satisfaction in fewer prompts during creative tasks,
thereby prompting them to deliberate and reflect more on a wider range of
possible produced results during the co-creative process. Focusing on visual
creativity, POET offers a first glimpse of how interaction techniques of future
text-to-image generation tools may support and align with more pluralistic
values and the needs of end users during the ideation stages of their work.

**Relevance Score:** 3.00

---

## 145. OneSearch: A Preliminary Exploration of the Unified End-to-End   Generative Framework for E-commerce Search

**Authors:** Ben Chen, Xian Guo, Siyuan Wang, Zihan Liang, Yue Lv, Yufei Ma, Xinlong Xiao, Bowen Xue, Xuxin Zhang, Ying Yang, Huangyu Dai, Xing Xu, Tong Zhao, Mingcan Peng, Xiaoyang Zheng, Chao Wang, Qihang Zhao, Zhixin Zhai, Yang Zhao, Bochao Liu, Jingshan Lv, Jing Chen, Xiao Liang, Yuqing Ding, Chenyi Lei, Wenwu Ou, Han Li, Kun Gai

**Categories:** cs.IR

**Links:** [arXiv](http://arxiv.org/abs/2509.03236v2) | [PDF](http://arxiv.org/pdf/2509.03236v2)

**Abstract:** Traditional e-commerce search systems employ multi-stage cascading
architectures (MCA) that progressively filter items through recall,
pre-ranking, and ranking stages. While effective at balancing computational
efficiency with business conversion, these systems suffer from fragmented
computation and optimization objective collisions across stages, which
ultimately limit their performance ceiling. To address these, we propose
\textbf{OneSearch}, the first industrial-deployed end-to-end generative
framework for e-commerce search. This framework introduces three key
innovations: (1) a Keyword-enhanced Hierarchical Quantization Encoding (KHQE)
module, to preserve both hierarchical semantics and distinctive item attributes
while maintaining strong query-item relevance constraints; (2) a multi-view
user behavior sequence injection strategy that constructs behavior-driven user
IDs and incorporates both explicit short-term and implicit long-term sequences
to model user preferences comprehensively; and (3) a Preference-Aware Reward
System (PARS) featuring multi-stage supervised fine-tuning and adaptive
reward-weighted ranking to capture fine-grained user preferences. Extensive
offline evaluations on large-scale industry datasets demonstrate OneSearch's
superior performance for high-quality recall and ranking. The rigorous online
A/B tests confirm its ability to enhance relevance in the same exposure
position, achieving statistically significant improvements: +1.67\% item CTR,
+2.40\% buyer, and +3.22\% order volume. Furthermore, OneSearch reduces
operational expenditure by 75.40\% and improves Model FLOPs Utilization from
3.26\% to 27.32\%. The system has been successfully deployed across multiple
search scenarios in Kuaishou, serving millions of users, generating tens of
millions of PVs daily.

**Relevance Score:** 3.00

---

## 146. Demographic-aware fine-grained classification of pediatric wrist   fractures

**Authors:** Ammar Ahmed, Ali Shariq Imran, Zenun Kastrati, Sher Muhammad Daudpota

**Categories:** cs.AI, cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2507.12964v5) | [PDF](http://arxiv.org/pdf/2507.12964v5)

**Abstract:** Wrist pathologies are frequently observed, particularly among children who
constitute the majority of fracture cases. Computer vision presents a promising
avenue, contingent upon the availability of extensive datasets, a notable
challenge in medical imaging. Therefore, reliance solely on one modality, such
as images, proves inadequate, especially in an era of diverse and plentiful
data types. This study addresses the problem using a multifaceted approach:
framing it as a fine-grained recognition task, fusing patient metadata with
X-rays, and leveraging weights from a separate fine-grained dataset rather than
from a coarse-grained dataset like ImageNet. Unlike prior work, this is the
first application of metadata integration for wrist pathology recognition. Our
results show that combining fine-grained transformer approach, fine-grained
pre-training, and metadata integration improves diagnostic accuracy by 2% on
small custom curated dataset and over 10% on a larger fracture dataset.

**Relevance Score:** 2.50

---

## 147. SSGaussian: Semantic-Aware and Structure-Preserving 3D Style Transfer

**Authors:** Jimin Xu, Bosheng Qin, Tao Jin, Zhou Zhao, Zhenhui Ye, Jun Yu, Fei Wu

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04379v1) | [PDF](http://arxiv.org/pdf/2509.04379v1)

**Abstract:** Recent advancements in neural representations, such as Neural Radiance Fields
and 3D Gaussian Splatting, have increased interest in applying style transfer
to 3D scenes. While existing methods can transfer style patterns onto
3D-consistent neural representations, they struggle to effectively extract and
transfer high-level style semantics from the reference style image.
Additionally, the stylized results often lack structural clarity and
separation, making it difficult to distinguish between different instances or
objects within the 3D scene. To address these limitations, we propose a novel
3D style transfer pipeline that effectively integrates prior knowledge from
pretrained 2D diffusion models. Our pipeline consists of two key stages: First,
we leverage diffusion priors to generate stylized renderings of key viewpoints.
Then, we transfer the stylized key views onto the 3D representation. This
process incorporates two innovative designs. The first is cross-view style
alignment, which inserts cross-view attention into the last upsampling block of
the UNet, allowing feature interactions across multiple key views. This ensures
that the diffusion model generates stylized key views that maintain both style
fidelity and instance-level consistency. The second is instance-level style
transfer, which effectively leverages instance-level consistency across
stylized key views and transfers it onto the 3D representation. This results in
a more structured, visually coherent, and artistically enriched stylization.
Extensive qualitative and quantitative experiments demonstrate that our 3D
style transfer pipeline significantly outperforms state-of-the-art methods
across a wide range of scenes, from forward-facing to challenging 360-degree
environments. Visit our project page https://jm-xu.github.io/SSGaussian for
immersive visualization.

**Relevance Score:** 2.50

---

## 148. Pilot Study on Generative AI and Critical Thinking in Higher Education   Classrooms

**Authors:** W. F. Lamberti, S. R. Lawrence, D. White, S. Kim, S. Abdullah

**Categories:** stat.AP, cs.HC, cs.AI, cs.CY

**Links:** [arXiv](http://arxiv.org/abs/2509.00167v2) | [PDF](http://arxiv.org/pdf/2509.00167v2)

**Abstract:** Generative AI (GAI) tools have seen rapid adoption in educational settings,
yet their role in fostering critical thinking remains underexplored. While
previous studies have examined GAI as a tutor for specific lessons or as a tool
for completing assignments, few have addressed how students critically evaluate
the accuracy and appropriateness of GAI-generated responses. This pilot study
investigates students' ability to apply structured critical thinking when
assessing Generative AI outputs in introductory Computational and Data Science
courses. Given that GAI tools often produce contextually flawed or factually
incorrect answers, we designed learning activities that require students to
analyze, critique, and revise AI-generated solutions. Our findings offer
initial insights into students' ability to engage critically with GAI content
and lay the groundwork for more comprehensive studies in future semesters.

**Relevance Score:** 2.50

---

## 149. Reinforcement Learning for Robust Ageing-Aware Control of Li-ion Battery   Systems with Data-Driven Formal Verification

**Authors:** Rudi Coppola, Hovsep Touloujian, Pierfrancesco Ombrini, Manuel Mazo Jr

**Categories:** eess.SY, cs.AI, cs.SY

**Links:** [arXiv](http://arxiv.org/abs/2509.04288v1) | [PDF](http://arxiv.org/pdf/2509.04288v1)

**Abstract:** Rechargeable lithium-ion (Li-ion) batteries are a ubiquitous element of
modern technology. In the last decades, the production and design of such
batteries and their adjacent embedded charging and safety protocols, denoted by
Battery Management Systems (BMS), has taken central stage. A fundamental
challenge to be addressed is the trade-off between the speed of charging and
the ageing behavior, resulting in the loss of capacity in the battery cell. We
rely on a high-fidelity physics-based battery model and propose an approach to
data-driven charging and safety protocol design. Following a
Counterexample-Guided Inductive Synthesis scheme, we combine Reinforcement
Learning (RL) with recent developments in data-driven formal methods to obtain
a hybrid control strategy: RL is used to synthesise the individual controllers,
and a data-driven abstraction guides their partitioning into a switched
structure, depending on the initial output measurements of the battery. The
resulting discrete selection among RL-based controllers, coupled with the
continuous battery dynamics, realises a hybrid system. When a design meets the
desired criteria, the abstraction provides probabilistic guarantees on the
closed-loop performance of the cell.

**Relevance Score:** 2.50

---

## 150. Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE   Solvers

**Authors:** Hang Zhou, Yuezhou Ma, Haixu Wu, Haowen Wang, Mingsheng Long

**Categories:** cs.NA, math.NA, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2405.17527v5) | [PDF](http://arxiv.org/pdf/2405.17527v5)

**Abstract:** Deep models have recently emerged as promising tools to solve partial
differential equations (PDEs), known as neural PDE solvers. While neural
solvers trained from either simulation data or physics-informed loss can solve
PDEs reasonably well, they are mainly restricted to a few instances of PDEs,
e.g. a certain equation with a limited set of coefficients. This limits their
generalization to diverse PDEs, preventing them from being practical surrogate
models of numerical solvers. In this paper, we present Unisolver, a novel
Transformer model trained on diverse data and conditioned on diverse PDEs,
aiming towards a universal neural PDE solver capable of solving a wide scope of
PDEs. Instead of purely scaling up data and parameters, Unisolver stems from
the theoretical analysis of the PDE-solving process. Inspired by the
mathematical structure of PDEs that a PDE solution is fundamentally governed by
a series of PDE components such as equation symbols and boundary conditions, we
define a complete set of PDE components and flexibly embed them as domain-wise
and point-wise deep conditions for Transformer PDE solvers. Integrating
physical insights with recent Transformer advances, Unisolver achieves
consistent state-of-the-art on three challenging large-scale benchmarks,
showing impressive performance and generalizability. Code is available at
https://github.com/thuml/Unisolver.

**Relevance Score:** 2.50

---

## 151. An Empirical Study of Vulnerabilities in Python Packages and Their   Detection

**Authors:** Haowei Quan, Junjie Wang, Xinzhe Li, Terry Yue Zhuo, Xiao Chen, Xiaoning Du

**Categories:** cs.SE, cs.AI, cs.CR

**Links:** [arXiv](http://arxiv.org/abs/2509.04260v1) | [PDF](http://arxiv.org/pdf/2509.04260v1)

**Abstract:** In the rapidly evolving software development landscape, Python stands out for
its simplicity, versatility, and extensive ecosystem. Python packages, as units
of organization, reusability, and distribution, have become a pressing concern,
highlighted by the considerable number of vulnerability reports. As a scripting
language, Python often cooperates with other languages for performance or
interoperability. This adds complexity to the vulnerabilities inherent to
Python packages, and the effectiveness of current vulnerability detection tools
remains underexplored. This paper addresses these gaps by introducing PyVul,
the first comprehensive benchmark suite of Python-package vulnerabilities.
PyVul includes 1,157 publicly reported, developer-verified vulnerabilities,
each linked to its affected packages. To accommodate diverse detection
techniques, it provides annotations at both commit and function levels. An
LLM-assisted data cleansing method is incorporated to improve label accuracy,
achieving 100% commit-level and 94% function-level accuracy, establishing PyVul
as the most precise large-scale Python vulnerability benchmark. We further
carry out a distribution analysis of PyVul, which demonstrates that
vulnerabilities in Python packages involve multiple programming languages and
exhibit a wide variety of types. Moreover, our analysis reveals that
multi-lingual Python packages are potentially more susceptible to
vulnerabilities. Evaluation of state-of-the-art detectors using this benchmark
reveals a significant discrepancy between the capabilities of existing tools
and the demands of effectively identifying real-world security issues in Python
packages. Additionally, we conduct an empirical review of the top-ranked CWEs
observed in Python packages, to diagnose the fine-grained limitations of
current detection tools and highlight the necessity for future advancements in
the field.

**Relevance Score:** 2.50

---

## 152. A Survey of Graph Retrieval-Augmented Generation for Customized Large   Language Models

**Authors:** Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Hao Chen, Yilin Xiao, Chuang Zhou, Yi Chang, Xiao Huang

**Categories:** cs.CL, cs.IR, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2501.13958v2) | [PDF](http://arxiv.org/pdf/2501.13958v2)

**Abstract:** Large language models (LLMs) have demonstrated remarkable capabilities in a
wide range of tasks, yet their application to specialized domains remains
challenging due to the need for deep expertise. Retrieval-Augmented generation
(RAG) has emerged as a promising solution to customize LLMs for professional
fields by seamlessly integrating external knowledge bases, enabling real-time
access to domain-specific expertise during inference. Despite its potential,
traditional RAG systems, based on flat text retrieval, face three critical
challenges: (i) complex query understanding in professional contexts, (ii)
difficulties in knowledge integration across distributed sources, and (iii)
system efficiency bottlenecks at scale. This survey presents a systematic
analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new
paradigm that revolutionizes domain-specific LLM applications. GraphRAG
addresses traditional RAG limitations through three key innovations: (i)
graph-structured knowledge representation that explicitly captures entity
relationships and domain hierarchies, (ii) efficient graph-based retrieval
techniques that enable context-preserving knowledge retrieval with multihop
reasoning ability, and (iii) structure-aware knowledge integration algorithms
that leverage retrieved knowledge for accurate and logical coherent generation
of LLMs. In this survey, we systematically analyze the technical foundations of
GraphRAG and examine current implementations across various professional
domains, identifying key technical challenges and promising research
directions. All the related resources of GraphRAG, including research papers,
open-source data, and projects, are collected for the community in
https://github.com/DEEP-PolyU/Awesome-GraphRAG.

**Relevance Score:** 2.50

---

## 153. How many patients could we save with LLM priors?

**Authors:** Shota Arai, David Selby, Andrew Vargo, Sebastian Vollmer

**Categories:** stat.AP, cs.ET, stat.ME, cs.IR, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04250v1) | [PDF](http://arxiv.org/pdf/2509.04250v1)

**Abstract:** Imagine a world where clinical trials need far fewer patients to achieve the
same statistical power, thanks to the knowledge encoded in large language
models (LLMs). We present a novel framework for hierarchical Bayesian modeling
of adverse events in multi-center clinical trials, leveraging LLM-informed
prior distributions. Unlike data augmentation approaches that generate
synthetic data points, our methodology directly obtains parametric priors from
the model. Our approach systematically elicits informative priors for
hyperparameters in hierarchical Bayesian models using a pre-trained LLM,
enabling the incorporation of external clinical expertise directly into
Bayesian safety modeling. Through comprehensive temperature sensitivity
analysis and rigorous cross-validation on real-world clinical trial data, we
demonstrate that LLM-derived priors consistently improve predictive performance
compared to traditional meta-analytical approaches. This methodology paves the
way for more efficient and expert-informed clinical trial design, enabling
substantial reductions in the number of patients required to achieve robust
safety assessment and with the potential to transform drug safety monitoring
and regulatory decision making.

**Comment:** 9 pages, 4 figures

**Relevance Score:** 2.50

---

## 154. FFHFlow: Diverse and Uncertainty-Aware Dexterous Grasp Generation via   Flow Variational Inference

**Authors:** Qian Feng, Jianxiang Feng, Zhaopeng Chen, Rudolph Triebel, Alois Knoll

**Categories:** cs.AI, cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2407.15161v4) | [PDF](http://arxiv.org/pdf/2407.15161v4)

**Abstract:** Synthesizing diverse, uncertainty-aware grasps for multi-fingered hands from
partial observations remains a critical challenge in robot learning. Prior
generative methods struggle to model the intricate grasp distribution of
dexterous hands and often fail to reason about shape uncertainty inherent in
partial point clouds, leading to unreliable or overly conservative grasps. We
propose FFHFlow, a flow-based variational framework that generates diverse,
robust multi-finger grasps while explicitly quantifying perceptual uncertainty
in the partial point clouds. Our approach leverages a normalizing flow-based
deep latent variable model to learn a hierarchical grasp manifold, overcoming
the mode collapse and rigid prior limitations of conditional Variational
Autoencoders (cVAEs). By exploiting the invertibility and exact likelihoods of
flows, FFHFlow introspects shape uncertainty in partial observations and
identifies novel object structures, enabling risk-aware grasp synthesis. To
further enhance reliability, we integrate a discriminative grasp evaluator with
the flow likelihoods, formulating an uncertainty-aware ranking strategy that
prioritizes grasps robust to shape ambiguity. Extensive experiments in
simulation and real-world setups demonstrate that FFHFlow outperforms
state-of-the-art baselines (including diffusion models) in grasp diversity and
success rate, while achieving run-time efficient sampling. We also showcase its
practical value in cluttered and confined environments, where diversity-driven
sampling excels by mitigating collisions (Project Page:
https://sites.google.com/view/ffhflow/home/).

**Comment:** First two authors contributed equally, whose ordering decided via
  coin-tossing. Accepted for CoRL 2025

**Relevance Score:** 2.50

---

## 155. Kolb-Based Experiential Learning for Generalist Agents with Human-Level   Kaggle Data Science Performance

**Authors:** Antoine Grosnit, Alexandre Maraval, Refinath S N, Zichao Zhao, James Dora, Giuseppe Paolo, Albert Thomas, Jonas Gonzalez, Abhineet Kumar, Khyati Khandelwal, Abdelhakim Benechehab, Hamza Cherkaoui, Youssef Attia El-Hili, Kun Shao, Jianye Hao, Jun Yao, Balázs Kégl, Jun Wang

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.03562v2) | [PDF](http://arxiv.org/pdf/2411.03562v2)

**Abstract:** Human expertise emerges through iterative cycles of interaction, reflection,
and internal model updating, which are central to cognitive theories such as
Kolb's experiential learning and Vygotsky's zone of proximal development. In
contrast, current AI systems, particularly LLM agents, rely on static
pre-training or rigid workflows, lacking mechanisms for continual adaptation.
Recent studies identified early cognitive traits in LLM agents (reflection,
revision, and self-correction) suggesting foundational elements of human-like
experiential learning. Thus the key question: Can we design LLM agents capable
of structured, cognitively grounded learning similar to human processes? In
response, we propose a computational framework of Kolb's learning cycle with
Vygotsky's ZPD for autonomous agents. Our architecture separates extrinsic
(environment interaction) and intrinsic (internal reflection/abstraction)
functions, enabling cognitively grounded scaffolded learning, where the agent
initially learns within structured environments, followed by open-ended
generalisation. This approach empowers agents to master complex tasks ; domains
that traditional fine-tuning or simple reflective methods could not tackle
effectively. Its potential is powerfully demonstrated via direct comparison
with humans in real-world Kaggle data science competitions. Learning fully
automated data science code generation across 81 tasks, our system, Agent K,
demonstrated the ability to perform the entire workflow autonomously, achieving
an Elo-MMR score of 1694, beyond median score of the Kaggle Masters (the top 2%
among 200,000 users) of our study. With 9 gold, 8 silver, and 12 bronze medals
level performance - including 4 gold and 4 silver on prize-awarding
competitions - Agent K is the 1st AI system to successfully integrate Kolb- and
Vygotsky-inspired human cognitive learning, marking a major step toward
generalist AI.

**Relevance Score:** 2.50

---

## 156. StreetViewAI: Making Street View Accessible Using Context-Aware   Multimodal AI

**Authors:** Jon E. Froehlich, Alexander Fiannaca, Nimer Jaber, Victor Tsaran, Shaun Kane

**Categories:** cs.AI, H.5; I.2, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2508.08524v3) | [PDF](http://arxiv.org/pdf/2508.08524v3)

**Abstract:** Interactive streetscape mapping tools such as Google Street View (GSV) and
Meta Mapillary enable users to virtually navigate and experience real-world
environments via immersive 360{\deg} imagery but remain fundamentally
inaccessible to blind users. We introduce StreetViewAI, the first-ever
accessible street view tool, which combines context-aware, multimodal AI,
accessible navigation controls, and conversational speech. With StreetViewAI,
blind users can virtually examine destinations, engage in open-world
exploration, or virtually tour any of the over 220 billion images and 100+
countries where GSV is deployed. We iteratively designed StreetViewAI with a
mixed-visual ability team and performed an evaluation with eleven blind users.
Our findings demonstrate the value of an accessible street view in supporting
POI investigations and remote route planning. We close by enumerating key
guidelines for future work.

**Comment:** Accepted to UIST'25 v2. Fixed a missing word in the PDF v3. Fixed a
  typo in an author's name

**Relevance Score:** 2.50

---

## 157. LOTS of Fashion! Multi-Conditioning for Image Generation via Sketch-Text   Pairing

**Authors:** Federico Girella, Davide Talon, Ziyue Liu, Zanxi Ruan, Yiming Wang, Marco Cristani

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2507.22627v2) | [PDF](http://arxiv.org/pdf/2507.22627v2)

**Abstract:** Fashion design is a complex creative process that blends visual and textual
expressions. Designers convey ideas through sketches, which define spatial
structure and design elements, and textual descriptions, capturing material,
texture, and stylistic details. In this paper, we present LOcalized Text and
Sketch for fashion image generation (LOTS), an approach for compositional
sketch-text based generation of complete fashion outlooks. LOTS leverages a
global description with paired localized sketch + text information for
conditioning and introduces a novel step-based merging strategy for diffusion
adaptation. First, a Modularized Pair-Centric representation encodes sketches
and text into a shared latent space while preserving independent localized
features; then, a Diffusion Pair Guidance phase integrates both local and
global conditioning via attention-based guidance within the diffusion model's
multi-step denoising process. To validate our method, we build on Fashionpedia
to release Sketchy, the first fashion dataset where multiple text-sketch pairs
are provided per image. Quantitative results show LOTS achieves
state-of-the-art image generation performance on both global and localized
metrics, while qualitative examples and a human evaluation study highlight its
unprecedented level of design customization.

**Comment:** Accepted at ICCV25 (Oral). Project page:
  https://intelligolabs.github.io/lots/

**Relevance Score:** 2.50

---

## 158. Attention as an Adaptive Filter

**Authors:** Peter Racioppo

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04154v1) | [PDF](http://arxiv.org/pdf/2509.04154v1)

**Abstract:** We introduce Adaptive Filter Attention (AFA), a novel attention mechanism
that incorporates a learnable dynamics model directly into the computation of
attention weights. Rather than comparing queries and keys directly, we model
the input sequence as discrete observations of a linear stochastic differential
equation (SDE). By imposing a linear dynamics model with simultaneously
diagonalizable state matrices and noise covariances, we can make use of a
closed-form solution to the differential Lyapunov equation to efficiently
propagate pairwise uncertainties through the dynamics. Attention naturally
arises as the maximum likelihood solution for this linear SDE, with attention
weights corresponding to robust residual-based reweightings of the propagated
pairwise precisions. Imposing an additional constraint on the state matrix's
eigenvalues leads to a simplified variant with the same computational and
memory complexity as standard attention. In the limit of vanishing dynamics and
process noise, and using a small-angle approximation, we recover ordinary
dot-product attention.

**Relevance Score:** 2.50

---

## 159. Quantifying Calibration Error in Neural Networks Through Evidence-Based   Theory

**Authors:** Koffi Ismael Ouattara, Ioannis Krontiris, Theo Dimitrakos, Frank Kargl

**Categories:** math.LO, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.00265v3) | [PDF](http://arxiv.org/pdf/2411.00265v3)

**Abstract:** Trustworthiness in neural networks is crucial for their deployment in
critical applications, where reliability, confidence, and uncertainty play
pivotal roles in decision-making. Traditional performance metrics such as
accuracy and precision fail to capture these aspects, particularly in cases
where models exhibit overconfidence. To address these limitations, this paper
introduces a novel framework for quantifying the trustworthiness of neural
networks by incorporating subjective logic into the evaluation of Expected
Calibration Error (ECE). This method provides a comprehensive measure of trust,
disbelief, and uncertainty by clustering predicted probabilities and fusing
opinions using appropriate fusion operators. We demonstrate the effectiveness
of this approach through experiments on MNIST and CIFAR-10 datasets, where
post-calibration results indicate improved trustworthiness. The proposed
framework offers a more interpretable and nuanced assessment of AI models, with
potential applications in sensitive domains such as healthcare and autonomous
systems.

**Comment:** This is the preprint of the paper accepted to Fusion 2025 (28th
  International Conference on Information Fusion, Rio de Janeiro, Brazil, July
  7-10, 2025). The published version is available at
  https://doi.org/10.23919/FUSION65864.2025.11124121

**Relevance Score:** 2.50

---

## 160. Hybrid Reinforcement Learning and Search for Flight Trajectory Planning

**Authors:** Alberto Luise, Michele Lombardi, Florent Teichteil Koenigsbuch

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04100v1) | [PDF](http://arxiv.org/pdf/2509.04100v1)

**Abstract:** This paper explores the combination of Reinforcement Learning (RL) and
search-based path planners to speed up the optimization of flight paths for
airliners, where in case of emergency a fast route re-calculation can be
crucial. The fundamental idea is to train an RL Agent to pre-compute
near-optimal paths based on location and atmospheric data and use those at
runtime to constrain the underlying path planning solver and find a solution
within a certain distance from the initial guess. The approach effectively
reduces the size of the solver's search space, significantly speeding up route
optimization. Although global optimality is not guaranteed, empirical results
conducted with Airbus aircraft's performance models show that fuel consumption
remains nearly identical to that of an unconstrained solver, with deviations
typically within 1%. At the same time, computation speed can be improved by up
to 50% as compared to using a conventional solver alone.

**Relevance Score:** 2.50

---

## 161. PIN: A Knowledge-Intensive Dataset for Paired and Interleaved Multimodal   Documents

**Authors:** Junjie Wang, Yuxiang Zhang, Minghao Liu, Yin Zhang, Yatai Ji, Weihao Xuan, Nie Lin, Kang Zhu, Zhiqiang Lin, Yiming Ren, Chunyang Jiang, Yiyao Yu, Zekun Wang, Tiezhen Wang, Wenhao Huang, Jie Fu, Qunshu Liu, Yujiu Yang, Ge Zhang, Ruibin Yuan, Bei Chen, Wenhu Chen

**Categories:** cs.AI, cs.MM, cs.CL, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2406.13923v2) | [PDF](http://arxiv.org/pdf/2406.13923v2)

**Abstract:** Recent advancements in large multimodal models (LMMs) have leveraged
extensive multimodal datasets to enhance capabilities in complex
knowledge-driven tasks. However, persistent challenges in perceptual and
reasoning errors limit their efficacy, particularly in interpreting intricate
visual data and deducing multimodal relationships. To address these issues, we
introduce PIN (Paired and INterleaved multimodal documents), a novel data
format designed to foster a deeper integration of visual and textual knowledge.
The PIN format uniquely combines semantically rich Markdown files, which
preserve fine-grained textual structures, with holistic overall images that
capture the complete document layout. Following this format, we construct and
release two large-scale, open-source datasets: PIN-200M (~200 million
documents) and PIN-14M (~14 million), compiled from diverse web and scientific
sources in both English and Chinese. To maximize usability, we provide detailed
statistical analyses and equip the datasets with quality signals, enabling
researchers to easily filter and select data for specific tasks. Our work
provides the community with a versatile data format and substantial resources,
offering a foundation for new research in pre-training strategies and the
development of more powerful knowledge-intensive LMMs.

**Comment:** Technical report v1.0

**Relevance Score:** 2.50

---

## 162. Conditional Video Generation for High-Efficiency Video Compression

**Authors:** Fangqiu Yi, Jingyu Xu, Jiawei Shao, Chi Zhang, Xuelong Li

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2507.15269v2) | [PDF](http://arxiv.org/pdf/2507.15269v2)

**Abstract:** Perceptual studies demonstrate that conditional diffusion models excel at
reconstructing video content aligned with human visual perception. Building on
this insight, we propose a video compression framework that leverages
conditional diffusion models for perceptually optimized reconstruction.
Specifically, we reframe video compression as a conditional generation task,
where a generative model synthesizes video from sparse, yet informative
signals. Our approach introduces three key modules: (1) Multi-granular
conditioning that captures both static scene structure and dynamic
spatio-temporal cues; (2) Compact representations designed for efficient
transmission without sacrificing semantic richness; (3) Multi-condition
training with modality dropout and role-aware embeddings, which prevent
over-reliance on any single modality and enhance robustness. Extensive
experiments show that our method significantly outperforms both traditional and
neural codecs on perceptual quality metrics such as Fr\'echet Video Distance
(FVD) and LPIPS, especially under high compression ratios.

**Comment:** Critical methodology flaws invalidate key results

**Relevance Score:** 2.50

---

## 163. Unveiling the Role of Data Uncertainty in Tabular Deep Learning

**Authors:** Nikolay Kartashev, Ivan Rubachev, Artem Babenko

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04430v1) | [PDF](http://arxiv.org/pdf/2509.04430v1)

**Abstract:** Recent advancements in tabular deep learning have demonstrated exceptional
practical performance, yet the field often lacks a clear understanding of why
these techniques actually succeed. To address this gap, our paper highlights
the importance of the concept of data uncertainty for explaining the
effectiveness of the recent tabular DL methods. In particular, we reveal that
the success of many beneficial design choices in tabular DL, such as numerical
feature embeddings, retrieval-augmented models and advanced ensembling
strategies, can be largely attributed to their implicit mechanisms for managing
high data uncertainty. By dissecting these mechanisms, we provide a unifying
understanding of the recent performance improvements. Furthermore, the insights
derived from this data-uncertainty perspective directly allowed us to develop
more effective numerical feature embeddings as an immediate practical outcome
of our analysis. Overall, our work paves the way to foundational understanding
of the benefits introduced by modern tabular methods that results in the
concrete advancements of existing techniques and outlines future research
directions for tabular DL.

**Relevance Score:** 2.50

---

## 164. Rethinking Layer-wise Gaussian Noise Injection: Bridging Implicit   Objectives and Privacy Budget Allocation

**Authors:** Qifeng Tan, Shusen Yang, Xuebin Ren, Yikai Zhang

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04232v1) | [PDF](http://arxiv.org/pdf/2509.04232v1)

**Abstract:** Layer-wise Gaussian mechanisms (LGM) enhance flexibility in differentially
private deep learning by injecting noise into partitioned gradient vectors.
However, existing methods often rely on heuristic noise allocation strategies,
lacking a rigorous understanding of their theoretical grounding in connecting
noise allocation to formal privacy-utility tradeoffs. In this paper, we present
a unified analytical framework that systematically connects layer-wise noise
injection strategies with their implicit optimization objectives and associated
privacy budget allocations. Our analysis reveals that several existing
approaches optimize ill-posed objectives -- either ignoring inter-layer
signal-to-noise ratio (SNR) consistency or leading to inefficient use of the
privacy budget. In response, we propose a SNR-Consistent noise allocation
strategy that unifies both aspects, yielding a noise allocation scheme that
achieves better signal preservation and more efficient privacy budget
utilization. Extensive experiments in both centralized and federated learning
settings demonstrate that our method consistently outperforms existing
allocation strategies, achieving better privacy-utility tradeoffs. Our
framework not only offers diagnostic insights into prior methods but also
provides theoretical guidance for designing adaptive and effective noise
injection schemes in deep models.

**Relevance Score:** 2.50

---

## 165. Set Block Decoding is a Language Model Inference Accelerator

**Authors:** Itai Gat, Heli Ben-Hamu, Marton Havasi, Daniel Haziza, Jeremy Reizenstein, Gabriel Synnaeve, David Lopez-Paz, Brian Karrer, Yaron Lipman

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04185v1) | [PDF](http://arxiv.org/pdf/2509.04185v1)

**Abstract:** Autoregressive next token prediction language models offer powerful
capabilities but face significant challenges in practical deployment due to the
high computational and memory costs of inference, particularly during the
decoding stage. We introduce Set Block Decoding (SBD), a simple and flexible
paradigm that accelerates generation by integrating standard next token
prediction (NTP) and masked token prediction (MATP) within a single
architecture. SBD allows the model to sample multiple, not necessarily
consecutive, future tokens in parallel, a key distinction from previous
acceleration methods. This flexibility allows the use of advanced solvers from
the discrete diffusion literature, offering significant speedups without
sacrificing accuracy. SBD requires no architectural changes or extra training
hyperparameters, maintains compatibility with exact KV-caching, and can be
implemented by fine-tuning existing next token prediction models. By
fine-tuning Llama-3.1 8B and Qwen-3 8B, we demonstrate that SBD enables a 3-5x
reduction in the number of forward passes required for generation while
achieving same performance as equivalent NTP training.

**Relevance Score:** 2.50

---

## 166. Comment on "A Note on Over-Smoothing for Graph Neural Networks"

**Authors:** Razi Hasson, Reuven Guetta

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04178v1) | [PDF](http://arxiv.org/pdf/2509.04178v1)

**Abstract:** We comment on Cai and Wang (2020, arXiv:2006.13318), who analyze
over-smoothing in GNNs via Dirichlet energy. We show that under mild spectral
conditions (including with Leaky-ReLU), the Dirichlet energy of node embeddings
decreases exponentially with depth; we further extend the result to spectral
polynomial filters and provide a short proof for the Leaky-ReLU case.
Experiments on edge deletion and weight amplification illustrate when Dirichlet
energy increases, hinting at practical ways to relieve over-smoothing.

**Comment:** Comment on arXiv:2006.13318 (Cai & Wang, 2020). Revisits their
  Dirichlet-energy analysis of over-smoothing and extends it to Leaky-ReLU and
  spectral polynomial filters; includes Proposition 7.1 and a new proof of
  Lemma 3.3 for Leaky-ReLU. 7 pages

**Relevance Score:** 2.50

---

## 167. Probabilistic QoS Metric Forecasting in Delay-Tolerant Networks Using   Conditional Diffusion Models on Latent Dynamics

**Authors:** Jianhua Liu, Zheng Liu, Yu Xiang, Yanwen Qu

**Categories:** stat.ML, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2504.08821v2) | [PDF](http://arxiv.org/pdf/2504.08821v2)

**Abstract:** Active QoS metric prediction, commonly employed in the maintenance and
operation of DTN, could enhance network performance regarding latency,
throughput, energy consumption, and dependability. Naturally formulated as a
multivariate time series forecasting problem, it attracts substantial research
efforts. Traditional mean regression methods for time series forecasting cannot
capture the data complexity adequately, resulting in deteriorated performance
in operational tasks in DTNs such as routing. This paper formulates the
prediction of QoS metrics in DTN as a probabilistic forecasting problem on
multivariate time series, where one could quantify the uncertainty of forecasts
by characterizing the distribution of these samples. The proposed approach
hires diffusion models and incorporates the latent temporal dynamics of
non-stationary and multi-mode data into them. Extensive experiments demonstrate
the efficacy of the proposed approach by showing that it outperforms the
popular probabilistic time series forecasting methods.

**Relevance Score:** 2.50

---

## 168. One Flight Over the Gap: A Survey from Perspective to Panoramic Vision

**Authors:** Xin Lin, Xian Ge, Dizhe Zhang, Zhaoliang Wan, Xianshun Wang, Xiangtai Li, Wenjie Jiang, Bo Du, Dacheng Tao, Ming-Hsuan Yang, Lu Qi

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04444v1) | [PDF](http://arxiv.org/pdf/2509.04444v1)

**Abstract:** Driven by the demand for spatial intelligence and holistic scene perception,
omnidirectional images (ODIs), which provide a complete 360\textdegree{} field
of view, are receiving growing attention across diverse applications such as
virtual reality, autonomous driving, and embodied robotics. Despite their
unique characteristics, ODIs exhibit remarkable differences from perspective
images in geometric projection, spatial distribution, and boundary continuity,
making it challenging for direct domain adaption from perspective methods. This
survey reviews recent panoramic vision techniques with a particular emphasis on
the perspective-to-panorama adaptation. We first revisit the panoramic imaging
pipeline and projection methods to build the prior knowledge required for
analyzing the structural disparities. Then, we summarize three challenges of
domain adaptation: severe geometric distortions near the poles, non-uniform
sampling in Equirectangular Projection (ERP), and periodic boundary continuity.
Building on this, we cover 20+ representative tasks drawn from more than 300
research papers in two dimensions. On one hand, we present a cross-method
analysis of representative strategies for addressing panoramic specific
challenges across different tasks. On the other hand, we conduct a cross-task
comparison and classify panoramic vision into four major categories: visual
quality enhancement and assessment, visual understanding, multimodal
understanding, and visual generation. In addition, we discuss open challenges
and future directions in data, models, and applications that will drive the
advancement of panoramic vision research. We hope that our work can provide new
insight and forward looking perspectives to advance the development of
panoramic vision technologies. Our project page is
https://insta360-research-team.github.io/Survey-of-Panorama

**Relevance Score:** 2.50

---

## 169. Durian: Dual Reference-guided Portrait Animation with Attribute Transfer

**Authors:** Hyunsoo Cha, Byungjun Kim, Hanbyul Joo

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04434v1) | [PDF](http://arxiv.org/pdf/2509.04434v1)

**Abstract:** We present Durian, the first method for generating portrait animation videos
with facial attribute transfer from a given reference image to a target
portrait in a zero-shot manner. To enable high-fidelity and spatially
consistent attribute transfer across frames, we introduce dual reference
networks that inject spatial features from both the portrait and attribute
images into the denoising process of a diffusion model. We train the model
using a self-reconstruction formulation, where two frames are sampled from the
same portrait video: one is treated as the attribute reference and the other as
the target portrait, and the remaining frames are reconstructed conditioned on
these inputs and their corresponding masks. To support the transfer of
attributes with varying spatial extent, we propose a mask expansion strategy
using keypoint-conditioned image generation for training. In addition, we
further augment the attribute and portrait images with spatial and
appearance-level transformations to improve robustness to positional
misalignment between them. These strategies allow the model to effectively
generalize across diverse attributes and in-the-wild reference combinations,
despite being trained without explicit triplet supervision. Durian achieves
state-of-the-art performance on portrait animation with attribute transfer, and
notably, its dual reference design enables multi-attribute composition in a
single generation pass without additional training.

**Comment:** Project Page: https://hyunsoocha.github.io/durian

**Relevance Score:** 2.50

---

## 170. Few-step Flow for 3D Generation via Marginal-Data Transport Distillation

**Authors:** Zanwei Zhou, Taoran Yi, Jiemin Fang, Chen Yang, Lingxi Xie, Xinggang Wang, Wei Shen, Qi Tian

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04406v1) | [PDF](http://arxiv.org/pdf/2509.04406v1)

**Abstract:** Flow-based 3D generation models typically require dozens of sampling steps
during inference. Though few-step distillation methods, particularly
Consistency Models (CMs), have achieved substantial advancements in
accelerating 2D diffusion models, they remain under-explored for more complex
3D generation tasks. In this study, we propose a novel framework, MDT-dist, for
few-step 3D flow distillation. Our approach is built upon a primary objective:
distilling the pretrained model to learn the Marginal-Data Transport. Directly
learning this objective needs to integrate the velocity fields, while this
integral is intractable to be implemented. Therefore, we propose two
optimizable objectives, Velocity Matching (VM) and Velocity Distillation (VD),
to equivalently convert the optimization target from the transport level to the
velocity and the distribution level respectively. Velocity Matching (VM) learns
to stably match the velocity fields between the student and the teacher, but
inevitably provides biased gradient estimates. Velocity Distillation (VD)
further enhances the optimization process by leveraging the learned velocity
fields to perform probability density distillation. When evaluated on the
pioneer 3D generation framework TRELLIS, our method reduces sampling steps of
each flow transformer from 25 to 1 or 2, achieving 0.68s (1 step x 2) and 0.94s
(2 steps x 2) latency with 9.0x and 6.5x speedup on A800, while preserving high
visual and geometric fidelity. Extensive experiments demonstrate that our
method significantly outperforms existing CM distillation methods, and enables
TRELLIS to achieve superior performance in few-step 3D generation.

**Comment:** Project page: https://github.com/Zanue/MDT-dist

**Relevance Score:** 2.50

---

## 171. Completing Spatial Transcriptomics Data for Gene Expression Prediction   Benchmarking

**Authors:** Daniela Ruiz, Paula Cárdenas, Leonardo Manrique, Daniela Vega, Gabriel M. Mejia, Pablo Arbeláez

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2505.02980v2) | [PDF](http://arxiv.org/pdf/2505.02980v2)

**Abstract:** Spatial Transcriptomics is a groundbreaking technology that integrates
histology images with spatially resolved gene expression profiles. Among the
various Spatial Transcriptomics techniques available, Visium has emerged as the
most widely adopted. However, its accessibility is limited by high costs, the
need for specialized expertise, and slow clinical integration. Additionally,
gene capture inefficiencies lead to significant dropout, corrupting acquired
data. To address these challenges, the deep learning community has explored the
gene expression prediction task directly from histology images. Yet,
inconsistencies in datasets, preprocessing, and training protocols hinder fair
comparisons between models. To bridge this gap, we introduce SpaRED, a
systematically curated database comprising 26 public datasets, providing a
standardized resource for model evaluation. We further propose SpaCKLE, a
state-of-the-art transformer-based gene expression completion model that
reduces mean squared error by over 82.5% compared to existing approaches.
Finally, we establish the SpaRED benchmark, evaluating eight state-of-the-art
prediction models on both raw and SpaCKLE-completed data, demonstrating SpaCKLE
substantially improves the results across all the gene expression prediction
models. Altogether, our contributions constitute the most comprehensive
benchmark of gene expression prediction from histology images to date and a
stepping stone for future research on Spatial Transcriptomics.

**Comment:** arXiv admin note: substantial text overlap with arXiv:2407.13027

**Relevance Score:** 2.50

---

## 172. Drawing2CAD: Sequence-to-Sequence Learning for CAD Generation from   Vector Drawings

**Authors:** Feiwei Qin, Shichao Lu, Junhao Hou, Changmiao Wang, Meie Fang, Ligang Liu

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.18733v3) | [PDF](http://arxiv.org/pdf/2508.18733v3)

**Abstract:** Computer-Aided Design (CAD) generative modeling is driving significant
innovations across industrial applications. Recent works have shown remarkable
progress in creating solid models from various inputs such as point clouds,
meshes, and text descriptions. However, these methods fundamentally diverge
from traditional industrial workflows that begin with 2D engineering drawings.
The automatic generation of parametric CAD models from these 2D vector drawings
remains underexplored despite being a critical step in engineering design. To
address this gap, our key insight is to reframe CAD generation as a
sequence-to-sequence learning problem where vector drawing primitives directly
inform the generation of parametric CAD operations, preserving geometric
precision and design intent throughout the transformation process. We propose
Drawing2CAD, a framework with three key technical components: a
network-friendly vector primitive representation that preserves precise
geometric information, a dual-decoder transformer architecture that decouples
command type and parameter generation while maintaining precise correspondence,
and a soft target distribution loss function accommodating inherent flexibility
in CAD parameters. To train and evaluate Drawing2CAD, we create CAD-VGDrawing,
a dataset of paired engineering drawings and parametric CAD models, and conduct
thorough experiments to demonstrate the effectiveness of our method. Code and
dataset are available at https://github.com/lllssc/Drawing2CAD.

**Comment:** Accepted to ACM MM 2025

**Relevance Score:** 2.50

---

## 173. DVS-PedX: Synthetic-and-Real Event-Based Pedestrian Dataset

**Authors:** Mustafa Sakhai, Kaung Sithu, Min Khant Soe Oke, Maciej Wielgosz

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04117v1) | [PDF](http://arxiv.org/pdf/2509.04117v1)

**Abstract:** Event cameras like Dynamic Vision Sensors (DVS) report micro-timed brightness
changes instead of full frames, offering low latency, high dynamic range, and
motion robustness. DVS-PedX (Dynamic Vision Sensor Pedestrian eXploration) is a
neuromorphic dataset designed for pedestrian detection and crossing-intention
analysis in normal and adverse weather conditions across two complementary
sources: (1) synthetic event streams generated in the CARLA simulator for
controlled "approach-cross" scenes under varied weather and lighting; and (2)
real-world JAAD dash-cam videos converted to event streams using the v2e tool,
preserving natural behaviors and backgrounds. Each sequence includes paired RGB
frames, per-frame DVS "event frames" (33 ms accumulations), and frame-level
labels (crossing vs. not crossing). We also provide raw AEDAT 2.0/AEDAT 4.0
event files and AVI DVS video files and metadata for flexible re-processing.
Baseline spiking neural networks (SNNs) using SpikingJelly illustrate dataset
usability and reveal a sim-to-real gap, motivating domain adaptation and
multimodal fusion. DVS-PedX aims to accelerate research in event-based
pedestrian safety, intention prediction, and neuromorphic perception.

**Comment:** 12 pages, 8 figures, 3 tables; dataset descriptor paper introducing
  DVS-PedX (synthetic-and-real event-based pedestrian dataset with baselines)
  External URL: https://doi.org/10.5281/zenodo.17030898

**Relevance Score:** 2.50

---

## 174. MUNBa: Machine Unlearning via Nash Bargaining

**Authors:** Jing Wu, Mehrtash Harandi

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2411.15537v4) | [PDF](http://arxiv.org/pdf/2411.15537v4)

**Abstract:** Machine Unlearning (MU) aims to selectively erase harmful behaviors from
models while retaining the overall utility of the model. As a multi-task
learning problem, MU involves balancing objectives related to forgetting
specific concepts/data and preserving general performance. A naive integration
of these forgetting and preserving objectives can lead to gradient conflicts
and dominance, impeding MU algorithms from reaching optimal solutions. To
address the gradient conflict and dominance issue, we reformulate MU as a
two-player cooperative game, where the two players, namely, the forgetting
player and the preservation player, contribute via their gradient proposals to
maximize their overall gain and balance their contributions. To this end,
inspired by the Nash bargaining theory, we derive a closed-form solution to
guide the model toward the Pareto stationary point. Our formulation of MU
guarantees an equilibrium solution, where any deviation from the final state
would lead to a reduction in the overall objectives for both players, ensuring
optimality in each objective. We evaluate our algorithm's effectiveness on a
diverse set of tasks across image classification and image generation.
Extensive experiments with ResNet, vision-language model CLIP, and
text-to-image diffusion models demonstrate that our method outperforms
state-of-the-art MU algorithms, achieving a better trade-off between forgetting
and preserving. Our results also highlight improvements in forgetting
precision, preservation of generalization, and robustness against adversarial
attacks.

**Relevance Score:** 2.50

---

## 175. Accurate and lightweight dehazing via multi-receptive-field non-local   network and novel contrastive regularization

**Authors:** Zewei He, Zixuan Chen, Jinlei Li, Ziqian Lu, Xuecheng Sun, Hao Luo, Zhe-Ming Lu, Evangelos K. Markakis

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2309.16494v3) | [PDF](http://arxiv.org/pdf/2309.16494v3)

**Abstract:** Recently, deep learning-based methods have dominated image dehazing domain. A
multi-receptive-field non-local network (MRFNLN) consisting of the multi-stream
feature attention block (MSFAB) and the cross non-local block (CNLB) is
presented in this paper to further enhance the performance. We start with
extracting richer features for dehazing. Specifically, a multi-stream feature
extraction (MSFE) sub-block, which contains three parallel convolutions with
different receptive fields (i.e., $1\times 1$, $3\times 3$, $5\times 5$), is
designed for extracting multi-scale features. Following MSFE, an attention
sub-block is employed to make the model adaptively focus on important
channels/regions. These two sub-blocks constitute our MSFAB. Then, we design a
cross non-local block (CNLB), which can capture long-range dependencies beyond
the query. Instead of the same input source of query branch, the key and value
branches are enhanced by fusing more preceding features. CNLB is
computation-friendly by leveraging a spatial pyramid down-sampling (SPDS)
strategy to reduce the computation and memory consumption without sacrificing
the performance. Last but not least, a novel detail-focused contrastive
regularization (DFCR) is presented by emphasizing the low-level details and
ignoring the high-level semantic information in a representation space
specially designed for dehazing. Comprehensive experimental results demonstrate
that the proposed MRFNLN model outperforms recent state-of-the-art dehazing
methods with less than 1.5 Million parameters.

**Comment:** submitted to the IEEE Journal for possible publication

**Relevance Score:** 2.50

---

## 176. SWiFT: Soft-Mask Weight Fine-tuning for Bias Mitigation

**Authors:** Junyu Yan, Feng Chen, Yuyang Xue, Yuning Du, Konstantinos Vilouras, Sotirios A. Tsaftaris, Steven McDonagh

**Categories:** cs.LG, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.18826v2) | [PDF](http://arxiv.org/pdf/2508.18826v2)

**Abstract:** Recent studies have shown that Machine Learning (ML) models can exhibit bias
in real-world scenarios, posing significant challenges in ethically sensitive
domains such as healthcare. Such bias can negatively affect model fairness,
model generalization abilities and further risks amplifying social
discrimination. There is a need to remove biases from trained models. Existing
debiasing approaches often necessitate access to original training data and
need extensive model retraining; they also typically exhibit trade-offs between
model fairness and discriminative performance. To address these challenges, we
propose Soft-Mask Weight Fine-Tuning (SWiFT), a debiasing framework that
efficiently improves fairness while preserving discriminative performance with
much less debiasing costs. Notably, SWiFT requires only a small external
dataset and only a few epochs of model fine-tuning. The idea behind SWiFT is to
first find the relative, and yet distinct, contributions of model parameters to
both bias and predictive performance. Then, a two-step fine-tuning process
updates each parameter with different gradient flows defined by its
contribution. Extensive experiments with three bias sensitive attributes
(gender, skin tone, and age) across four dermatological and two chest X-ray
datasets demonstrate that SWiFT can consistently reduce model bias while
achieving competitive or even superior diagnostic accuracy under common
fairness and accuracy metrics, compared to the state-of-the-art. Specifically,
we demonstrate improved model generalization ability as evidenced by superior
performance on several out-of-distribution (OOD) datasets.

**Comment:** Accepted for publication at the Journal of Machine Learning for
  Biomedical Imaging (MELBA) https://melba-journal.org/2025:015

**Relevance Score:** 2.50

---

## 177. BuzzSet v1.0: A Dataset for Pollinator Detection in Field Conditions

**Authors:** Ahmed Emam, Mohamed Elbassiouny, Julius Miller, Patrick Donworth, Sabine Seidel, Ribana Roscher

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.19762v3) | [PDF](http://arxiv.org/pdf/2508.19762v3)

**Abstract:** Pollinator insects such as honeybees and bumblebees are vital to global food
production and ecosystem stability, yet their populations are declining due to
anthropogenic and environmental stressors. Scalable, automated monitoring in
agricultural environments remains an open challenge due to the difficulty of
detecting small, fast-moving, and often camouflaged insects. To address this,
we present BuzzSet v1.0, a large-scale dataset of high-resolution pollinator
images collected under real field conditions. BuzzSet contains 7,856 manually
verified images with more than 8,000 annotated instances across three classes:
honeybees, bumblebees, and unidentified insects. Initial annotations were
produced using a YOLOv12 model trained on external data and refined through
human verification with open-source tools. All images were preprocessed into
256 x 256 tiles to improve the detection of small insects. We provide baselines
using the RF-DETR transformer-based object detector. The model achieves strong
classification accuracy with F1 scores of 0.94 and 0.92 for honeybees and
bumblebees, with minimal confusion between these categories. The unidentified
class remains more difficult due to label ambiguity and fewer samples, yet
still contributes insights for robustness evaluation. Overall detection
performance (mAP at 0.50 of 0.559) illustrates the challenging nature of the
dataset and its potential to drive advances in small object detection under
realistic ecological conditions. Future work focuses on expanding the dataset
to version 2.0 with additional annotations and evaluating further detection
strategies. BuzzSet establishes a benchmark for ecological computer vision,
with the primary challenge being reliable detection of insects frequently
camouflaged within natural vegetation, highlighting an open problem for future
research.

**Relevance Score:** 2.50

---

## 178. Integrating Intermediate Layer Optimization and Projected Gradient   Descent for Solving Inverse Problems with Diffusion Models

**Authors:** Yang Zheng, Wen Li, Zhaoqiang Liu

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2505.20789v3) | [PDF](http://arxiv.org/pdf/2505.20789v3)

**Abstract:** Inverse problems (IPs) involve reconstructing signals from noisy
observations. Recently, diffusion models (DMs) have emerged as a powerful
framework for solving IPs, achieving remarkable reconstruction performance.
However, existing DM-based methods frequently encounter issues such as heavy
computational demands and suboptimal convergence. In this work, building upon
the idea of the recent work DMPlug, we propose two novel methods, DMILO and
DMILO-PGD, to address these challenges. Our first method, DMILO, employs
intermediate layer optimization (ILO) to alleviate the memory burden inherent
in DMPlug. Additionally, by introducing sparse deviations, we expand the range
of DMs, enabling the exploration of underlying signals that may lie outside the
range of the diffusion model. We further propose DMILO-PGD, which integrates
ILO with projected gradient descent (PGD), thereby reducing the risk of
suboptimal convergence. We provide an intuitive theoretical analysis of our
approaches under appropriate conditions and validate their superiority through
extensive experiments on diverse image datasets, encompassing both linear and
nonlinear IPs. Our results demonstrate significant performance gains over
state-of-the-art methods, highlighting the effectiveness of DMILO and DMILO-PGD
in addressing common challenges in DM-based IP solvers.

**Comment:** ICML 2025

**Relevance Score:** 2.50

---

## 179. Autoformalization in the Wild: Assessing LLMs on Real-World Mathematical   Definitions

**Authors:** Lan Zhang, Marco Valentino, Andre Freitas

**Categories:** cs.CL, cs.FL

**Links:** [arXiv](http://arxiv.org/abs/2502.12065v3) | [PDF](http://arxiv.org/pdf/2502.12065v3)

**Abstract:** Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge
the gap between informal mathematics and formal languages through
autoformalization. However, it is still unclear how well LLMs generalize to
sophisticated and naturally occurring mathematical statements. To address this
gap, we investigate the task of autoformalizing real-world mathematical
definitions: a critical component of mathematical discourse. Specifically, we
introduce two novel resources for autoformalization, collecting definitions
from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically
evaluate a range of LLMs, analyzing their ability to formalize definitions into
Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs'
performance including refinement through external feedback from Proof
Assistants, and formal definition grounding, where we augment LLMs'
formalizations through relevant contextual elements from formal mathematical
libraries. Our findings reveal that definitions present a greater challenge
compared to existing benchmarks, such as miniF2F. In particular, we found that
LLMs still struggle with self-correction, and aligning with relevant
mathematical libraries. At the same time, structured refinement methods and
definition grounding strategies yield notable improvements of up to 16% on
self-correction capabilities and 43% on the reduction of undefined errors,
highlighting promising directions for enhancing LLM-based autoformalization in
real-world scenarios.

**Comment:** EMNLP 2025 Camera-Ready Version

**Relevance Score:** 2.50

---

## 180. Towards Stable and Personalised Profiles for Lexical Alignment in Spoken   Human-Agent Dialogue

**Authors:** Keara Schaaij, Roel Boumans, Tibor Bosse, Iris Hendrickx

**Categories:** cs.HC, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04104v1) | [PDF](http://arxiv.org/pdf/2509.04104v1)

**Abstract:** Lexical alignment, where speakers start to use similar words across
conversation, is known to contribute to successful communication. However, its
implementation in conversational agents remains underexplored, particularly
considering the recent advancements in large language models (LLMs). As a first
step towards enabling lexical alignment in human-agent dialogue, this study
draws on strategies for personalising conversational agents and investigates
the construction of stable, personalised lexical profiles as a basis for
lexical alignment. Specifically, we varied the amounts of transcribed spoken
data used for construction as well as the number of items included in the
profiles per part-of-speech (POS) category and evaluated profile performance
across time using recall, coverage, and cosine similarity metrics. It was shown
that smaller and more compact profiles, created after 10 min of transcribed
speech containing 5 items for adjectives, 5 items for conjunctions, and 10
items for adverbs, nouns, pronouns, and verbs each, offered the best balance in
both performance and data efficiency. In conclusion, this study offers
practical insights into constructing stable, personalised lexical profiles,
taking into account minimal data requirements, serving as a foundational step
toward lexical alignment strategies in conversational agents.

**Comment:** Accepted for TSD 2025

**Relevance Score:** 2.50

---

## 181. Learning Optimal Prompt Ensemble for Multi-source Visual Prompt Transfer

**Authors:** Jianhua Liu, Liwen Cao, Yanru Wu, Zijie Zhao, Yang Li

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2504.12311v4) | [PDF](http://arxiv.org/pdf/2504.12311v4)

**Abstract:** Prompt tuning has emerged as a lightweight strategy for adapting foundation
models to downstream tasks, particularly for resource-constrained systems. As
pre-trained prompts become valuable assets, combining multiple source prompts
offers a promising approach to enhance generalization for new tasks by
leveraging complementary knowledge. However, naive aggregation often overlooks
different source prompts have different contribution potential to the target
task. To address this, we propose HGPrompt, a dynamic framework that learns
optimal ensemble weights. These weights are optimized by jointly maximizing an
information-theoretic metric for transferability and minimizing gradient
conflicts via a novel regularization strategy. Specifically, we propose a
differentiable prompt transferability metric to captures the discriminability
of prompt-induced features on the target task. Meanwhile, HGPrompt match the
gradient variances with respect to different source prompts based on Hessian
and Fisher Information, ensuring stable and coherent knowledge transfer while
suppressing gradient conflicts among them. Extensive experiments on the
large-scale VTAB benchmark demonstrate the state-of-the-art performance of
HGPrompt, validating its effectiveness in learning an optimal ensemble for
effective multi-source prompt transfer.

**Relevance Score:** 2.50

---

## 182. VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based   Role-Playing Agents

**Authors:** Weihao Wu, Liang Cao, Xinyu Wu, Zhiwei Lin, Rui Niu, Jingbei Li, Zhiyong Wu

**Categories:** cs.CL, cs.AI, cs.SD

**Links:** [arXiv](http://arxiv.org/abs/2509.03940v1) | [PDF](http://arxiv.org/pdf/2509.03940v1)

**Abstract:** Recent significant advancements in Large Language Models (LLMs) have greatly
propelled the development of Role-Playing Conversational Agents (RPCAs). These
systems aim to create immersive user experiences through consistent persona
adoption. However, current RPCA research faces dual limitations. First,
existing work predominantly focuses on the textual modality, entirely
overlooking critical paralinguistic features including intonation, prosody, and
rhythm in speech, which are essential for conveying character emotions and
shaping vivid identities. Second, the speech-based role-playing domain suffers
from a long-standing lack of standardized evaluation benchmarks. Most current
spoken dialogue datasets target only fundamental capability assessments,
featuring thinly sketched or ill-defined character profiles. Consequently, they
fail to effectively quantify model performance on core competencies like
long-term persona consistency. To address this critical gap, we introduce
VoxRole, the first comprehensive benchmark specifically designed for the
evaluation of speech-based RPCAs. The benchmark comprises 13335 multi-turn
dialogues, totaling 65.6 hours of speech from 1228 unique characters across 261
movies. To construct this resource, we propose a novel two-stage automated
pipeline that first aligns movie audio with scripts and subsequently employs an
LLM to systematically build multi-dimensional profiles for each character.
Leveraging VoxRole, we conduct a multi-dimensional evaluation of contemporary
spoken dialogue models, revealing crucial insights into their respective
strengths and limitations in maintaining persona consistency.

**Relevance Score:** 2.50

---

## 183. Decoding the Poetic Language of Emotion in Korean Modern Poetry:   Insights from a Human-Labeled Dataset and AI Modeling

**Authors:** Iro Lim, Haein Ji, Byungjun Kim

**Categories:** cs.CL, cs.CY, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03932v1) | [PDF](http://arxiv.org/pdf/2509.03932v1)

**Abstract:** This study introduces KPoEM (Korean Poetry Emotion Mapping) , a novel dataset
for computational emotion analysis in modern Korean poetry. Despite remarkable
progress in text-based emotion classification using large language models,
poetry-particularly Korean poetry-remains underexplored due to its figurative
language and cultural specificity. We built a multi-label emotion dataset of
7,662 entries, including 7,007 line-level entries from 483 poems and 615
work-level entries, annotated with 44 fine-grained emotion categories from five
influential Korean poets. A state-of-the-art Korean language model fine-tuned
on this dataset significantly outperformed previous models, achieving 0.60
F1-micro compared to 0.34 from models trained on general corpora. The KPoEM
model, trained through sequential fine-tuning-first on general corpora and then
on the KPoEM dataset-demonstrates not only an enhanced ability to identify
temporally and culturally specific emotional expressions, but also a strong
capacity to preserve the core sentiments of modern Korean poetry. This study
bridges computational methods and literary analysis, presenting new
possibilities for the quantitative exploration of poetic emotions through
structured data that faithfully retains the emotional and cultural nuances of
Korean literature.

**Comment:** 30 pages, 13 tables, 2 figures, Digital Humanities and Social
  Sciences Korea Conference, James Joo-Jin Kim Center for Korean Studies,
  University of Pennsylvania, Philadelphia, USA

**Relevance Score:** 2.50

---

## 184. False Sense of Security: Why Probing-based Malicious Input Detection   Fails to Generalize

**Authors:** Cheng Wang, Zeming Wei, Qin Liu, Muhao Chen

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03888v1) | [PDF](http://arxiv.org/pdf/2509.03888v1)

**Abstract:** Large Language Models (LLMs) can comply with harmful instructions, raising
serious safety concerns despite their impressive capabilities. Recent work has
leveraged probing-based approaches to study the separability of malicious and
benign inputs in LLMs' internal representations, and researchers have proposed
using such probing methods for safety detection. We systematically re-examine
this paradigm. Motivated by poor out-of-distribution performance, we
hypothesize that probes learn superficial patterns rather than semantic
harmfulness. Through controlled experiments, we confirm this hypothesis and
identify the specific patterns learned: instructional patterns and trigger
words. Our investigation follows a systematic approach, progressing from
demonstrating comparable performance of simple n-gram methods, to controlled
experiments with semantically cleaned datasets, to detailed analysis of pattern
dependencies. These results reveal a false sense of security around current
probing-based approaches and highlight the need to redesign both models and
evaluation protocols, for which we provide further discussions in the hope of
suggesting responsible further research in this direction. We have open-sourced
the project at https://github.com/WangCheng0116/Why-Probe-Fails.

**Relevance Score:** 2.50

---

## 185. NE-PADD: Leveraging Named Entity Knowledge for Robust Partial Audio   Deepfake Detection via Attention Aggregation

**Authors:** Huhong Xian, Rui Liu, Berrak Sisman, Haizhou Li

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03829v1) | [PDF](http://arxiv.org/pdf/2509.03829v1)

**Abstract:** Different from traditional sentence-level audio deepfake detection (ADD),
partial audio deepfake detection (PADD) requires frame-level positioning of the
location of fake speech. While some progress has been made in this area,
leveraging semantic information from audio, especially named entities, remains
an underexplored aspect. To this end, we propose NE-PADD, a novel method for
Partial Audio Deepfake Detection (PADD) that leverages named entity knowledge
through two parallel branches: Speech Name Entity Recognition (SpeechNER) and
PADD. The approach incorporates two attention aggregation mechanisms: Attention
Fusion (AF) for combining attention weights and Attention Transfer (AT) for
guiding PADD with named entity semantics using an auxiliary loss. Built on the
PartialSpoof-NER dataset, experiments show our method outperforms existing
baselines, proving the effectiveness of integrating named entity knowledge in
PADD. The code is available at https://github.com/AI-S2-Lab/NE-PADD.

**Relevance Score:** 2.50

---

## 186. Align-then-Slide: A complete evaluation framework for Ultra-Long   Document-Level Machine Translation

**Authors:** Jiaxin Guo, Daimeng Wei, Yuanchang Luo, Xiaoyu Chen, Zhanglin Wu, Huan Yang, Hengchao Shang, Zongyao Li, Zhiqiang Rao, Jinlong Yang, Hao Yang

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03809v1) | [PDF](http://arxiv.org/pdf/2509.03809v1)

**Abstract:** Large language models (LLMs) have ushered in a new era for document-level
machine translation (\textit{doc}-mt), yet their whole-document outputs
challenge existing evaluation methods that assume sentence-by-sentence
alignment. We introduce \textit{\textbf{Align-then-Slide}}, a complete
evaluation framework for ultra-long doc-mt. In the Align stage, we
automatically infer sentence-level source-target correspondences and rebuild
the target to match the source sentence number, resolving omissions and
many-to-one/one-to-many mappings. In the n-Chunk Sliding Evaluate stage, we
calculate averaged metric scores under 1-, 2-, 3- and 4-chunk for
multi-granularity assessment. Experiments on the WMT benchmark show a Pearson
correlation of 0.929 between our method with expert MQM rankings. On a newly
curated real-world test set, our method again aligns closely with human
judgments. Furthermore, preference data produced by Align-then-Slide enables
effective CPO training and its direct use as a reward model for GRPO, both
yielding translations preferred over a vanilla SFT baseline. The results
validate our framework as an accurate, robust, and actionable evaluation tool
for doc-mt systems.

**Comment:** under preview

**Relevance Score:** 2.50

---

## 187. Evaluating the Robustness of Retrieval-Augmented Generation to   Adversarial Evidence in the Health Domain

**Authors:** Shakiba Amirshahi, Amin Bigdeli, Charles L. A. Clarke, Amira Ghenai

**Categories:** cs.IR, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.03787v1) | [PDF](http://arxiv.org/pdf/2509.03787v1)

**Abstract:** Retrieval augmented generation (RAG) systems provide a method for factually
grounding the responses of a Large Language Model (LLM) by providing retrieved
evidence, or context, as support. Guided by this context, RAG systems can
reduce hallucinations and expand the ability of LLMs to accurately answer
questions outside the scope of their training data. Unfortunately, this design
introduces a critical vulnerability: LLMs may absorb and reproduce
misinformation present in retrieved evidence. This problem is magnified if
retrieved evidence contains adversarial material explicitly intended to
promulgate misinformation. This paper presents a systematic evaluation of RAG
robustness in the health domain and examines alignment between model outputs
and ground-truth answers. We focus on the health domain due to the potential
for harm caused by incorrect responses, as well as the availability of
evidence-based ground truth for many common health-related questions. We
conduct controlled experiments using common health questions, varying both the
type and composition of the retrieved documents (helpful, harmful, and
adversarial) as well as the framing of the question by the user (consistent,
neutral, and inconsistent). Our findings reveal that adversarial documents
substantially degrade alignment, but robustness can be preserved when helpful
evidence is also present in the retrieval pool. These findings offer actionable
insights for designing safer RAG systems in high-stakes domains by highlighting
the need for retrieval safeguards. To enable reproducibility and facilitate
future research, all experimental results are publicly available in our github
repository.
  https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL

**Relevance Score:** 2.50

---

## 188. Integrating Pruning with Quantization for Efficient Deep Neural Networks   Compression

**Authors:** Sara Makenali, Babak Rokh, Ali Azarpeyvand

**Categories:** cs.NE

**Links:** [arXiv](http://arxiv.org/abs/2509.04244v1) | [PDF](http://arxiv.org/pdf/2509.04244v1)

**Abstract:** Deep Neural Networks (DNNs) have achieved significant advances in a wide
range of applications. However, their deployment on resource-constrained
devices remains a challenge due to the large number of layers and parameters,
which result in considerable computational and memory demands. To address this
issue, pruning and quantization are two widely used compression techniques,
commonly applied individually in most studies to reduce model size and enhance
processing speed. Nevertheless, combining these two techniques can yield even
greater compression benefits. Effectively integrating pruning and quantization
to harness their complementary advantages poses a challenging task, primarily
due to their potential impact on model accuracy and the complexity of jointly
optimizing both processes. In this paper, we propose two approaches that
integrate similarity-based filter pruning with Adaptive Power-of-Two (APoT)
quantization to achieve higher compression efficiency while preserving model
accuracy. In the first approach, pruning and quantization are applied
simultaneously during training. In the second approach, pruning is performed
first to remove less important parameters, followed by quantization of the
pruned model using low-bit representations. Experimental results demonstrate
that our proposed approaches achieve effective model compression with minimal
accuracy degradation, making them well-suited for deployment on devices with
limited computational resources.

**Relevance Score:** 2.00

---

## 189. An invertible generative model for forward and inverse problems

**Authors:** Tristan van Leeuwen, Christoph Brune, Marcello Carioni

**Categories:** stat.ML, cs.LG, math.PR

**Links:** [arXiv](http://arxiv.org/abs/2509.03910v1) | [PDF](http://arxiv.org/pdf/2509.03910v1)

**Abstract:** We formulate the inverse problem in a Bayesian framework and aim to train a
generative model that allows us to simulate (i.e., sample from the likelihood)
and do inference (i.e., sample from the posterior). We review the use of
triangular normalizing flows for conditional sampling in this context and show
how to combine two such triangular maps (an upper and a lower one) in to one
invertible mapping that can be used for simulation and inference. We work out
several useful properties of this invertible generative model and propose a
possible training loss for training the map directly. We illustrate the
workings of this new approach to conditional generative modeling numerically on
a few stylized examples.

**Relevance Score:** 2.00

---

## 190. Asymptotic convexity of wide and shallow neural networks

**Authors:** Vivek Borkar, Parthe Pandit

**Categories:** stat.ML, cs.LG, math.PR, 68T07

**Links:** [arXiv](http://arxiv.org/abs/2507.01044v2) | [PDF](http://arxiv.org/pdf/2507.01044v2)

**Abstract:** For a simple model of shallow and wide neural networks, we show that the
epigraph of its input-output map as a function of the network parameters
approximates epigraph of a. convex function in a precise sense. This leads to a
plausible explanation of their observed good performance.

**Comment:** 5 pages

**Relevance Score:** 2.00

---

## 191. Solving Robotics Tasks with Prior Demonstration via   Exploration-Efficient Deep Reinforcement Learning

**Authors:** Chengyandan Shen, Christoffer Sloth

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04069v1) | [PDF](http://arxiv.org/pdf/2509.04069v1)

**Abstract:** This paper proposes an exploration-efficient Deep Reinforcement Learning with
Reference policy (DRLR) framework for learning robotics tasks that incorporates
demonstrations. The DRLR framework is developed based on an algorithm called
Imitation Bootstrapped Reinforcement Learning (IBRL). We propose to improve
IBRL by modifying the action selection module. The proposed action selection
module provides a calibrated Q-value, which mitigates the bootstrapping error
that otherwise leads to inefficient exploration. Furthermore, to prevent the RL
policy from converging to a sub-optimal policy, SAC is used as the RL policy
instead of TD3. The effectiveness of our method in mitigating bootstrapping
error and preventing overfitting is empirically validated by learning two
robotics tasks: bucket loading and open drawer, which require extensive
interactions with the environment. Simulation results also demonstrate the
robustness of the DRLR framework across tasks with both low and high
state-action dimensions, and varying demonstration qualities. To evaluate the
developed framework on a real-world industrial robotics task, the bucket
loading task is deployed on a real wheel loader. The sim2real results validate
the successful deployment of the DRLR framework.

**Relevance Score:** 2.00

---

## 192. ClutterDexGrasp: A Sim-to-Real System for General Dexterous Grasping in   Cluttered Scenes

**Authors:** Zeyuan Chen, Qiyang Yan, Yuanpei Chen, Tianhao Wu, Jiyao Zhang, Zihan Ding, Jinzhou Li, Yaodong Yang, Hao Dong

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2506.14317v3) | [PDF](http://arxiv.org/pdf/2506.14317v3)

**Abstract:** Dexterous grasping in cluttered scenes presents significant challenges due to
diverse object geometries, occlusions, and potential collisions. Existing
methods primarily focus on single-object grasping or grasp-pose prediction
without interaction, which are insufficient for complex, cluttered scenes.
Recent vision-language-action models offer a potential solution but require
extensive real-world demonstrations, making them costly and difficult to scale.
To address these limitations, we revisit the sim-to-real transfer pipeline and
develop key techniques that enable zero-shot deployment in reality while
maintaining robust generalization. We propose ClutterDexGrasp, a two-stage
teacher-student framework for closed-loop target-oriented dexterous grasping in
cluttered scenes. The framework features a teacher policy trained in simulation
using clutter density curriculum learning, incorporating both a geometry and
spatially-embedded scene representation and a novel comprehensive safety
curriculum, enabling general, dynamic, and safe grasping behaviors. Through
imitation learning, we distill the teacher's knowledge into a student 3D
diffusion policy (DP3) that operates on partial point cloud observations. To
the best of our knowledge, this represents the first zero-shot sim-to-real
closed-loop system for target-oriented dexterous grasping in cluttered scenes,
demonstrating robust performance across diverse objects and layouts. More
details and videos are available at https://clutterdexgrasp.github.io/.

**Comment:** Accepted at CoRL 2025

**Relevance Score:** 2.00

---

## 193. Spiking Neural Network Decoders of Finger Forces from High-Density   Intramuscular Microelectrode Arrays

**Authors:** Farah Baracat, Agnese Grison, Dario Farina, Giacomo Indiveri, Elisa Donati

**Categories:** cs.HC, eess.SP

**Links:** [arXiv](http://arxiv.org/abs/2509.04088v1) | [PDF](http://arxiv.org/pdf/2509.04088v1)

**Abstract:** Restoring naturalistic finger control in assistive technologies requires the
continuous decoding of motor intent with high accuracy, efficiency, and
robustness. Here, we present a spike-based decoding framework that integrates
spiking neural networks (SNNs) with motor unit activity extracted from
high-density intramuscular microelectrode arrays. We demonstrate simultaneous
and proportional decoding of individual finger forces from motor unit spike
trains during isometric contractions at 15% of maximum voluntary contraction
using SNNs. We systematically evaluated alternative SNN decoder configurations
and compared two possible input modalities: physiologically grounded motor unit
spike trains and spike-encoded intramuscular EMG signals. Through this
comparison, we quantified trade-offs between decoding accuracy, memory
footprint, and robustness to input errors. The results showed that shallow SNNs
can reliably decode finger-level motor intent with competitive accuracy and
minimal latency, while operating with reduced memory requirements and without
the need for external preprocessing buffers. This work provides a practical
blueprint for integrating SNNs into finger-level force decoding systems,
demonstrating how the choice of input representation can be strategically
tailored to meet application-specific requirements for accuracy, robustness,
and memory efficiency.

**Relevance Score:** 2.00

---

## 194. Temporal Interest-Driven Multimodal Personalized Content Generation

**Authors:** Tian Miao

**Categories:** cs.IR

**Links:** [arXiv](http://arxiv.org/abs/2509.04330v1) | [PDF](http://arxiv.org/pdf/2509.04330v1)

**Abstract:** With the dynamic evolution of user interests and the increasing multimodal
demands in internet applications, personalized content generation strategies
based on static interest preferences struggle to meet practical application
requirements. The proposed TIMGen (Temporal Interest-driven Multimodal
Generation) model addresses this challenge by modeling the long-term temporal
evolution of users' interests and capturing dynamic interest representations
with strong temporal dependencies. This model also supports the fusion of
multimodal features, such as text, images, video, and audio, and delivers
customized content based on multimodal preferences. TIMGen jointly learns
temporal dependencies and modal preferences to obtain a unified interest
representation, which it then generates to meet users' personalized content
needs. TIMGen overcomes the shortcomings of personalized content recommendation
methods based on static preferences, enabling flexible and dynamic modeling of
users' multimodal interests, better understanding and capturing their interests
and preferences. It can be extended to a variety of practical application
scenarios, including e-commerce, advertising, online education, and precision
medicine, providing insights for future research.

**Relevance Score:** 2.00

---

## 195. PianoBind: A Multimodal Joint Embedding Model for Pop-piano Music

**Authors:** Hayeon Bang, Eunjin Choi, Seungheon Doh, Juhan Nam

**Categories:** cs.SD, cs.IR, cs.MM

**Links:** [arXiv](http://arxiv.org/abs/2509.04215v1) | [PDF](http://arxiv.org/pdf/2509.04215v1)

**Abstract:** Solo piano music, despite being a single-instrument medium, possesses
significant expressive capabilities, conveying rich semantic information across
genres, moods, and styles. However, current general-purpose music
representation models, predominantly trained on large-scale datasets, often
struggle to captures subtle semantic distinctions within homogeneous solo piano
music. Furthermore, existing piano-specific representation models are typically
unimodal, failing to capture the inherently multimodal nature of piano music,
expressed through audio, symbolic, and textual modalities. To address these
limitations, we propose PianoBind, a piano-specific multimodal joint embedding
model. We systematically investigate strategies for multi-source training and
modality utilization within a joint embedding framework optimized for capturing
fine-grained semantic distinctions in (1) small-scale and (2) homogeneous piano
datasets. Our experimental results demonstrate that PianoBind learns multimodal
representations that effectively capture subtle nuances of piano music,
achieving superior text-to-music retrieval performance on in-domain and
out-of-domain piano datasets compared to general-purpose music joint embedding
models. Moreover, our design choices offer reusable insights for multimodal
representation learning with homogeneous datasets beyond piano music.

**Comment:** Accepted for publication at the 26th International Society for Music
  Information Retrieval Conference (ISMIR 2025)

**Relevance Score:** 2.00

---

## 196. ChronoGraph: A Real-World Graph-Based Multivariate Time Series Dataset

**Authors:** Adrian Catalin Lutu, Ioana Pintilie, Elena Burceanu, Andrei Manolache

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04449v1) | [PDF](http://arxiv.org/pdf/2509.04449v1)

**Abstract:** We present ChronoGraph, a graph-structured multivariate time series
forecasting dataset built from real-world production microservices. Each node
is a service that emits a multivariate stream of system-level performance
metrics, capturing CPU, memory, and network usage patterns, while directed
edges encode dependencies between services. The primary task is forecasting
future values of these signals at the service level. In addition, ChronoGraph
provides expert-annotated incident windows as anomaly labels, enabling
evaluation of anomaly detection methods and assessment of forecast robustness
during operational disruptions. Compared to existing benchmarks from industrial
control systems or traffic and air-quality domains, ChronoGraph uniquely
combines (i) multivariate time series, (ii) an explicit, machine-readable
dependency graph, and (iii) anomaly labels aligned with real incidents. We
report baseline results spanning forecasting models, pretrained time-series
foundation models, and standard anomaly detectors. ChronoGraph offers a
realistic benchmark for studying structure-aware forecasting and incident-aware
evaluation in microservice systems.

**Relevance Score:** 1.50

---

## 197. Understanding Space Is Rocket Science -- Only Top Reasoning Models Can   Solve Spatial Understanding Tasks

**Authors:** Nils Hoehing, Mayug Maniparambil, Ellen Rushe, Noel E. O'Connor, Anthony Ventresque

**Categories:** cs.CL, cs.AI, cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.02175v2) | [PDF](http://arxiv.org/pdf/2509.02175v2)

**Abstract:** We propose RocketScience, an open-source contrastive VLM benchmark that tests
for spatial relation understanding. It is comprised of entirely new real-world
image-text pairs covering mostly relative spatial understanding and the order
of objects. The benchmark is designed to be very easy for humans and hard for
the current generation of VLMs, and this is empirically verified. Our results
show a striking lack of spatial relation understanding in open source and
frontier commercial VLMs and a surprisingly high performance of reasoning
models. Additionally, we perform a disentanglement analysis to separate the
contributions of object localization and spatial reasoning in
chain-of-thought-based models and find that the performance on the benchmark is
bottlenecked by spatial reasoning and not object localization capabilities. We
release the dataset with a CC-BY-4.0 license and make the evaluation code
available at: https://github.com/nilshoehing/rocketscience

**Relevance Score:** 1.50

---

## 198. AUDETER: A Large-scale Dataset for Deepfake Audio Detection in Open   Worlds

**Authors:** Qizhou Wang, Hanxun Huang, Guansong Pang, Sarah Erfani, Christopher Leckie

**Categories:** cs.SD, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04345v1) | [PDF](http://arxiv.org/pdf/2509.04345v1)

**Abstract:** Speech generation systems can produce remarkably realistic vocalisations that
are often indistinguishable from human speech, posing significant authenticity
challenges. Although numerous deepfake detection methods have been developed,
their effectiveness in real-world environments remains unrealiable due to the
domain shift between training and test samples arising from diverse human
speech and fast evolving speech synthesis systems. This is not adequately
addressed by current datasets, which lack real-world application challenges
with diverse and up-to-date audios in both real and deep-fake categories. To
fill this gap, we introduce AUDETER (AUdio DEepfake TEst Range), a large-scale,
highly diverse deepfake audio dataset for comprehensive evaluation and robust
development of generalised models for deepfake audio detection. It consists of
over 4,500 hours of synthetic audio generated by 11 recent TTS models and 10
vocoders with a broad range of TTS/vocoder patterns, totalling 3 million audio
clips, making it the largest deepfake audio dataset by scale. Through extensive
experiments with AUDETER, we reveal that i) state-of-the-art (SOTA) methods
trained on existing datasets struggle to generalise to novel deepfake audio
samples and suffer from high false positive rates on unseen human voice,
underscoring the need for a comprehensive dataset; and ii) these methods
trained on AUDETER achieve highly generalised detection performance and
significantly reduce detection error rate by 44.1% to 51.6%, achieving an error
rate of only 4.17% on diverse cross-domain samples in the popular In-the-Wild
dataset, paving the way for training generalist deepfake audio detectors.
AUDETER is available on GitHub.

**Relevance Score:** 1.50

---

## 199. Improving Robustness of AlphaZero Algorithms to Test-Time Environment   Changes

**Authors:** Isidoro Tamassia, Wendelin Böhmer

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04317v1) | [PDF](http://arxiv.org/pdf/2509.04317v1)

**Abstract:** The AlphaZero framework provides a standard way of combining Monte Carlo
planning with prior knowledge provided by a previously trained policy-value
neural network. AlphaZero usually assumes that the environment on which the
neural network was trained will not change at test time, which constrains its
applicability. In this paper, we analyze the problem of deploying AlphaZero
agents in potentially changed test environments and demonstrate how the
combination of simple modifications to the standard framework can significantly
boost performance, even in settings with a low planning budget available. The
code is publicly available on GitHub.

**Relevance Score:** 1.50

---

## 200. Evaluating Quality of Gaming Narratives Co-created with AI

**Authors:** Arturo Valdivia, Paolo Burelli

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04239v1) | [PDF](http://arxiv.org/pdf/2509.04239v1)

**Abstract:** This paper proposes a structured methodology to evaluate AI-generated game
narratives, leveraging the Delphi study structure with a panel of narrative
design experts. Our approach synthesizes story quality dimensions from
literature and expert insights, mapping them into the Kano model framework to
understand their impact on player satisfaction. The results can inform game
developers on prioritizing quality aspects when co-creating game narratives
with generative AI.

**Relevance Score:** 1.50

---

## 201. Theory of Mind Using Active Inference: A Framework for Multi-Agent   Cooperation

**Authors:** Riddhi J. Pitliya, Ozan Çatal, Toon Van de Maele, Corrado Pezzato, Tim Verbelen

**Categories:** cs.AI, cs.MA

**Links:** [arXiv](http://arxiv.org/abs/2508.00401v2) | [PDF](http://arxiv.org/pdf/2508.00401v2)

**Abstract:** Theory of Mind (ToM) -- the ability to understand that others can have
differing knowledge and goals -- enables agents to reason about others' beliefs
while planning their own actions. We present a novel approach to multi-agent
cooperation by implementing ToM within active inference. Unlike previous active
inference approaches to multi-agent cooperation, our method neither relies on
task-specific shared generative models nor requires explicit communication. In
our framework, ToM-equipped agents maintain distinct representations of their
own and others' beliefs and goals. ToM agents then use an extended and adapted
version of the sophisticated inference tree-based planning algorithm to
systematically explore joint policy spaces through recursive reasoning. We
evaluate our approach through collision avoidance and foraging simulations.
Results suggest that ToM agents cooperate better compared to non-ToM
counterparts by being able to avoid collisions and reduce redundant efforts.
Crucially, ToM agents accomplish this by inferring others' beliefs solely from
observable behaviour and considering them when planning their own actions. Our
approach shows potential for generalisable and scalable multi-agent systems
while providing computational insights into ToM mechanisms.

**Relevance Score:** 1.50

---

## 202. First Order Model-Based RL through Decoupled Backpropagation

**Authors:** Joseph Amigo, Rooholla Khorrambakht, Elliot Chane-Sane, Nicolas Mansard, Ludovic Righetti

**Categories:** cs.AI, cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.00215v2) | [PDF](http://arxiv.org/pdf/2509.00215v2)

**Abstract:** There is growing interest in reinforcement learning (RL) methods that
leverage the simulator's derivatives to improve learning efficiency. While
early gradient-based approaches have demonstrated superior performance compared
to derivative-free methods, accessing simulator gradients is often impractical
due to their implementation cost or unavailability. Model-based RL (MBRL) can
approximate these gradients via learned dynamics models, but the solver
efficiency suffers from compounding prediction errors during training rollouts,
which can degrade policy performance. We propose an approach that decouples
trajectory generation from gradient computation: trajectories are unrolled
using a simulator, while gradients are computed via backpropagation through a
learned differentiable model of the simulator. This hybrid design enables
efficient and consistent first-order policy optimization, even when simulator
gradients are unavailable, as well as learning a critic from simulation
rollouts, which is more accurate. Our method achieves the sample efficiency and
speed of specialized optimizers such as SHAC, while maintaining the generality
of standard approaches like PPO and avoiding ill behaviors observed in other
first-order MBRL methods. We empirically validate our algorithm on benchmark
control tasks and demonstrate its effectiveness on a real Go2 quadruped robot,
across both quadrupedal and bipedal locomotion tasks.

**Comment:** CoRL 2025. Project website: https://machines-in-motion.github.io/DMO/

**Relevance Score:** 1.50

---

## 203. YOLO Ensemble for UAV-based Multispectral Defect Detection in Wind   Turbine Components

**Authors:** Serhii Svystun, Pavlo Radiuk, Oleksandr Melnychenko, Oleg Savenko, Anatoliy Sachenko

**Categories:** I.2.10; I.4.8; I.5.4; I.2.9, cs.AI, cs.CV, cs.RO, 68T07, 68T45, 68U10, 68T40

**Links:** [arXiv](http://arxiv.org/abs/2509.04156v1) | [PDF](http://arxiv.org/pdf/2509.04156v1)

**Abstract:** Unmanned aerial vehicles (UAVs) equipped with advanced sensors have opened up
new opportunities for monitoring wind power plants, including blades, towers,
and other critical components. However, reliable defect detection requires
high-resolution data and efficient methods to process multispectral imagery. In
this research, we aim to enhance defect detection accuracy through the
development of an ensemble of YOLO-based deep learning models that integrate
both visible and thermal channels. We propose an ensemble approach that
integrates a general-purpose YOLOv8 model with a specialized thermal model,
using a sophisticated bounding box fusion algorithm to combine their
predictions. Our experiments show this approach achieves a mean Average
Precision (mAP@.5) of 0.93 and an F1-score of 0.90, outperforming a standalone
YOLOv8 model, which scored an mAP@.5 of 0.91. These findings demonstrate that
combining multiple YOLO architectures with fused multispectral data provides a
more reliable solution, improving the detection of both visual and thermal
defects.

**Comment:** The 13th IEEE International Conference on Intelligent Data
  Acquisition and Advanced Computing Systems: Technology and Applications, 4-6
  September, 2025, Gliwice, Poland

**Relevance Score:** 1.50

---

## 204. Extending FKG.in: Towards a Food Claim Traceability Network

**Authors:** Saransh Kumar Gupta, Rizwan Gulzar Mir, Lipika Dey, Partha Pratim Das, Anirban Sen, Ramesh Jain

**Categories:** cs.AI, cs.IR, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.16117v2) | [PDF](http://arxiv.org/pdf/2508.16117v2)

**Abstract:** The global food landscape is rife with scientific, cultural, and commercial
claims about what foods are, what they do, what they should not do, or should
not do. These range from rigorously studied health benefits (probiotics improve
gut health) and misrepresentations (soaked almonds make one smarter) to vague
promises (superfoods boost immunity) and culturally rooted beliefs (cold foods
cause coughs). Despite their widespread influence, the infrastructure for
tracing, verifying, and contextualizing these claims remains fragmented and
underdeveloped. In this paper, we propose a Food Claim-Traceability Network
(FCN) as an extension of FKG[.]in, a knowledge graph of Indian food that we
have been incrementally building. We also present the ontology design and the
semi-automated knowledge curation workflow that we used to develop a proof of
concept of FKG[.]in-FCN using Reddit data and Large Language Models. FCN
integrates curated data inputs, structured schemas, and provenance-aware
pipelines for food-related claim extraction and validation. While directly
linked to the Indian food knowledge graph as an application, our methodology
remains application-agnostic and adaptable to other geographic, culinary, or
regulatory settings. By modeling food claims and their traceability in a
structured, verifiable, and explainable way, we aim to contribute to more
transparent and accountable food knowledge ecosystems, supporting researchers,
policymakers, and most importantly, everyday consumers in navigating a world
saturated with dietary assertions.

**Comment:** 10 pages, 3 figures, 1 table, 45 references, ACM International
  Conference on Multimedia 2025 - Multi-modal Food Computing Workshop

**Relevance Score:** 1.50

---

## 205. Enhancing FKG.in: automating Indian food composition analysis

**Authors:** Saransh Kumar Gupta, Lipika Dey, Partha Pratim Das, Geeta Trilok-Kumar, Ramesh Jain

**Categories:** cs.AI, cs.IR, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2412.05248v3) | [PDF](http://arxiv.org/pdf/2412.05248v3)

**Abstract:** This paper presents a novel approach to compute food composition data for
Indian recipes using a knowledge graph for Indian food (FKG[.]in) and LLMs. The
primary focus is to provide a broad overview of an automated food composition
analysis workflow and describe its core functionalities: nutrition data
aggregation, food composition analysis, and LLM-augmented information
resolution. This workflow aims to complement FKG[.]in and iteratively
supplement food composition data from verified knowledge bases. Additionally,
this paper highlights the challenges of representing Indian food and accessing
food composition data digitally. It also reviews three key sources of food
composition data: the Indian Food Composition Tables, the Indian Nutrient
Databank, and the Nutritionix API. Furthermore, it briefly outlines how users
can interact with the workflow to obtain diet-based health recommendations and
detailed food composition information for numerous recipes. We then explore the
complex challenges of analyzing Indian recipe information across dimensions
such as structure, multilingualism, and uncertainty as well as present our
ongoing work on LLM-based solutions to address these issues. The methods
proposed in this workshop paper for AI-driven knowledge curation and
information resolution are application-agnostic, generalizable, and replicable
for any domain.

**Comment:** 15 pages, 5 figures, 30 references, International Conference on
  Pattern Recognition 2024 - Multimedia Assisted Dietary Management Workshop

**Relevance Score:** 1.50

---

## 206. Stochastic Parameter Decomposition

**Authors:** Lucius Bushnaq, Dan Braun, Lee Sharkey

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2506.20790v2) | [PDF](http://arxiv.org/pdf/2506.20790v2)

**Abstract:** A key step in reverse engineering neural networks is to decompose them into
simpler parts that can be studied in relative isolation. Linear parameter
decomposition -- a framework that has been proposed to resolve several issues
with current decomposition methods -- decomposes neural network parameters into
a sum of sparsely used vectors in parameter space. However, the current main
method in this framework, Attribution-based Parameter Decomposition (APD), is
impractical on account of its computational cost and sensitivity to
hyperparameters. In this work, we introduce \textit{Stochastic Parameter
Decomposition} (SPD), a method that is more scalable and robust to
hyperparameters than APD, which we demonstrate by decomposing models that are
slightly larger and more complex than was possible to decompose with APD. We
also show that SPD avoids other issues, such as shrinkage of the learned
parameters, and better identifies ground truth mechanisms in toy models. By
bridging causal mediation analysis and network decomposition methods, this
demonstration opens up new research possibilities in mechanistic
interpretability by removing barriers to scaling linear parameter decomposition
methods to larger models. We release a library for running SPD and reproducing
our experiments at https://github.com/goodfire-ai/spd/tree/spd-paper.

**Relevance Score:** 1.50

---

## 207. Analysis of Bluffing by DQN and CFR in Leduc Hold'em Poker

**Authors:** Tarik Zaciragic, Aske Plaat, K. Joost Batenburg

**Categories:** cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04125v1) | [PDF](http://arxiv.org/pdf/2509.04125v1)

**Abstract:** In the game of poker, being unpredictable, or bluffing, is an essential
skill. When humans play poker, they bluff. However, most works on
computer-poker focus on performance metrics such as win rates, while bluffing
is overlooked. In this paper we study whether two popular algorithms, DQN
(based on reinforcement learning) and CFR (based on game theory), exhibit
bluffing behavior in Leduc Hold'em, a simplified version of poker. We designed
an experiment where we let the DQN and CFR agent play against each other while
we log their actions. We find that both DQN and CFR exhibit bluffing behavior,
but they do so in different ways. Although both attempt to perform bluffs at
different rates, the percentage of successful bluffs (where the opponent folds)
is roughly the same. This suggests that bluffing is an essential aspect of the
game, not of the algorithm. Future work should look at different bluffing
styles and at the full game of poker. Code at
https://github.com/TarikZ03/Bluffing-by-DQN-and-CFR-in-Leduc-Hold-em-Poker-Codebase.

**Relevance Score:** 1.50

---

## 208. EHVC: Efficient Hierarchical Reference and Quality Structure for Neural   Video Coding

**Authors:** Junqi Liao, Yaojun Wu, Chaoyi Lin, Zhipin Deng, Li Li, Dong Liu, Xiaoyan Sun

**Categories:** eess.IV, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04118v1) | [PDF](http://arxiv.org/pdf/2509.04118v1)

**Abstract:** Neural video codecs (NVCs), leveraging the power of end-to-end learning, have
demonstrated remarkable coding efficiency improvements over traditional video
codecs. Recent research has begun to pay attention to the quality structures in
NVCs, optimizing them by introducing explicit hierarchical designs. However,
less attention has been paid to the reference structure design, which
fundamentally should be aligned with the hierarchical quality structure. In
addition, there is still significant room for further optimization of the
hierarchical quality structure. To address these challenges in NVCs, we propose
EHVC, an efficient hierarchical neural video codec featuring three key
innovations: (1) a hierarchical multi-reference scheme that draws on
traditional video codec design to align reference and quality structures,
thereby addressing the reference-quality mismatch; (2) a lookahead strategy to
utilize an encoder-side context from future frames to enhance the quality
structure; (3) a layer-wise quality scale with random quality training strategy
to stabilize quality structures during inference. With these improvements, EHVC
achieves significantly superior performance to the state-of-the-art NVCs. Code
will be released in: https://github.com/bytedance/NEVC.

**Comment:** 9 pages, 8 figures, Accepted to ACMMM 2025

**Relevance Score:** 1.50

---

## 209. WASP: A Weight-Space Approach to Detecting Learned Spuriousness

**Authors:** Cristian Daniel Păduraru, Antonio Bărbălau, Radu Filipescu, Andrei Liviu Nicolicioiu, Elena Burceanu

**Categories:** cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2410.18970v4) | [PDF](http://arxiv.org/pdf/2410.18970v4)

**Abstract:** It is of crucial importance to train machine learning models such that they
clearly understand what defines each class in a given task. Though there is a
sum of works dedicated to identifying the spurious correlations featured by a
dataset that may impact the model's understanding of the classes, all current
approaches rely solely on data or error analysis. That is, they cannot point
out spurious correlations learned by the model that are not already pointed out
by the counterexamples featured in the validation or training sets. We propose
a method that transcends this limitation, switching the focus from analyzing a
model's predictions to analyzing the model's weights, the mechanism behind the
making of the decisions, which proves to be more insightful. Our proposed
Weight-space Approach to detecting Spuriousness (WASP) relies on analyzing the
weights of foundation models as they drift towards capturing various (spurious)
correlations while being fine-tuned on a given dataset. We demonstrate that
different from previous works, our method (i) can expose spurious correlations
featured by a dataset even when they are not exposed by training or validation
counterexamples, (ii) it works for multiple modalities such as image and text,
and (iii) it can uncover previously untapped spurious correlations learned by
ImageNet-1k classifiers.

**Comment:** under review

**Relevance Score:** 1.50

---

## 210. Beyond holography: the entropic quantum gravity foundations of image   processing

**Authors:** Ginestra Bianconi

**Categories:** cond-mat.dis-nn, cond-mat.stat-mech, cs.AI, gr-qc, quant-ph

**Links:** [arXiv](http://arxiv.org/abs/2503.14048v2) | [PDF](http://arxiv.org/pdf/2503.14048v2)

**Abstract:** Recently, thanks to the development of artificial intelligence (AI) there is
increasing scientific attention in establishing the connections between
theoretical physics and AI. Traditionally, these connections have been focusing
mostly on the relation between string theory and image processing and involve
important theoretical paradigms such as holography. Recently G. Bianconi has
formulated the Gravity from Entropy (GfE) approach to quantum gravity in which
gravity is derived from the geometric quantum relative entropy (GQRE) between
two metrics associated with the Lorentzian spacetime. Here it is demonstrated
that the famous Perona-Malik algorithm for image processing is the gradient
flow of the GfE action in its simple warm-up scenario. Specifically, this
algorithm is the outcome of the minimization of the GQRE between two Euclidean
metrics: the one of the support of the image and the one induced by the image.
As the Perona-Malik algorithm is known to preserve sharp contours, this implies
that the GfE action, does not in general lead to uniform images upon iteration
of the gradient flow dynamics as it would be intuitively expected from entropic
actions maximising classical entropies. Rather, the outcome of the minimization
of the GQRE is compatible with the preservation of complex structures. These
results provide the geometrical and information theory foundations for the
Perona-Malik algorithm and might contribute to establish deeper connections
between GfE, machine learning and brain research.

**Comment:** (7 pages, 1 figure)

**Relevance Score:** 1.50

---

## 211. SAC-MIL: Spatial-Aware Correlated Multiple Instance Learning for   Histopathology Whole Slide Image Classification

**Authors:** Yu Bai, Zitong Yu, Haowen Tian, Xijing Wang, Shuo Yan, Lin Wang, Honglin Li, Xitong Ling, Bo Zhang, Zheng Zhang, Wufan Wang, Hui Gao, Xiangyang Gong, Wendong Wang

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03973v1) | [PDF](http://arxiv.org/pdf/2509.03973v1)

**Abstract:** We propose Spatial-Aware Correlated Multiple Instance Learning (SAC-MIL) for
performing WSI classification. SAC-MIL consists of a positional encoding module
to encode position information and a SAC block to perform full instance
correlations. The positional encoding module utilizes the instance coordinates
within the slide to encode the spatial relationships instead of the instance
index in the input WSI sequence. The positional encoding module can also handle
the length extrapolation issue where the training and testing sequences have
different lengths. The SAC block is an MLP-based method that performs full
instance correlation in linear time complexity with respect to the sequence
length. Due to the simple structure of MLP, it is easy to deploy since it does
not require custom CUDA kernels, compared to Transformer-based methods for WSI
classification. SAC-MIL has achieved state-of-the-art performance on the
CAMELYON-16, TCGA-LUNG, and TCGA-BRAC datasets. The code will be released upon
acceptance.

**Relevance Score:** 1.50

---

## 212. Virtual Fitting Room: Generating Arbitrarily Long Videos of Virtual   Try-On from a Single Image -- Technical Preview

**Authors:** Jun-Kun Chen, Aayush Bansal, Minh Phuoc Vo, Yu-Xiong Wang

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04450v1) | [PDF](http://arxiv.org/pdf/2509.04450v1)

**Abstract:** We introduce the Virtual Fitting Room (VFR), a novel video generative model
that produces arbitrarily long virtual try-on videos. Our VFR models long video
generation tasks as an auto-regressive, segment-by-segment generation process,
eliminating the need for resource-intensive generation and lengthy video data,
while providing the flexibility to generate videos of arbitrary length. The key
challenges of this task are twofold: ensuring local smoothness between adjacent
segments and maintaining global temporal consistency across different segments.
To address these challenges, we propose our VFR framework, which ensures
smoothness through a prefix video condition and enforces consistency with the
anchor video -- a 360-degree video that comprehensively captures the human's
wholebody appearance. Our VFR generates minute-scale virtual try-on videos with
both local smoothness and global temporal consistency under various motions,
making it a pioneering work in long virtual try-on video generation.

**Comment:** Project Page: https://immortalco.github.io/VirtualFittingRoom/

**Relevance Score:** 1.50

---

## 213. Understanding sparse autoencoder scaling in the presence of feature   manifolds

**Authors:** Eric J. Michaud, Liv Gorton, Tom McGrath

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.02565v2) | [PDF](http://arxiv.org/pdf/2509.02565v2)

**Abstract:** Sparse autoencoders (SAEs) model the activations of a neural network as
linear combinations of sparsely occurring directions of variation (latents).
The ability of SAEs to reconstruct activations follows scaling laws w.r.t. the
number of latents. In this work, we adapt a capacity-allocation model from the
neural scaling literature (Brill, 2024) to understand SAE scaling, and in
particular, to understand how "feature manifolds" (multi-dimensional features)
influence scaling behavior. Consistent with prior work, the model recovers
distinct scaling regimes. Notably, in one regime, feature manifolds have the
pathological effect of causing SAEs to learn far fewer features in data than
there are latents in the SAE. We provide some preliminary discussion on whether
or not SAEs are in this pathological regime in the wild.

**Comment:** 13 pages, 8 figures, short workshop submission

**Relevance Score:** 1.50

---

## 214. Moco: A Learnable Meta Optimizer for Combinatorial Optimization

**Authors:** Tim Dernedde, Daniela Thyssens, Sören Dittrich, Maximilian Stubbemann, Lars Schmidt-Thieme

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2402.04915v3) | [PDF](http://arxiv.org/pdf/2402.04915v3)

**Abstract:** Relevant combinatorial optimization problems (COPs) are often NP-hard. While
they have been tackled mainly via handcrafted heuristics in the past, advances
in neural networks have motivated the development of general methods to learn
heuristics from data. Many approaches utilize a neural network to directly
construct a solution, but are limited in further improving based on already
constructed solutions at inference time. Our approach, Moco, defines a
lightweight solution construction procedure, guided by a single continuous
vector $\theta$ (called heatmap) and learns a neural network to update $\theta$
for a single instance of a COP at inference time. The update is based on
various features of the current search state. The training procedure is budget
aware, targeting the overall best solution found during the entire search. Moco
is a fully learnable meta optimizer not utilizing problem specific heuristics
or requiring optimal solutions for training. We test Moco on the Traveling
Salesman Problem (TSP) and Maximum Independent Set (MIS) and show that it
significantly improves over other heatmap based methods.

**Comment:** 20 pages, 2 figures. A prior version was published in Advances in
  Knowledge Discovery and Data Mining. PAKDD 2025. Lecture Notes in Computer
  Science, vol 15872. Springer, Singapore

**Relevance Score:** 1.50

---

## 215. Characteristic Energy Behavior Profiling of Non-Residential Buildings

**Authors:** Haley Dozier, Althea Henslee

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04322v1) | [PDF](http://arxiv.org/pdf/2509.04322v1)

**Abstract:** Due to the threat of changing climate and extreme weather events, the
infrastructure of the United States Army installations is at risk. More than
ever, climate resilience measures are needed to protect facility assets that
support critical missions and help generate readiness. As most of the Army
installations within the continental United States rely on commercial energy
and water sources, resilience to the vulnerabilities within independent energy
resources (electricity grids, natural gas pipelines, etc) along with a baseline
understanding of energy usage within installations must be determined. This
paper will propose a data-driven behavioral model to determine behavior
profiles of energy usage on installations. These profiles will be used 1) to
create a baseline assessment of the impact of unexpected disruptions on energy
systems and 2) to benchmark future resiliency measures. In this methodology,
individual building behavior will be represented with models that can
accurately analyze, predict, and cluster multimodal data collected from energy
usage of non-residential buildings. Due to the nature of Army installation
energy usage data, similarly structured open access data will be used to
illustrate this methodology.

**Relevance Score:** 1.50

---

## 216. Generalized and Unified Equivalences between Hardness and Pseudoentropy

**Authors:** Lunjia Hu, Salil Vadhan

**Categories:** cs.CC, cs.CR, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2507.05972v2) | [PDF](http://arxiv.org/pdf/2507.05972v2)

**Abstract:** Pseudoentropy characterizations provide a quantitatively precise
demonstration of the close relationship between computational hardness and
computational randomness. We prove a unified pseudoentropy characterization
that generalizes and strengthens previous results for both uniform and
non-uniform models of computation. Our characterization holds for a general
family of entropy notions that encompasses the common notions of Shannon
entropy and min entropy as special cases. Moreover, we show that the
characterizations for different entropy notions can be simultaneously achieved
by a single, universal function that simultaneously witnesses computational
hardness and computational randomness. A key technical insight of our work is
that the notion of weight-restricted calibration from the recent literature on
algorithm fairness, along with standard computational indistinguishability
(known as multiaccuracy in the fairness literature), suffices for proving
pseudoentropy characterizations for general entropy notions. This demonstrates
the power of weight-restricted calibration to enhance the classic
Complexity-Theoretic Regularity Lemma (Trevisan, Tulsiani, and Vadhan, 2009)
and Leakage Simulation Lemma (Jetchev and Pietrzak, 2014) and allows us to
achieve an exponential improvement in the complexity dependency on the alphabet
size compared to the pseudoentropy characterizations by Casacuberta, Dwork, and
Vadhan (2024) based on the much stronger notion of multicalibration. We show
that the exponential dependency on the alphabet size is inevitable for
multicalibration as well as for the weaker notion of calibrated multiaccuracy.

**Comment:** Accepted to TCC 2025

**Relevance Score:** 1.50

---

## 217. Federated Isolation Forest for Efficient Anomaly Detection on Edge IoT   Systems

**Authors:** Pavle Vasiljevic, Milica Matic, Miroslav Popovic

**Categories:** cs.LG, cs.DC

**Links:** [arXiv](http://arxiv.org/abs/2506.05138v2) | [PDF](http://arxiv.org/pdf/2506.05138v2)

**Abstract:** Recently, federated learning frameworks such as Python TestBed for Federated
Learning Algorithms and MicroPython TestBed for Federated Learning Algorithms
have emerged to tackle user privacy concerns and efficiency in embedded
systems. Even more recently, an efficient federated anomaly detection
algorithm, FLiForest, based on Isolation Forests has been developed, offering a
low-resource, unsupervised method well-suited for edge deployment and
continuous learning. In this paper, we present an application of Isolation
Forest-based temperature anomaly detection, developed using the previously
mentioned federated learning frameworks, aimed at small edge devices and IoT
systems running MicroPython. The system has been experimentally evaluated,
achieving over 96% accuracy in distinguishing normal from abnormal readings and
above 78% precision in detecting anomalies across all tested configurations,
while maintaining a memory usage below 160 KB during model training. These
results highlight its suitability for resource-constrained environments and
edge systems, while upholding federated learning principles of data privacy and
collaborative learning.

**Comment:** 6 pages, 4 algorithms, 5 figures, 2 tables

**Relevance Score:** 1.50

---

## 218. An Interactive Framework for Finding the Optimal Trade-off in   Differential Privacy

**Authors:** Yaohong Yang, Aki Rehn, Sammie Katt, Antti Honkela, Samuel Kaski

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04290v1) | [PDF](http://arxiv.org/pdf/2509.04290v1)

**Abstract:** Differential privacy (DP) is the standard for privacy-preserving analysis,
and introduces a fundamental trade-off between privacy guarantees and model
performance. Selecting the optimal balance is a critical challenge that can be
framed as a multi-objective optimization (MOO) problem where one first
discovers the set of optimal trade-offs (the Pareto front) and then learns a
decision-maker's preference over them. While a rich body of work on interactive
MOO exists, the standard approach -- modeling the objective functions with
generic surrogates and learning preferences from simple pairwise feedback -- is
inefficient for DP because it fails to leverage the problem's unique structure:
a point on the Pareto front can be generated directly by maximizing accuracy
for a fixed privacy level. Motivated by this property, we first derive the
shape of the trade-off theoretically, which allows us to model the Pareto front
directly and efficiently. To address inefficiency in preference learning, we
replace pairwise comparisons with a more informative interaction. In
particular, we present the user with hypothetical trade-off curves and ask them
to pick their preferred trade-off. Our experiments on differentially private
logistic regression and deep transfer learning across six real-world datasets
show that our method converges to the optimal privacy-accuracy trade-off with
significantly less computational cost and user interaction than baselines.

**Comment:** 20 pages, 12 figures

**Relevance Score:** 1.50

---

## 219. Privacy Risks in Time Series Forecasting: User- and Record-Level   Membership Inference

**Authors:** Nicolas Johansson, Tobias Olsson, Daniel Nilsson, Johan Östman, Fazeleh Hoseini

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04169v1) | [PDF](http://arxiv.org/pdf/2509.04169v1)

**Abstract:** Membership inference attacks (MIAs) aim to determine whether specific data
were used to train a model. While extensively studied on classification models,
their impact on time series forecasting remains largely unexplored. We address
this gap by introducing two new attacks: (i) an adaptation of multivariate
LiRA, a state-of-the-art MIA originally developed for classification models, to
the time-series forecasting setting, and (ii) a novel end-to-end learning
approach called Deep Time Series (DTS) attack. We benchmark these methods
against adapted versions of other leading attacks from the classification
setting.
  We evaluate all attacks in realistic settings on the TUH-EEG and ELD
datasets, targeting two strong forecasting architectures, LSTM and the
state-of-the-art N-HiTS, under both record- and user-level threat models. Our
results show that forecasting models are vulnerable, with user-level attacks
often achieving perfect detection. The proposed methods achieve the strongest
performance in several settings, establishing new baselines for privacy risk
assessment in time series forecasting. Furthermore, vulnerability increases
with longer prediction horizons and smaller training populations, echoing
trends observed in large language models.

**Relevance Score:** 1.50

---

## 220. Uncertainty-Guided Likelihood Tree Search

**Authors:** Julia Grosse, Ruotian Wu, Ahmad Rashid, Cheng Zhang, Philipp Hennig, Pascal Poupart, Agustinus Kristiadi

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2407.03951v3) | [PDF](http://arxiv.org/pdf/2407.03951v3)

**Abstract:** Tree search is a fundamental tool for planning, as many sequential
decision-making problems can be framed as searching over tree-structured
spaces. We propose an uncertainty-guided tree search algorithm for settings
where the reward function is a log-likelihood function of the paths. Due to the
combinatorial explosion of the tree size, the set of paths for which one can
obtain rewards is sparse, particularly when the likelihood is obtained through
expensive evaluations, such as by querying a large language model. We address
this challenge by deriving an probabilistic search heuristic based on
regularity assumptions for the likelihood. Unlike existing tree search methods,
the proposed method can perform backtracking and trade-off exploration with
exploitation, and yet does not require expensive roll-outs, or sophisticated
Bayesian inference. Through extensive on-model and off-model experiments on
timely, large-scale practical applications, we demonstrate that our method
identifies paths with high likelihood while requiring fewer costly evaluations.

**Comment:** 10 pages

**Relevance Score:** 1.50

---

## 221. Bayesian Additive Regression Trees for functional ANOVA model

**Authors:** Seokhun Park, Insung Kong, Yongdai Kim

**Categories:** stat.ML, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03317v2) | [PDF](http://arxiv.org/pdf/2509.03317v2)

**Abstract:** Bayesian Additive Regression Trees (BART) is a powerful statistical model
that leverages the strengths of Bayesian inference and regression trees. It has
received significant attention for capturing complex non-linear relationships
and interactions among predictors. However, the accuracy of BART often comes at
the cost of interpretability. To address this limitation, we propose ANOVA
Bayesian Additive Regression Trees (ANOVA-BART), a novel extension of BART
based on the functional ANOVA decomposition, which is used to decompose the
variability of a function into different interactions, each representing the
contribution of a different set of covariates or factors. Our proposed
ANOVA-BART enhances interpretability, preserves and extends the theoretical
guarantees of BART, and achieves superior predictive performance. Specifically,
we establish that the posterior concentration rate of ANOVA-BART is nearly
minimax optimal, and further provides the same convergence rates for each
interaction that are not available for BART. Moreover, comprehensive
experiments confirm that ANOVA-BART surpasses BART in both accuracy and
uncertainty quantification, while also demonstrating its effectiveness in
component selection. These results suggest that ANOVA-BART offers a compelling
alternative to BART by balancing predictive accuracy, interpretability, and
theoretical consistency.

**Relevance Score:** 1.50

---

## 222. Shuffling Heuristic in Variational Inequalities: Establishing New   Convergence Guarantees

**Authors:** Daniil Medyakov, Gleb Molodtsov, Grigoriy Evseev, Egor Petrov, Aleksandr Beznosikov

**Categories:** math.OC, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04133v1) | [PDF](http://arxiv.org/pdf/2509.04133v1)

**Abstract:** Variational inequalities have gained significant attention in machine
learning and optimization research. While stochastic methods for solving these
problems typically assume independent data sampling, we investigate an
alternative approach -- the shuffling heuristic. This strategy involves
permuting the dataset before sequential processing, ensuring equal
consideration of all data points. Despite its practical utility, theoretical
guarantees for shuffling in variational inequalities remain unexplored. We
address this gap by providing the first theoretical convergence estimates for
shuffling methods in this context. Our analysis establishes rigorous bounds and
convergence rates, extending the theoretical framework for this important class
of algorithms. We validate our findings through extensive experiments on
diverse benchmark variational inequality problems, demonstrating faster
convergence of shuffling methods compared to independent sampling approaches.

**Comment:** 25 pages, 5 figures, 2 tables

**Relevance Score:** 1.50

---

## 223. FedQuad: Federated Stochastic Quadruplet Learning to Mitigate Data   Heterogeneity

**Authors:** Ozgu Goksu, Nicolas Pugeault

**Categories:** cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04107v1) | [PDF](http://arxiv.org/pdf/2509.04107v1)

**Abstract:** Federated Learning (FL) provides decentralised model training, which
effectively tackles problems such as distributed data and privacy preservation.
However, the generalisation of global models frequently faces challenges from
data heterogeneity among clients. This challenge becomes even more pronounced
when datasets are limited in size and class imbalance. To address data
heterogeneity, we propose a novel method, \textit{FedQuad}, that explicitly
optimises smaller intra-class variance and larger inter-class variance across
clients, thereby decreasing the negative impact of model aggregation on the
global model over client representations. Our approach minimises the distance
between similar pairs while maximising the distance between negative pairs,
effectively disentangling client data in the shared feature space. We evaluate
our method on the CIFAR-10 and CIFAR-100 datasets under various data
distributions and with many clients, demonstrating superior performance
compared to existing approaches. Furthermore, we provide a detailed analysis of
metric learning-based strategies within both supervised and federated learning
paradigms, highlighting their efficacy in addressing representational learning
challenges in federated settings.

**Comment:** The 3rd IEEE International Conference on Federated Learning
  Technologies and Applications (FLTA25)

**Relevance Score:** 1.50

---

## 224. Deliberate Planning of 3D Bin Packing on Packing Configuration Trees

**Authors:** Hang Zhao, Juzhan Xu, Kexiong Yu, Ruizhen Hu, Chenyang Zhu, Bo Du, Kai Xu

**Categories:** cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2504.04421v4) | [PDF](http://arxiv.org/pdf/2504.04421v4)

**Abstract:** Online 3D Bin Packing Problem (3D-BPP) has widespread applications in
industrial automation. Existing methods usually solve the problem with limited
resolution of spatial discretization, and/or cannot deal with complex practical
constraints well. We propose to enhance the practical applicability of online
3D-BPP via learning on a novel hierarchical representation, packing
configuration tree (PCT). PCT is a full-fledged description of the state and
action space of bin packing which can support packing policy learning based on
deep reinforcement learning (DRL). The size of the packing action space is
proportional to the number of leaf nodes, making the DRL model easy to train
and well-performing even with continuous solution space. We further discover
the potential of PCT as tree-based planners in deliberately solving packing
problems of industrial significance, including large-scale packing and
different variations of BPP setting. A recursive packing method is proposed to
decompose large-scale packing into smaller sub-trees while a spatial ensemble
mechanism integrates local solutions into global. For different BPP variations
with additional decision variables, such as lookahead, buffering, and offline
packing, we propose a unified planning framework enabling out-of-the-box
problem solving. Extensive evaluations demonstrate that our method outperforms
existing online BPP baselines and is versatile in incorporating various
practical constraints. The planning process excels across large-scale problems
and diverse problem variations. We develop a real-world packing robot for
industrial warehousing, with careful designs accounting for constrained
placement and transportation stability. Our packing robot operates reliably and
efficiently on unprotected pallets at 10 seconds per box. It achieves averagely
19 boxes per pallet with 57.4% space utilization for relatively large-size
boxes.

**Comment:** International Journal of Robotics Research

**Relevance Score:** 1.50

---

## 225. Emergence of Quantised Representations Isolated to Anisotropic Functions

**Authors:** George Bird

**Categories:** cs.LG, I.5.1; F.1.1; I.2.6

**Links:** [arXiv](http://arxiv.org/abs/2507.12070v3) | [PDF](http://arxiv.org/pdf/2507.12070v3)

**Abstract:** This paper presents a novel methodology for determining representational
structure, which builds upon the existing Spotlight Resonance method. This new
tool is used to gain insight into how discrete representations can emerge and
organise in autoencoder models, through a controlled ablation study in which
only the activation function is altered. Using this technique, the validity of
whether function-driven symmetries can act as implicit inductive biases on
representations is determined. Representations are found to tend to discretise
when the activation functions are defined through a discrete algebraic
permutation-equivariant symmetry. In contrast, they remain continuous under a
continuous algebraic orthogonal-equivariant definition. This confirms the
hypothesis that the symmetries of network primitives can carry unintended
inductive biases, which produce task-independent artefactual structures in
representations. The discrete symmetry of contemporary forms is shown to be a
strong predictor for the production of discrete representations emerging from
otherwise continuous distributions -- a quantisation effect. This motivates
further reassessment of functional forms in common usage due to such unintended
consequences. Moreover, this supports a general causal model for one mode in
which discrete representations may form, and could constitute a prerequisite
for downstream interpretability phenomena, including grandmother neurons,
discrete coding schemes, general linear features and possibly Superposition.
Hence, this tool and proposed mechanism for the influence of functional form on
representations may provide insights into interpretability research. Finally,
preliminary results indicate that quantisation of representations appears to
correlate with a measurable increase in reconstruction error, reinforcing
previous conjectures that this collapse can be detrimental.

**Comment:** 41 pages, 37 figures, edited some introductory phrasing and
  appendices on hyperoctahedral LeakyReLU

**Relevance Score:** 1.50

---

## 226. On Aligning Prediction Models with Clinical Experiential Learning: A   Prostate Cancer Case Study

**Authors:** Jacqueline J. Vallon, William Overman, Wanqiao Xu, Neil Panjwani, Xi Ling, Sushmita Vij, Hilary P. Bagshaw, John T. Leppert, Sumit Shah, Geoffrey Sonn, Sandy Srinivas, Erqi Pollom, Mark K. Buyyounouski, Mohsen Bayati

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04053v1) | [PDF](http://arxiv.org/pdf/2509.04053v1)

**Abstract:** Over the past decade, the use of machine learning (ML) models in healthcare
applications has rapidly increased. Despite high performance, modern ML models
do not always capture patterns the end user requires. For example, a model may
predict a non-monotonically decreasing relationship between cancer stage and
survival, keeping all other features fixed. In this paper, we present a
reproducible framework for investigating this misalignment between model
behavior and clinical experiential learning, focusing on the effects of
underspecification of modern ML pipelines. In a prostate cancer outcome
prediction case study, we first identify and address these inconsistencies by
incorporating clinical knowledge, collected by a survey, via constraints into
the ML model, and subsequently analyze the impact on model performance and
behavior across degrees of underspecification. The approach shows that aligning
the ML model with clinical experiential learning is possible without
compromising performance. Motivated by recent literature in generative AI, we
further examine the feasibility of a feedback-driven alignment approach in
non-generative AI clinical risk prediction models through a randomized
experiment with clinicians. Our findings illustrate that, by eliciting
clinicians' model preferences using our proposed methodology, the larger the
difference in how the constrained and unconstrained models make predictions for
a patient, the more apparent the difference is in clinical interpretation.

**Relevance Score:** 1.50

---

## 227. TensoIS: A Step Towards Feed-Forward Tensorial Inverse Subsurface   Scattering for Perlin Distributed Heterogeneous Media

**Authors:** Ashish Tiwari, Satyam Bhardwaj, Yash Bachwana, Parag Sarvoday Sahu, T. M. Feroz Ali, Bhargava Chintalapati, Shanmuganathan Raman

**Categories:** cs.GR, cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04047v1) | [PDF](http://arxiv.org/pdf/2509.04047v1)

**Abstract:** Estimating scattering parameters of heterogeneous media from images is a
severely under-constrained and challenging problem. Most of the existing
approaches model BSSRDF either through an analysis-by-synthesis approach,
approximating complex path integrals, or using differentiable volume rendering
techniques to account for heterogeneity. However, only a few studies have
applied learning-based methods to estimate subsurface scattering parameters,
but they assume homogeneous media. Interestingly, no specific distribution is
known to us that can explicitly model the heterogeneous scattering parameters
in the real world. Notably, procedural noise models such as Perlin and Fractal
Perlin noise have been effective in representing intricate heterogeneities of
natural, organic, and inorganic surfaces. Leveraging this, we first create
HeteroSynth, a synthetic dataset comprising photorealistic images of
heterogeneous media whose scattering parameters are modeled using Fractal
Perlin noise. Furthermore, we propose Tensorial Inverse Scattering (TensoIS), a
learning-based feed-forward framework to estimate these Perlin-distributed
heterogeneous scattering parameters from sparse multi-view image observations.
Instead of directly predicting the 3D scattering parameter volume, TensoIS uses
learnable low-rank tensor components to represent the scattering volume. We
evaluate TensoIS on unseen heterogeneous variations over shapes from the
HeteroSynth test set, smoke and cloud geometries obtained from open-source
realistic volumetric simulations, and some real-world samples to establish its
effectiveness for inverse scattering. Overall, this study is an attempt to
explore Perlin noise distribution, given the lack of any such well-defined
distribution in literature, to potentially model real-world heterogeneous
scattering in a feed-forward manner.

**Comment:** To appear in Pacific Graphics 2025 (CGF Journal Track), Project page:
  https://yashbachwana.github.io/TensoIS/

**Relevance Score:** 1.50

---

## 228. What if I ask in \textit{alia lingua}? Measuring Functional Similarity   Across Languages

**Authors:** Debangan Mishra, Arihant Rastogi, Agyeya Negi, Shashwat Goel, Ponnurangam Kumaraguru

**Categories:** cs.CL, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04032v1) | [PDF](http://arxiv.org/pdf/2509.04032v1)

**Abstract:** How similar are model outputs across languages? In this work, we study this
question using a recently proposed model similarity metric $\kappa_p$ applied
to 20 languages and 47 subjects in GlobalMMLU. Our analysis reveals that a
model's responses become increasingly consistent across languages as its size
and capability grow. Interestingly, models exhibit greater cross-lingual
consistency within themselves than agreement with other models prompted in the
same language. These results highlight not only the value of $\kappa_p$ as a
practical tool for evaluating multilingual reliability, but also its potential
to guide the development of more consistent multilingual systems.

**Comment:** Preprint, 11 Pages

**Relevance Score:** 1.50

---

## 229. Learning neural representations for X-ray ptychography reconstruction   with unknown probes

**Authors:** Tingyou Li, Zixin Xu, Zirui Gao, Hanfei Yan, Xiaojing Huang, Jizhou Li

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04402v1) | [PDF](http://arxiv.org/pdf/2509.04402v1)

**Abstract:** X-ray ptychography provides exceptional nanoscale resolution and is widely
applied in materials science, biology, and nanotechnology. However, its full
potential is constrained by the critical challenge of accurately reconstructing
images when the illuminating probe is unknown. Conventional iterative methods
and deep learning approaches are often suboptimal, particularly under the
low-signal conditions inherent to low-dose and high-speed experiments. These
limitations compromise reconstruction fidelity and restrict the broader
adoption of the technique. In this work, we introduce the Ptychographic
Implicit Neural Representation (PtyINR), a self-supervised framework that
simultaneously addresses the object and probe recovery problem. By
parameterizing both as continuous neural representations, PtyINR performs
end-to-end reconstruction directly from raw diffraction patterns without
requiring any pre-characterization of the probe. Extensive evaluations
demonstrate that PtyINR achieves superior reconstruction quality on both
simulated and experimental data, with remarkable robustness under challenging
low-signal conditions. Furthermore, PtyINR offers a generalizable,
physics-informed framework for addressing probe-dependent inverse problems,
making it applicable to a wide range of computational microscopy problems.

**Relevance Score:** 1.50

---

## 230. Ecological Legacies of Pre-Columbian Settlements Evident in Palm   Clusters of Neotropical Mountain Forests

**Authors:** Sebastian Fajardo, Sina Mohammadi, Jonas Gregorio de Souza, César Ardila, Alan Tapscott Baltar, Shaddai Heidgen, Maria Isabel Mayorga Hernández, Sylvia Mota de Oliveira, Fernando Montejo, Marco Moderato, Vinicius Peripato, Katy Puche, Carlos Reina, Juan Carlos Vargas, Frank W. Takes, Marco Madella

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2507.06949v2) | [PDF](http://arxiv.org/pdf/2507.06949v2)

**Abstract:** Ancient populations markedly transformed Neotropical forests, yet the spatial
extent of their ecological influence remains underexplored at high resolution.
Here we present a deep learning and remote sensing based approach to estimate
areas of pre-Columbian forest modification based on modern vegetation. We apply
this method to high-resolution satellite imagery from the Sierra Nevada de
Santa Marta, Colombia, as a demonstration of a scalable approach, to evaluate
palm tree distributions in relation to archaeological infrastructure. Palms
were significantly more abundant near archaeological sites with large
infrastructure investment. The extent of the largest palm cluster indicates
that ancient human-managed areas linked to major infrastructure sites may be up
to two orders of magnitude bigger than indicated by current archaeological
evidence alone. Our findings suggest that pre-Columbian populations influenced
vegetation, fostering conditions conducive to palm proliferation, leaving a
lasting ecological footprint. This may have lowered the logistical costs of
establishing infrastructure-heavy settlements in less accessible locations.

**Relevance Score:** 1.50

---

## 231. Stitching the Story: Creating Panoramic Incident Summaries from   Body-Worn Footage

**Authors:** Dor Cohen, Inga Efrosman, Yehudit Aperstein, Alexander Apartsin

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04370v1) | [PDF](http://arxiv.org/pdf/2509.04370v1)

**Abstract:** First responders widely adopt body-worn cameras to document incident scenes
and support post-event analysis. However, reviewing lengthy video footage is
impractical in time-critical situations. Effective situational awareness
demands a concise visual summary that can be quickly interpreted. This work
presents a computer vision pipeline that transforms body-camera footage into
informative panoramic images summarizing the incident scene. Our method
leverages monocular Simultaneous Localization and Mapping (SLAM) to estimate
camera trajectories and reconstruct the spatial layout of the environment. Key
viewpoints are identified by clustering camera poses along the trajectory, and
representative frames from each cluster are selected. These frames are fused
into spatially coherent panoramic images using multi-frame stitching
techniques. The resulting summaries enable rapid understanding of complex
environments and facilitate efficient decision-making and incident review.

**Comment:** 5 pages, 3 figures

**Relevance Score:** 1.50

---

## 232. MICACL: Multi-Instance Category-Aware Contrastive Learning for   Long-Tailed Dynamic Facial Expression Recognition

**Authors:** Feng-Qi Cui, Zhen Lin, Xinlong Rao, Anyang Tong, Shiyao Li, Fei Wang, Changlin Chen, Bin Liu

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04344v1) | [PDF](http://arxiv.org/pdf/2509.04344v1)

**Abstract:** Dynamic facial expression recognition (DFER) faces significant challenges due
to long-tailed category distributions and complexity of spatio-temporal feature
modeling. While existing deep learning-based methods have improved DFER
performance, they often fail to address these issues, resulting in severe model
induction bias. To overcome these limitations, we propose a novel
multi-instance learning framework called MICACL, which integrates
spatio-temporal dependency modeling and long-tailed contrastive learning
optimization. Specifically, we design the Graph-Enhanced Instance Interaction
Module (GEIIM) to capture intricate spatio-temporal between adjacent instances
relationships through adaptive adjacency matrices and multiscale convolutions.
To enhance instance-level feature aggregation, we develop the Weighted Instance
Aggregation Network (WIAN), which dynamically assigns weights based on instance
importance. Furthermore, we introduce a Multiscale Category-aware Contrastive
Learning (MCCL) strategy to balance training between major and minor
categories. Extensive experiments on in-the-wild datasets (i.e., DFEW and
FERV39k) demonstrate that MICACL achieves state-of-the-art performance with
superior robustness and generalization.

**Comment:** Accepted by IEEE ISPA2025

**Relevance Score:** 1.50

---

## 233. DIO: Refining Mutual Information and Causal Chain to Enhance Machine   Abstract Reasoning Ability

**Authors:** Ruizhuo Song, Beiming Yuan

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.15387v4) | [PDF](http://arxiv.org/pdf/2508.15387v4)

**Abstract:** Despite the outstanding performance of current deep learning models across
various domains, their fundamental bottleneck in abstract reasoning remains
unresolved. To address this challenge, the academic community has introduced
Raven's Progressive Matrices (RPM) problems as an authoritative benchmark for
evaluating the abstract reasoning capabilities of deep learning algorithms,
with a focus on core intelligence dimensions such as abstract reasoning,
pattern recognition, and complex problem-solving. Therefore, this paper centers
on solving RPM problems, aiming to contribute to enhancing the abstract
reasoning abilities of machine intelligence. Firstly, this paper adopts a
``causal chain modeling'' perspective to systematically analyze the complete
causal chain in RPM tasks: image $\rightarrow$ abstract attributes
$\rightarrow$ progressive attribute patterns $\rightarrow$ pattern consistency
$\rightarrow$ correct answer. Based on this analysis, the network architecture
of the baseline model DIO is designed. However, experiments reveal that the
optimization objective formulated for DIO, namely maximizing the variational
lower bound of mutual information between the context and the correct option,
fails to enable the model to genuinely acquire the predefined human reasoning
logic. This is attributed to two main reasons: the tightness of the lower bound
significantly impacts the effectiveness of mutual information maximization, and
mutual information, as a statistical measure, does not capture the causal
relationship between subjects and objects. To overcome these limitations, this
paper progressively proposes three improvement methods:

**Comment:** 15 pages, 9 figures, 8 tables

**Relevance Score:** 1.50

---

## 234. TEn-CATS: Text-Enriched Audio-Visual Video Parsing with Multi-Scale   Category-Aware Temporal Graph

**Authors:** Yaru Chen, Faegheh Sardari, Peiliang Zhang, Ruohao Guo, Yang Xiang, Zhenbo Li, Wenwu Wang

**Categories:** cs.CV, cs.MM

**Links:** [arXiv](http://arxiv.org/abs/2509.04086v1) | [PDF](http://arxiv.org/pdf/2509.04086v1)

**Abstract:** Audio-Visual Video Parsing (AVVP) task aims to identify event categories and
their occurrence times in a given video with weakly supervised labels. Existing
methods typically fall into two categories: (i) designing enhanced
architectures based on attention mechanism for better temporal modeling, and
(ii) generating richer pseudo-labels to compensate for the absence of
frame-level annotations. However, the first type methods treat noisy
segment-level pseudo labels as reliable supervision and the second type methods
let indiscriminate attention spread them across all frames, the initial errors
are repeatedly amplified during training. To address this issue, we propose a
method that combines the Bi-Directional Text Fusion (BiT) module and
Category-Aware Temporal Graph (CATS) module. Specifically, we integrate the
strengths and complementarity of the two previous research directions. We first
perform semantic injection and dynamic calibration on audio and visual modality
features through the BiT module, to locate and purify cleaner and richer
semantic cues. Then, we leverage the CATS module for semantic propagation and
connection to enable precise semantic information dissemination across time.
Experimental results demonstrate that our proposed method achieves
state-of-the-art (SOTA) performance in multiple key indicators on two benchmark
datasets, LLP and UnAV-100.

**Relevance Score:** 1.50

---

## 235. A Re-ranking Method using K-nearest Weighted Fusion for Person   Re-identification

**Authors:** Quang-Huy Che, Le-Chuong Nguyen, Gia-Nghia Tran, Dinh-Duy Phan, Vinh-Tiep Nguyen

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04050v1) | [PDF](http://arxiv.org/pdf/2509.04050v1)

**Abstract:** In person re-identification, re-ranking is a crucial step to enhance the
overall accuracy by refining the initial ranking of retrieved results. Previous
studies have mainly focused on features from single-view images, which can
cause view bias and issues like pose variation, viewpoint changes, and
occlusions. Using multi-view features to present a person can help reduce view
bias. In this work, we present an efficient re-ranking method that generates
multi-view features by aggregating neighbors' features using K-nearest Weighted
Fusion (KWF) method. Specifically, we hypothesize that features extracted from
re-identification models are highly similar when representing the same
identity. Thus, we select K neighboring features in an unsupervised manner to
generate multi-view features. Additionally, this study explores the weight
selection strategies during feature aggregation, allowing us to identify an
effective strategy. Our re-ranking approach does not require model fine-tuning
or extra annotations, making it applicable to large-scale datasets. We evaluate
our method on the person re-identification datasets Market1501, MSMT17, and
Occluded-DukeMTMC. The results show that our method significantly improves
Rank@1 and mAP when re-ranking the top M candidates from the initial ranking
results. Specifically, compared to the initial results, our re-ranking method
achieves improvements of 9.8%/22.0% in Rank@1 on the challenging datasets:
MSMT17 and Occluded-DukeMTMC, respectively. Furthermore, our approach
demonstrates substantial enhancements in computational efficiency compared to
other re-ranking methods.

**Comment:** Published in ICPRAM 2025, ISBN 978-989-758-730-6, ISSN 2184-4313

**Relevance Score:** 1.50

---

## 236. Encoder-Only Image Registration

**Authors:** Xiang Chen, Renjiu Hu, Jinwei Zhang, Yuxi Zhang, Xinyao Yue, Min Liu, Yaonan Wang, Hang Zhang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.00451v2) | [PDF](http://arxiv.org/pdf/2509.00451v2)

**Abstract:** Learning-based techniques have significantly improved the accuracy and speed
of deformable image registration. However, challenges such as reducing
computational complexity and handling large deformations persist. To address
these challenges, we analyze how convolutional neural networks (ConvNets)
influence registration performance using the Horn-Schunck optical flow
equation. Supported by prior studies and our empirical experiments, we observe
that ConvNets play two key roles in registration: linearizing local intensities
and harmonizing global contrast variations. Based on these insights, we propose
the Encoder-Only Image Registration (EOIR) framework, designed to achieve a
better accuracy-efficiency trade-off. EOIR separates feature learning from flow
estimation, employing only a 3-layer ConvNet for feature extraction and a set
of 3-layer flow estimators to construct a Laplacian feature pyramid,
progressively composing diffeomorphic deformations under a large-deformation
model. Results on five datasets across different modalities and anatomical
regions demonstrate EOIR's effectiveness, achieving superior
accuracy-efficiency and accuracy-smoothness trade-offs. With comparable
accuracy, EOIR provides better efficiency and smoothness, and vice versa. The
source code of EOIR is publicly available on
https://github.com/XiangChen1994/EOIR.

**Relevance Score:** 1.50

---

## 237. Hallo4: High-Fidelity Dynamic Portrait Animation via Direct Preference   Optimization and Temporal Motion Modulation

**Authors:** Jiahao Cui, Yan Chen, Mingwang Xu, Hanlin Shang, Yuxuan Chen, Yun Zhan, Zilong Dong, Yao Yao, Jingdong Wang, Siyu Zhu

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2505.23525v2) | [PDF](http://arxiv.org/pdf/2505.23525v2)

**Abstract:** Generating highly dynamic and photorealistic portrait animations driven by
audio and skeletal motion remains challenging due to the need for precise lip
synchronization, natural facial expressions, and high-fidelity body motion
dynamics. We propose a human-preference-aligned diffusion framework that
addresses these challenges through two key innovations. First, we introduce
direct preference optimization tailored for human-centric animation, leveraging
a curated dataset of human preferences to align generated outputs with
perceptual metrics for portrait motion-video alignment and naturalness of
expression. Second, the proposed temporal motion modulation resolves
spatiotemporal resolution mismatches by reshaping motion conditions into
dimensionally aligned latent features through temporal channel redistribution
and proportional feature expansion, preserving the fidelity of high-frequency
motion details in diffusion-based synthesis. The proposed mechanism is
complementary to existing UNet and DiT-based portrait diffusion approaches, and
experiments demonstrate obvious improvements in lip-audio synchronization,
expression vividness, body motion coherence over baseline methods, alongside
notable gains in human preference metrics. Our model and source code can be
found at: https://github.com/xyz123xyz456/hallo4.

**Relevance Score:** 1.50

---

## 238. Defending LVLMs Against Vision Attacks through Partial-Perception   Supervision

**Authors:** Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Categories:** cs.CV, cs.AI, cs.CR

**Links:** [arXiv](http://arxiv.org/abs/2412.12722v2) | [PDF](http://arxiv.org/pdf/2412.12722v2)

**Abstract:** Recent studies have raised significant concerns regarding the vulnerability
of Large Vision Language Models (LVLMs) to maliciously injected or perturbed
input images, which can mislead their responses. Existing defense methods show
that such vision attacks are sensitive to image modifications especially
cropping, using majority voting across responses of modified images as
corrected responses. However, these modifications often result in partial
images and distort the semantics, which reduces response quality on clean
images after voting. Instead of directly using responses from partial images
for voting, we investigate using them to supervise the LVLM's responses to the
original images. We propose a black-box, training-free method called DPS
(Defense through Partial-Perception Supervision). In this approach, the model
is prompted using the responses generated by a model that perceives only a
partial image. With DPS, the model can adjust its response based on partial
image understanding when under attack, while confidently maintaining its
original response for clean input. Our findings show that the weak model can
supervise the strong model: when faced with an attacked input, the strong model
becomes less confident and adjusts its response based on the weak model's
partial understanding, effectively defending against the attack. With clean
input, it confidently maintains its original response. Empirical experiments
show our method outperforms the baseline, cutting the average attack success
rate by 76.3% across six datasets on three popular models.

**Comment:** Accepted to ICML 2025

**Relevance Score:** 1.50

---

## 239. LMVC: An End-to-End Learned Multiview Video Coding Framework

**Authors:** Xihua Sheng, Yingwen Zhang, Long Xu, Shiqi Wang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03922v1) | [PDF](http://arxiv.org/pdf/2509.03922v1)

**Abstract:** Multiview video is a key data source for volumetric video, enabling immersive
3D scene reconstruction but posing significant challenges in storage and
transmission due to its massive data volume. Recently, deep learning-based
end-to-end video coding has achieved great success, yet most focus on
single-view or stereo videos, leaving general multiview scenarios
underexplored. This paper proposes an end-to-end learned multiview video coding
(LMVC) framework that ensures random access and backward compatibility while
enhancing compression efficiency. Our key innovation lies in effectively
leveraging independent-view motion and content information to enhance
dependent-view compression. Specifically, to exploit the inter-view motion
correlation, we propose a feature-based inter-view motion vector prediction
method that conditions dependent-view motion encoding on decoded
independent-view motion features, along with an inter-view motion entropy model
that learns inter-view motion priors. To exploit the inter-view content
correlation, we propose a disparity-free inter-view context prediction module
that predicts inter-view contexts from decoded independent-view content
features, combined with an inter-view contextual entropy model that captures
inter-view context priors. Experimental results show that our proposed LMVC
framework outperforms the reference software of the traditional MV-HEVC
standard by a large margin, establishing a strong baseline for future research
in this field.

**Relevance Score:** 1.50

---

## 240. MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation

**Authors:** Gowen Loo, Chang Liu, Qinghong Yin, Xiang Chen, Jiawei Chen, Jingyuan Zhang, Yu Tian

**Categories:** cs.CL, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03891v1) | [PDF](http://arxiv.org/pdf/2509.03891v1)

**Abstract:** Smartphones have become indispensable in people's daily lives, permeating
nearly every aspect of modern society. With the continuous advancement of large
language models (LLMs), numerous LLM-based mobile agents have emerged. These
agents are capable of accurately parsing diverse user queries and automatically
assisting users in completing complex or repetitive operations. However,
current agents 1) heavily rely on the comprehension ability of LLMs, which can
lead to errors caused by misoperations or omitted steps during tasks, 2) lack
interaction with the external environment, often terminating tasks when an app
cannot fulfill user queries, and 3) lack memory capabilities, requiring each
instruction to reconstruct the interface and being unable to learn from and
correct previous mistakes. To alleviate the above issues, we propose MobileRAG,
a mobile agents framework enhanced by Retrieval-Augmented Generation (RAG),
which includes InterRAG, LocalRAG, and MemRAG. It leverages RAG to more quickly
and accurately identify user queries and accomplish complex and long-sequence
mobile tasks. Additionally, to more comprehensively assess the performance of
MobileRAG, we introduce MobileRAG-Eval, a more challenging benchmark
characterized by numerous complex, real-world mobile tasks that require
external knowledge assistance. Extensive experimental results on MobileRAG-Eval
demonstrate that MobileRAG can easily handle real-world mobile tasks, achieving
10.3\% improvement over state-of-the-art methods with fewer operational steps.
Our code is publicly available at:
https://github.com/liuxiaojieOutOfWorld/MobileRAG_arxiv

**Relevance Score:** 1.50

---

## 241. Contextualized Token Discrimination for Speech Search Query Correction

**Authors:** Junyu Lu, Di Jiang, Mengze Hong, Victor Junqiu Wei, Qintian Guo, Zhiyang Su

**Categories:** cs.SD, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04393v1) | [PDF](http://arxiv.org/pdf/2509.04393v1)

**Abstract:** Query spelling correction is an important function of modern search engines
since it effectively helps users express their intentions clearly. With the
growing popularity of speech search driven by Automated Speech Recognition
(ASR) systems, this paper introduces a novel method named Contextualized Token
Discrimination (CTD) to conduct effective speech query correction. In CTD, we
first employ BERT to generate token-level contextualized representations and
then construct a composition layer to enhance semantic information. Finally, we
produce the correct query according to the aggregated token representation,
correcting the incorrect tokens by comparing the original token representations
and the contextualized representations. Extensive experiments demonstrate the
superior performance of our proposed method across all metrics, and we further
present a new benchmark dataset with erroneous ASR transcriptions to offer
comprehensive evaluations for audio query correction.

**Relevance Score:** 1.50

---

## 242. DynaSaur: Large Language Agents Beyond Predefined Actions

**Authors:** Dang Nguyen, Viet Dac Lai, Seunghyun Yoon, Ryan A. Rossi, Handong Zhao, Ruiyi Zhang, Puneet Mathur, Nedim Lipka, Yu Wang, Trung Bui, Franck Dernoncourt, Tianyi Zhou

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2411.01747v3) | [PDF](http://arxiv.org/pdf/2411.01747v3)

**Abstract:** Existing LLM agent systems typically select actions from a fixed and
predefined set at every step. While this approach is effective in closed,
narrowly scoped environments, it presents two major challenges for real-world,
open-ended scenarios: (1) it significantly restricts the planning and acting
capabilities of LLM agents, and (2) it requires substantial human effort to
enumerate and implement all possible actions, which is impractical in complex
environments with a vast number of potential actions. To address these
limitations, we propose an LLM agent framework that can dynamically create and
compose actions as needed. In this framework, the agent interacts with its
environment by generating and executing programs written in a general-purpose
programming language. Moreover, generated actions are accumulated over time for
future reuse. Our extensive experiments across multiple benchmarks show that
this framework significantly improves flexibility and outperforms prior methods
that rely on a fixed action set. Notably, it enables LLM agents to adapt and
recover in scenarios where predefined actions are insufficient or fail due to
unforeseen edge cases. Our code can be found in
https://github.com/adobe-research/dynasaur.

**Comment:** Published as a conference paper at COLM 2025

**Relevance Score:** 1.50

---

## 243. Can Compact Language Models Search Like Agents? Distillation-Guided   Policy Optimization for Preserving Agentic RAG Capabilities

**Authors:** Rikuto Kotoge, Mai Nishimura, Jiaxin Ma

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2508.20324v2) | [PDF](http://arxiv.org/pdf/2508.20324v2)

**Abstract:** Reinforcement Learning has emerged as a post-training approach to elicit
agentic RAG behaviors such as search and planning from language models.
However, compact language models (e.g., 0.5B parameters) struggle due to poor
reasoning ability, resulting in sparse rewards and unstable training. To
overcome these difficulties, we propose Distillation-Guided Policy Optimization
(DGPO), which addresses the challenges through cold-start initialization from
teacher demonstrations and continuous teacher guidance during policy
optimization. To systematically evaluate our approach, we introduce Agentic RAG
Capabilities (ARC), a fine-grained metric analyzing reasoning, search
coordination, and response synthesis. Comprehensive experiments demonstrate
that DGPO enables compact models to achieve sophisticated agentic search
behaviors, even outperforming the larger teacher model in some cases. DGPO
makes agentic RAG feasible in computing resource-constrained environments.

**Relevance Score:** 1.50

---

## 244. Explicit and Implicit Data Augmentation for Social Event Detection

**Authors:** Congbo Ma, Yuxia Wang, Jia Wu, Jian Yang, Jing Du, Zitai Qiu, Qing Li, Hu Wang, Preslav Nakov

**Categories:** cs.CL, cs.SI

**Links:** [arXiv](http://arxiv.org/abs/2509.04202v1) | [PDF](http://arxiv.org/pdf/2509.04202v1)

**Abstract:** Social event detection involves identifying and categorizing important events
from social media, which relies on labeled data, but annotation is costly and
labor-intensive. To address this problem, we propose Augmentation framework for
Social Event Detection (SED-Aug), a plug-and-play dual augmentation framework,
which combines explicit text-based and implicit feature-space augmentation to
enhance data diversity and model robustness. The explicit augmentation utilizes
large language models to enhance textual information through five diverse
generation strategies. For implicit augmentation, we design five novel
perturbation techniques that operate in the feature space on structural fused
embeddings. These perturbations are crafted to keep the semantic and relational
properties of the embeddings and make them more diverse. Specifically, SED-Aug
outperforms the best baseline model by approximately 17.67% on the Twitter2012
dataset and by about 15.57% on the Twitter2018 dataset in terms of the average
F1 score. The code is available at GitHub: https://github.com/congboma/SED-Aug.

**Relevance Score:** 1.50

---

## 245. Joint Modeling of Entities and Discourse Relations for Coherence   Assessment

**Authors:** Wei Liu, Michael Strube

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04182v1) | [PDF](http://arxiv.org/pdf/2509.04182v1)

**Abstract:** In linguistics, coherence can be achieved by different means, such as by
maintaining reference to the same set of entities across sentences and by
establishing discourse relations between them. However, most existing work on
coherence modeling focuses exclusively on either entity features or discourse
relation features, with little attention given to combining the two. In this
study, we explore two methods for jointly modeling entities and discourse
relations for coherence assessment. Experiments on three benchmark datasets
show that integrating both types of features significantly enhances the
performance of coherence models, highlighting the benefits of modeling both
simultaneously for coherence evaluation.

**Comment:** EMNLP 2025

**Relevance Score:** 1.50

---

## 246. SLM-Bench: A Comprehensive Benchmark of Small Language Models on   Environmental Impacts--Extended Version

**Authors:** Nghiem Thanh Pham, Tung Kieu, Duc-Manh Nguyen, Son Ha Xuan, Nghia Duong-Trung, Danh Le-Phuoc

**Categories:** cs.CL, cs.CY, cs.PF

**Links:** [arXiv](http://arxiv.org/abs/2508.15478v2) | [PDF](http://arxiv.org/pdf/2508.15478v2)

**Abstract:** Small Language Models (SLMs) offer computational efficiency and
accessibility, yet a systematic evaluation of their performance and
environmental impact remains lacking. We introduce SLM-Bench, the first
benchmark specifically designed to assess SLMs across multiple dimensions,
including accuracy, computational efficiency, and sustainability metrics.
SLM-Bench evaluates 15 SLMs on 9 NLP tasks using 23 datasets spanning 14
domains. The evaluation is conducted on 4 hardware configurations, providing a
rigorous comparison of their effectiveness. Unlike prior benchmarks, SLM-Bench
quantifies 11 metrics across correctness, computation, and consumption,
enabling a holistic assessment of efficiency trade-offs. Our evaluation
considers controlled hardware conditions, ensuring fair comparisons across
models. We develop an open-source benchmarking pipeline with standardized
evaluation protocols to facilitate reproducibility and further research. Our
findings highlight the diverse trade-offs among SLMs, where some models excel
in accuracy while others achieve superior energy efficiency. SLM-Bench sets a
new standard for SLM evaluation, bridging the gap between resource efficiency
and real-world applicability.

**Comment:** 24 pages. An extended version of "SLM-Bench: A Comprehensive
  Benchmark of Small Language Models on Environmental Impacts" accepted at
  EMNLP 2025

**Relevance Score:** 1.50

---

## 247. MultiWikiQA: A Reading Comprehension Benchmark in 300+ Languages

**Authors:** Dan Saattrup Smart

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04111v1) | [PDF](http://arxiv.org/pdf/2509.04111v1)

**Abstract:** We introduce a new reading comprehension dataset, dubbed MultiWikiQA, which
covers 306 languages. The context data comes from Wikipedia articles, with
questions generated by an LLM and the answers appearing verbatim in the
Wikipedia articles. We conduct a crowdsourced human evaluation of the fluency
of the generated questions across 30 of the languages, providing evidence that
the questions are of good quality. We evaluate 6 different language models,
both decoder and encoder models of varying sizes, showing that the benchmark is
sufficiently difficult and that there is a large performance discrepancy
amongst the languages. The dataset and survey evaluations are freely available.

**Relevance Score:** 1.50

---

## 248. Exploring Linguistic Features for Turkish Text Readability

**Authors:** Ahmet Yavuz Uluslu, Gerold Schneider

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2306.03774v4) | [PDF](http://arxiv.org/pdf/2306.03774v4)

**Abstract:** This paper presents the first comprehensive study on automatic readability
assessment of Turkish texts. We combine state-of-the-art neural network models
with linguistic features at lexical, morphological, syntactic and discourse
levels to develop an advanced readability tool. We evaluate the effectiveness
of traditional readability formulas compared to modern automated methods and
identify key linguistic features that determine the readability of Turkish
texts.

**Relevance Score:** 1.50

---

## 249. A Comprehensive Survey on Trustworthiness in Reasoning with Large   Language Models

**Authors:** Yanbo Wang, Yongcan Yu, Jian Liang, Ran He

**Categories:** cs.CL, cs.AI, cs.CR

**Links:** [arXiv](http://arxiv.org/abs/2509.03871v1) | [PDF](http://arxiv.org/pdf/2509.03871v1)

**Abstract:** The development of Long-CoT reasoning has advanced LLM performance across
various tasks, including language understanding, complex problem solving, and
code generation. This paradigm enables models to generate intermediate
reasoning steps, thereby improving both accuracy and interpretability. However,
despite these advancements, a comprehensive understanding of how CoT-based
reasoning affects the trustworthiness of language models remains
underdeveloped. In this paper, we survey recent work on reasoning models and
CoT techniques, focusing on five core dimensions of trustworthy reasoning:
truthfulness, safety, robustness, fairness, and privacy. For each aspect, we
provide a clear and structured overview of recent studies in chronological
order, along with detailed analyses of their methodologies, findings, and
limitations. Future research directions are also appended at the end for
reference and discussion. Overall, while reasoning techniques hold promise for
enhancing model trustworthiness through hallucination mitigation, harmful
content detection, and robustness improvement, cutting-edge reasoning models
themselves often suffer from comparable or even greater vulnerabilities in
safety, robustness, and privacy. By synthesizing these insights, we hope this
work serves as a valuable and timely resource for the AI safety community to
stay informed on the latest progress in reasoning trustworthiness. A full list
of related papers can be found at
\href{https://github.com/ybwang119/Awesome-reasoning-safety}{https://github.com/ybwang119/Awesome-reasoning-safety}.

**Comment:** 38 pages. This survey considers papers published up to June 30, 2025.
  Work in progress

**Relevance Score:** 1.50

---

## 250. Measuring How (Not Just Whether) VLMs Build Common Ground

**Authors:** Saki Imai, Mert İnan, Anthony Sicilia, Malihe Alikhani

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03805v1) | [PDF](http://arxiv.org/pdf/2509.03805v1)

**Abstract:** Large vision language models (VLMs) increasingly claim reasoning skills, yet
current benchmarks evaluate them in single-turn or question answering settings.
However, grounding is an interactive process in which people gradually develop
shared understanding through ongoing communication. We introduce a four-metric
suite (grounding efficiency, content alignment, lexical adaptation, and
human-likeness) to systematically evaluate VLM performance in interactive
grounding contexts. We deploy the suite on 150 self-play sessions of
interactive referential games between three proprietary VLMs and compare them
with human dyads. All three models diverge from human patterns on at least
three metrics, while GPT4o-mini is the closest overall. We find that (i) task
success scores do not indicate successful grounding and (ii) high
image-utterance alignment does not necessarily predict task success. Our metric
suite and findings offer a framework for future research on VLM grounding.

**Relevance Score:** 1.50

---

## 251. HalluEntity: Benchmarking and Understanding Entity-Level Hallucination   Detection

**Authors:** Min-Hsuan Yeh, Max Kamachee, Seongheon Park, Yixuan Li

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2502.11948v3) | [PDF](http://arxiv.org/pdf/2502.11948v3)

**Abstract:** To mitigate the impact of hallucination nature of LLMs, many studies propose
detecting hallucinated generation through uncertainty estimation. However,
these approaches predominantly operate at the sentence or paragraph level,
failing to pinpoint specific spans or entities responsible for hallucinated
content. This lack of granularity is especially problematic for long-form
outputs that mix accurate and fabricated information. To address this
limitation, we explore entity-level hallucination detection. We propose a new
data set, HalluEntity, which annotates hallucination at the entity level. Based
on the dataset, we comprehensively evaluate uncertainty-based hallucination
detection approaches across 17 modern LLMs. Our experimental results show that
uncertainty estimation approaches focusing on individual token probabilities
tend to over-predict hallucinations, while context-aware methods show better
but still suboptimal performance. Through an in-depth qualitative study, we
identify relationships between hallucination tendencies and linguistic
properties and highlight important directions for future research.
  HalluEntity: https://huggingface.co/datasets/samuelyeh/HalluEntity

**Comment:** TMLR 2025

**Relevance Score:** 1.50

---

## 252. SiLVERScore: Semantically-Aware Embeddings for Sign Language Generation   Evaluation

**Authors:** Saki Imai, Mert İnan, Anthony Sicilia, Malihe Alikhani

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03791v1) | [PDF](http://arxiv.org/pdf/2509.03791v1)

**Abstract:** Evaluating sign language generation is often done through back-translation,
where generated signs are first recognized back to text and then compared to a
reference using text-based metrics. However, this two-step evaluation pipeline
introduces ambiguity: it not only fails to capture the multimodal nature of
sign language-such as facial expressions, spatial grammar, and prosody-but also
makes it hard to pinpoint whether evaluation errors come from sign generation
model or the translation system used to assess it. In this work, we propose
SiLVERScore, a novel semantically-aware embedding-based evaluation metric that
assesses sign language generation in a joint embedding space. Our contributions
include: (1) identifying limitations of existing metrics, (2) introducing
SiLVERScore for semantically-aware evaluation, (3) demonstrating its robustness
to semantic and prosodic variations, and (4) exploring generalization
challenges across datasets. On PHOENIX-14T and CSL-Daily datasets, SiLVERScore
achieves near-perfect discrimination between correct and random pairs (ROC AUC
= 0.99, overlap < 7%), substantially outperforming traditional metrics.

**Relevance Score:** 1.50

---

## 253. Improved sampling algorithms and Poincaré inequalities for   non-log-concave distributions

**Authors:** Yuchen He, Zhehan Lei, Jianan Shao, Chihao Zhang

**Categories:** cs.DS, cs.LG, math.PR, stat.ML

**Links:** [arXiv](http://arxiv.org/abs/2507.11236v2) | [PDF](http://arxiv.org/pdf/2507.11236v2)

**Abstract:** We study the problem of sampling from a distribution $\mu$ with density
$\propto e^{-V}$ for some potential function $V:\mathbb R^d\to \mathbb R$ with
query access to $V$ and $\nabla V$. We start with the following standard
assumptions:
  (1) The potential function $V$ is $L$-smooth.
  (2) The second moment $\mathbf{E}_{X\sim \mu}[\|X\|^2]\leq M$.
  Recently, He and Zhang (COLT'25) showed that the query complexity of sampling
from such distributions is at least
$\left(\frac{LM}{d\epsilon}\right)^{\Omega(d)}$ where $\epsilon$ is the desired
accuracy in total variation distance, and the Poincar\'e constant can be
arbitrarily large.
  Meanwhile, another common assumption in the study of diffusion based samplers
(see e.g., the work of Chen, Chewi, Li, Li, Salim and Zhang (ICLR'23))
strengthens the smoothness condition (1) to the following:
  (1*) The potential function of *every* distribution along the
Ornstein-Uhlenbeck process starting from $\mu$ is $L$-smooth.
  We show that under the assumptions (1*) and (2), the query complexity of
sampling from $\mu$ can be $\mathrm{poly}(L,d)\cdot
\left(\frac{Ld+M}{\epsilon^2}\right)^{\mathcal{O}(L+1)}$, which is polynomial
in $d$ and $\frac{1}{\epsilon}$ when $L=\mathcal{O}(1)$ and
$M=\mathrm{poly}(d)$. This improves the algorithm with quasi-polynomial query
complexity developed by Huang et al. (COLT'24). Our results imply that the
seemly moderate strengthening of the smoothness condition (1) to (1*) can lead
to an exponential gap in the query complexity of sampling algorithms.
  Moreover, we show that together with the assumption (1*) and the stronger
moment assumption that $\|X\|$ is $\lambda$-sub-Gaussian for $X\sim\mu$, the
Poincar\'e constant of $\mu$ is at most $\mathcal{O}(\lambda)^{2(L+1)}$. As an
application of our technique, we obtain improved estimate of the Poincar\'e
constant for mixture of Gaussians with the same covariance.

**Relevance Score:** 1.00

---

## 254. Towards understanding Accelerated Stein Variational Gradient Flow --   Analysis of Generalized Bilinear Kernels for Gaussian target distributions

**Authors:** Viktor Stein, Wuchen Li

**Categories:** math.OC, stat.ML, 46N10 (Primary) 46E22 94A15 37Lxx 37A50 (Secondary)

**Links:** [arXiv](http://arxiv.org/abs/2509.04008v1) | [PDF](http://arxiv.org/pdf/2509.04008v1)

**Abstract:** Stein variational gradient descent (SVGD) is a kernel-based and
non-parametric particle method for sampling from a target distribution, such as
in Bayesian inference and other machine learning tasks. Different from other
particle methods, SVGD does not require estimating the score, which is the
gradient of the log-density. However, in practice, SVGD can be slow compared to
score-estimation-based sampling algorithms. To design a fast and efficient
high-dimensional sampling algorithm with the advantages of SVGD, we introduce
accelerated SVGD (ASVGD), based on an accelerated gradient flow in a metric
space of probability densities following Nesterov's method. We then derive a
momentum-based discrete-time sampling algorithm, which evolves a set of
particles deterministically. To stabilize the particles' position update, we
also include a Wasserstein metric regularization. This paper extends the
conference version \cite{SL2025}. For the bilinear kernel and Gaussian target
distributions, we study the kernel parameter and damping parameters with an
optimal convergence rate of the proposed dynamics. This is achieved by
analyzing the linearized accelerated gradient flows at the equilibrium.
Interestingly, the optimal parameter is a constant, which does not depend on
the covariance of the target distribution. For the generalized kernel
functions, such as the Gaussian kernel, numerical examples with varied target
distributions demonstrate the effectiveness of ASVGD compared to SVGD and other
popular sampling methods. Furthermore, we show that in the setting of Bayesian
neural networks, ASVGD outperforms SVGD significantly in terms of
log-likelihood and total iteration times.

**Comment:** 46 pages, 4 figures, 4 algorithms, 4 tables, comments welcome!

**Relevance Score:** 1.00

---

## 255. Simulation-based Inference via Langevin Dynamics with Score Matching

**Authors:** Haoyu Jiang, Yuexi Wang, Yun Yang

**Categories:** stat.ME, stat.CO, stat.ML

**Links:** [arXiv](http://arxiv.org/abs/2509.03853v1) | [PDF](http://arxiv.org/pdf/2509.03853v1)

**Abstract:** Simulation-based inference (SBI) enables Bayesian analysis when the
likelihood is intractable but model simulations are available. Recent advances
in statistics and machine learning, including Approximate Bayesian Computation
and deep generative models, have expanded the applicability of SBI, yet these
methods often face challenges in moderate to high-dimensional parameter spaces.
Motivated by the success of gradient-based Monte Carlo methods in Bayesian
sampling, we propose a novel SBI method that integrates score matching with
Langevin dynamics to explore complex posterior landscapes more efficiently in
such settings. Our approach introduces tailored score-matching procedures for
SBI, including a localization scheme that reduces simulation costs and an
architectural regularization that embeds the statistical structure of
log-likelihood scores to improve score-matching accuracy. We provide
theoretical analysis of the method and illustrate its practical benefits on
benchmark tasks and on more challenging problems in moderate to high
dimensions, where it performs favorably compared to existing approaches.

**Relevance Score:** 1.00

---

## 256. Estimation of High-Dimensional Markov-Switching VAR Models with an   Approximate EM Algorithm

**Authors:** Xiudi Li, Abolfazl Safikhani, Ali Shojaie

**Categories:** stat.ME, stat.ML

**Links:** [arXiv](http://arxiv.org/abs/2210.07456v3) | [PDF](http://arxiv.org/pdf/2210.07456v3)

**Abstract:** Regime shifts in high-dimensional time series arise naturally in many
applications, from neuroimaging to finance. This problem has received
considerable attention in low-dimensional settings, with both Bayesian and
frequentist methods used extensively for parameter estimation. The EM algorithm
is a particularly popular strategy for parameter estimation in low-dimensional
settings, although the statistical properties of the resulting estimates have
not been well understood. Furthermore, its extension to high-dimensional time
series has proved challenging. To overcome these challenges, in this paper we
propose an approximate EM algorithm for Markov-switching VAR models that leads
to efficient computation and also facilitates the investigation of asymptotic
properties of the resulting parameter estimates. We establish the consistency
of the proposed EM algorithm in high dimensions and investigate its performance
via simulation studies. We also demonstrate the algorithm by analyzing a brain
electroencephalography (EEG) dataset recorded on a patient experiencing
epileptic seizure.

**Relevance Score:** 1.00

---

## 257. HITTER: A HumanoId Table TEnnis Robot via Hierarchical Planning and   Learning

**Authors:** Zhi Su, Bike Zhang, Nima Rahmanian, Yuman Gao, Qiayuan Liao, Caitlin Regan, Koushil Sreenath, S. Shankar Sastry

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2508.21043v2) | [PDF](http://arxiv.org/pdf/2508.21043v2)

**Abstract:** Humanoid robots have recently achieved impressive progress in locomotion and
whole-body control, yet they remain constrained in tasks that demand rapid
interaction with dynamic environments through manipulation. Table tennis
exemplifies such a challenge: with ball speeds exceeding 5 m/s, players must
perceive, predict, and act within sub-second reaction times, requiring both
agility and precision. To address this, we present a hierarchical framework for
humanoid table tennis that integrates a model-based planner for ball trajectory
prediction and racket target planning with a reinforcement learning-based
whole-body controller. The planner determines striking position, velocity and
timing, while the controller generates coordinated arm and leg motions that
mimic human strikes and maintain stability and agility across consecutive
rallies. Moreover, to encourage natural movements, human motion references are
incorporated during training. We validate our system on a general-purpose
humanoid robot, achieving up to 106 consecutive shots with a human opponent and
sustained exchanges against another humanoid. These results demonstrate
real-world humanoid table tennis with sub-second reactive control, marking a
step toward agile and interactive humanoid behaviors.

**Comment:** add more references

**Relevance Score:** 1.00

---

## 258. Classification of Vision-Based Tactile Sensors: A Review

**Authors:** Haoran Li, Yijiong Lin, Chenghua Lu, Max Yang, Efi Psomopoulou, Nathan F Lepora

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.02478v2) | [PDF](http://arxiv.org/pdf/2509.02478v2)

**Abstract:** Vision-based tactile sensors (VBTS) have gained widespread application in
robotic hands, grippers and prosthetics due to their high spatial resolution,
low manufacturing costs, and ease of customization. While VBTSs have common
design features, such as a camera module, they can differ in a rich diversity
of sensing principles, material compositions, multimodal approaches, and data
interpretation methods. Here, we propose a novel classification of VBTS that
categorizes the technology into two primary sensing principles based on the
underlying transduction of contact into a tactile image: the Marker-Based
Transduction Principle and the Intensity-Based Transduction Principle.
Marker-Based Transduction interprets tactile information by detecting marker
displacement and changes in marker density. In contrast, Intensity-Based
Transduction maps external disturbances with variations in pixel values.
Depending on the design of the contact module, Marker-Based Transduction can be
further divided into two subtypes: Simple Marker-Based (SMB) and Morphological
Marker-Based (MMB) mechanisms. Similarly, the Intensity-Based Transduction
Principle encompasses the Reflective Layer-based (RLB) and Transparent
Layer-Based (TLB) mechanisms. This paper provides a comparative study of the
hardware characteristics of these four types of sensors including various
combination types, and discusses the commonly used methods for interpreting
tactile information. This~comparison reveals some current challenges faced by
VBTS technology and directions for future research.

**Comment:** 15 pages

**Relevance Score:** 1.00

---

## 259. FPC-VLA: A Vision-Language-Action Framework with a Supervisor for   Failure Prediction and Correction

**Authors:** Yifan Yang, Zhixiang Duan, Tianshi Xie, Fuyu Cao, Pinxi Shen, Peili Song, Piaopiao Jin, Guokang Sun, Shaoqing Xu, Yangwei You, Jingtai Liu

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04018v1) | [PDF](http://arxiv.org/pdf/2509.04018v1)

**Abstract:** Robotic manipulation is a fundamental component of automation. However,
traditional perception-planning pipelines often fall short in open-ended tasks
due to limited flexibility, while the architecture of a single end-to-end
Vision-Language-Action (VLA) offers promising capabilities but lacks crucial
mechanisms for anticipating and recovering from failure. To address these
challenges, we propose FPC-VLA, a dual-model framework that integrates VLA with
a supervisor for failure prediction and correction. The supervisor evaluates
action viability through vision-language queries and generates corrective
strategies when risks arise, trained efficiently without manual labeling. A
similarity-guided fusion module further refines actions by leveraging past
predictions. Evaluation results on multiple simulation platforms (SIMPLER and
LIBERO) and robot embodiments (WidowX, Google Robot, Franka) show that FPC-VLA
outperforms state-of-the-art models in both zero-shot and fine-tuned settings.
By activating the supervisor only at keyframes, our approach significantly
increases task success rates with minimal impact on execution time. Successful
real-world deployments on diverse, long-horizon tasks confirm FPC-VLA's strong
generalization and practical utility for building more reliable autonomous
systems.

**Relevance Score:** 1.00

---

## 260. Odometry Calibration and Pose Estimation of a 4WIS4WID Mobile Wall   Climbing Robot

**Authors:** Branimir Ćaran, Vladimir Milić, Marko Švaco, Bojan Jerbić

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04016v1) | [PDF](http://arxiv.org/pdf/2509.04016v1)

**Abstract:** This paper presents the design of a pose estimator for a four wheel
independent steer four wheel independent drive (4WIS4WID) wall climbing mobile
robot, based on the fusion of multimodal measurements, including wheel
odometry, visual odometry, and an inertial measurement unit (IMU) data using
Extended Kalman Filter (EKF) and Unscented Kalman Filter (UKF). The pose
estimator is a critical component of wall climbing mobile robots, as their
operational environment involves carrying precise measurement equipment and
maintenance tools in construction, requiring information about pose on the
building at the time of measurement. Due to the complex geometry and material
properties of building facades, the use of traditional localization sensors
such as laser, ultrasonic, or radar is often infeasible for wall-climbing
robots. Moreover, GPS-based localization is generally unreliable in these
environments because of signal degradation caused by reinforced concrete and
electromagnetic interference. Consequently, robot odometry remains the primary
source of velocity and position information, despite being susceptible to drift
caused by both systematic and non-systematic errors. The calibrations of the
robot's systematic parameters were conducted using nonlinear optimization and
Levenberg-Marquardt methods as Newton-Gauss and gradient-based model fitting
methods, while Genetic algorithm and Particle swarm were used as
stochastic-based methods for kinematic parameter calibration. Performance and
results of the calibration methods and pose estimators were validated in detail
with experiments on the experimental mobile wall climbing robot.

**Comment:** ACCEPTED FOR IEEE EUROPEAN CONFERENCE ON MOBILE ROBOTS 2025. PREPRINT
  VERSION. ACCEPTED JUNE, 2025 AND PRESENTED SEPTEMBER, 2025

**Relevance Score:** 1.00

---

## 261. Emancipatory Information Retrieval

**Authors:** Bhaskar Mitra

**Categories:** cs.IR, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2501.19241v5) | [PDF](http://arxiv.org/pdf/2501.19241v5)

**Abstract:** Our world today is facing a confluence of several mutually reinforcing crises
each of which intersects with concerns of social justice and emancipation. This
paper is a provocation for the role of computer-mediated information access in
our emancipatory struggles. We define emancipatory information retrieval as the
study and development of information access methods that challenge various
forms of human oppression, and situates its activities within broader
collective emancipatory praxis. The term "emancipatory" here signifies the
moral concerns of universal humanization of all peoples and the elimination of
oppression to create the conditions under which we can collectively flourish.
To develop an emancipatory research agenda for information retrieval (IR), in
this paper we speculate about the practices that the community can adopt,
enumerate some of the projects that the field should undertake, and discuss
provocations to spark new ideas and directions for research. We challenge the
field of IR research to embrace humanistic values and commit to universal
emancipation and social justice. We also invite scholars from fields such as
human-computer interaction, information sciences, media studies, design,
science and technology studies, social and political sciences, philosophy, law,
environmental sciences, public health, educational sciences, as well as legal
and policy experts, civil rights advocates, social justice activists and
movement organizers, and artists to join us in realizing this transformation.
In this process, we must both imagine post-oppressive worlds, and reimagine the
role of IR in that world and in the journey that leads us there.

**Relevance Score:** 1.00

---

## 262. Would I regret being different? The influence of social norms on   attitudes toward AI usage

**Authors:** Jaroslaw Kornowicz, Maurice Pape, Kirsten Thommes

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04241v1) | [PDF](http://arxiv.org/pdf/2509.04241v1)

**Abstract:** Prior research shows that social norms can reduce algorithm aversion, but
little is known about how such norms become established. Most accounts
emphasize technological and individual determinants, yet AI adoption unfolds
within organizational social contexts shaped by peers and supervisors. We ask
whether the source of the norm-peers or supervisors-shapes AI usage behavior.
This question is practically relevant for organizations seeking to promote
effective AI adoption. We conducted an online vignette experiment, complemented
by qualitative data on participants' feelings and justifications after
(counter-)normative behavior. In line with the theory, counter-normative
choices elicited higher regret than norm-adherent choices. On average, choosing
AI increased regret compared to choosing an human. This aversion was weaker
when AI use was presented as the prevailing norm, indicating a statistically
significant interaction between AI use and an AI-favoring norm. Participants
also attributed less blame to technology than to humans, which increased regret
when AI was chosen over human expertise. Both peer and supervisor influence
emerged as relevant factors, though contrary to expectations they did not
significantly affect regret. Our findings suggest that regret aversion,
embedded in social norms, is a central mechanism driving imitation in
AI-related decision-making.

**Comment:** 30 pages, 5 figures

**Relevance Score:** 1.00

---

## 263. The MolecularWeb Universe: Web-Based, Immersive, Multiuser Molecular   Graphics And Modeling, for Education and Work in Chemistry, Structural   Biology, and Materials Sciences

**Authors:** Luciano A. Abriata

**Categories:** physics.chem-ph, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.04056v1) | [PDF](http://arxiv.org/pdf/2509.04056v1)

**Abstract:** Molecular visualization software has long supported research and education in
chemical and structural sciences, but consumer devices constrained to 2D inputs
and outputs pose two major challenges: they poorly convey 3D nature, and 3D
manipulation is very difficult. eXtended Reality (XR, including AR and VR)
offers new ways to see and interact with molecules in three dimensions. This
chapter presents the "MolecularWeb" ecosystem (https://molecularweb.org), a set
of web-based tools for immersive visualization, modeling, and simulations,
already widely used in education and science communication and now expanding
toward research applications. We cover moleculARweb, which provides AR
educational activities via phones, tablets, and computers; MolecularWebXR, a
multiuser WebXR platform accessible from both headsets and simpler devices,
supporting immersive education, outreach, and scientific discussion; and
PDB2AR, which enables users to generate custom content for MolecularWebXR and
standalone AR/VR. Finally, we introduce a prototype and an upcoming version of
HandMol, our latest WebXR software which allows concurrent multiuser immersive
visualization and modeling of molecules with bare hands supported by real-time
molecular mechanics, natural language input via a language model, and access
through both high-end headsets or consumer devices like smartphones and
laptops. Together, these tools demonstrate the present and near-future of
accessible, interactive molecular science on the web.

**Comment:** 37 pages, 7 figures

**Relevance Score:** 1.00

---

## 264. Beyond-Voice: Towards Continuous 3D Hand Pose Tracking on Commercial   Home Assistant Devices

**Authors:** Yin Li, Rohan Reddy, Cheng Zhang, Rajalakshmi Nandakumar

**Categories:** cs.SD, cs.HC, eess.AS

**Links:** [arXiv](http://arxiv.org/abs/2306.17477v3) | [PDF](http://arxiv.org/pdf/2306.17477v3)

**Abstract:** The surging popularity of home assistants and their voice user interface
(VUI) have made them an ideal central control hub for smart home devices.
However, current form factors heavily rely on VUI, which poses accessibility
and usability issues; some latest ones are equipped with additional cameras and
displays, which are costly and raise privacy concerns. These concerns jointly
motivate Beyond-Voice, a novel high-fidelity acoustic sensing system that
allows commodity home assistant devices to track and reconstruct hand poses
continuously. It transforms the home assistant into an active sonar system
using its existing onboard microphones and speakers. We feed a high-resolution
range profile to the deep learning model that can analyze the motions of
multiple body parts and predict the 3D positions of 21 finger joints, bringing
the granularity for acoustic hand tracking to the next level. It operates
across different environments and users without the need for personalized
training data. A user study with 11 participants in 3 different environments
shows that Beyond-Voice can track joints with an average mean absolute error of
16.47mm without any training data provided by the testing subject.

**Comment:** Accepted by IPSN 2024

**Relevance Score:** 1.00

---

## 265. Towards an Understanding of Developer Experience-Driven Transparency in   Software Ecosystems

**Authors:** Rodrigo Oliveira Zacarias, Rodrigo Pereira dos Santos, Patricia Lago

**Categories:** cs.SE, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.03848v1) | [PDF](http://arxiv.org/pdf/2509.03848v1)

**Abstract:** Software ecosystems (SECO) have become a dominant paradigm in the software
industry, enabling third-party developers to co-create value through
complementary components and services. While Developer Experience (DX) is
increasingly recognized as critical for sustainable SECO, transparency remains
an underexplored factor shaping how developers perceive and interact with
ecosystems. Existing studies acknowledge transparency as essential for trust,
fairness, and engagement, yet its relationship with DX has not been
systematically conceptualized. Hence, this work aims to advance the
understanding of transparency in SECO from a developer-centered perspective. To
this end, we propose SECO-TransDX (Transparency in Software Ecosystems from a
Developer Experience Perspective), a conceptual model that introduces the
notion of DX-driven transparency. The model identifies 63 interrelated
concepts, including conditioning factors, ecosystem procedures, artifacts, and
relational dynamics that influence how transparency is perceived and
constructed during developer interactions. SECO-TransDX was built upon prior
research and refined through a Delphi study with experts from academia and
industry. It offers a structured lens to examine how transparency mediates DX
across technical, social, and organizational layers. For researchers, it lays
the groundwork for future studies and tool development; for practitioners, it
supports the design of trustworthy, developer-centered platforms that improve
transparency and foster long-term engagement in SECO.

**Comment:** 36 pages Submitted to the ACM Transactions on Software Engineering
  and Methodology. 2025

**Relevance Score:** 1.00

---

## 266. Exploring the Integration of Extended Reality and Artificial   Intelligence (AI) for Remote STEM Education and Assessment

**Authors:** Shadeeb Hossain, Natalie Sommer, Neda Adib

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.03812v1) | [PDF](http://arxiv.org/pdf/2509.03812v1)

**Abstract:** This paper presents a dynamic gamification architecture for an Extended
Reality Artificial Intelligence virtual training environment designed to
enhance STEM education through immersive adaptive, and kinesthetic learning.
The proposed system can be introduced in four phases: Introduction Phase,
Component Development Phase, Fault Introduction and Correction Phase and
Generative AI XR scenarios Phase. Security and privacy are discussed via a
defense-in-depth approach spanning client, middleware, and backend layers,
incorporating AES 256 encryption, multi-factor authentication, role-based
access control and GDPR or FERPA compliance. Risks such as sensor exploitation,
perceptual manipulation, and virtual physical harm are identified, with
mitigation strategies embedded at the design stage. Potential barriers to large
scale adoption-including technical complexity, cost of deployment, and need for
cybersecurity expertise are discussed.

**Comment:** 9 pages, 5 figures, 1 table

**Relevance Score:** 1.00

---

## 267. DEXOP: A Device for Robotic Transfer of Dexterous Human Manipulation

**Authors:** Hao-Shu Fang, Branden Romero, Yichen Xie, Arthur Hu, Bo-Ruei Huang, Juan Alvarez, Matthew Kim, Gabriel Margolis, Kavya Anbarasu, Masayoshi Tomizuka, Edward Adelson, Pulkit Agrawal

**Categories:** cs.CV, cs.HC, cs.AI, cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04441v1) | [PDF](http://arxiv.org/pdf/2509.04441v1)

**Abstract:** We introduce perioperation, a paradigm for robotic data collection that
sensorizes and records human manipulation while maximizing the transferability
of the data to real robots. We implement this paradigm in DEXOP, a passive hand
exoskeleton designed to maximize human ability to collect rich sensory (vision
+ tactile) data for diverse dexterous manipulation tasks in natural
environments. DEXOP mechanically connects human fingers to robot fingers,
providing users with direct contact feedback (via proprioception) and mirrors
the human hand pose to the passive robot hand to maximize the transfer of
demonstrated skills to the robot. The force feedback and pose mirroring make
task demonstrations more natural for humans compared to teleoperation,
increasing both speed and accuracy. We evaluate DEXOP across a range of
dexterous, contact-rich tasks, demonstrating its ability to collect
high-quality demonstration data at scale. Policies learned with DEXOP data
significantly improve task performance per unit time of data collection
compared to teleoperation, making DEXOP a powerful tool for advancing robot
dexterity. Our project page is at https://dex-op.github.io.

**Comment:** project page: https://dex-op.github.io

**Relevance Score:** 0.50

---

## 268. PARCO: Phoneme-Augmented Robust Contextual ASR via Contrastive Entity   Disambiguation

**Authors:** Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda

**Categories:** cs.CL, cs.SD, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04357v1) | [PDF](http://arxiv.org/pdf/2509.04357v1)

**Abstract:** Automatic speech recognition (ASR) systems struggle with domain-specific
named entities, especially homophones. Contextual ASR improves recognition but
often fails to capture fine-grained phoneme variations due to limited entity
diversity. Moreover, prior methods treat entities as independent tokens,
leading to incomplete multi-token biasing. To address these issues, we propose
Phoneme-Augmented Robust Contextual ASR via COntrastive entity disambiguation
(PARCO), which integrates phoneme-aware encoding, contrastive entity
disambiguation, entity-level supervision, and hierarchical entity filtering.
These components enhance phonetic discrimination, ensure complete entity
retrieval, and reduce false positives under uncertainty. Experiments show that
PARCO achieves CER of 4.22% on Chinese AISHELL-1 and WER of 11.14% on English
DATA2 under 1,000 distractors, significantly outperforming baselines. PARCO
also demonstrates robust gains on out-of-domain datasets like THCHS-30 and
LibriSpeech.

**Comment:** Accepted by ASRU 2025

**Relevance Score:** 0.50

---

## 269. Decoupled Entity Representation Learning for Pinterest Ads Ranking

**Authors:** Jie Liu, Yinrui Li, Jiankai Sun, Kungang Li, Han Sun, Sihan Wang, Huasen Wu, Siyuan Gao, Paulo Soares, Nan Li, Zhifang Liu, Haoyang Li, Siping Ji, Ling Leng, Prathibha Deshikachar

**Categories:** cs.IR, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04337v1) | [PDF](http://arxiv.org/pdf/2509.04337v1)

**Abstract:** In this paper, we introduce a novel framework following an
upstream-downstream paradigm to construct user and item (Pin) embeddings from
diverse data sources, which are essential for Pinterest to deliver personalized
Pins and ads effectively. Our upstream models are trained on extensive data
sources featuring varied signals, utilizing complex architectures to capture
intricate relationships between users and Pins on Pinterest. To ensure
scalability of the upstream models, entity embeddings are learned, and
regularly refreshed, rather than real-time computation, allowing for
asynchronous interaction between the upstream and downstream models. These
embeddings are then integrated as input features in numerous downstream tasks,
including ad retrieval and ranking models for CTR and CVR predictions. We
demonstrate that our framework achieves notable performance improvements in
both offline and online settings across various downstream tasks. This
framework has been deployed in Pinterest's production ad ranking systems,
resulting in significant gains in online metrics.

**Relevance Score:** 0.50

---

## 270. Autonomation, Not Automation: Activities and Needs of European   Fact-checkers as a Basis for Designing Human-Centered AI Systems

**Authors:** Andrea Hrckova, Robert Moro, Ivan Srba, Jakub Simko, Maria Bielikova

**Categories:** cs.HC, cs.AI, cs.CY

**Links:** [arXiv](http://arxiv.org/abs/2211.12143v3) | [PDF](http://arxiv.org/pdf/2211.12143v3)

**Abstract:** To mitigate the negative effects of false information more effectively, the
development of Artificial Intelligence (AI) systems to assist fact-checkers is
needed. Nevertheless, the lack of focus on the needs of these stakeholders
results in their limited acceptance and skepticism toward automating the whole
fact-checking process. In this study, we conducted semi-structured in-depth
interviews with Central European fact-checkers. Their activities and problems
were analyzed using iterative content analysis. The most significant problems
were validated with a survey of European fact-checkers, in which we collected
24 responses from 20 countries, i.e., 62% of active European signatories of the
International Fact-Checking Network (IFCN). Our contributions include an
in-depth examination of the variability of fact-checking work in
non-English-speaking regions, which still remained largely uncovered. By
aligning them with the knowledge from prior studies, we created conceptual
models that help to understand the fact-checking processes. In addition, we
mapped our findings on the fact-checkers' activities and needs to the relevant
tasks for AI research, while providing a discussion on three AI tasks that were
not covered by previous similar studies. The new opportunities identified for
AI researchers and developers have implications for the focus of AI research in
this domain.

**Comment:** 44 pages, 13 figures, 2 annexes. Accepted to ACM Journal on
  Responsible Computing

**Relevance Score:** 0.50

---

## 271. The KG-ER Conceptual Schema Language

**Authors:** Enrico Franconi, Benoît Groz, Jan Hidders, Nina Pardal, Sławek Staworko, Jan Van den Bussche, Piotr Wieczorek

**Categories:** cs.DB, cs.AI, 68P15

**Links:** [arXiv](http://arxiv.org/abs/2508.02548v2) | [PDF](http://arxiv.org/pdf/2508.02548v2)

**Abstract:** We propose KG-ER, a conceptual schema language for knowledge graphs that
describes the structure of knowledge graphs independently of their
representation (relational databases, property graphs, RDF) while helping to
capture the semantics of the information stored in a knowledge graph.

**Relevance Score:** 0.50

---

## 272. Image Embedding Sampling Method for Diverse Captioning

**Authors:** Sania Waheed, Na Min An

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2502.10118v2) | [PDF](http://arxiv.org/pdf/2502.10118v2)

**Abstract:** Image Captioning for state-of-the-art VLMs has significantly improved over
time; however, this comes at the cost of increased computational complexity,
making them less accessible for resource-constrained applications such as
mobile devices and assistive technologies. Alternatively, comparably smaller
VLMs prioritize high-level scene descriptions, overlooking finer details that
contribute to a richer understanding of an image. In this paper, we introduce a
training-free framework that enhances caption diversity and informativeness by
explicitly attending to distinct image regions using a comparably small VLM,
BLIP, as the backbone. Our approach leverages structured segmentation to
produce hierarchical representations that capture both global and localized
semantics. Without requiring additional model training, we demonstrate that our
method allows smaller VLMs to achieve performance comparable to larger models
in terms of image-caption alignment, semantic integrity, and diversity. We
evaluate our framework on MSCOCO, Flickr30k, and Nocaps test datasets,
achieving a Div-2 score of 0.735, 0.750, and 0.748 for each dataset,
respectively, while maintaining strong image-caption relevancy and semantic
integrity with the human-annotated captions.

**Comment:** 17 pages, 5 figures, 9 tables

**Relevance Score:** 0.50

---

## 273. Learning Active Perception via Self-Evolving Preference Optimization for   GUI Grounding

**Authors:** Wanfu Wang, Qipeng Huang, Guangquan Xue, Xiaobo Liang, Juntao Li

**Categories:** cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04243v1) | [PDF](http://arxiv.org/pdf/2509.04243v1)

**Abstract:** Vision Language Models (VLMs) have recently achieved significant progress in
bridging visual perception and linguistic reasoning. Recently, OpenAI o3 model
introduced a zoom-in search strategy that effectively elicits active perception
capabilities in VLMs, improving downstream task performance. However, enabling
VLMs to reason effectively over appropriate image regions remains a core
challenge in GUI grounding, particularly under high-resolution inputs and
complex multi-element visual interactions. In this work, we propose LASER, a
self-evolving framework that progressively endows VLMs with multi-step
perception capabilities, enabling precise coordinate prediction. Specifically,
our approach integrate Monte Carlo quality estimation with
Intersection-over-Union (IoU)-based region quality evaluation to jointly
encourage both accuracy and diversity in constructing high-quality preference
data. This combination explicitly guides the model to focus on
instruction-relevant key regions while adaptively allocating reasoning steps
based on task complexity. Comprehensive experiments on the ScreenSpot Pro and
ScreenSpot-v2 benchmarks demonstrate consistent performance gains, validating
the effectiveness of our method. Furthermore, when fine-tuned on GTA1-7B, LASER
achieves a score of 55.7 on the ScreenSpot-Pro benchmark, establishing a new
state-of-the-art (SoTA) among 7B-scale models.

**Relevance Score:** 0.50

---

## 274. AutoPETIII: The Tracer Frontier. What Frontier?

**Authors:** Zacharia Mesbah, Léo Mottay, Romain Modzelewski, Pierre Decazes, Sébastien Hapdey, Su Ruan, Sébastien Thureau

**Categories:** eess.IV, cs.AI, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2410.02807v2) | [PDF](http://arxiv.org/pdf/2410.02807v2)

**Abstract:** For the last three years, the AutoPET competition gathered the medical
imaging community around a hot topic: lesion segmentation on Positron Emitting
Tomography (PET) scans. Each year a different aspect of the problem is
presented; in 2024 the multiplicity of existing and used tracers was at the
core of the challenge. Specifically, this year's edition aims to develop a
fully automatic algorithm capable of performing lesion segmentation on a PET/CT
scan, without knowing the tracer, which can either be a FDG or PSMA-based
tracer. In this paper we describe how we used the nnUNetv2 framework to train
two sets of 6 fold ensembles of models to perform fully automatic PET/CT lesion
segmentation as well as a MIP-CNN to choose which set of models to use for
segmentation.

**Relevance Score:** 0.50

---

## 275. Domain size asymptotics for Markov logic networks

**Authors:** Vera Koponen

**Categories:** cs.AI, cs.LO, math.LO, 68T27, 68T30, 68T37, 03C13, I.2; F.4; G.3

**Links:** [arXiv](http://arxiv.org/abs/2509.04192v1) | [PDF](http://arxiv.org/pdf/2509.04192v1)

**Abstract:** A Markov logic network (MLN) determines a probability distribution on the set
of structures, or ``possible worlds'', with an arbitrary finite domain. We
study the properties of such distributions as the domain size tends to
infinity. Three types of concrete examples of MLNs will be considered, and the
properties of random structures with domain sizes tending to infinity will be
studied: (1) Arbitrary quantifier-free MLNs over a language with only one
relation symbol which has arity 1. In this case we give a pretty complete
characterization of the possible limit behaviours of random structures. (2) An
MLN that favours graphs with fewer triangles (or more generally, fewer
k-cliques). As a corollary of the analysis a ``$\delta$-approximate 0-1 law''
for first-order logic is obtained. (3) An MLN that favours graphs with fewer
vertices with degree higher than a fixed (but arbitrary) number. The analysis
shows that depending on which ``soft constraints'' an MLN uses the limit
behaviour of random structures can be quite different, and the weights of the
soft constraints may, or may not, have influence on the limit behaviour. It
will also be demonstrated, using (1), that quantifier-free MLNs and lifted
Bayesian networks (in a broad sense) are asymptotically incomparable, roughly
meaning that there is a sequence of distributions on possible worlds with
increasing domain sizes that can be defined by one of the formalisms but not
even approximated by the other. In a rather general context it is also shown
that on large domains the distribution determined by an MLN concentrates almost
all its probability mass on a totally different part of the space of possible
worlds than the uniform distribution does.

**Relevance Score:** 0.50

---

## 276. Towards an Action-Centric Ontology for Cooking Procedures Using Temporal   Graphs

**Authors:** Aarush Kumbhakern, Saransh Kumar Gupta, Lipika Dey, Partha Pratim Das

**Categories:** cs.CL, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04159v1) | [PDF](http://arxiv.org/pdf/2509.04159v1)

**Abstract:** Formalizing cooking procedures remains a challenging task due to their
inherent complexity and ambiguity. We introduce an extensible domain-specific
language for representing recipes as directed action graphs, capturing
processes, transfers, environments, concurrency, and compositional structure.
Our approach enables precise, modular modeling of complex culinary workflows.
Initial manual evaluation on a full English breakfast recipe demonstrates the
DSL's expressiveness and suitability for future automated recipe analysis and
execution. This work represents initial steps towards an action-centric
ontology for cooking, using temporal graphs to enable structured machine
understanding, precise interpretation, and scalable automation of culinary
processes - both in home kitchens and professional culinary settings.

**Comment:** 6 pages, 3 figures, 1 table, 11 references, ACM International
  Conference on Multimedia 2025 - Multi-modal Food Computing Workshop

**Relevance Score:** 0.50

---

## 277. The human biological advantage over AI

**Authors:** William Stewart

**Categories:** cs.AI, cs.CY, I.2.0

**Links:** [arXiv](http://arxiv.org/abs/2509.04130v1) | [PDF](http://arxiv.org/pdf/2509.04130v1)

**Abstract:** Recent advances in AI raise the possibility that AI systems will one day be
able to do anything humans can do, only better. If artificial general
intelligence (AGI) is achieved, AI systems may be able to understand, reason,
problem solve, create, and evolve at a level and speed that humans will
increasingly be unable to match, or even understand. These possibilities raise
a natural question as to whether AI will eventually become superior to humans,
a successor "digital species", with a rightful claim to assume leadership of
the universe. However, a deeper consideration suggests the overlooked
differentiator between human beings and AI is not the brain, but the central
nervous system (CNS), providing us with an immersive integration with physical
reality. It is our CNS that enables us to experience emotion including pain,
joy, suffering, and love, and therefore to fully appreciate the consequences of
our actions on the world around us. And that emotional understanding of the
consequences of our actions is what is required to be able to develop
sustainable ethical systems, and so be fully qualified to be the leaders of the
universe. A CNS cannot be manufactured or simulated; it must be grown as a
biological construct. And so, even the development of consciousness will not be
sufficient to make AI systems superior to humans. AI systems may become more
capable than humans on almost every measure and transform our society. However,
the best foundation for leadership of our universe will always be DNA, not
silicon.

**Comment:** 12 pages

**Relevance Score:** 0.50

---

## 278. Simplicity Lies in the Eye of the Beholder: A Strategic Perspective on   Controllers in Reactive Synthesis

**Authors:** Mickael Randour

**Categories:** cs.LO, cs.AI, cs.FL, math.PR

**Links:** [arXiv](http://arxiv.org/abs/2509.04129v1) | [PDF](http://arxiv.org/pdf/2509.04129v1)

**Abstract:** In the game-theoretic approach to controller synthesis, we model the
interaction between a system to be controlled and its environment as a game
between these entities, and we seek an appropriate (e.g., winning or optimal)
strategy for the system. This strategy then serves as a formal blueprint for a
real-world controller. A common belief is that simple (e.g., using limited
memory) strategies are better: corresponding controllers are easier to conceive
and understand, and cheaper to produce and maintain.
  This invited contribution focuses on the complexity of strategies in a
variety of synthesis contexts. We discuss recent results concerning memory and
randomness, and take a brief look at what lies beyond our traditional notions
of complexity for strategies.

**Comment:** Invited paper at RP 2025

**Relevance Score:** 0.50

---

## 279. Auto-Regressive vs Flow-Matching: a Comparative Study of Modeling   Paradigms for Text-to-Music Generation

**Authors:** Or Tal, Felix Kreuk, Yossi Adi

**Categories:** cs.SD, eess.AS, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2506.08570v3) | [PDF](http://arxiv.org/pdf/2506.08570v3)

**Abstract:** Recent progress in text-to-music generation has enabled models to synthesize
high-quality musical segments, full compositions, and even respond to
fine-grained control signals, e.g. chord progressions. State-of-the-art (SOTA)
systems differ significantly in many dimensions, such as training datasets,
modeling paradigms, and architectural choices. This diversity complicates
efforts to evaluate models fairly and identify which design choices influence
performance the most. While factors like data and architecture are important,
in this study we focus exclusively on the modeling paradigm. We conduct a
systematic empirical analysis to isolate its effects, offering insights into
associated trade-offs and emergent behaviors that can guide future
text-to-music generation systems. Specifically, we compare the two arguably
most common modeling paradigms: auto-regressive decoding and conditional
flow-matching. We conduct a controlled comparison by training all models from
scratch using identical datasets, training configurations, and similar backbone
architectures. Performance is evaluated across multiple axes, including
generation quality, robustness to inference configurations, scalability,
adherence to both textual and temporally aligned conditioning, and editing
capabilities in the form of audio inpainting. This comparative study sheds
light on distinct strengths and limitations of each paradigm, providing
actionable insights that can inform future architectural and training decisions
in the evolving landscape of text-to-music generation. Audio sampled examples
are available at: https://huggingface.co/spaces/ortal1602/ARvsFM

**Relevance Score:** 0.50

---

## 280. Neural Video Compression with In-Loop Contextual Filtering and   Out-of-Loop Reconstruction Enhancement

**Authors:** Yaojun Wu, Chaoyi Lin, Yiming Wang, Semih Esenlik, Zhaobin Zhang, Kai Zhang, Li Zhang

**Categories:** eess.IV, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.04051v1) | [PDF](http://arxiv.org/pdf/2509.04051v1)

**Abstract:** This paper explores the application of enhancement filtering techniques in
neural video compression. Specifically, we categorize these techniques into
in-loop contextual filtering and out-of-loop reconstruction enhancement based
on whether the enhanced representation affects the subsequent coding loop.
In-loop contextual filtering refines the temporal context by mitigating error
propagation during frame-by-frame encoding. However, its influence on both the
current and subsequent frames poses challenges in adaptively applying filtering
throughout the sequence. To address this, we introduce an adaptive coding
decision strategy that dynamically determines filtering application during
encoding. Additionally, out-of-loop reconstruction enhancement is employed to
refine the quality of reconstructed frames, providing a simple yet effective
improvement in coding efficiency. To the best of our knowledge, this work
presents the first systematic study of enhancement filtering in the context of
conditional-based neural video compression. Extensive experiments demonstrate a
7.71% reduction in bit rate compared to state-of-the-art neural video codecs,
validating the effectiveness of the proposed approach.

**Comment:** 9 pages, 8 figures, Accepted to ACMMM 2025

**Relevance Score:** 0.50

---

## 281. Oruga: An Avatar of Representational Systems Theory

**Authors:** Daniel Raggi, Gem Stapleton, Mateja Jamnik, Aaron Stockdill, Grecia Garcia Garcia, Peter C-H. Cheng

**Categories:** cs.AI, cs.LO, 68T30, 68T27, 03B35, I.2.4; I.2.3; F.4.1; F.4.3

**Links:** [arXiv](http://arxiv.org/abs/2509.04041v1) | [PDF](http://arxiv.org/pdf/2509.04041v1)

**Abstract:** Humans use representations flexibly. We draw diagrams, change representations
and exploit creative analogies across different domains. We want to harness
this kind of power and endow machines with it to make them more compatible with
human use. Previously we developed Representational Systems Theory (RST) to
study the structure and transformations of representations. In this paper we
present Oruga (caterpillar in Spanish; a symbol of transformation), an
implementation of various aspects of RST. Oruga consists of a core of data
structures corresponding to concepts in RST, a language for communicating with
the core, and an engine for producing transformations using a method we call
structure transfer. In this paper we present an overview of the core and
language of Oruga, with a brief example of the kind of transformation that
structure transfer can execute.

**Relevance Score:** 0.50

---

## 282. Axiomatics of Restricted Choices by Linear Orders of Sets with Minimum   as Fallback

**Authors:** Kai Sauerwald, Kenneth Skiba, Eduardo Fermé, Thomas Meyer

**Categories:** cs.AI, cs.LO, 03E99, 91B14, I.2.4

**Links:** [arXiv](http://arxiv.org/abs/2506.03315v2) | [PDF](http://arxiv.org/pdf/2506.03315v2)

**Abstract:** We study how linear orders can be employed to realise choice functions for
which the set of potential choices is restricted, i.e., the possible choice is
not possible among the full powerset of all alternatives. In such restricted
settings, constructing a choice function via a relation on the alternatives is
not always possible. However, we show that one can always construct a choice
function via a linear order on sets of alternatives, even when a fallback value
is encoded as the minimal element in the linear order. The axiomatics of such
choice functions are presented for the general case and the case of
union-closed input restrictions. Restricted choice structures have applications
in knowledge representation and reasoning, and here we discuss their
applications for theory change and abstract argumentation.

**Relevance Score:** 0.50

---

## 283. Structure Transfer: an Inference-Based Calculus for the Transformation   of Representations

**Authors:** Daniel Raggi, Gem Stapleton, Mateja Jamnik, Aaron Stockdill, Grecia Garcia Garcia, Peter C-H. Cheng

**Categories:** cs.LG, cs.AI, cs.LO, 68T30, 68T27, 03B35, I.2.4; I.2.3; F.4.1; F.4.3

**Links:** [arXiv](http://arxiv.org/abs/2509.03249v2) | [PDF](http://arxiv.org/pdf/2509.03249v2)

**Abstract:** Representation choice is of fundamental importance to our ability to
communicate and reason effectively. A major unsolved problem, addressed in this
paper, is how to devise representational-system (RS) agnostic techniques that
drive representation transformation and choice. We present a novel calculus,
called structure transfer, that enables representation transformation across
diverse RSs. Specifically, given a source representation drawn from a source
RS, the rules of structure transfer allow us to generate a target
representation for a target RS. The generality of structure transfer comes in
part from its ability to ensure that the source representation and the
generated target representation satisfy any specified relation (such as
semantic equivalence). This is done by exploiting schemas, which encode
knowledge about RSs. Specifically, schemas can express preservation of
information across relations between any pair of RSs, and this knowledge is
used by structure transfer to derive a structure for the target representation
which ensures that the desired relation holds. We formalise this using
Representational Systems Theory, building on the key concept of a construction
space. The abstract nature of construction spaces grants them the generality to
model RSs of diverse kinds, including formal languages, geometric figures and
diagrams, as well as informal notations. Consequently, structure transfer is a
system-agnostic calculus that can be used to identify alternative
representations in a wide range of practical settings.

**Relevance Score:** 0.50

---

## 284. Towards Cognitively-Faithful Decision-Making Models to Improve AI   Alignment

**Authors:** Cyrus Cousins, Vijay Keswani, Vincent Conitzer, Hoda Heidari, Jana Schaich Borg, Walter Sinnott-Armstrong

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04445v1) | [PDF](http://arxiv.org/pdf/2509.04445v1)

**Abstract:** Recent AI work trends towards incorporating human-centric objectives, with
the explicit goal of aligning AI models to personal preferences and societal
values. Using standard preference elicitation methods, researchers and
practitioners build models of human decisions and judgments, which are then
used to align AI behavior with that of humans. However, models commonly used in
such elicitation processes often do not capture the true cognitive processes of
human decision making, such as when people use heuristics to simplify
information associated with a decision problem. As a result, models learned
from people's decisions often do not align with their cognitive processes, and
can not be used to validate the learning framework for generalization to other
decision-making tasks. To address this limitation, we take an axiomatic
approach to learning cognitively faithful decision processes from pairwise
comparisons. Building on the vast literature characterizing the cognitive
processes that contribute to human decision-making, and recent work
characterizing such processes in pairwise comparison tasks, we define a class
of models in which individual features are first processed and compared across
alternatives, and then the processed features are then aggregated via a fixed
rule, such as the Bradley-Terry rule. This structured processing of information
ensures such models are realistic and feasible candidates to represent
underlying human decision-making processes. We demonstrate the efficacy of this
modeling approach in learning interpretable models of human decision making in
a kidney allocation task, and show that our proposed models match or surpass
the accuracy of prior models of human pairwise decision-making.

**Relevance Score:** 0.50

---

## 285. Echo State Networks as State-Space Models: A Systems Perspective

**Authors:** Pradeep Singh, Balasubramanian Raman

**Categories:** cs.LG, 93C10, 68T07, 93C05, 93E11, 93B30, 93B05, 93B07, 62M10, I.2.6; I.5.1; I.6.5; I.6.4; G.3

**Links:** [arXiv](http://arxiv.org/abs/2509.04422v1) | [PDF](http://arxiv.org/pdf/2509.04422v1)

**Abstract:** Echo State Networks (ESNs) are typically presented as efficient,
readout-trained recurrent models, yet their dynamics and design are often
guided by heuristics rather than first principles. We recast ESNs explicitly as
state-space models (SSMs), providing a unified systems-theoretic account that
links reservoir computing with classical identification and modern kernelized
SSMs. First, we show that the echo-state property is an instance of
input-to-state stability for a contractive nonlinear SSM and derive verifiable
conditions in terms of leak, spectral scaling, and activation Lipschitz
constants. Second, we develop two complementary mappings: (i) small-signal
linearizations that yield locally valid LTI SSMs with interpretable poles and
memory horizons; and (ii) lifted/Koopman random-feature expansions that render
the ESN a linear SSM in an augmented state, enabling transfer-function and
convolutional-kernel analyses. This perspective yields frequency-domain
characterizations of memory spectra and clarifies when ESNs emulate structured
SSM kernels. Third, we cast teacher forcing as state estimation and propose
Kalman/EKF-assisted readout learning, together with EM for hyperparameters
(leak, spectral radius, process/measurement noise) and a hybrid subspace
procedure for spectral shaping under contraction constraints.

**Comment:** 27 pages, 1 figure

**Relevance Score:** 0.50

---

## 286. Interpretable Clustering with Adaptive Heterogeneous Causal Structure   Learning in Mixed Observational Data

**Authors:** Wenrui Li, Qinghao Zhang, Xiaowo Wang

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04415v1) | [PDF](http://arxiv.org/pdf/2509.04415v1)

**Abstract:** Understanding causal heterogeneity is essential for scientific discovery in
domains such as biology and medicine. However, existing methods lack causal
awareness, with insufficient modeling of heterogeneity, confounding, and
observational constraints, leading to poor interpretability and difficulty
distinguishing true causal heterogeneity from spurious associations. We propose
an unsupervised framework, HCL (Interpretable Causal Mechanism-Aware Clustering
with Adaptive Heterogeneous Causal Structure Learning), that jointly infers
latent clusters and their associated causal structures from mixed-type
observational data without requiring temporal ordering, environment labels,
interventions or other prior knowledge. HCL relaxes the homogeneity and
sufficiency assumptions by introducing an equivalent representation that
encodes both structural heterogeneity and confounding. It further develops a
bi-directional iterative strategy to alternately refine causal clustering and
structure learning, along with a self-supervised regularization that balance
cross-cluster universality and specificity. Together, these components enable
convergence toward interpretable, heterogeneous causal patterns. Theoretically,
we show identifiability of heterogeneous causal structures under mild
conditions. Empirically, HCL achieves superior performance in both clustering
and structure learning tasks, and recovers biologically meaningful mechanisms
in real-world single-cell perturbation data, demonstrating its utility for
discovering interpretable, mechanism-level causal heterogeneity.

**Relevance Score:** 0.50

---

## 287. SAFE--MA--RRT: Multi-Agent Motion Planning with Data-Driven Safety   Certificates

**Authors:** Babak Esmaeili, Hamidreza Modares

**Categories:** cs.MA, cs.SY, math.OC, eess.SY, cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04413v1) | [PDF](http://arxiv.org/pdf/2509.04413v1)

**Abstract:** This paper proposes a fully data-driven motion-planning framework for
homogeneous linear multi-agent systems that operate in shared, obstacle-filled
workspaces without access to explicit system models. Each agent independently
learns its closed-loop behavior from experimental data by solving convex
semidefinite programs that generate locally invariant ellipsoids and
corresponding state-feedback gains. These ellipsoids, centered along grid-based
waypoints, certify the dynamic feasibility of short-range transitions and
define safe regions of operation. A sampling-based planner constructs a tree of
such waypoints, where transitions are allowed only when adjacent ellipsoids
overlap, ensuring invariant-to-invariant transitions and continuous safety. All
agents expand their trees simultaneously and are coordinated through a
space-time reservation table that guarantees inter-agent safety by preventing
simultaneous occupancy and head-on collisions. Each successful edge in the tree
is equipped with its own local controller, enabling execution without
re-solving optimization problems at runtime. The resulting trajectories are not
only dynamically feasible but also provably safe with respect to both
environmental constraints and inter-agent collisions. Simulation results
demonstrate the effectiveness of the approach in synthesizing synchronized,
safe trajectories for multiple agents under shared dynamics and constraints,
using only data and convex optimization tools.

**Comment:** Submitted to IEEE Transactions on Automation Science and Engineering

**Relevance Score:** 0.50

---

## 288. Closed-Loop Neural Operator-Based Observer of Traffic Density

**Authors:** Alice Harting, Karl Henrik Johansson, Matthieu Barreau

**Categories:** math.OC, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2504.04873v2) | [PDF](http://arxiv.org/pdf/2504.04873v2)

**Abstract:** We consider the problem of traffic density estimation with sparse
measurements from stationary roadside sensors. Our approach uses Fourier neural
operators to learn macroscopic traffic flow dynamics from high-fidelity data.
During inference, the operator functions as an open-loop predictor of traffic
evolution. To close the loop, we couple the open-loop operator with a
correction operator that combines the predicted density with sparse
measurements from the sensors. Simulations with the SUMO software indicate
that, compared to open-loop observers, the proposed closed-loop observer
exhibits classical closed-loop properties such as robustness to noise and
ultimate boundedness of the error. This shows the advantages of combining
learned physics with real-time corrections, and opens avenues for accurate,
efficient, and interpretable data-driven observers.

**Relevance Score:** 0.50

---

## 289. When three experiments are better than two: Avoiding intractable   correlated aleatoric uncertainty by leveraging a novel bias--variance   tradeoff

**Authors:** Paul Scherer, Andreas Kirsch, Jake P. Taylor-King

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04363v1) | [PDF](http://arxiv.org/pdf/2509.04363v1)

**Abstract:** Real-world experimental scenarios are characterized by the presence of
heteroskedastic aleatoric uncertainty, and this uncertainty can be correlated
in batched settings. The bias--variance tradeoff can be used to write the
expected mean squared error between a model distribution and a ground-truth
random variable as the sum of an epistemic uncertainty term, the bias squared,
and an aleatoric uncertainty term. We leverage this relationship to propose
novel active learning strategies that directly reduce the bias between
experimental rounds, considering model systems both with and without noise.
Finally, we investigate methods to leverage historical data in a quadratic
manner through the use of a novel cobias--covariance relationship, which
naturally proposes a mechanism for batching through an eigendecomposition
strategy. When our difference-based method leveraging the cobias--covariance
relationship is utilized in a batched setting (with a quadratic estimator), we
outperform a number of canonical methods including BALD and Least Confidence.

**Comment:** 16 pages, 5 figures

**Relevance Score:** 0.50

---

## 290. The Strong, Weak and Benign Goodhart's law. An independence-free and   paradigm-agnostic formalisation

**Authors:** Adrien Majka, El-Mahdi El-Mhamdi

**Categories:** math.ST, stat.ML, cs.LG, stat.TH

**Links:** [arXiv](http://arxiv.org/abs/2505.23445v2) | [PDF](http://arxiv.org/pdf/2505.23445v2)

**Abstract:** Goodhart's law is a famous adage in policy-making that states that ``When a
measure becomes a target, it ceases to be a good measure''. As machine learning
models and the optimisation capacity to train them grow, growing empirical
evidence reinforced the belief in the validity of this law without however
being formalised. Recently, a few attempts were made to formalise Goodhart's
law, either by categorising variants of it, or by looking at how optimising a
proxy metric affects the optimisation of an intended goal. In this work, we
alleviate the simplifying independence assumption, made in previous works, and
the assumption on the learning paradigm made in most of them, to study the
effect of the coupling between the proxy metric and the intended goal on
Goodhart's law. Our results show that in the case of light tailed goal and
light tailed discrepancy, dependence does not change the nature of Goodhart's
effect. However, in the light tailed goal and heavy tailed discrepancy case, we
exhibit an example where over-optimisation occurs at a rate inversely
proportional to the heavy tailedness of the discrepancy between the goal and
the metric. %

**Comment:** 32 pages, 1 figure

**Relevance Score:** 0.50

---

## 291. Using causal abstractions to accelerate decision-making in complex   bandit problems

**Authors:** Joel Dyer, Nicholas Bishop, Anisoara Calinescu, Michael Wooldridge, Fabio Massimo Zennaro

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04296v1) | [PDF](http://arxiv.org/pdf/2509.04296v1)

**Abstract:** Although real-world decision-making problems can often be encoded as causal
multi-armed bandits (CMABs) at different levels of abstraction, a general
methodology exploiting the information and computational advantages of each
abstraction level is missing. In this paper, we propose AT-UCB, an algorithm
which efficiently exploits shared information between CMAB problem instances
defined at different levels of abstraction. More specifically, AT-UCB leverages
causal abstraction (CA) theory to explore within a cheap-to-simulate and
coarse-grained CMAB instance, before employing the traditional upper confidence
bound (UCB) algorithm on a restricted set of potentially optimal actions in the
CMAB of interest, leading to significant reductions in cumulative regret when
compared to the classical UCB algorithm. We illustrate the advantages of AT-UCB
theoretically, through a novel upper bound on the cumulative regret, and
empirically, by applying AT-UCB to epidemiological simulators with varying
resolution and computational cost.

**Relevance Score:** 0.50

---

## 292. A Primer on Causal and Statistical Dataset Biases for Fair and Robust   Image Analysis

**Authors:** Charles Jones, Ben Glocker

**Categories:** stat.ML, cs.LG, cs.CY

**Links:** [arXiv](http://arxiv.org/abs/2509.04295v1) | [PDF](http://arxiv.org/pdf/2509.04295v1)

**Abstract:** Machine learning methods often fail when deployed in the real world. Worse
still, they fail in high-stakes situations and across socially sensitive lines.
These issues have a chilling effect on the adoption of machine learning methods
in settings such as medical diagnosis, where they are arguably best-placed to
provide benefits if safely deployed. In this primer, we introduce the causal
and statistical structures which induce failure in machine learning methods for
image analysis. We highlight two previously overlooked problems, which we call
the \textit{no fair lunch} problem and the \textit{subgroup separability}
problem. We elucidate why today's fair representation learning methods fail to
adequately solve them and propose potential paths forward for the field.

**Comment:** Excerpt from C. Jones' PhD thesis. Winner of the G-Research PhD prize
  2025

**Relevance Score:** 0.50

---

## 293. Pulling Back the Curtain on ReLU Networks

**Authors:** Maciej Satkiewicz

**Categories:** I.2.6; I.4.10, cs.NE, cs.CV, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2507.22832v3) | [PDF](http://arxiv.org/pdf/2507.22832v3)

**Abstract:** Since any ReLU network is piecewise affine, its hidden units can be
characterized by their pullbacks through the active subnetwork, i.e., by their
gradients (up to bias terms). However, gradients of deeper neurons are
notoriously misaligned, which obscures the network's internal representations.
We posit that models do align gradients with data, yet this is concealed by the
intrinsic noise of the ReLU hard gating. We validate this intuition by applying
soft gating in the backward pass only, reducing the local impact of weakly
excited neurons. The resulting modified gradients, which we call "excitation
pullbacks", exhibit striking perceptual alignment on a number of
ImageNet-pretrained architectures, while the rudimentary pixel-space gradient
ascent quickly produces easily interpretable input- and target-specific
features. Inspired by these findings, we formulate the "path stability"
hypothesis, claiming that the binary activation patterns largely stabilize
during training and get encoded in the pre-activation distribution of the final
model. When true, excitation pullbacks become aligned with the gradients of a
kernel machine that mainly determines the network's decision. This provides a
theoretical justification for the apparent faithfulness of the feature
attributions based on these pullbacks, potentially even leading to mechanistic
interpretability of deeper models. Incidentally, we give a possible explanation
for the effectiveness of Batch Normalization and Deep Features, together with a
novel perspective on the network's internal memory and generalization
properties. We release the code and an interactive app for easier exploration
of the excitation pullbacks.

**Comment:** 12 pages, 3-page appendix, 4 figures, preprint; v3 changes: changed
  title, improved abstract, expanded introduction, added section on
  implications of the path stability

**Relevance Score:** 0.50

---

## 294. Mitigating Message Imbalance in Fraud Detection with Dual-View Graph   Representation Learning

**Authors:** Yudan Song, Yuecen Wei, Yuhang Lu, Qingyun Sun, Minglai Shao, Li-e Wang, Chunming Hu, Xianxian Li, Xingcheng Fu

**Categories:** cs.LG, cs.SI

**Links:** [arXiv](http://arxiv.org/abs/2507.06469v3) | [PDF](http://arxiv.org/pdf/2507.06469v3)

**Abstract:** Graph representation learning has become a mainstream method for fraud
detection due to its strong expressive power, which focuses on enhancing node
representations through improved neighborhood knowledge capture. However, the
focus on local interactions leads to imbalanced transmission of global
topological information and increased risk of node-specific information being
overwhelmed during aggregation due to the imbalance between fraud and benign
nodes. In this paper, we first summarize the impact of topology and class
imbalance on downstream tasks in GNN-based fraud detection, as the problem of
imbalanced supervisory messages is caused by fraudsters' topological behavior
obfuscation and identity feature concealment. Based on statistical validation,
we propose a novel dual-view graph representation learning method to mitigate
Message imbalance in Fraud Detection (MimbFD). Specifically, we design a
topological message reachability module for high-quality node representation
learning to penetrate fraudsters' camouflage and alleviate insufficient
propagation. Then, we introduce a local confounding debiasing module to adjust
node representations, enhancing the stable association between node
representations and labels to balance the influence of different classes.
Finally, we conducted experiments on three public fraud datasets, and the
results demonstrate that MimbFD exhibits outstanding performance in fraud
detection.

**Relevance Score:** 0.50

---

## 295. Single-seed generation of Brownian paths and integrals for adaptive and   high order SDE solvers

**Authors:** Andraž Jelinčič, James Foster, Patrick Kidger

**Categories:** math.NA, cs.LG, cs.NA, math.PR, stat.CO

**Links:** [arXiv](http://arxiv.org/abs/2405.06464v5) | [PDF](http://arxiv.org/pdf/2405.06464v5)

**Abstract:** Despite the success of adaptive time-stepping in ODE simulation, it has so
far seen few applications for Stochastic Differential Equations (SDEs). To
simulate SDEs adaptively, methods such as the Virtual Brownian Tree (VBT) have
been developed, which can generate Brownian motion (BM) non-chronologically.
However, in most applications, knowing only the values of Brownian motion is
not enough to achieve a high order of convergence; for that, we must compute
time-integrals of BM such as $\int_s^t W_r \, dr$. With the aim of using high
order SDE solvers adaptively, we extend the VBT to generate these integrals of
BM in addition to the Brownian increments. A JAX-based implementation of our
construction is included in the popular Diffrax library
(https://github.com/patrick-kidger/diffrax).
  Since the entire Brownian path produced by VBT is uniquely determined by a
single PRNG seed, previously generated samples need not be stored, which
results in a constant memory footprint and enables experiment repeatability and
strong error estimation. Based on binary search, the VBT's time complexity is
logarithmic in the tolerance parameter $\varepsilon$. Unlike the original VBT
algorithm, which was only precise at some dyadic times, we prove that our
construction exactly matches the joint distribution of the Brownian motion and
its time integrals at any query times, provided they are at least $\varepsilon$
apart.
  We present two applications of adaptive high order solvers enabled by our new
VBT. Using adaptive solvers to simulate a high-volatility CIR model, we achieve
more than twice the convergence order of constant stepping. We apply an
adaptive third order underdamped or kinetic Langevin solver to an MCMC problem,
where our approach outperforms the No U-Turn Sampler, while using only a tenth
of its function evaluations.

**Relevance Score:** 0.50

---

## 296. Reservoir kernels and Volterra series

**Authors:** Lukas Gonon, Lyudmila Grigoryeva, Juan-Pablo Ortega

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2212.14641v2) | [PDF](http://arxiv.org/pdf/2212.14641v2)

**Abstract:** A universal kernel is constructed whose sections approximate any causal and
time-invariant filter in the fading memory category with inputs and outputs in
a finite-dimensional Euclidean space. This kernel is built using the reservoir
functional associated with a state-space representation of the Volterra series
expansion available for any analytic fading memory filter, and it is hence
called the Volterra reservoir kernel. Even though the state-space
representation and the corresponding reservoir feature map are defined on an
infinite-dimensional tensor algebra space, the kernel map is characterized by
explicit recursions that are readily computable for specific data sets when
employed in estimation problems using the representer theorem. The empirical
performance of the Volterra reservoir kernel is showcased and compared to other
standard static and sequential kernels in a multidimensional and highly
nonlinear learning task for the conditional covariances of financial asset
returns.

**Comment:** 11 pages, 2 tables

**Relevance Score:** 0.50

---

## 297. Revealing the empirical flexibility of gas units through deep clustering

**Authors:** Chiara Fusar Bassini, Alice Lixuan Xu, Jorge Sánchez Canales, Lion Hirth, Lynn H. Kaack

**Categories:** cs.CY, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2504.16943v2) | [PDF](http://arxiv.org/pdf/2504.16943v2)

**Abstract:** The flexibility of a power generation unit determines how quickly and often
it can ramp up or down. In energy models, it depends on assumptions on the
technical characteristics of the unit, such as its installed capacity or
turbine technology. In this paper, we learn the empirical flexibility of gas
units from their electricity generation, revealing how real-world limitations
can lead to substantial differences between units with similar technical
characteristics. Using a novel deep clustering approach, we transform 5 years
(2019-2023) of unit-level hourly generation data for 49 German units from 100
MWp of installed capacity into low-dimensional embeddings. Our unsupervised
approach identifies two clusters of peaker units (high flexibility) and two
clusters of non-peaker units (low flexibility). The estimated ramp rates of
non-peakers, which constitute half of the sample, display a low empirical
flexibility, comparable to coal units. Non-peakers, predominantly owned by
industry and municipal utilities, show limited response to low residual load
and negative prices, generating on average 1.3 GWh during those hours. As the
transition to renewables increases market variability, regulatory changes will
be needed to unlock this flexibility potential.

**Comment:** 19 pages, 4 figures, 3 tables

**Relevance Score:** 0.50

---

## 298. Why Can't I See My Clusters? A Precision-Recall Approach to   Dimensionality Reduction Validation

**Authors:** Diede P. M. van der Hoorn, Alessio Arleo, Fernando V. Paulovich

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04222v1) | [PDF](http://arxiv.org/pdf/2509.04222v1)

**Abstract:** Dimensionality Reduction (DR) is widely used for visualizing high-dimensional
data, often with the goal of revealing expected cluster structure. However,
such a structure may not always appear in the projections. Existing DR quality
metrics assess projection reliability (to some extent) or cluster structure
quality, but do not explain why expected structures are missing. Visual
Analytics solutions can help, but are often time-consuming due to the large
hyperparameter space. This paper addresses this problem by leveraging a recent
framework that divides the DR process into two phases: a relationship phase,
where similarity relationships are modeled, and a mapping phase, where the data
is projected accordingly. We introduce two supervised metrics, precision and
recall, to evaluate the relationship phase. These metrics quantify how well the
modeled relationships align with an expected cluster structure based on some
set of labels representing this structure. We illustrate their application
using t-SNE and UMAP, and validate the approach through various usage
scenarios. Our approach can guide hyperparameter tuning, uncover projection
artifacts, and determine if the expected structure is captured in the
relationships, making the DR process faster and more reliable.

**Relevance Score:** 0.50

---

## 299. Batched Stochastic Matching Bandits

**Authors:** Jung-hun Kim, Min-hwan Oh

**Categories:** stat.ML, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04194v1) | [PDF](http://arxiv.org/pdf/2509.04194v1)

**Abstract:** In this study, we introduce a novel bandit framework for stochastic matching
based on the Multi-nomial Logit (MNL) choice model. In our setting, $N$ agents
on one side are assigned to $K$ arms on the other side, where each arm
stochastically selects an agent from its assigned pool according to an unknown
preference and yields a corresponding reward. The objective is to minimize
regret by maximizing the cumulative revenue from successful matches across all
agents. This task requires solving a combinatorial optimization problem based
on estimated preferences, which is NP-hard and leads a naive approach to incur
a computational cost of $O(K^N)$ per round. To address this challenge, we
propose batched algorithms that limit the frequency of matching updates,
thereby reducing the amortized computational cost (i.e., the average cost per
round) to $O(1)$ while still achieving a regret bound of $\tilde{O}(\sqrt{T})$.

**Relevance Score:** 0.50

---

## 300. Unobtrusive In-Situ Measurement of Behavior Change by Deep Metric   Similarity Learning of Motion Patterns

**Authors:** Christian Merz, Lukas Schach, Marie Luisa Fiedler, Jean-Luc Lugrin, Carolin Wienrich, Marc Erich Latoschik

**Categories:** cs.HC, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04174v1) | [PDF](http://arxiv.org/pdf/2509.04174v1)

**Abstract:** This paper introduces an unobtrusive in-situ measurement method to detect
user behavior changes during arbitrary exposures in XR systems. Here, such
behavior changes are typically associated with the Proteus effect or bodily
affordances elicited by different avatars that the users embody in XR. We
present a biometric user model based on deep metric similarity learning, which
uses high-dimensional embeddings as reference vectors to identify behavior
changes of individual users. We evaluate our model against two alternative
approaches: a (non-learned) motion analysis based on central tendencies of
movement patterns and subjective post-exposure embodiment questionnaires
frequently used in various XR exposures. In a within-subject study,
participants performed a fruit collection task while embodying avatars of
different body heights (short, actual-height, and tall). Subjective assessments
confirmed the effective manipulation of perceived body schema, while the
(non-learned) objective analyses of head and hand movements revealed
significant differences across conditions. Our similarity learning model
trained on the motion data successfully identified the elicited behavior change
for various query and reference data pairings of the avatar conditions. The
approach has several advantages in comparison to existing methods: 1) In-situ
measurement without additional user input, 2) generalizable and scalable motion
analysis for various use cases, 3) user-specific analysis on the individual
level, and 4) with a trained model, users can be added and evaluated in real
time to study how avatar changes affect behavior.

**Relevance Score:** 0.50

---

## 301. Exposing Synthetic Speech: Model Attribution and Detection of   AI-generated Speech via Audio Fingerprints

**Authors:** Matías Pizarro, Mike Laszkiewicz, Shawkat Hesso, Dorothea Kolossa, Asja Fischer

**Categories:** eess.AS, cs.CR, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2411.14013v3) | [PDF](http://arxiv.org/pdf/2411.14013v3)

**Abstract:** As speech generation technologies continue to advance in quality and
accessibility, the risk of malicious use cases, including impersonation,
misinformation, and spoofing, increases rapidly. This work addresses this
threat by introducing a simple, training-free, yet effective approach for
detecting AI-generated speech and attributing it to its source model.
Specifically, we tackle three key tasks: (1) single-model attribution in an
open-world setting, where the goal is to determine whether a given audio sample
was generated by a specific target neural speech synthesis system (with access
only to data from that system); (2) multi-model attribution in a closed-world
setting, where the objective is to identify the generating system from a known
pool of candidates; and last but not least (3) detection of synthetic versus
real speech. Our approach leverages standardized average residuals-the
difference between an input audio signal and its filtered version using either
a low-pass filter or the EnCodec audio autoencoder. We demonstrate that these
residuals consistently capture artifacts introduced by diverse speech synthesis
systems, serving as distinctive, model-agnostic fingerprints for attribution.
Across extensive experiments, our approach achieves AUROC scores exceeding 99%
in most scenarios, evaluated on augmented benchmark datasets that pair real
speech with synthetic audio generated by multiple synthesis systems. In
addition, our robustness analysis underscores the method's ability to maintain
high performance even in the presence of moderate additive noise. Due to its
simplicity, efficiency, and strong generalization across speech synthesis
systems and languages, this technique offers a practical tool for digital
forensics and security applications.

**Relevance Score:** 0.50

---

## 302. Who Pays for Fairness? Rethinking Recourse under Social Burden

**Authors:** Ainhize Barrainkua, Giovanni De Toni, Jose Antonio Lozano, Novi Quadrianto

**Categories:** cs.LG, cs.CY

**Links:** [arXiv](http://arxiv.org/abs/2509.04128v1) | [PDF](http://arxiv.org/pdf/2509.04128v1)

**Abstract:** Machine learning based predictions are increasingly used in sensitive
decision-making applications that directly affect our lives. This has led to
extensive research into ensuring the fairness of classifiers. Beyond just fair
classification, emerging legislation now mandates that when a classifier
delivers a negative decision, it must also offer actionable steps an individual
can take to reverse that outcome. This concept is known as algorithmic
recourse. Nevertheless, many researchers have expressed concerns about the
fairness guarantees within the recourse process itself. In this work, we
provide a holistic theoretical characterization of unfairness in algorithmic
recourse, formally linking fairness guarantees in recourse and classification,
and highlighting limitations of the standard equal cost paradigm. We then
introduce a novel fairness framework based on social burden, along with a
practical algorithm (MISOB), broadly applicable under real-world conditions.
Empirical results on real-world datasets show that MISOB reduces the social
burden across all groups without compromising overall classifier accuracy.

**Relevance Score:** 0.50

---

## 303. Synthetic Counterfactual Labels for Efficient Conformal Counterfactual   Inference

**Authors:** Amirmohammad Farzaneh, Matteo Zecchin, Osvaldo Simeone

**Categories:** cs.LG, cs.IT, math.IT

**Links:** [arXiv](http://arxiv.org/abs/2509.04112v1) | [PDF](http://arxiv.org/pdf/2509.04112v1)

**Abstract:** This work addresses the problem of constructing reliable prediction intervals
for individual counterfactual outcomes. Existing conformal counterfactual
inference (CCI) methods provide marginal coverage guarantees but often produce
overly conservative intervals, particularly under treatment imbalance when
counterfactual samples are scarce. We introduce synthetic data-powered CCI
(SP-CCI), a new framework that augments the calibration set with synthetic
counterfactual labels generated by a pre-trained counterfactual model. To
ensure validity, SP-CCI incorporates synthetic samples into a conformal
calibration procedure based on risk-controlling prediction sets (RCPS) with a
debiasing step informed by prediction-powered inference (PPI). We prove that
SP-CCI achieves tighter prediction intervals while preserving marginal
coverage, with theoretical guarantees under both exact and approximate
importance weighting. Empirical results on different datasets confirm that
SP-CCI consistently reduces interval width compared to standard CCI across all
settings.

**Relevance Score:** 0.50

---

## 304. Gromov-Wasserstein and optimal transport: from assignment problems to   probabilistic numeric

**Authors:** Iman Seyedi, Antonio Candelieri, Enza Messina, Francesco Archetti

**Categories:** math.OC, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.04089v1) | [PDF](http://arxiv.org/pdf/2509.04089v1)

**Abstract:** The assignment problem, a cornerstone of operations research, seeks an
optimal one-to-one mapping between agents and tasks to minimize total cost.
This work traces its evolution from classical formulations and algorithms to
modern optimal transport (OT) theory, positioning the Quadratic Assignment
Problem (QAP) and related structural matching tasks within this framework. We
connect the linear assignment problem to Monge's transport problem,
Kantorovich's relaxation, and Wasserstein distances, then extend to cases where
source and target lie in different metric-measure spaces requiring
Gromov-Wasserstein (GW) distances. GW formulations, including the fused GW
variant that integrates structural and feature information, naturally address
QAP-like problems by optimizing alignment based on both intra-domain distances
and cross-domain attributes. Applications include graph matching, keypoint
correspondence, and feature-based assignments. We present exact solvers,
Genetic Algorithms (GA), and multiple GW variants, including a proposed
multi-initialization strategy (GW-MultiInit) that mitigates the risk of getting
stuck in local optima alongside entropic Sinkhorn-based approximations and
fused GW. Computational experiments on capacitated QAP instances show that
GW-MultiInit consistently achieves near-optimal solutions and scales
efficiently to large problems where exact methods become impractical, while
parameterized EGW and FGW variants provide flexible trade-offs between accuracy
and runtime. Our findings provide theoretical foundations, computational
insights, and practical guidelines for applying OT and GW methods to QAP and
other real-world matching problems, such as those in machine learning and
logistics.

**Relevance Score:** 0.50

---

## 305. One Small Step with Fingerprints, One Giant Leap for De Novo Molecule   Generation from Mass Spectra

**Authors:** Neng Kai Nigel Neo, Lim Jing, Ngoui Yong Zhau Preston, Koh Xue Ting Serene, Bingquan Shen

**Categories:** cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2508.04180v3) | [PDF](http://arxiv.org/pdf/2508.04180v3)

**Abstract:** A common approach to the de novo molecular generation problem from mass
spectra involves a two-stage pipeline: (1) encoding mass spectra into molecular
fingerprints, followed by (2) decoding these fingerprints into molecular
structures. In our work, we adopt MIST as the encoder and MolForge as the
decoder, leveraging additional training data to enhance performance. We also
threshold the probabilities of each fingerprint bit to focus on the presence of
substructures. This results in a tenfold improvement over previous
state-of-the-art methods, generating top-1 28% / top-10 36% of molecular
structures correctly from mass spectra in MassSpecGym. We position this as a
strong baseline for future research in de novo molecule elucidation from mass
spectra.

**Relevance Score:** 0.50

---

## 306. The Telephone Game: Evaluating Semantic Drift in Unified Models

**Authors:** Sabbir Mollah, Rohit Gupta, Sirnam Swetha, Qingyang Liu, Ahnaf Munir, Mubarak Shah

**Categories:** cs.CL, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04438v1) | [PDF](http://arxiv.org/pdf/2509.04438v1)

**Abstract:** Employing a single, unified model (UM) for both visual understanding
(image-to-text: I2T) and and visual generation (text-to-image: T2I) has opened
a new direction in Visual Language Model (VLM) research. While UMs can also
support broader unimodal tasks (e.g., text-to-text, image-to-image), we focus
on the core cross-modal pair T2I and I2T, as consistency between understanding
and generation is critical for downstream use. Existing evaluations consider
these capabilities in isolation: FID and GenEval for T2I, and benchmarks such
as MME, MMBench for I2T. These single-pass metrics do not reveal whether a
model that understands a concept can also render it, nor whether meaning is
preserved when cycling between image and text modalities. To address this, we
introduce the Unified Consistency Framework for Unified Models (UCF-UM), a
cyclic evaluation protocol that alternates I2T and T2I over multiple
generations to quantify semantic drift. UCF formulates 3 metrics: (i) Mean
Cumulative Drift (MCD), an embedding-based measure of overall semantic loss;
(ii) Semantic Drift Rate (SDR), that summarizes semantic decay rate; and (iii)
Multi-Generation GenEval (MGG), an object-level compliance score extending
GenEval. To assess generalization beyond COCO, which is widely used in
training; we create a new benchmark ND400, sampled from NoCaps and DOCCI and
evaluate on seven recent models. UCF-UM reveals substantial variation in
cross-modal stability: some models like BAGEL maintain semantics over many
alternations, whereas others like Vila-u drift quickly despite strong
single-pass scores. Our results highlight cyclic consistency as a necessary
complement to standard I2T and T2I evaluations, and provide practical metrics
to consistently assess unified model's cross-modal stability and strength of
their shared representations. Code:
https://github.com/mollahsabbir/Semantic-Drift-in-Unified-Models

**Relevance Score:** 0.50

---

## 307. From Lines to Shapes: Geometric-Constrained Segmentation of X-Ray   Collimators via Hough Transform

**Authors:** Benjamin El-Zein, Dominik Eckert, Andreas Fieselmann, Christopher Syben, Ludwig Ritschl, Steffen Kappler, Sebastian Stober

**Categories:** cs.CV, physics.med-ph

**Links:** [arXiv](http://arxiv.org/abs/2509.04437v1) | [PDF](http://arxiv.org/pdf/2509.04437v1)

**Abstract:** Collimation in X-ray imaging restricts exposure to the region-of-interest
(ROI) and minimizes the radiation dose applied to the patient. The detection of
collimator shadows is an essential image-based preprocessing step in digital
radiography posing a challenge when edges get obscured by scattered X-ray
radiation. Regardless, the prior knowledge that collimation forms
polygonal-shaped shadows is evident. For this reason, we introduce a deep
learning-based segmentation that is inherently constrained to its geometry. We
achieve this by incorporating a differentiable Hough transform-based network to
detect the collimation borders and enhance its capability to extract the
information about the ROI center. During inference, we combine the information
of both tasks to enable the generation of refined, line-constrained
segmentation masks. We demonstrate robust reconstruction of collimated regions
achieving median Hausdorff distances of 4.3-5.0mm on diverse test sets of real
Xray images. While this application involves at most four shadow borders, our
method is not fundamentally limited by a specific number of edges.

**Relevance Score:** 0.50

---

## 308. Sat-DN: Implicit Surface Reconstruction from Multi-View Satellite Images   with Depth and Normal Supervision

**Authors:** Tianle Liu, Shuangming Zhao, Wanshou Jiang, Bingxuan Guo

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2502.08352v2) | [PDF](http://arxiv.org/pdf/2502.08352v2)

**Abstract:** With advancements in satellite imaging technology, acquiring high-resolution
multi-view satellite imagery has become increasingly accessible, enabling rapid
and location-independent ground model reconstruction. However, traditional
stereo matching methods struggle to capture fine details, and while neural
radiance fields (NeRFs) achieve high-quality reconstructions, their training
time is prohibitively long. Moreover, challenges such as low visibility of
building facades, illumination and style differences between pixels, and weakly
textured regions in satellite imagery further make it hard to reconstruct
reasonable terrain geometry and detailed building facades. To address these
issues, we propose Sat-DN, a novel framework leveraging a progressively trained
multi-resolution hash grid reconstruction architecture with explicit depth
guidance and surface normal consistency constraints to enhance reconstruction
quality. The multi-resolution hash grid accelerates training, while the
progressive strategy incrementally increases the learning frequency, using
coarse low-frequency geometry to guide the reconstruction of fine
high-frequency details. The depth and normal constraints ensure a clear
building outline and correct planar distribution. Extensive experiments on the
DFC2019 dataset demonstrate that Sat-DN outperforms existing methods, achieving
state-of-the-art results in both qualitative and quantitative evaluations. The
code is available at https://github.com/costune/SatDN.

**Relevance Score:** 0.50

---

## 309. FADE: A Dataset for Detecting Falling Objects around Buildings in Video

**Authors:** Zhigang Tu, Zitao Gao, Zhengbo Zhang, Chunluan Zhou, Junsong Yuan, Bo Du

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2408.05750v2) | [PDF](http://arxiv.org/pdf/2408.05750v2)

**Abstract:** Falling objects from buildings can cause severe injuries to pedestrians due
to the great impact force they exert. Although surveillance cameras are
installed around some buildings, it is challenging for humans to capture such
events in surveillance videos due to the small size and fast motion of falling
objects, as well as the complex background. Therefore, it is necessary to
develop methods to automatically detect falling objects around buildings in
surveillance videos. To facilitate the investigation of falling object
detection, we propose a large, diverse video dataset called FADE (FAlling
Object DEtection around Buildings) for the first time. FADE contains 1,881
videos from 18 scenes, featuring 8 falling object categories, 4 weather
conditions, and 4 video resolutions. Additionally, we develop a new object
detection method called FADE-Net, which effectively leverages motion
information and produces small-sized but high-quality proposals for detecting
falling objects around buildings. Importantly, our method is extensively
evaluated and analyzed by comparing it with the previous approaches used for
generic object detection, video object detection, and moving object detection
on the FADE dataset. Experimental results show that the proposed FADE-Net
significantly outperforms other methods, providing an effective baseline for
future research. The dataset and code are publicly available at
https://fadedataset.github.io/FADE.github.io/.

**Comment:** Accepted by IEEE Transactions on Information Forensics and Security
  (TIFS), 2025

**Relevance Score:** 0.50

---

## 310. Global-to-Local or Local-to-Global? Enhancing Image Retrieval with   Efficient Local Search and Effective Global Re-ranking

**Authors:** Dror Aiger, Bingyi Cao, Kaifeng Chen, Andre Araujo

**Categories:** cs.IR, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04351v1) | [PDF](http://arxiv.org/pdf/2509.04351v1)

**Abstract:** The dominant paradigm in image retrieval systems today is to search large
databases using global image features, and re-rank those initial results with
local image feature matching techniques. This design, dubbed global-to-local,
stems from the computational cost of local matching approaches, which can only
be afforded for a small number of retrieved images. However, emerging efficient
local feature search approaches have opened up new possibilities, in particular
enabling detailed retrieval at large scale, to find partial matches which are
often missed by global feature search. In parallel, global feature-based
re-ranking has shown promising results with high computational efficiency. In
this work, we leverage these building blocks to introduce a local-to-global
retrieval paradigm, where efficient local feature search meets effective global
feature re-ranking. Critically, we propose a re-ranking method where global
features are computed on-the-fly, based on the local feature retrieval
similarities. Such re-ranking-only global features leverage multidimensional
scaling techniques to create embeddings which respect the local similarities
obtained during search, enabling a significant re-ranking boost.
Experimentally, we demonstrate solid retrieval performance, setting new
state-of-the-art results on the Revisited Oxford and Paris datasets.

**Relevance Score:** 0.50

---

## 311. GeoArena: An Open Platform for Benchmarking Large Vision-language Models   on WorldWide Image Geolocalization

**Authors:** Pengyue Jia, Yingyi Zhang, Xiangyu Zhao, Yixuan Li

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04334v1) | [PDF](http://arxiv.org/pdf/2509.04334v1)

**Abstract:** Image geolocalization aims to predict the geographic location of images
captured anywhere on Earth, but its global nature presents significant
challenges. Current evaluation methodologies suffer from two major limitations.
First, data leakage: advanced approaches often rely on large vision-language
models (LVLMs) to predict image locations, yet these models are frequently
pretrained on the test datasets, compromising the accuracy of evaluating a
model's actual geolocalization capability. Second, existing metrics primarily
rely on exact geographic coordinates to assess predictions, which not only
neglects the reasoning process but also raises privacy concerns when user-level
location data is required. To address these issues, we propose GeoArena, a
first open platform for evaluating LVLMs on worldwide image geolocalization
tasks, offering true in-the-wild and human-centered benchmarking. GeoArena
enables users to upload in-the-wild images for a more diverse evaluation
corpus, and it leverages pairwise human judgments to determine which model
output better aligns with human expectations. Our platform has been deployed
online for two months, during which we collected over thousands voting records.
Based on this data, we conduct a detailed analysis and establish a leaderboard
of different LVLMs on the image geolocalization task.

**Relevance Score:** 0.50

---

## 312. Noisy Label Refinement with Semantically Reliable Synthetic Images

**Authors:** Yingxuan Li, Jiafeng Mao, Yusuke Matsui

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04298v1) | [PDF](http://arxiv.org/pdf/2509.04298v1)

**Abstract:** Semantic noise in image classification datasets, where visually similar
categories are frequently mislabeled, poses a significant challenge to
conventional supervised learning approaches. In this paper, we explore the
potential of using synthetic images generated by advanced text-to-image models
to address this issue. Although these high-quality synthetic images come with
reliable labels, their direct application in training is limited by domain gaps
and diversity constraints. Unlike conventional approaches, we propose a novel
method that leverages synthetic images as reliable reference points to identify
and correct mislabeled samples in noisy datasets. Extensive experiments across
multiple benchmark datasets show that our approach significantly improves
classification accuracy under various noise conditions, especially in
challenging scenarios with semantic label noise. Additionally, since our method
is orthogonal to existing noise-robust learning techniques, when combined with
state-of-the-art noise-robust training methods, it achieves superior
performance, improving accuracy by 30% on CIFAR-10 and by 11% on CIFAR-100
under 70% semantic noise, and by 24% on ImageNet-100 under real-world noise
conditions.

**Comment:** Accepted to ICIP2025

**Relevance Score:** 0.50

---

## 313. PAOLI: Pose-free Articulated Object Learning from Sparse-view Images

**Authors:** Jianning Deng, Kartic Subr, Hakan Bilen

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04276v1) | [PDF](http://arxiv.org/pdf/2509.04276v1)

**Abstract:** We present a novel self-supervised framework for learning articulated object
representations from sparse-view, unposed images. Unlike prior methods that
require dense multi-view observations and ground-truth camera poses, our
approach operates with as few as four views per articulation and no camera
supervision. To address the inherent challenges, we first reconstruct each
articulation independently using recent advances in sparse-view 3D
reconstruction, then learn a deformation field that establishes dense
correspondences across poses. A progressive disentanglement strategy further
separates static from moving parts, enabling robust separation of camera and
object motion. Finally, we jointly optimize geometry, appearance, and
kinematics with a self-supervised loss that enforces cross-view and cross-pose
consistency. Experiments on the standard benchmark and real-world examples
demonstrate that our method produces accurate and detailed articulated object
representations under significantly weaker input assumptions than existing
approaches.

**Relevance Score:** 0.50

---

## 314. Dual-Scale Volume Priors with Wasserstein-Based Consistency for   Semi-Supervised Medical Image Segmentation

**Authors:** Junying Meng, Gangxuan Zhou, Jun Liu, Weihong Guo

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04273v1) | [PDF](http://arxiv.org/pdf/2509.04273v1)

**Abstract:** Despite signi cant progress in semi-supervised medical image segmentation,
most existing segmentation networks overlook e ective methodological guidance
for feature extraction and important prior information from
  datasets. In this paper, we develop a semi-supervised medical image
segmentation framework that e ectively integrates spatial regularization
methods and volume priors. Speci cally, our approach integrates a strong
explicit volume prior at the image scale and Threshold Dynamics spatial
regularization, both derived from variational models, into the backbone
segmentation network. The target region volumes for each unlabeled image are
estimated by a regression network, which e ectively regularizes the backbone
segmentation network through an image-scale Wasserstein distance constraint,
ensuring that the class ratios in the segmentation results for each unlabeled
image match those predicted by the regression network. Additionally, we design
a dataset-scale Wasserstein distance loss function based on a weak implicit
volume prior, which enforces that the volume distribution predicted for the
unlabeled dataset is similar to that of labeled dataset. Experimental results
on the 2017 ACDC dataset, PROMISE12 dataset, and thigh muscle MR image dataset
show the superiority of the proposed method.

**Relevance Score:** 0.50

---

## 315. Revisiting Simple Baselines for In-The-Wild Deepfake Detection

**Authors:** Orlando Castaneda, Kevin So-Tang, Kshitij Gurung

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04150v1) | [PDF](http://arxiv.org/pdf/2509.04150v1)

**Abstract:** The widespread adoption of synthetic media demands accessible deepfake
detectors and realistic benchmarks. While most existing research evaluates
deepfake detectors on highly controlled datasets, we focus on the recently
released "in-the-wild" benchmark, Deepfake-Eval-2024. Initial reporting on
Deepfake-Eval-2024 showed that three finetuned open-source models achieve
accuracies between 61% and 69%, significantly lagging behind the leading
commercial deepfake detector with 82% accuracy. Our work revisits one of these
baseline approaches, originally introduced by Ojha et al., which adapts
standard pretrained vision backbones to produce generalizable deepfake
detectors. We demonstrate that with better-tuned hyperparameters, this simple
approach actually yields much higher performance -- 81% accuracy on
Deepfake-Eval-2024 -- surpassing the previously reported accuracy of this
baseline approach by 18% and competing with commercial deepfake detectors. We
discuss tradeoffs in accuracy, computational costs, and interpretability,
focusing on how practical these deepfake detectors might be when deployed in
real-world settings. Our code can be found at
https://github.com/Deepfake-Detection-KKO/deepfake-detection.

**Relevance Score:** 0.50

---

## 316. Replication Study and Benchmarking of Real-Time Object Detection Models

**Authors:** Pierre-Luc Asselin, Vincent Coulombe, William Guimont-Martin, William Larrivée-Hardy

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2405.06911v2) | [PDF](http://arxiv.org/pdf/2405.06911v2)

**Abstract:** This work examines the reproducibility and benchmarking of state-of-the-art
real-time object detection models. As object detection models are often used in
real-world contexts, such as robotics, where inference time is paramount,
simply measuring models' accuracy is not enough to compare them. We thus
compare a large variety of object detection models' accuracy and inference
speed on multiple graphics cards. In addition to this large benchmarking
attempt, we also reproduce the following models from scratch using PyTorch on
the MS COCO 2017 dataset: DETR, RTMDet, ViTDet and YOLOv7. More importantly, we
propose a unified training and evaluation pipeline, based on MMDetection's
features, to better compare models. Our implementation of DETR and ViTDet could
not achieve accuracy or speed performances comparable to what is declared in
the original papers. On the other hand, reproduced RTMDet and YOLOv7 could
match such performances. Studied papers are also found to be generally lacking
for reproducibility purposes. As for MMDetection pretrained models, speed
performances are severely reduced with limited computing resources (larger,
more accurate models even more so). Moreover, results exhibit a strong
trade-off between accuracy and speed, prevailed by anchor-free models - notably
RTMDet or YOLOx models. The code used is this paper and all the experiments is
available in the repository at https://github.com/willGuimont/segdet_mlcr2024.

**Comment:** Authors are presented in alphabetical order, each having equal
  contribution to the work

**Relevance Score:** 0.50

---

## 317. TriLiteNet: Lightweight Model for Multi-Task Visual Perception

**Authors:** Quang-Huy Che, Duc-Khai Lam

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04092v1) | [PDF](http://arxiv.org/pdf/2509.04092v1)

**Abstract:** Efficient perception models are essential for Advanced Driver Assistance
Systems (ADAS), as these applications require rapid processing and response to
ensure safety and effectiveness in real-world environments. To address the
real-time execution needs of such perception models, this study introduces the
TriLiteNet model. This model can simultaneously manage multiple tasks related
to panoramic driving perception. TriLiteNet is designed to optimize performance
while maintaining low computational costs. Experimental results on the BDD100k
dataset demonstrate that the model achieves competitive performance across
three key tasks: vehicle detection, drivable area segmentation, and lane line
segmentation. Specifically, the TriLiteNet_{base} demonstrated a recall of
85.6% for vehicle detection, a mean Intersection over Union (mIoU) of 92.4% for
drivable area segmentation, and an Acc of 82.3% for lane line segmentation with
only 2.35M parameters and a computational cost of 7.72 GFLOPs. Our proposed
model includes a tiny configuration with just 0.14M parameters, which provides
a multi-task solution with minimal computational demand. Evaluated for latency
and power consumption on embedded devices, TriLiteNet in both configurations
shows low latency and reasonable power during inference. By balancing
performance, computational efficiency, and scalability, TriLiteNet offers a
practical and deployable solution for real-world autonomous driving
applications. Code is available at https://github.com/chequanghuy/TriLiteNet.

**Relevance Score:** 0.50

---

## 318. SPARE: Symmetrized Point-to-Plane Distance for Robust Non-Rigid   Registration

**Authors:** Yuxin Yao, Bailin Deng, Junhui Hou, Juyong Zhang

**Categories:** cs.CV, cs.GR

**Links:** [arXiv](http://arxiv.org/abs/2405.20188v2) | [PDF](http://arxiv.org/pdf/2405.20188v2)

**Abstract:** Existing optimization-based methods for non-rigid registration typically
minimize an alignment error metric based on the point-to-point or
point-to-plane distance between corresponding point pairs on the source surface
and target surface. However, these metrics can result in slow convergence or a
loss of detail. In this paper, we propose SPARE, a novel formulation that
utilizes a symmetrized point-to-plane distance for robust non-rigid
registration. The symmetrized point-to-plane distance relies on both the
positions and normals of the corresponding points, resulting in a more accurate
approximation of the underlying geometry and can achieve higher accuracy than
existing methods. To solve this optimization problem efficiently, we introduce
an as-rigid-as-possible regulation term to estimate the deformed normals and
propose an alternating minimization solver using a majorization-minimization
strategy. Moreover, for effective initialization of the solver, we incorporate
a deformation graph-based coarse alignment that improves registration quality
and efficiency. Extensive experiments show that the proposed method greatly
improves the accuracy of non-rigid registration problems and maintains
relatively high solution efficiency. The code is publicly available at
https://github.com/yaoyx689/spare.

**Comment:** Accepted to IEEE Transactions on Pattern Analysis and Machine
  Intelligence

**Relevance Score:** 0.50

---

## 319. Millisecond-Response Tracking and Gazing System for UAVs: A Domestic   Solution Based on "Phytium + Cambricon"

**Authors:** Yuchen Zhu, Longxiang Yin, Kai Zhao

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04043v1) | [PDF](http://arxiv.org/pdf/2509.04043v1)

**Abstract:** In the frontier research and application of current video surveillance
technology, traditional camera systems exhibit significant limitations of
response delay exceeding 200 ms in dynamic scenarios due to the insufficient
deep feature extraction capability of automatic recognition algorithms and the
efficiency bottleneck of computing architectures, failing to meet the real-time
requirements in complex scenes. To address this issue, this study proposes a
heterogeneous computing architecture based on Phytium processors and Cambricon
accelerator cards, constructing a UAV tracking and gazing system with
millisecond-level response capability. At the hardware level, the system adopts
a collaborative computing architecture of Phytium FT-2000/4 processors and
MLU220 accelerator cards, enhancing computing power through multi-card
parallelism. At the software level, it innovatively integrates a lightweight
YOLOv5s detection network with a DeepSORT cascaded tracking algorithm, forming
a closed-loop control chain of "detection-tracking-feedback". Experimental
results demonstrate that the system achieves a stable single-frame
comprehensive processing delay of 50-100 ms in 1920*1080 resolution video
stream processing, with a multi-scale target recognition accuracy of over
98.5%, featuring both low latency and high precision. This study provides an
innovative solution for UAV monitoring and the application of domestic chips.

**Comment:** 16 pages,17 figures

**Relevance Score:** 0.50

---

## 320. HLG: Comprehensive 3D Room Construction via Hierarchical Layout   Generation

**Authors:** Xiping Wang, Yuxi Wang, Mengqi Zhou, Junsong Fan, Zhaoxiang Zhang

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2508.17832v2) | [PDF](http://arxiv.org/pdf/2508.17832v2)

**Abstract:** Realistic 3D indoor scene generation is crucial for virtual reality, interior
design, embodied intelligence, and scene understanding. While existing methods
have made progress in coarse-scale furniture arrangement, they struggle to
capture fine-grained object placements, limiting the realism and utility of
generated environments. This gap hinders immersive virtual experiences and
detailed scene comprehension for embodied AI applications. To address these
issues, we propose Hierarchical Layout Generation (HLG), a novel method for
fine-grained 3D scene generation. HLG is the first to adopt a coarse-to-fine
hierarchical approach, refining scene layouts from large-scale furniture
placement to intricate object arrangements. Specifically, our fine-grained
layout alignment module constructs a hierarchical layout through vertical and
horizontal decoupling, effectively decomposing complex 3D indoor scenes into
multiple levels of granularity. Additionally, our trainable layout optimization
network addresses placement issues, such as incorrect positioning, orientation
errors, and object intersections, ensuring structurally coherent and physically
plausible scene generation. We demonstrate the effectiveness of our approach
through extensive experiments, showing superior performance in generating
realistic indoor scenes compared to existing methods. This work advances the
field of scene generation and opens new possibilities for applications
requiring detailed 3D environments. We will release our code upon publication
to encourage future research.

**Relevance Score:** 0.50

---

## 321. Learning from Majority Label: A Novel Problem in Multi-class   Multiple-Instance Learning

**Authors:** Shiku Kaito, Shinnosuke Matsuo, Daiki Suehiro, Ryoma Bise

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.04023v1) | [PDF](http://arxiv.org/pdf/2509.04023v1)

**Abstract:** The paper proposes a novel multi-class Multiple-Instance Learning (MIL)
problem called Learning from Majority Label (LML). In LML, the majority class
of instances in a bag is assigned as the bag-level label. The goal of LML is to
train a classification model that estimates the class of each instance using
the majority label. This problem is valuable in a variety of applications,
including pathology image segmentation, political voting prediction, customer
sentiment analysis, and environmental monitoring. To solve LML, we propose a
Counting Network trained to produce bag-level majority labels, estimated by
counting the number of instances in each class. Furthermore, analysis
experiments on the characteristics of LML revealed that bags with a high
proportion of the majority class facilitate learning. Based on this result, we
developed a Majority Proportion Enhancement Module (MPEM) that increases the
proportion of the majority class by removing minority class instances within
the bags. Experiments demonstrate the superiority of the proposed method on
four datasets compared to conventional MIL methods. Moreover, ablation studies
confirmed the effectiveness of each module. The code is available at
\href{https://github.com/Shiku-Kaito/Learning-from-Majority-Label-A-Novel-Problem-in-Multi-class-Multiple-Instance-Learning}{here}.

**Comment:** 35 pages, 9 figures, Accepted in Pattern recognition

**Relevance Score:** 0.50

---

## 322. Vision-based Manipulation from Single Human Video with Open-World Object   Graphs

**Authors:** Yifeng Zhu, Arisrei Lim, Peter Stone, Yuke Zhu

**Categories:** cs.CV, cs.RO, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2405.20321v2) | [PDF](http://arxiv.org/pdf/2405.20321v2)

**Abstract:** This work presents an object-centric approach to learning vision-based
manipulation skills from human videos. We investigate the problem of robot
manipulation via imitation in the open-world setting, where a robot learns to
manipulate novel objects from a single video demonstration. We introduce ORION,
an algorithm that tackles the problem by extracting an object-centric
manipulation plan from a single RGB or RGB-D video and deriving a policy that
conditions on the extracted plan. Our method enables the robot to learn from
videos captured by daily mobile devices and to generalize the policies to
deployment environments with varying visual backgrounds, camera angles, spatial
layouts, and novel object instances. We systematically evaluate our method on
both short-horizon and long-horizon tasks, using RGB-D and RGB-only
demonstration videos. Across varied tasks and demonstration types (RGB-D /
RGB), we observe an average success rate of 74.4%, demonstrating the efficacy
of ORION in learning from a single human video in the open world. Additional
materials can be found on our project website:
https://ut-austin-rpl.github.io/ORION-release.

**Comment:** Extended version of paper adding results with RGB-only demonstration
  videos uploaded on 09/04/2025

**Relevance Score:** 0.50

---

## 323. Improving Vessel Segmentation with Multi-Task Learning and Auxiliary   Data Available Only During Model Training

**Authors:** Daniel Sobotka, Alexander Herold, Matthias Perkonigg, Lucian Beer, Nina Bastati, Alina Sablatnig, Ahmed Ba-Ssalamah, Georg Langs

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03975v1) | [PDF](http://arxiv.org/pdf/2509.03975v1)

**Abstract:** Liver vessel segmentation in magnetic resonance imaging data is important for
the computational analysis of vascular remodelling, associated with a wide
spectrum of diffuse liver diseases. Existing approaches rely on contrast
enhanced imaging data, but the necessary dedicated imaging sequences are not
uniformly acquired. Images without contrast enhancement are acquired more
frequently, but vessel segmentation is challenging, and requires large-scale
annotated data. We propose a multi-task learning framework to segment vessels
in liver MRI without contrast. It exploits auxiliary contrast enhanced MRI data
available only during training to reduce the need for annotated training
examples. Our approach draws on paired native and contrast enhanced data with
and without vessel annotations for model training. Results show that auxiliary
data improves the accuracy of vessel segmentation, even if they are not
available during inference. The advantage is most pronounced if only few
annotations are available for training, since the feature representation
benefits from the shared task structure. A validation of this approach to
augment a model for brain tumor segmentation confirms its benefits across
different domains. An auxiliary informative imaging modality can augment expert
annotations even if it is only available during training.

**Relevance Score:** 0.50

---

## 324. Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer   Learning in a U-Net Architecture

**Authors:** Alvaro Aranibar Roque, Helga Sebastian

**Categories:** cs.CV, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2509.03950v1) | [PDF](http://arxiv.org/pdf/2509.03950v1)

**Abstract:** Pneumothorax, the abnormal accumulation of air in the pleural space, can be
life-threatening if undetected. Chest X-rays are the first-line diagnostic
tool, but small cases may be subtle. We propose an automated deep-learning
pipeline using a U-Net with an EfficientNet-B4 encoder to segment pneumothorax
regions. Trained on the SIIM-ACR dataset with data augmentation and a combined
binary cross-entropy plus Dice loss, the model achieved an IoU of 0.7008 and
Dice score of 0.8241 on the independent PTX-498 dataset. These results
demonstrate that the model can accurately localize pneumothoraces and support
radiologists.

**Comment:** 10 page, 5 figures

**Relevance Score:** 0.50

---

## 325. TopoSculpt: Betti-Steered Topological Sculpting of 3D Fine-grained   Tubular Shapes

**Authors:** Minghui Zhang, Yaoyu Liu, Junyang Wu, Xin You, Hanxiao Zhang, Junjun He, Yun Gu

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03938v1) | [PDF](http://arxiv.org/pdf/2509.03938v1)

**Abstract:** Medical tubular anatomical structures are inherently three-dimensional
conduits with lumens, enclosing walls, and complex branching topologies.
Accurate reconstruction of their geometry and topology is crucial for
applications such as bronchoscopic navigation and cerebral arterial
connectivity assessment. Existing methods often rely on voxel-wise overlap
measures, which fail to capture topological correctness and completeness.
Although topology-aware losses and persistent homology constraints have shown
promise, they are usually applied patch-wise and cannot guarantee global
preservation or correct geometric errors at inference. To address these
limitations, we propose a novel TopoSculpt, a framework for topological
refinement of 3D fine-grained tubular structures. TopoSculpt (i) adopts a
holistic whole-region modeling strategy to capture full spatial context, (ii)
first introduces a Topological Integrity Betti (TIB) constraint that jointly
enforces Betti number priors and global integrity, and (iii) employs a
curriculum refinement scheme with persistent homology to progressively correct
errors from coarse to fine scales. Extensive experiments on challenging
pulmonary airway and Circle of Willis datasets demonstrate substantial
improvements in both geometry and topology. For instance, $\beta_{0}$ errors
are reduced from 69.00 to 3.40 on the airway dataset and from 1.65 to 0.30 on
the CoW dataset, with Tree length detected and branch detected rates improving
by nearly 10\%. These results highlight the effectiveness of TopoSculpt in
correcting critical topological errors and advancing the high-fidelity modeling
of complex 3D tubular anatomy. The project homepage is available at:
https://github.com/Puzzled-Hui/TopoSculpt.

**Relevance Score:** 0.50

---

## 326. A Framework for Supervised and Unsupervised Segmentation and   Classification of Materials Microstructure Images

**Authors:** Kungang Zhang, Wei Chen, Wing K. Liu, L. Catherine Brinson, Daniel W. Apley

**Categories:** stat.AP, stat.ML, cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2502.07107v2) | [PDF](http://arxiv.org/pdf/2502.07107v2)

**Abstract:** Microstructure of materials is often characterized through image analysis to
understand processing-structure-properties linkages. We propose a largely
automated framework that integrates unsupervised and supervised learning
methods to classify micrographs according to microstructure phase/class and,
for multiphase microstructures, segments them into different homogeneous
regions. With the advance of manufacturing and imaging techniques, the
ultra-high resolution of imaging that reveals the complexity of microstructures
and the rapidly increasing quantity of images (i.e., micrographs) enables and
necessitates a more powerful and automated framework to extract materials
characteristics and knowledge. The framework we propose can be used to
gradually build a database of microstructure classes relevant to a particular
process or group of materials, which can help in analyzing and
discovering/identifying new materials. The framework has three steps: (1)
segmentation of multiphase micrographs through a recently developed score-based
method so that different microstructure homogeneous regions can be identified
in an unsupervised manner; (2) {identification and classification of}
homogeneous regions of micrographs through an uncertainty-aware supervised
classification network trained using the segmented micrographs from Step $1$
with their identified labels verified via the built-in uncertainty
quantification and minimal human inspection; (3) supervised segmentation (more
powerful than the segmentation in Step $1$) of multiphase microstructures
through a segmentation network trained with micrographs and the results from
Steps $1$-$2$ using a form of data augmentation. This framework can iteratively
characterize/segment new homogeneous or multiphase materials while expanding
the database to enhance performance. The framework is demonstrated on various
sets of materials and texture images.

**Relevance Score:** 0.50

---

## 327. Weakly-Supervised Learning of Dense Functional Correspondences

**Authors:** Stefan Stojanov, Linan Zhao, Yunzhi Zhang, Daniel L. K. Yamins, Jiajun Wu

**Categories:** cs.CV

**Links:** [arXiv](http://arxiv.org/abs/2509.03893v1) | [PDF](http://arxiv.org/pdf/2509.03893v1)

**Abstract:** Establishing dense correspondences across image pairs is essential for tasks
such as shape reconstruction and robot manipulation. In the challenging setting
of matching across different categories, the function of an object, i.e., the
effect that an object can cause on other objects, can guide how correspondences
should be established. This is because object parts that enable specific
functions often share similarities in shape and appearance. We derive the
definition of dense functional correspondence based on this observation and
propose a weakly-supervised learning paradigm to tackle the prediction task.
The main insight behind our approach is that we can leverage vision-language
models to pseudo-label multi-view images to obtain functional parts. We then
integrate this with dense contrastive learning from pixel correspondences to
distill both functional and spatial knowledge into a new model that can
establish dense functional correspondence. Further, we curate synthetic and
real evaluation datasets as task benchmarks. Our results demonstrate the
advantages of our approach over baseline solutions consisting of off-the-shelf
self-supervised image representations and grounded vision language models.

**Comment:** Accepted at ICCV 2025. Project website:
  https://dense-functional-correspondence.github.io/

**Relevance Score:** 0.50

---

## 328. Can Language Models Handle a Non-Gregorian Calendar?

**Authors:** Mutsumi Sasaki, Go Kamoda, Ryosuke Takahashi, Kosuke Sato, Kentaro Inui, Keisuke Sakaguchi, Benjamin Heinzerling

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2509.04432v1) | [PDF](http://arxiv.org/pdf/2509.04432v1)

**Abstract:** Temporal reasoning and knowledge are essential capabilities for language
models (LMs). While much prior work has analyzed and improved temporal
reasoning in LMs, most studies have focused solely on the Gregorian calendar.
However, many non-Gregorian systems, such as the Japanese, Hijri, and Hebrew
calendars, are in active use and reflect culturally grounded conceptions of
time. If and how well current LMs can accurately handle such non-Gregorian
calendars has not been evaluated so far. Here, we present a systematic
evaluation of how well open-source LMs handle one such non-Gregorian system:
the Japanese calendar. For our evaluation, we create datasets for four tasks
that require both temporal knowledge and temporal reasoning. Evaluating a range
of English-centric and Japanese-centric LMs, we find that some models can
perform calendar conversions, but even Japanese-centric models struggle with
Japanese-calendar arithmetic and with maintaining consistency across calendars.
Our results highlight the importance of developing LMs that are better equipped
for culture-specific calendar understanding.

**Relevance Score:** 0.50

---

## 329. That is Unacceptable: the Moral Foundations of Canceling

**Authors:** Soda Marem Lo, Oscar Araque, Rajesh Sharma, Marco Antonio Stranisci

**Categories:** cs.CY, cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2503.05720v3) | [PDF](http://arxiv.org/pdf/2503.05720v3)

**Abstract:** Canceling is a morally-driven phenomenon that hinders the development of safe
social media platforms and contributes to ideological polarization. To address
this issue we present the Canceling Attitudes Detection (CADE) dataset, an
annotated corpus of canceling incidents aimed at exploring the factors of
disagreements in evaluating people canceling attitudes on social media.
Specifically, we study the impact of annotators' morality in their perception
of canceling, showing that morality is an independent axis for the explanation
of disagreement on this phenomenon. Annotator's judgments heavily depend on the
type of controversial events and involved celebrities. This shows the need to
develop more event-centric datasets to better understand how harms are
perpetrated in social media and to develop more aware technologies for their
detection.

**Relevance Score:** 0.50

---

## 330. MyProfessors: Mining Turkish Student Reviews

**Authors:** Ibrahim Faruk Ceylan, Necmettin Bera Calik

**Categories:** cs.CL

**Links:** [arXiv](http://arxiv.org/abs/2109.02325v5) | [PDF](http://arxiv.org/pdf/2109.02325v5)

**Abstract:** We introduce Hocalarim (MyProfessors), the largest student review dataset
available for the Turkish language. It consists of over 5000 professor reviews
left online by students, with different aspects of education rated on a scale
of 1 to 5 stars. We investigate the properties of the dataset and present its
statistics. We examine the impact of students' institution type on their
ratings and the correlation of students' bias to give positive or negative
feedback.

**Comment:** The paper is withdrawn due to the scraping errors in the dataset
  collection process and affected results

**Relevance Score:** 0.50

---

## 331. We Have It Covered: A Resampling-based Method for Uplift Model   Comparison

**Authors:** Yang Liu, Chaoyu Yuan

**Categories:** stat.ME, stat.ML

**Links:** [arXiv](http://arxiv.org/abs/2509.04315v1) | [PDF](http://arxiv.org/pdf/2509.04315v1)

**Abstract:** Uplift models play a critical role in modern marketing applications to help
understand the incremental benefits of interventions and identify optimal
targeting strategies. A variety of techniques exist for building uplift models,
and it is essential to understand the model differences in the context of
intended applications. The uplift curve is a widely adopted tool for assessing
uplift model performance on the selection universe when observations are
available for the entire population. However, when it is uneconomical or
infeasible to select the entire population, it becomes difficult or even
impossible to estimate the uplift curve without appropriate sampling design. To
the best of our knowledge, no prior work has addressed uncertainty
quantification of uplift curve estimates, which is essential for model
comparisons. We propose a two-step sampling procedure and a resampling-based
approach to compare uplift models with uncertainty quantification, examine the
proposed method via simulations and real data applications, and conclude with a
discussion.

**Relevance Score:** 0.00

---

## 332. Sharp Convergence Rates of Empirical Unbalanced Optimal Transport for   Spatio-Temporal Point Processes

**Authors:** Marina Struleva, Shayan Hundrieser, Dominic Schuhmacher, Axel Munk

**Categories:** math.ST, stat.ML, stat.TH, primary 62G05, 62G07, 62R20, secondary: 60D05, 60G60

**Links:** [arXiv](http://arxiv.org/abs/2509.04225v1) | [PDF](http://arxiv.org/pdf/2509.04225v1)

**Abstract:** We statistically analyze empirical plug-in estimators for unbalanced optimal
transport (UOT) formalisms, focusing on the Kantorovich-Rubinstein distance,
between general intensity measures based on observations from spatio-temporal
point processes. Specifically, we model the observations by two weakly
time-stationary point processes with spatial intensity measures $\mu$ and $\nu$
over the expanding window $(0,t]$ as $t$ increases to infinity, and establish
sharp convergence rates of the empirical UOT in terms of the intrinsic
dimensions of the measures. We assume a sub-quadratic temporal growth condition
of the variance of the process, which allows for a wide range of temporal
dependencies. As the growth approaches quadratic, the convergence rate becomes
slower. This variance assumption is related to the time-reduced factorial
covariance measure, and we exemplify its validity for various point processes,
including the Poisson cluster, Hawkes, Neyman-Scott, and log-Gaussian Cox
processes. Complementary to our upper bounds, we also derive matching lower
bounds for various spatio-temporal point processes of interest and establish
near minimax rate optimality of the empirical Kantorovich-Rubinstein distance.

**Comment:** The first two authors contributed equally, 76 pages, 7 figures

**Relevance Score:** 0.00

---

## 333. FastPart: Over-Parameterized Stochastic Gradient Descent for Sparse   optimisation on Measures

**Authors:** Yohann De Castro, Sébastien Gadat, Clément Marteau

**Categories:** math.OC, stat.ML

**Links:** [arXiv](http://arxiv.org/abs/2312.05993v2) | [PDF](http://arxiv.org/pdf/2312.05993v2)

**Abstract:** This paper presents a novel algorithm that leverages Stochastic Gradient
Descent strategies in conjunction with Random Features to augment the
scalability of Conic Particle Gradient Descent (CPGD) specifically tailored for
solving sparse optimization problems on measures. By formulating the CPGD steps
within a variational framework, we provide rigorous mathematical proofs
demonstrating the following key findings: $\mathrm{(i)}$ The total variation
norms of the solution measures along the descent trajectory remain bounded,
ensuring stability and preventing undesirable divergence; $\mathrm{(ii)}$ We
establish a global convergence guarantee with a convergence rate of
${O}(\log(K)/\sqrt{K})$ over $K$ iterations, showcasing the efficiency and
effectiveness of our algorithm, $\mathrm{(iii)}$ Additionally, we analyse and
establish local control over the first-order condition discrepancy,
contributing to a deeper understanding of the algorithm's behaviour and
reliability in practical applications.

**Comment:** 45 pages, 4 figures

**Relevance Score:** 0.00

---

## 334. Prob-GParareal: A Probabilistic Numerical Parallel-in-Time Solver for   Differential Equations

**Authors:** Guglielmo Gattiglio, Lyudmila Grigoryeva, Massimiliano Tamborrino

**Categories:** stat.CO, cs.DC, cs.NA, math.NA, stat.ML, 65M55, 65M22, 65L05, 50G15, 65Y05

**Links:** [arXiv](http://arxiv.org/abs/2509.03945v1) | [PDF](http://arxiv.org/pdf/2509.03945v1)

**Abstract:** We introduce Prob-GParareal, a probabilistic extension of the GParareal
algorithm designed to provide uncertainty quantification for the
Parallel-in-Time (PinT) solution of (ordinary and partial) differential
equations (ODEs, PDEs). The method employs Gaussian processes (GPs) to model
the Parareal correction function, as GParareal does, further enabling the
propagation of numerical uncertainty across time and yielding probabilistic
forecasts of system's evolution. Furthermore, Prob-GParareal accommodates
probabilistic initial conditions and maintains compatibility with classical
numerical solvers, ensuring its straightforward integration into existing
Parareal frameworks. Here, we first conduct a theoretical analysis of the
computational complexity and derive error bounds of Prob-GParareal. Then, we
numerically demonstrate the accuracy and robustness of the proposed algorithm
on five benchmark ODE systems, including chaotic, stiff, and bifurcation
problems. To showcase the flexibility and potential scalability of the proposed
algorithm, we also consider Prob-nnGParareal, a variant obtained by replacing
the GPs in Parareal with the nearest-neighbors GPs, illustrating its increased
performance on an additional PDE example. This work bridges a critical gap in
the development of probabilistic counterparts to established PinT methods.

**Relevance Score:** 0.00

---

## 335. EMMA: Scaling Mobile Manipulation via Egocentric Human Data

**Authors:** Lawrence Y. Zhu, Pranav Kuppili, Ryan Punamiya, Patcharapong Aphiwetsa, Dhruv Patel, Simar Kareer, Sehoon Ha, Danfei Xu

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04443v1) | [PDF](http://arxiv.org/pdf/2509.04443v1)

**Abstract:** Scaling mobile manipulation imitation learning is bottlenecked by expensive
mobile robot teleoperation. We present Egocentric Mobile MAnipulation (EMMA),
an end-to-end framework training mobile manipulation policies from human mobile
manipulation data with static robot data, sidestepping mobile teleoperation. To
accomplish this, we co-train human full-body motion data with static robot
data. In our experiments across three real-world tasks, EMMA demonstrates
comparable performance to baselines trained on teleoperated mobile robot data
(Mobile ALOHA), achieving higher or equivalent task performance in full task
success. We find that EMMA is able to generalize to new spatial configurations
and scenes, and we observe positive performance scaling as we increase the
hours of human data, opening new avenues for scalable robotic learning in
real-world environments. Details of this project can be found at
https://ego-moma.github.io/.

**Relevance Score:** 0.00

---

## 336. Taming High-Dimensional Dynamics: Learning Optimal Projections onto   Spectral Submanifolds

**Authors:** Hugo Buurmeijer, Luis A. Pabon, John Irvin Alora, Roshan S. Kaundinya, George Haller, Marco Pavone

**Categories:** eess.SY, cs.RO, cs.SY

**Links:** [arXiv](http://arxiv.org/abs/2504.03157v2) | [PDF](http://arxiv.org/pdf/2504.03157v2)

**Abstract:** High-dimensional nonlinear systems pose considerable challenges for modeling
and control across many domains, from fluid mechanics to advanced robotics.
Such systems are typically approximated with reduced-order models, which often
rely on orthogonal projections, a simplification that may lead to large
prediction errors. In this work, we derive optimality of fiber-aligned
projections onto spectral submanifolds, preserving the nonlinear geometric
structure and minimizing long-term prediction error. We propose a data-driven
procedure to learn these projections from trajectories and demonstrate its
effectiveness through a 180-dimensional robotic system. Our reduced-order
models achieve up to fivefold improvement in trajectory tracking accuracy under
model predictive control compared to the state of the art.

**Relevance Score:** 0.00

---

## 337. Leveraging Equivariances and Symmetries in the Control Barrier Function   Synthesis

**Authors:** Adrian Wiltz, Dimos V. Dimarogonas

**Categories:** eess.SY, cs.RO, cs.SY

**Links:** [arXiv](http://arxiv.org/abs/2509.04399v1) | [PDF](http://arxiv.org/pdf/2509.04399v1)

**Abstract:** The synthesis of Control Barrier Functions (CBFs) often involves demanding
computations or a meticulous construction. However, structural properties of
the system dynamics and constraints have the potential to mitigate these
challenges. In this paper, we explore how equivariances in the dynamics,
loosely speaking a form of symmetry, can be leveraged in the CBF synthesis.
Although CBFs are generally not inherently symmetric, we show how equivariances
in the dynamics and symmetries in the constraints induce symmetries in CBFs
derived through reachability analysis. This insight allows us to infer their
CBF values across the entire domain from their values on a subset, leading to
significant computational savings. Interestingly, equivariances can be even
leveraged to the CBF synthesis for non-symmetric constraints. Specifically, we
show how a partially known CBF can be leveraged together with equivariances to
construct a CBF for various new constraints. Throughout the paper, we provide
examples illustrating the theoretical findings. Furthermore, a numerical study
investigates the computational gains from invoking equivariances into the CBF
synthesis.

**Comment:** 15 pages

**Relevance Score:** 0.00

---

## 338. Robotic Manipulation via Imitation Learning: Taxonomy, Evolution,   Benchmark, and Challenges

**Authors:** Zezeng Li, Alexandre Chapin, Enda Xiang, Rui Yang, Bruno Machado, Na Lei, Emmanuel Dellandrea, Di Huang, Liming Chen

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2508.17449v2) | [PDF](http://arxiv.org/pdf/2508.17449v2)

**Abstract:** Robotic Manipulation (RM) is central to the advancement of autonomous robots,
enabling them to interact with and manipulate objects in real-world
environments. This survey focuses on RM methodologies that leverage imitation
learning, a powerful technique that allows robots to learn complex manipulation
skills by mimicking human demonstrations. We identify and analyze the most
influential studies in this domain, selected based on community impact and
intrinsic quality. For each paper, we provide a structured summary, covering
the research purpose, technical implementation, hierarchical classification,
input formats, key priors, strengths and limitations, and citation metrics.
Additionally, we trace the chronological development of imitation learning
techniques within RM policy (RMP), offering a timeline of key technological
advancements. Where available, we report benchmark results and perform
quantitative evaluations to compare existing methods. By synthesizing these
insights, this review provides a comprehensive resource for researchers and
practitioners, highlighting both the state of the art and the challenges that
lie ahead in the field of robotic manipulation through imitation learning.

**Relevance Score:** 0.00

---

## 339. Privacy Perceptions in Robot-Assisted Well-Being Coaching: Examining the   Roles of Information Transparency, User Control, and Proactivity

**Authors:** Atikkhan Faridkhan Nilgar, Manuel Dietrich, Kristof Van Laerhoven

**Categories:** cs.HC, cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04358v1) | [PDF](http://arxiv.org/pdf/2509.04358v1)

**Abstract:** Social robots are increasingly recognized as valuable supporters in the field
of well-being coaching. They can function as independent coaches or provide
support alongside human coaches, and healthcare professionals. In coaching
interactions, these robots often handle sensitive information shared by users,
making privacy a relevant issue. Despite this, little is known about the
factors that shape users' privacy perceptions. This research aims to examine
three key factors systematically: (1) the transparency about information usage,
(2) the level of specific user control over how the robot uses their
information, and (3) the robot's behavioral approach - whether it acts
proactively or only responds on demand. Our results from an online study (N =
200) show that even when users grant the robot general access to personal data,
they additionally expect the ability to explicitly control how that information
is interpreted and shared during sessions. Experimental conditions that
provided such control received significantly higher ratings for perceived
privacy appropriateness and trust. Compared to user control, the effects of
transparency and proactivity on privacy appropriateness perception were low,
and we found no significant impact. The results suggest that merely informing
users or proactive sharing is insufficient without accompanying user control.
These insights underscore the need for further research on mechanisms that
allow users to manage robots' information processing and sharing, especially
when social robots take on more proactive roles alongside humans.

**Relevance Score:** 0.00

---

## 340. Compatibility of Multiple Control Barrier Functions for Constrained   Nonlinear Systems

**Authors:** Max H. Cohen, Eugene Lavretsky, Aaron D. Ames

**Categories:** eess.SY, cs.RO, cs.SY, math.OC

**Links:** [arXiv](http://arxiv.org/abs/2509.04220v1) | [PDF](http://arxiv.org/pdf/2509.04220v1)

**Abstract:** Control barrier functions (CBFs) are a powerful tool for the constrained
control of nonlinear systems; however, the majority of results in the
literature focus on systems subject to a single CBF constraint, making it
challenging to synthesize provably safe controllers that handle multiple state
constraints. This paper presents a framework for constrained control of
nonlinear systems subject to box constraints on the systems' vector-valued
outputs using multiple CBFs. Our results illustrate that when the output has a
vector relative degree, the CBF constraints encoding these box constraints are
compatible, and the resulting optimization-based controller is locally
Lipschitz continuous and admits a closed-form expression. Additional results
are presented to characterize the degradation of nominal tracking objectives in
the presence of safety constraints. Simulations of a planar quadrotor are
presented to demonstrate the efficacy of the proposed framework.

**Comment:** To appear at IEEE CDC 2025

**Relevance Score:** 0.00

---

## 341. Lightweight Kinematic and Static Modeling of Cable-Driven Continuum   Robots via Actuation-Space Energy Formulation

**Authors:** Ke Wu, Yuhao Wang, Kevin Henry, Cesare Stefanini, Gang Zheng

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04119v1) | [PDF](http://arxiv.org/pdf/2509.04119v1)

**Abstract:** Continuum robots, inspired by octopus arms and elephant trunks, combine
dexterity with intrinsic compliance, making them well suited for unstructured
and confined environments. Yet their continuously deformable morphology poses
challenges for motion planning and control, calling for accurate but
lightweight models. We propose the Lightweight Actuation Space Energy Modeling
(LASEM) framework for cable driven continuum robots, which formulates actuation
potential energy directly in actuation space. LASEM yields an analytical
forward model derived from geometrically nonlinear beam and rod theories via
Hamilton's principle, while avoiding explicit modeling of cable backbone
contact. It accepts both force and displacement inputs, thereby unifying
kinematic and static formulations. Assuming the friction is neglected, the
framework generalizes to nonuniform geometries, arbitrary cable routings,
distributed loading and axial extensibility, while remaining computationally
efficient for real-time use. Numerical simulations validate its accuracy, and a
semi-analytical iterative scheme is developed for inverse kinematics. To
address discretization in practical robots, LASEM further reformulates the
functional minimization as a numerical optimization, which also naturally
incorporates cable potential energy without explicit contact modeling.

**Comment:** Journal

**Relevance Score:** 0.00

---

## 342. Cloud-Assisted Remote Control for Aerial Robots: From Theory to   Proof-of-Concept Implementation

**Authors:** Achilleas Santi Seisa, Viswa Narayanan Sankaranarayanan, Gerasimos Damigos, Sumeet Gajanan Satpute, George Nikolakopoulos

**Categories:** cs.RO, cs.DC

**Links:** [arXiv](http://arxiv.org/abs/2509.04095v1) | [PDF](http://arxiv.org/pdf/2509.04095v1)

**Abstract:** Cloud robotics has emerged as a promising technology for robotics
applications due to its advantages of offloading computationally intensive
tasks, facilitating data sharing, and enhancing robot coordination. However,
integrating cloud computing with robotics remains a complex challenge due to
network latency, security concerns, and the need for efficient resource
management. In this work, we present a scalable and intuitive framework for
testing cloud and edge robotic systems. The framework consists of two main
components enabled by containerized technology: (a) a containerized cloud
cluster and (b) the containerized robot simulation environment. The system
incorporates two endpoints of a User Datagram Protocol (UDP) tunnel, enabling
bidirectional communication between the cloud cluster container and the robot
simulation environment, while simulating realistic network conditions. To
achieve this, we consider the use case of cloud-assisted remote control for
aerial robots, while utilizing Linux-based traffic control to introduce
artificial delay and jitter, replicating variable network conditions
encountered in practical cloud-robot deployments.

**Comment:** 6 pages, 7 figures, CCGridW 2025

**Relevance Score:** 0.00

---

## 343. Object-Reconstruction-Aware Whole-body Control of Mobile Manipulators

**Authors:** Fatih Dursun, Bruno Vilhena Adorno, Simon Watson, Wei Pan

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04094v1) | [PDF](http://arxiv.org/pdf/2509.04094v1)

**Abstract:** Object reconstruction and inspection tasks play a crucial role in various
robotics applications. Identifying paths that reveal the most unknown areas of
the object becomes paramount in this context, as it directly affects
efficiency, and this problem is known as the view path planning problem.
Current methods often use sampling-based path planning techniques, evaluating
potential views along the path to enhance reconstruction performance. However,
these methods are computationally expensive as they require evaluating several
candidate views on the path. To this end, we propose a computationally
efficient solution that relies on calculating a focus point in the most
informative (unknown) region and having the robot maintain this point in the
camera field of view along the path. We incorporated this strategy into the
whole-body control of a mobile manipulator employing a visibility constraint
without the need for an additional path planner. We conducted comprehensive and
realistic simulations using a large dataset of 114 diverse objects of varying
sizes from 57 categories to compare our method with a sampling-based planning
strategy using Bayesian data analysis. Furthermore, we performed real-world
experiments with an 8-DoF mobile manipulator to demonstrate the proposed
method's performance in practice. Our results suggest that there is no
significant difference in object coverage and entropy. In contrast, our method
is approximately nine times faster than the baseline sampling-based method in
terms of the average time the robot spends between views.

**Comment:** 14 pages, 13 figures, 3 tables. Under Review for the IEEE
  Transactions on Robotics (T-RO)

**Relevance Score:** 0.00

---

## 344. Segmented Trajectory Optimization for Autonomous Parking in Unstructured   Environments

**Authors:** Hang Yu, Renjie Li

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2504.05041v2) | [PDF](http://arxiv.org/pdf/2504.05041v2)

**Abstract:** This paper presents a Segmented Trajectory Optimization (STO) method for
autonomous parking, which refines an initial trajectory into a dynamically
feasible and collision-free one using an iterative SQP-based approach. STO
maintains the maneuver strategy of the high-level global planner while allowing
curvature discontinuities at switching points to improve maneuver efficiency.
To ensure safety, a convex corridor is constructed via GJK-accelerated ellipse
shrinking and expansion, serving as safety constraints in each iteration.
Numerical simulations in perpendicular and reverse-angled parking scenarios
demonstrate that STO enhances maneuver efficiency while ensuring safety.
Moreover, computational performance confirms its practicality for real-world
applications.

**Comment:** 8 pages, 6 figures

**Relevance Score:** 0.00

---

## 345. Integrated Wheel Sensor Communication using ESP32 -- A Contribution   towards a Digital Twin of the Road System

**Authors:** Ventseslav Yordanov, Simon Schäfer, Alexander Mann, Stefan Kowalewski, Bassam Alrifaee, Lutz Eckstein

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.04061v1) | [PDF](http://arxiv.org/pdf/2509.04061v1)

**Abstract:** While current onboard state estimation methods are adequate for most driving
and safety-related applications, they do not provide insights into the
interaction between tires and road surfaces. This paper explores a novel
communication concept for efficiently transmitting integrated wheel sensor data
from an ESP32 microcontroller. Our proposed approach utilizes a
publish-subscribe system, surpassing comparable solutions in the literature
regarding data transmission volume. We tested this approach on a drum tire test
rig with our prototype sensors system utilizing a diverse selection of sample
frequencies between 1 Hz and 32 000 Hz to demonstrate the efficacy of our
communication concept. The implemented prototype sensor showcases minimal data
loss, approximately 0.1 % of the sampled data, validating the reliability of
our developed communication system. This work contributes to advancing
real-time data acquisition, providing insights into optimizing integrated wheel
sensor communication.

**Comment:** 6 pages, 2 figures, this work was submitted to and accepted by IEEE
  International Conference on Intelligent Transportation Systems (ITSC) 2025

**Relevance Score:** 0.00

---

## 346. A Digital Twin for Robotic Post Mortem Tissue Sampling using Virtual   Reality

**Authors:** Maximilian Neidhardt, Ludwig Bosse, Vidas Raudonis, Kristina Allgoewer, Axel Heinemann, Benjamin Ondruschka, Alexander Schlaefer

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.02760v2) | [PDF](http://arxiv.org/pdf/2509.02760v2)

**Abstract:** Studying tissue samples obtained during autopsies is the gold standard when
diagnosing the cause of death and for understanding disease pathophysiology.
Recently, the interest in post mortem minimally invasive biopsies has grown
which is a less destructive approach in comparison to an open autopsy and
reduces the risk of infection. While manual biopsies under ultrasound guidance
are more widely performed, robotic post mortem biopsies have been recently
proposed. This approach can further reduce the risk of infection for
physicians. However, planning of the procedure and control of the robot need to
be efficient and usable. We explore a virtual reality setup with a digital twin
to realize fully remote planning and control of robotic post mortem biopsies.
The setup is evaluated with forensic pathologists in a usability study for
three interaction methods. Furthermore, we evaluate clinical feasibility and
evaluate the system with three human cadavers. Overall, 132 needle insertions
were performed with an off-axis needle placement error of 5.30+-3.25 mm. Tissue
samples were successfully biopsied and histopathologically verified. Users
reported a very intuitive needle placement approach, indicating that the system
is a promising, precise, and low-risk alternative to conventional approaches.

**Relevance Score:** 0.00

---

## 347. Reactive In-Air Clothing Manipulation with Confidence-Aware Dense   Correspondence and Visuotactile Affordance

**Authors:** Neha Sunil, Megha Tippur, Arnau Saumell, Edward Adelson, Alberto Rodriguez

**Categories:** cs.RO, cs.AI, cs.LG

**Links:** [arXiv](http://arxiv.org/abs/2509.03889v1) | [PDF](http://arxiv.org/pdf/2509.03889v1)

**Abstract:** Manipulating clothing is challenging due to complex configurations, variable
material dynamics, and frequent self-occlusion. Prior systems often flatten
garments or assume visibility of key features. We present a dual-arm
visuotactile framework that combines confidence-aware dense visual
correspondence and tactile-supervised grasp affordance to operate directly on
crumpled and suspended garments. The correspondence model is trained on a
custom, high-fidelity simulated dataset using a distributional loss that
captures cloth symmetries and generates correspondence confidence estimates.
These estimates guide a reactive state machine that adapts folding strategies
based on perceptual uncertainty. In parallel, a visuotactile grasp affordance
network, self-supervised using high-resolution tactile feedback, determines
which regions are physically graspable. The same tactile classifier is used
during execution for real-time grasp validation. By deferring action in
low-confidence states, the system handles highly occluded table-top and in-air
configurations. We demonstrate our task-agnostic grasp selection module in
folding and hanging tasks. Moreover, our dense descriptors provide a reusable
intermediate representation for other planning modalities, such as extracting
grasp targets from human video demonstrations, paving the way for more
generalizable and scalable garment manipulation.

**Comment:** Accepted at CoRL 2025. Project website:
  https://mhtippur.github.io/inairclothmanipulation/

**Relevance Score:** 0.00

---

## 348. Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator

**Authors:** Haichao Zhang, Haonan Yu, Le Zhao, Andrew Choi, Qinxun Bai, Yiqing Yang, Wei Xu

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2509.03859v1) | [PDF](http://arxiv.org/pdf/2509.03859v1)

**Abstract:** Quadruped-based mobile manipulation presents significant challenges in
robotics due to the diversity of required skills, the extended task horizon,
and partial observability. After presenting a multi-stage pick-and-place task
as a succinct yet sufficiently rich setup that captures key desiderata for
quadruped-based mobile manipulation, we propose an approach that can train a
visuo-motor policy entirely in simulation, and achieve nearly 80\% success in
the real world. The policy efficiently performs search, approach, grasp,
transport, and drop into actions, with emerged behaviors such as re-grasping
and task chaining. We conduct an extensive set of real-world experiments with
ablation studies highlighting key techniques for efficient training and
effective sim-to-real transfer. Additional experiments demonstrate deployment
across a variety of indoor and outdoor environments. Demo videos and additional
resources are available on the project page:
https://horizonrobotics.github.io/gail/SLIM.

**Comment:** Project: https://horizonrobotics.github.io/gail/SLIM

**Relevance Score:** 0.00

---

## 349. GMT: General Motion Tracking for Humanoid Whole-Body Control

**Authors:** Zixuan Chen, Mazeyu Ji, Xuxin Cheng, Xuanbin Peng, Xue Bin Peng, Xiaolong Wang

**Categories:** cs.RO

**Links:** [arXiv](http://arxiv.org/abs/2506.14770v2) | [PDF](http://arxiv.org/pdf/2506.14770v2)

**Abstract:** The ability to track general whole-body motions in the real world is a useful
way to build general-purpose humanoid robots. However, achieving this can be
challenging due to the temporal and kinematic diversity of the motions, the
policy's capability, and the difficulty of coordination of the upper and lower
bodies. To address these issues, we propose GMT, a general and scalable
motion-tracking framework that trains a single unified policy to enable
humanoid robots to track diverse motions in the real world. GMT is built upon
two core components: an Adaptive Sampling strategy and a Motion
Mixture-of-Experts (MoE) architecture. The Adaptive Sampling automatically
balances easy and difficult motions during training. The MoE ensures better
specialization of different regions of the motion manifold. We show through
extensive experiments in both simulation and the real world the effectiveness
of GMT, achieving state-of-the-art performance across a broad spectrum of
motions using a unified general policy. Videos and additional information can
be found at https://gmt-humanoid.github.io.

**Relevance Score:** 0.00

---

## 350. Robust Offline Imitation Learning Through State-level Trajectory   Stitching

**Authors:** Shuze Wang, Yunpeng Mei, Hongjie Cao, Yetian Yuan, Gang Wang, Jian Sun, Jie Chen

**Categories:** cs.RO, cs.AI

**Links:** [arXiv](http://arxiv.org/abs/2503.22524v2) | [PDF](http://arxiv.org/pdf/2503.22524v2)

**Abstract:** Imitation learning (IL) has proven effective for enabling robots to acquire
visuomotor skills through expert demonstrations. However, traditional IL
methods are limited by their reliance on high-quality, often scarce, expert
data, and suffer from covariate shift. To address these challenges, recent
advances in offline IL have incorporated suboptimal, unlabeled datasets into
the training. In this paper, we propose a novel approach to enhance policy
learning from mixed-quality offline datasets by leveraging task-relevant
trajectory fragments and rich environmental dynamics. Specifically, we
introduce a state-based search framework that stitches state-action pairs from
imperfect demonstrations, generating more diverse and informative training
trajectories. Experimental results on standard IL benchmarks and real-world
robotic tasks showcase that our proposed method significantly improves both
generalization and performance.

**Relevance Score:** 0.00

---

## 351. Real-Time Buoyancy Estimation for AUV Simulations Using Convex   Hull-Based Submerged Volume Calculation

**Authors:** Ad-Deen Mahbub, Md Ragib Shaharear

**Categories:** cs.RO, cs.SY, eess.SY

**Links:** [arXiv](http://arxiv.org/abs/2509.03804v1) | [PDF](http://arxiv.org/pdf/2509.03804v1)

**Abstract:** Accurate real-time buoyancy modeling is essential for high-fidelity
Autonomous Underwater Vehicle (AUV) simulations, yet NVIDIA Isaac Sim lacks a
native buoyancy system, requiring external solutions for precise underwater
physics. This paper presents a novel convex hull-based approach to dynamically
compute the submerged volume of an AUV in real time. By extracting mesh
geometry from the simulation environment and calculating the hull portion
intersecting the water level along the z-axis, our method enhances accuracy
over traditional geometric approximations. A cross-sectional area extension
reduces computational overhead, enabling efficient buoyant force updates that
adapt to orientation, depth, and sinusoidal wave fluctuations (+-0.3 m). Tested
on a custom AUV design for SAUVC 2025, this approach delivers real-time
performance and scalability, improving simulation fidelity for underwater
robotics research without precomputed hydrodynamic models.

**Comment:** 7 pages, 10 figures

**Relevance Score:** 0.00

---

## 352. Fuzzy to Clear: Elucidating the Threat Hunter Cognitive Process and   Cognitive Support Needs

**Authors:** Alessandra Maciel Paz Milani, Arty Starr, Samantha Hill, Callum Curtis, Norman Anderson, David Moreno-Lumbreras, Margaret-Anne Storey

**Categories:** cs.CR, cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2408.04348v3) | [PDF](http://arxiv.org/pdf/2408.04348v3)

**Abstract:** With security threats increasing in frequency and severity, it is critical
that we consider the important role of threat hunters. These highly-trained
security professionals learn to see, identify, and intercept security threats.
Many recent works and existing tools in cybersecurity are focused on automating
the threat hunting process, often overlooking the critical human element. Our
study shifts this paradigm by emphasizing a human-centered approach to
understanding the lived experiences of threat hunters. By observing threat
hunters during hunting sessions and analyzing the rich insights they provide,
we seek to advance the understanding of their cognitive processes and the tool
support they need. Through an in-depth observational study of threat hunters,
we introduce a model of how they build and refine their mental models during
threat hunting sessions. We also present 23 themes that provide a foundation to
better understand threat hunter needs and suggest five actionable design
propositions to enhance the tools that support them. Through these
contributions, our work enriches the theoretical understanding of threat
hunting and provides practical insights for designing more effective,
human-centered cybersecurity tools.

**Comment:** 22 Pages; 5 Figures; 8 Tables

**Relevance Score:** 0.00

---

## 353. Human causal perception in a cube-stacking task

**Authors:** Nikolai Bahr, Christoph Zetzsche

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2505.05923v2) | [PDF](http://arxiv.org/pdf/2505.05923v2)

**Abstract:** In intuitive physics the process of stacking cubes has become a paradigmatic,
canonical task. Even though it gets employed in various shades and
complexities, the very fundamental setting with two cubes has not been
thoroughly investigated. Furthermore, the majority of settings feature only a
reduced, one dimensional (1D) decision space. In this paper an experiment is
conducted in which participants judge the stability of two cubes stacked on top
of each other. It is performed in the full 3D setting which features a 2D
decision surface. The analysis yield a shape of a rotated square for the
perceived stability area instead of the commonly reported safety margin in 1D.
This implies a more complex decision behavior in human than previously assumed.

**Comment:** 7 pages, 6 figures

**Relevance Score:** 0.00

---

## 354. "Low Frequency Tweeters Have More to Say!" A New Approach to Identify   Importance of Tweets

**Authors:** Gautam Khannaa, Yeliz Yesilada, Sukru Eraslan, Simon Harper

**Categories:** cs.HC, cs.SI

**Links:** [arXiv](http://arxiv.org/abs/2509.03931v1) | [PDF](http://arxiv.org/pdf/2509.03931v1)

**Abstract:** Twitter is one of the most popular social media platforms.With a large number
of tweets, the activity feed of users becomes noisy, challenging to read, and
most importantly tweets often get lost. We present a new approach to
personalise the ranking of the tweets toward solving the problem of information
overload which is achieved by analysing the relationship between the importance
of tweets to the frequency at which the author tweets. The hypothesis tested is
that "low-frequency tweeters have more to say", i.e. if a user who tweets
infrequently actually goes to the effort of tweeting, then it is more likely to
be of more importance or contain more "meaning" than a tweet by a user who
tweets continuously. We propose six new measures to evaluate the importance of
tweets based on the ability of the tweet to drive interaction among its
readers, which is measured through metrics such as retweets, favourites, and
comments, and the extent of the author's network interacting with the tweet.
Our study shows that users who tweeted less than ten tweets per week were more
likely to be perceived as important by their followers and have the most
important messages. This identified tweet-frequency band could be used to
reorder the activity feed of users and such reordering would ensure the
messages of low-frequency tweeters do not get lost in the stream of tweets.
This could also serve as a scoring index for Twitter users to identify users
frequently tweeting important messages.

**Comment:** 12 pages

**Relevance Score:** 0.00

---

## 355. Map as a By-product: Collective Landmark Mapping from IMU Data and   User-provided Texts in Situated Tasks

**Authors:** Ryo Yonetani, Kotaro Hara

**Categories:** cs.HC

**Links:** [arXiv](http://arxiv.org/abs/2509.03792v1) | [PDF](http://arxiv.org/pdf/2509.03792v1)

**Abstract:** This paper presents Collective Landmark Mapper, a novel map-as-a-by-product
system for generating semantic landmark maps of indoor environments. Consider
users engaged in situated tasks that require them to navigate these
environments and regularly take notes on their smartphones. Collective Landmark
Mapper exploits the smartphone's IMU data and the user's free text input during
these tasks to identify a set of landmarks encountered by the user. The
identified landmarks are then aggregated across multiple users to generate a
unified map representing the positions and semantic information of all
landmarks. In developing the proposed system, we focused specifically on retail
applications and conducted a formative interview with stakeholders to confirm
their practical needs that motivate the map-as-a-byproduct approach. Our user
study demonstrates the feasibility of the proposed system and its superior
mapping performance in two different setups: creating a product availability
map from restocking checklist tasks at a retail store and constructing a room
usage map from office inspection tasks, further demonstrating the potential
applicability to non-retail applications.

**Comment:** (c) 2025 Copyright held by the owner/author(s). Publication rights
  licensed to ACM. This is the author's version of the work. It is posted here
  for your personal use. Not for redistribution. The definitive Version of
  Record was published in Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.
  9, 3, Article 146 (September 2025), https://doi.org/10.1145/3749455

**Relevance Score:** 0.00

---

## 356. Safeguarding Patient Trust in the Age of AI: Tackling Health   Misinformation with Explainable AI

**Authors:** Sueun Hong, Shuojie Fu, Ovidiu Serban, Brianna Bao, James Kinross, Francesa Toni, Guy Martin, Uddhav Vaghela

**Categories:** cs.IR

**Links:** [arXiv](http://arxiv.org/abs/2509.04052v1) | [PDF](http://arxiv.org/pdf/2509.04052v1)

**Abstract:** AI-generated health misinformation poses unprecedented threats to patient
safety and healthcare system trust globally. This white paper presents an
explainable AI framework developed through the EPSRC INDICATE project to combat
medical misinformation while enhancing evidence-based healthcare delivery. Our
systematic review of 17 studies reveals the urgent need for transparent AI
systems in healthcare. The proposed solution demonstrates 95% recall in
clinical evidence retrieval and integrates novel trustworthiness classifiers
achieving 76% F1 score in detecting biomedical misinformation. Results show
that explainable AI can transform traditional 6-month expert review processes
into real-time, automated evidence synthesis while maintaining clinical rigor.
This approach offers a critical intervention to preserve healthcare integrity
in the AI era.

**Relevance Score:** 0.00

---

## 357. dsld: A Socially Relevant Tool for Teaching Statistics

**Authors:** Aditya Mittal, Taha Abdullah, Arjun Ashok, Brandon Zarate Estrada, Shubhada Martha, Billy Ouattara, Jonathan Tran, Norman Matloff

**Categories:** stat.ME, cs.IR, cs.LG, stat.AP

**Links:** [arXiv](http://arxiv.org/abs/2411.04228v3) | [PDF](http://arxiv.org/pdf/2411.04228v3)

**Abstract:** The growing influence of data science in statistics education requires tools
that make key concepts accessible through real-world applications. We introduce
"Data Science Looks At Discrimination" (dsld), an R package that provides a
comprehensive set of analytical and graphical methods for examining issues of
discrimination involving attributes such as race, gender, and age. By
positioning fairness analysis as a teaching tool, the package enables
instructors to demonstrate confounder effects, model bias, and related topics
through applied examples. An accompanying 80-page Quarto book guides students
and legal professionals in understanding these principles and applying them to
real data. We describe the implementation of the package functions and
illustrate their use with examples. Python interfaces are also available.

**Comment:** preprint

**Relevance Score:** 0.00

---

