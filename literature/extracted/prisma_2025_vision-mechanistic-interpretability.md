---
id: "prisma_2025_vision-mechanistic-interpretability"
source_pdf: "../pdf/prisma_2025_vision-mechanistic-interpretability.pdf"
source_filename: "prisma_2025_vision-mechanistic-interpretability.pdf"
format: "academic-paper"
---

## Prisma : An Open Source Toolkit for Mechanistic Interpretability in Vision and Video

Sonia Joseph 1 , 2 , 3 Praneet Suresh 1 , 2 Lorenz Hufe 6

Edward Stevinson 5 Robert Graham 2 Yash Vadi 1 , 4 Danilo Bzdok 1 , 2 Sebastian Lapuschkin 6 , 7 Lee Sharkey 8 Blake Aaron Richards 1 , 2

1 Mila Quebec 2 McGill University 3 Meta 4 Universit´ e de Montr´ eal 5 Imperial College London 6 Fraunhofer Heinrich Hertz Institute 7 Technological University Dublin 8 Apollo Research

sonia.joseph@mila.quebec (Corresponding author)

Figure 1. Prisma adapts popular vision model repositories for key mechanistic interpretability techniques.

<!-- image -->

In contrast, vision mechanistic interpretability has lagged behind partially due to limited access to tools, pretrained weights, and standardized implementation code necessary for scalable research.

Prisma bridges this gap by providing:

1. Hooked Vision Transformers. A unified interface for accessing and modifying Vision Transformer (ViT) [14] and Video Transformer activations and weights across 75+ models from timm, OpenCLIP, and Huggingface, enabling diverse downstream interpretability tasks, similar to TransformerLens [52].
2. SAEs, Transcoders, and Crosscoders. Support for training and evaluating SAEs, and their variants, transcoders and crosscoders [5, 11, 16, 47]. We also open source 80+ pre-trained SAE weights for CLIPB and DINO-B ViTs [6, 29, 57, 58] for all layers, transcoders for CLIP for all layers, with accompanying evaluations for all models (Sec. 10).
3. Interpretability Tools. Asuite of tools, including methods for circuit analysis, attention head visualization, and logit lens techniques [55].
4. Educational Resources. Tutorial notebooks, and documentation to ensure rapid onboarding. Additionally, toy vision transformers checkpoints are provided in the spirit

## Abstract

Robust tooling and publicly available pre-trained models have helped drive recent advances in mechanistic interpretability for language models. However, similar progress in vision mechanistic interpretability has been hindered by the lack of accessible frameworks and pre-trained weights. We present Prisma 1 , an open-source framework designed to accelerate vision mechanistic interpretability research, providing a unified toolkit for accessing 75+ vision and video transformers; support for sparse autoencoder (SAE), transcoder, and crosscoder training; a suite of 80+ pretrained SAE weights; activation caching, circuit analysis tools, and visualization tools; and educational resources. Our analysis reveals surprising findings, including that effective vision SAEs can exhibit substantially lower sparsity patterns than language SAEs, and that in some instances, SAE reconstructions can decrease model loss. Prisma enables new research directions for understanding vision model internals while lowering barriers to entry in this emerging field.

## 1. Introduction

Mechanistic interpretability has emerged as a powerful approach for understanding language models [5, 42]. This progress is driven by robust tools for circuit analysis and Sparse Autoencoder (SAE) training, enabled by features such as activation caching [21, 52], and the public release of pre-trained SAEs and training frameworks [18, 36].

1 Access the codebase here: https://github.com/PrismaMultimodal/ViT-Prisma

of [20] to enable interpretability work in compute constrained environments.

In this paper, we detail the contributions listed above. Then, using the Prisma toolkit, we explore some surprising findings about vision SAEs, including the high density of vision SAEs compared to language SAEs, and cases where injecting SAE activations into the forward pass can decrease the original model's loss. We intend for the Prisma library to facilitate deeper investigations into vision transformer internals, accelerating the field of vision mechanistic interpretability.

## 2. Background and Related Work

## 2.1. Mechanistic interpretability for vision and video

Mechanistic interpretability in language has been flourishing, emphasizing causal circuit-tracing with sparse model representations to reverse engineer models [9, 20, 50, 66]. Prior work on vision model interpretability has generally used different techniques from those that characterize current language mechanistic interpretability, such as feature visualization and attention analysis [6, 54, 59]. Recently, mechanistic interpretability for vision models has been gaining traction [1, 23, 24, 33, 37, 39], but research has been limited by the lack of scalable tooling, a gap which Prisma seeks to address.

Mechanistic interpretability for video remains limited, hindered by high compute costs and a lack of dedicated tools [34, 37]. Some methods automatically discover spatiotemporal concepts [40], while others rely on proxy tasks to assess the model's use of dynamic information [27, 30], sensitivity to scene bias [7, 43, 44], and difference between static and dynamic signals in intermediate representations [41]. We integrate two widely used video encoders, ViViT [2] and V-JEPA [3], into Prisma to facilitate interpretability research in video models.

## 2.2. Mechanistic interpretability methods

Neural networks represent more features than available neurons, making individual neurons polysemantic and difficult to interpret [19]. SAEs decompose single-layer representations into interpretable features [5, 11], while transcoders extend this across layers for a more faithful approximation of feedforward computation [15, 49, 63]. Crosscoders further track persistent features across layers and models [47].

Circuit analysis identifies interactions between model components to explain behaviors [66]. Techniques include activation patching [28], which modifies activations in contrast pairs to isolate key components, and attribution patching [51], a faster gradient-based alternative. Interpretability has shifted from manual unit analysis to automated circuit discovery, with algorithms like ACDC [9] for language and vision transformers [62].

## 2.3. Vision and video architectures

Vision transformers (ViTs) process images by splitting them into tokenized patches, enabling self-attention analysis while achieving CNN-level performance [14]. CLIP [58] aligns visual and language representations via contrastive learning for zero-shot transfer, while DINO [6] employs self-distillation for self-supervised learning.

Video transformers extend these concepts to sequential frames, including supervised models like ViViT [2] and unsupervised models like V-JEPA [3]. We integrate these architectures into HookedViT, our vision model class that captures layer-wise activations for interpretability and sparse coder training.

## 2.4. Open source mechanistic interpretability tools

Mechanistic interpretability in language has a growing ecosystem of tools. TransformerLens enables circuit-style analysis of language models [52], while NNsight supports model intervention but is not vision-specific [21]. CircuitsVis enables mechanistic interpretability visualizations [10]. SAELens aids in training and analyzing SAEs [36], and Gao et al. [26] released Top-K SAE training code, later extended by Eleuther [18]. Gemma Scope provides open source SAEs on Gemma 2 [45]. For vision, Joseph and Nanda [33] introduced a circuit-based framework for transformers, which Prisma extends with sparse coder training. However, no other open-source toolkits exist specifically for vision mechanistic interpretability, which differs from language mechanistic interpretability in tokenization, architecture, and analysis.

## 2.5. Differences between vision and language mechanistic interpretability

Vision and language transformers, despite sharing decoderonly architecture, differ fundamentally in ways that impact interpretability methods [32, 33]. Language input is discrete while vision input is continuous, creating differences in tokenization, processing, and decoding. ViTs use spatial patch tokenization without a canonical dictionary and use a learnable CLS token, which leads to distinct statistical properties between the two token types. Vision models typically employ contrastive learning or unsupervised approaches rather than next-token prediction. Due to the richness of visual representation, the outputs typically call for specialized decoding like cosine similarity or diffusion decoding instead of analyzing the next-token logit prediction. These architectural and functional differences call for specialized vision interpretability tools rather than adapting existing language interpretability libraries.

## 3. The Prisma Open Source Library

In this section, we introduce Prisma, an open-source toolkit designed to advance mechanistic interpretability in vision models. Building on the successes of interpretability tools in the language domain, Prisma adapts and extends these techniques to tackle the unique challenges encountered in vision. By retaining the naming conventions and hook interfaces established by TransformerLens [52], Prisma enables researchers to easily transition between language and vision tooling.

## 3.1. Prisma Codebase

## 3.1.1. Models Supported

Our model registry integrates 75+ pretrained model configurations into HookedViT with TransformerLens-inspired activation caching for easy circuit analysis and SAE training. Support extends Hugging Face, OpenCLIP, and timm models (ViT, CLIP-VIT, DINO), and includes videospecific features like 3D tubelet embeddings for video models (ViViT, V-JEPA). We also adapt the Kandinsky ViT encoder with its pretrained SAE for steering diffusion models for image generation. The model registry structure enables convenient adoption of freshly released open-source models from the broader open-source community.

## 3.1.2. Codebase Structure

Built around HookedViT, key components include:

- Circuit Analysis: Supports ablations via forward pass interventions. The HookedSAEViT module enables sparse feature circuit analysis through layer replacement with multiple SAEs or sparse coders. The HookedViT module integrates with ACDC for automatic circuit discovery.
- Sparse Coder Architectures: Enables training SAEs, transcoders, and crosscoders from HookedViT representations, with ReLU [5, 11], Top-K [25], and JumpReLU [61], and Gated [60] implementations.
- Configurable SAE Training: Parameters for encoder and decoder initialization [8], activation normalization [64], ghost gradients [31], dead feature resampling [5], and early stopping criteria. We have options for activation caching for latency improvement [36, 48], or alternatively, on-the-fly computation for VRAM optimization [17]. We chose to make the config expressive rather than minimal to accommodate emerging vision SAE best practices.
- SAE Evaluation Tools: Measures metrics like sparsity (L0), reconstruction and substitution losses [5], and identifies maximally activating images [46] for trained sparse coders.
- Visualization: Provides logit lens and attention visualization for intermediate activations.

## 4. Prisma Open Source Sparse Coders

We release SAEs for all layers of the CLIP and DINO base models and transcoders for CLIP base trained on ImageNet1k ( Tab. 1). For CLIP, we provide three token configurations: all patches, CLS-only, and spatial tokens, with standard ReLu and Top-K ( k = 64 ) variants. Key training config details include:

- Expansion Factor: x64, mapping a 768-dimensional activation space to a dictionary of 49,152 features.
- Weight Initialization: Encoder weights are set as the transpose of the decoder weights.
- Training: Conducted on ImageNet1k [13] for one epoch using the Adam optimizer [38]. Learning rates were swept from 1 × 10 - 5 to 1 × 10 - 1 with a cosine annealing schedule preceeded by a 200-step warmup and a batch size of 4096.
- L1 coefficients: were swept over [10 - 11 , 1] ) for optimal sparsity reconstruction tradeoff.
- Auxiliary Loss: Ghost gradients were added to prevent dead features.

For a full list of the SAEs, see Table 1, and for evaluation metrics, see Appendix 4.

## 4.1. Open Source Toy Vision Transformers

We release toy ViTs trained on ImageNet with 1-4 layers (Sec. 9.1). These models include both full transformer architectures and attention-only variants, with multiple training checkpoints provided. Toy models enable researchers to validate hypotheses in a controlled environment before scaling to full-size networks-a strategy that has proven valuable in language mechanistic interpretability [20, 53, 56].

## 5. Preliminary SAE Analysis with Prisma

We use the Prisma toolkit to briefly present two unexpected observations about CLIP SAEs: 1) The internal representations of CLIP ViTs appear much less sparse than in language, leaving optimal sparsity for vision SAEs an open question. 2) There is sometimes a decrease in the model's cross-entropy loss when injecting SAE reconstructions into the forward pass, which may be explained by the denoising properties of SAEs. We present our analysis here and leave further exploration to the broader vision mechanistic interpretability community.

## 5.1. Vision SAE Sparsity is Higher than Language SAE Sparsity

High-quality SAEs balance low sparsity, measure by L0 (number of active SAE latents per token), with high explained variance. For language models like GPT-2 Small, the L0 ranges from 12-74 per token [4], with top-K SAEs performing best at K = 32-256 [25].

Vision SAEs exhibit significantly higher L0 [35]: approximately 500+ per patch in CLIP-B/32 while retaining a comparable explained variance to language (see SAE evaluation tables in Sec. 10). Even with improved training techniques, our top-K SAEs (K = 64, 128) still show polysemantic features, suggesting higher K values may be necessary. Analysis of a pretrained vision SAE [12] from the Kandinsky encoder reveals an extreme L0 of 122,631 per CLS token, despite its demonstrated effectiveness in diffusion model steering.

Wepropose four potential explanations for this disparity:

1. Information Density Differences. Visual inputs may have an inherently denser distribution than language, lack the power-law distribution of natural language, and may require more extensive preprocessing.
2. Patch Granularity. CLIP-B/32 processes approximately 50 patches compared to GPT-2's 1024 tokens, a 20× difference. When normalized (25 per patch), vision L0 becomes more comparable to language. The relationship between patch size and L0 warrants further investigation.
3. Patch Type Specialization. Spatial patches and CLS tokens exhibit distinct distributional properties [35, 46, 65]. The CLS token's high L0 encapsulates a global image representation. When normalized (L0 ≈ 479 per patch) and adjusted to language-scale tokenization (L0 ≈ 120 ), this aligns with language SAE sparsity. Our analysis of dead SAE features reveals complementary processing patterns: the CLS token has less dead features in deeper layers as it builds a global representation, while spatial patches show an inverse trend as information moves to the CLS token (Fig. 2). This suggests optimal top-K varies by token type and layer depth, with CLS tokens requiring higher K in later layers to accommodate their enriched representations.
4. Domain-Specific Optimization. Current vision SAE training methods are adapted from language mechanistic interpretability, potentially limiting achievable sparsity. Vision-specific SAE training techniques may yield sparser representations, but are yet to be developed.

This apparent sparsity discrepancy across modalities remains an open question, and we hope these insights serve as a starting point for further research.

## 5.2. SAE Reconstructions Can Improve Model Loss

Our analysis reveals the unexpected finding that inserting SAEreconstructions into the forward pass can reduce crossentropy loss in the original model. In particular, the Vanilla and Top K = 64 CLS-only CLIP SAE sets (Sec. 10, Tab. 4 and Tab. 6), and the vanilla DINO set (Tab. 9) have layers showing this phenomenon. For the Top K = 64 CLS-only SAEs, the effect intensifies as the CLS token representations develop through network layers (Fig. 3), suggesting a potential denoising mechanism at work, and aligning with a previous observation of SAE reconstructions reducing the loss in CLIP's final layer [22]. Interestingly, we do not observe significant improvements in the CLIP transcoders (Tab. 8), CLIP Vanilla Spatial Patch SAEs (Tab. 6), or CLIP Vanilla CLS-only SAEs (Tab. 5). The precise mechanism behind this improvement remains unclear and presents a compelling avenue for future research by the vision mechanistic interpretability community.

Figure 2. The percentage of alive features shows that CLS tokens require more features in deeper layers while spatial tokens show declining utilization, showing how information shifts from spatial patches to a global representation. Results are from the vanilla CLS SAEs (Tab. 5).

<!-- image -->

Figure 3. Using an SAE during the model's forward pass decreases the original model's loss, with results becoming more dramatic in later layers. Red dots indicate for which layers the performance is improved. Results are from the CLIP-B Top K = 64 SAEs (Tab. 6).

<!-- image -->

## 6. Conclusion

Prisma bridges language and vision interpretability techniques with an extensive toolkit including 80+ SAEs for CLIP, DINO, and video transformers, and experimental toy models. Our analysis reveals vision representations are less sparse than language models, and SAE reconstructions can improve model performance, especially for CLS tokens in deeper layers. We invite researchers to build on these findings and contribute to Prisma's open-source development

## 7. Acknowledgements

We thank Joseph Bloom, Neel Nanda, Jacob Dunefsky, and Leo Gao for the discussions during the early days of the repository.

## References

- [1] Reduan Achtibat, Maximilian Dreyer, Ilona Eisenbraun, Sebastian Bosse, Thomas Wiegand, Wojciech Samek, and Sebastian Lapuschkin. From attribution maps to human-understandable explanations through concept relevance propagation. Nature Machine Intelligence , 5(9):10061019, 2023. 2
- [2] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, and Cordelia Schmid. Vivit: A video vision transformer. CoRR , abs/2103.15691, 2021. 2
- [3] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video, 2024. 2
- [4] Joseph Bloom. Open source sparse autoencoders for all residual stream layers of gpt2 small. https://www.alignmentforum.org/posts/ f9EgfLSurAiqRJySD / open - source - sparse - autoencoders - for - all - residual - stream , 2024. 3
- [5] Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Thomas Conerly, Nick Turner, Cem Anil, Catherine Denison, Amanda Askell, Richard Lasenby, Yuntao Wu, Stephen Kravec, Noa Schiefer, Thomas Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Katy Nguyen, Brayden McLean, James E. Burke, Thomas Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread , 2023. 1, 2, 3
- [6] Mathilde Caron, Hugo Touvron, Ishan Misra, Herv´ e J´ egou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision , pages 9650-9660, 2021. 1, 2
- [7] Jinwoo Choi, Chen Gao, Joseph C. E. Messou, and Jia-Bin Huang. Why can't i dance in the mall? learning to mitigate scene bias in action recognition, 2019. 2
- [8] Tom Conerly, Hoagy Cunningham, Adly Templeton, Jack Lindsey, Basil Hosmer, and Adam Jermyn. Dictionary learning optimization techniques, 2025. Accessed: 2025-03-10. 3
- [9] Arthur Conmy, Augustine N Mavor-Parker, Aengus Lynch, Stefan Heimersheim, and Adri` a Garriga-Alonso. Towards automated circuit discovery for mechanistic interpretability. arXiv preprint arXiv:2304.14997 , 2023. 2
- [10] Alan Cooney and Neel Nanda. Circuitsvis. https://github.com/TransformerLensOrg/ CircuitsVis , 2023. 2
- [11] Hoagy Cunningham, Arthur Ewart, Lewis Smith, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable model directions. arXiv preprint arXiv:2309.08600 , 2023. 1, 2, 3
- [12] Gytis Daujotas. Interpreting and steering features in images. URL https://www. lesswrong. com/posts/Quqekpvx8BGMMcaem/interpreting-andsteering-features-in-images , 2024. 4
- [13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition , pages 248-255. Ieee, 2009. 3
- [14] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 , 2020. 1, 2
- [15] Jacob Dunefsky, Peter Chlenski, and Neel Nanda. Transcoders enable fine-grained interpretable circuit analysis for language models. 2024. 2
- [16] Jacob Dunefsky, Philippe Chlenski, and Neel Nanda. Transcoders find interpretable llm feature circuits. Advances in Neural Information Processing Systems , 37:24375-24410, 2025. 1
- [17] EleutherAI. Sparsify. GitHub, 2024. Accessed: 2025-03-10. 3
- [18] EleutherAI. Sparsify: Sparsify transformers with saes and transcoders, 2024. 1, 2
- [19] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac HatfieldDodds, Robert Lasenby, Dawn Drain, Carol Chen, Roger Grosse, Sam McCandlish, Jared Kaplan, Dario Amodei, Martin Wattenberg, and Christopher Olah. Toy models of superposition, 2022. 2
- [20] Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac HatfieldDodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652 , 2022. 2, 3
- [21] Jaden Fiotto-Kaufman, Alexander R Loftus, Eric Todd, Jannik Brinkmann, Caden Juang, Koyena Pal, Can Rager, Aaron Mueller, Samuel Marks, Arnab Sen Sharma, Francesca Lucchetti, Michael Ripa, Adam Belfki, Nikhil Prakash, Sumeet Multani, Carla Brodley, Arjun Guha, Jonathan Bell, Byron Wallace, and David Bau. Nnsight and ndif: Democratizing access to foundation model internals. 2024. 1, 2
- [22] Hugo Fry. Towards multimodal interpretability: Learning sparse interpretable features in vision transformers. https://www.lesswrong.com/posts/ bCtbuWraqYTDtuARg / towards - multimodal - interpretability-learning-sparse-2 , 2024. 4
- [23] Yossi Gandelsman, Alexei A. Efros, and Jacob Steinhardt. Interpreting the second-order effects of neurons in clip, 2024. 2
- [24] Yossi Gandelsman, Alexei A. Efros, and Jacob Steinhardt. Interpreting clip's image representation via text-based decomposition, 2024. 2

- [25] Leo Gao, Tom Dupr´ e la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093 , 2024. 3
- [26] Leo Gao, Tom Dupr´ e la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders, 2024. 2
- [27] Amir Ghodrati, Efstratios Gavves, and Cees G. M. Snoek. Video time: Properties, encoders and evaluation. CoRR , abs/1807.06980, 2018. 2
- [28] Stefan Heimersheim and Neel Nanda. How to use and interpret activation patching, 2024. 2
- [29] Gabriel Ilharco, Mitchell Wortsman, Ross Wightman, Cade Gordon, Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar, Hongseok Namkoong, John Miller, Hannaneh Hajishirzi, Ali Farhadi, and Ludwig Schmidt. Openclip, 2021. 1
- [30] Filip Ilic, Thomas Pock, and Richard P. Wildes. Is appearance free action recognition possible?, 2022. 2
- [31] A Jermyn and A Templeton. Ghost grads: An improvement on resampling, 2024. Accessed: 2025-03-10. 3
- [32] Sonia Joseph. Vit-prisma main demo. Google Colab Jupyter Notebook, 2024. Accessed: March 10, 2025. 2
- [33] Sonia Joseph and Neel Nanda. Laying the foundations for vision and multimodal mechanistic interpretability &amp; open problems. AI Alignment Forum , 2024. 2
- [34] S. Joseph, A. Zholus, M. R. Samsami, and B. A. Richards. Mining the diamond miner: Mechanistic interpretability on the video pretraining agent. ATTRIB Workshop at NeurIPS 2023, 2023. Available at: https://openreview.net/ pdf?id=lDeysxpH6W . 2
- [35] Sonia Joseph, Praneet Suresh, Ethan Goldfarb, Lorenz Hufe, Yossi Gandelsman, Robert Graham, Danilo Bzdok, Wojciech Samek, and Blake Aaron Richards. Steering clip's vision transformer with sparse autoencoders, 2025. 4
- [36] Curt Tigges Joseph Bloom and David Chanin. Saelens. https://github.com/jbloomAus/SAELens , 2024. 1, 2, 3
- [37] Karolis Jucys, George Adamopoulos, Mehrab Hamidi, Stephanie Milani, Mohammad Reza Samsami, Artem Zholus, Sonia Joseph, Blake Richards, Irina Rish, and ¨ Ozg¨ ur S ¸ ims ¸ek. Interpretability in action: Exploratory analysis of vpt, a minecraft agent. arXiv preprint arXiv:2407.12161 , 2024. 2
- [38] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980 , 2014. 3
- [39] Piotr Komorowski, Hubert Baniecki, and Przemyslaw Biecek. Towards evaluating explanations of vision transformers for medical imaging. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition , pages 3726-3732, 2023. 2
- [40] Matthew Kowal, Achal Dave, Rares Ambrus, Adrien Gaidon, Konstantinos G. Derpanis, and Pavel Tokmakov. Understanding video transformers via universal concept discovery. In Proceedings of the IEEE/CVF Conference on
17. Computer Vision and Pattern Recognition (CVPR) , pages 10946-10956, 2024. 2
- [41] Matthew Kowal, Mennatullah Siam, Md Amirul Islam, Neil D. B. Bruce, Richard P. Wildes, and Konstantinos G. Derpanis. Quantifying and learning static vs. dynamic information in deep spatiotemporal networks, 2024. 2
- [42] J´ anos Kram´ ar, Tom Lieberum, Rohin Shah, and Neel Nanda. Atp*: An efficient and scalable method for localizing llm behaviour to components, 2024. 1
- [43] Yi Li and Nuno Vasconcelos. Repair: Removing representation bias by dataset resampling, 2019. 2
- [44] Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European Conference on Computer Vision (ECCV) , 2018. 2
- [45] Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, J´ anos Kram´ ar, Anca Dragan, Rohin Shah, and Neel Nanda. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2, 2024. 2
- [46] Hyesu Lim, Jinho Choi, Jaegul Choo, and Steffen Schneider. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. arXiv preprint arXiv:2412.05276 , 2024. 3, 4
- [47] Jack Lindsey, Adly Templeton, Jonathan Marcus, Thomas Conerly, Joshua Batson, and Christopher Olah. Sparse crosscoders for cross-layer features and model diffing. Transformer Circuits Thread , 2024. Research Update. 1, 2
- [48] Sam Marks. Some open-source dictionaries and dictionary learning infrastructure, 2023. Accessed: 2025-03-10. 3
- [49] Samuel Marks. dictionary learning. 2024. 2
- [50] Samuel Marks, Can Rager, Eric J. Michaud, Yonatan Belinkov, David Bau, and Aaron Mueller. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models, 2024. 2
- [51] Neel Nanda. Attribution patching: Activation patching at industrial scale, 2023. 2
- [52] Neel Nanda and Joseph Bloom. Transformerlens. https://github.com/TransformerLensOrg/ TransformerLens , 2022. 1, 2, 3
- [53] Neel Nanda, Lawrence Chan, Tom Lieberum, Jess Smith, and Jacob Steinhardt. Progress measures for grokking via mechanistic interpretability. arXiv preprint arXiv:2301.05217 , 2023. 3
- [54] Muzammal Naseer, Kanchana Ranasinghe, Salman Khan, Fahad Shahbaz Khan, and Fatih Porikli. Intriguing properties of vision transformers. Advances in Neural Information Processing Systems , 34:23296-23308, 2021. 2
- [55] nostalgebraist. Interpreting gpt: The logit lens, 2020. LessWrong. 1
- [56] Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. Incontext learning and induction heads. arXiv preprint arXiv:2209.11895 , 2022. 3
- [57] Maxime Oquab, Timoth´ ee Darcet, Th´ eo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 , 2023. 1

- [58] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020 , 2021. 1, 2
- [59] Maithra Raghu, Thomas Unterthiner, Simon Kornblith, Chiyuan Zhang, and Alexey Dosovitskiy. Do vision transformers see like convolutional neural networks? Advances in Neural Information Processing Systems , 34:12116-12128, 2021. 2
- [60] Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Tom Lieberum, Vikrant Varma, J´ anos Kram´ ar, Rohin Shah, and Neel Nanda. Improving dictionary learning with gated sparse autoencoders, 2024. 3
- [61] Senthooran Rajamanoharan, Tom Lieberum, Nicolas Sonnerat, Arthur Conmy, Vikrant Varma, J´ anos Kram´ ar, and Neel Nanda. Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders, 2024. 3
- [62] Achyuta Rajaram, Neil Chowdhury, Antonio Torralba, Jacob Andreas, and Sarah Schwettmann. Automatic discovery of visual circuits, 2024. 2
- [63] Adly Templeton, Joshua Batson, Adam Jermyn, and Christopher Olah. Predicting future activations. 2024. 2
- [64] Adly Templeton, Tom Conerly, Jonathan Marcus, Tom Henighan, Anna Golubeva, and Trenton Bricken. Improvements to dictionary learning, 2024. Accessed: 2025-03-10. 3
- [65] Martina G Vilas, Timothy Schauml¨ offel, and Gemma Roig. Analyzing vision transformers for image classification in class embedding space. Advances in neural information processing systems , 36:40030-40041, 2023. 4
- [66] Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, and Jacob Steinhardt. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. arXiv preprint arXiv:2211.00593 , 2022. 2

## Prisma : An Open Source Toolkit for Mechanistic Interpretability in Vision and Video

## Supplementary Material

## 8. Open Source SAE Suite List

We train sparse coders on all twelve layers of the residual stream for CLIP-B-32 and DINO-B-32 in the following kinds:

Table 1. Open Source SAE and Transcoder Weights for All Layers

| Model | Coder | Version | Token |
| - | - | - | - |
| CLIP-B | SAE | ReLU | All |
| CLIP-B | SAE | ReLU | CLS |
| CLIP-B | SAE | Top K (K = 64) | CLS |
| CLIP-B | SAE | ReLU | Spatial |
| CLIP-B | SAE | Top K (K = 64) | Spatial |
| CLIP-B | SAE | Top K (K = 128) | Spatial |
| DINO-B | SAE | ReLU | All |
| CLIP | Transcoder | ReLU | All |

## 9. Details of Toy Vision Transformer Models

## 9.1. ImageNet

Our primary toy ViTs use a patch size of 16. For enhanced attention-head visualization, we include a subset with a patch size of 32 (see Appendix 9.1). The attention-only variants are more amenable to mathematical analysis than full transformer architectures.

## 9.1.1. Patch Size 16

Table 2. Accuracy [Top-1 | Top-5]

| Size | Num Layers | Attn+MLP | Attn+MLP | Attn-Only | Attn-Only |
| - | - | - | - | - | - |
| tiny | 1 | 0.16 | &#124; 0.33 | 0.11 | &#124; 0.25 |
| base | 2 | 0.23 | &#124; 0.44 | 0.16 | &#124; 0.34 |
| small | 3 | 0.28 | &#124; 0.51 | 0.17 | &#124; 0.35 |
| medium | 4 | 0.33 | &#124; 0.56 | 0.17 | &#124; 0.36 |

## 9.1.2. Patch Size 32

We trained a toy ViT with a larger patch size ViT (32), allowing for inspectable attention heads by the attention head visualizer, whereas patch size 16 attention heads are too large to easily render in JavaScript. This model includes fifty training checkpoints for analyzing training dynamics.

Table 3. ImageNet-1k Classification Checkpoints (patch size 32) [Top-1 | Top-5].

| Size | Num Layers | Attn+MLP | Attn-Only |
| - | - | - | - |
| tiny | 3 | 0.22 &#124; 0.42 | N/A |

## 10. SAE Evaluations

Below are our SAE evaluations from the SAEs that we trained in Sec. 4.

## 10.1. CLIP-ViT-B-32 SAEs

## Vanilla SAEs (All Patches)

Table 4. CLIP-ViT-B-32 vanilla sparse autoencoder performance metrics for all patches (CLS + spatial).

| Layer | Sublayer | l1 coeff. | %Explained var. | Avg L0 | Avg CLS L0 | Cos sim | Recon cos sim | CE | Recon CE | Zero abl CE | %CErecovered | %Alive features | Model |
| - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| 0 | mlp out | 1e-5 | 98.7 | 604.44 | 36.92 | 0.994 | 0.998 | 6.762 | 6.762 | 6.779 | 99.51 | 100 | link |
| 0 | resid post | 1e-5 | 98.6 | 1110.9 | 40.46 | 0.993 | 0.988 | 6.762 | 6.763 | 6.908 | 99.23 | 100 | link |
| 1 | mlp out | 1e-5 | 98.4 | 1476.8 | 97.82 | 0.992 | 0.994 | 6.762 | 6.762 | 6.889 | 99.40 | 100 | link |
| 1 | resid post | 1e-5 | 98.3 | 1508.4 | 27.39 | 0.991 | 0.989 | 6.762 | 6.763 | 6.908 | 99.02 | 100 | link |
| 2 | mlp out | 1e-5 | 98.0 | 1799.7 | 376.0 | 0.992 | 0.998 | 6.762 | 6.762 | 6.803 | 99.44 | 100 | link |
| 2 | resid post | 5e-5 | 90.6 | 717.84 | 10.11 | 0.944 | 0.960 | 6.762 | 6.767 | 6.908 | 96.34 | 100 | link |
| 3 | mlp out | 1e-5 | 98.1 | 1893.4 | 648.2 | 0.992 | 0.999 | 6.762 | 6.762 | 6.784 | 99.54 | 100 | link |
| 3 | resid post | 1e-5 | 98.1 | 2053.9 | 77.90 | 0.989 | 0.996 | 6.762 | 6.762 | 6.908 | 99.79 | 100 | link |
| 4 | mlp out | 1e-5 | 98.1 | 1901.2 | 1115.0 | 0.993 | 0.999 | 6.762 | 6.762 | 6.786 | 99.55 | 100 | link |
| 4 | resid post | 1e-5 | 98.0 | 2068.3 | 156.7 | 0.989 | 0.997 | 6.762 | 6.762 | 6.908 | 99.74 | 100 | link |
| 5 | mlp out | 1e-5 | 98.2 | 1761.5 | 1259.0 | 0.993 | 0.999 | 6.762 | 6.762 | 6.797 | 99.76 | 100 | link |
| 5 | resid post | 1e-5 | 98.1 | 1953.8 | 228.5 | 0.990 | 0.997 | 6.762 | 6.762 | 6.908 | 99.80 | 100 | link |
| 6 | mlp out | 1e-5 | 98.3 | 1598.0 | 1337.0 | 0.993 | 0.999 | 6.762 | 6.762 | 6.789 | 99.83 | 100 | link |
| 6 | resid post | 1e-5 | 98.2 | 1717.5 | 321.3 | 0.991 | 0.996 | 6.762 | 6.762 | 6.908 | 99.93 | 100 | link |
| 7 | mlp out | 1e-5 | 98.2 | 1535.3 | 1300.0 | 0.992 | 0.999 | 6.762 | 6.762 | 6.796 | 100.17 | 100 | link |
| 7 | resid post | 1e-5 | 98.2 | 1688.4 | 494.3 | 0.991 | 0.995 | 6.762 | 6.761 | 6.908 | 100.24 | 100 | link |
| 8 | mlp out | 1e-5 | 97.8 | 1074.5 | 1167.0 | 0.990 | 0.998 | 6.762 | 6.761 | 6.793 | 100.57 | 100 | link |
| 8 | resid post | 1e-5 | 98.2 | 1570.8 | 791.3 | 0.991 | 0.992 | 6.762 | 6.761 | 6.908 | 100.41 | 100 | link |
| 9 | mlp out | 1e-5 | 97.6 | 856.68 | 1076.0 | 0.989 | 0.998 | 6.762 | 6.762 | 6.792 | 100.28 | 100 | link |
| 9 | resid post | 1e-5 | 98.2 | 1533.5 | 1053.0 | 0.991 | 0.989 | 6.762 | 6.761 | 6.908 | 100.32 | 100 | link |
| 10 | mlp out | 1e-5 | 98.1 | 788.49 | 965.5 | 0.991 | 0.998 | 6.762 | 6.762 | 6.772 | 101.50 | 99.80 | link |
| 10 | resid post | 1e-5 | 98.4 | 1292.6 | 1010.0 | 0.992 | 0.987 | 6.762 | 6.760 | 6.908 | 100.83 | 99.99 | link |
| 11 | mlp out | 5e-5 | 89.7 | 748.14 | 745.5 | 0.972 | 0.993 | 6.762 | 6.759 | 6.768 | 135.77 | 100 | link |
| 11 | resid post | 1e-5 | 98.4 | 1405.0 | 1189.0 | 0.993 | 0.987 | 6.762 | 6.765 | 6.908 | 98.03 | 99.99 | link |

## Vanilla SAEs (CLS only)

Table 5. CLIP-ViT-B-32 vanilla sparse autoencoder performance metrics for CLS tokens.

| Layer | Sublayer | l1 coeff. | %Explained var. | Avg CLS L0 | Cos sim | Recon cos sim | CE | Recon CE | Zero abl CE | %CErecovered | %Alive features | Model |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| 0 | resid post | 2e-8 | 82 | 934.83 | 0.98008 | 0.99995 | 6.7622 | 6.7622 | 6.9084 | 99.9984 | 4.33 | link |
| 1 | resid post | 8e-6 | 85 | 314.13 | 0.97211 | 0.99994 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 2.82 | link |
| 2 | resid post | 9e-8 | 96 | 711.84 | 0.98831 | 0.99997 | 6.7622 | 6.7622 | 6.9083 | 99.9977 | 2.54 | link |
| 3 | resid post | 1e-8 | 95 | 687.41 | 0.98397 | 0.99994 | 6.7622 | 6.7622 | 6.9085 | 99.9998 | 4.49 | link |
| 4 | resid post | 9e-8 | 95 | 681.08 | 0.98092 | 0.99988 | 6.7622 | 6.7622 | 6.9082 | 100.00 | 15.75 | link |
| 5 | resid post | 1e-7 | 94 | 506.77 | 0.97404 | 0.99966 | 6.7622 | 6.7622 | 6.9081 | 99.9911 | 16.80 | link |
| 6 | resid post | 1e-8 | 92 | 423.70 | 0.96474 | 0.99913 | 6.7622 | 6.7622 | 6.9083 | 99.9971 | 29.46 | link |
| 7 | resid post | 2e-6 | 88 | 492.68 | 0.93899 | 0.99737 | 6.7622 | 6.7622 | 6.9082 | 99.9583 | 51.68 | link |
| 8 | resid post | 4e-8 | 76 | 623.01 | 0.89168 | 0.99110 | 6.7622 | 6.7625 | 6.9087 | 99.7631 | 82.07 | link |
| 9 | resid post | 1e-12 | 74 | 521.90 | 0.87076 | 0.98191 | 6.7622 | 6.7628 | 6.9083 | 99.5425 | 93.68 | link |
| 10 | resid post | 3e-7 | 74 | 533.94 | 0.87646 | 0.96514 | 6.7622 | 6.7635 | 6.9082 | 99.1070 | 99.98 | link |
| 11 | resid post | 1e-8 | 65 | 386.09 | 0.81890 | 0.89607 | 6.7622 | 6.7853 | 6.9086 | 84.1918 | 99.996 | link |

Top K SAEs (CLS only, k = 64 )

Table 6. CLIP-ViT-B-32 topk sparse autoencoder performance metrics for CLS tokens ( k = 64 ).

| Layer | Sublayer | %Explained var. | Avg CLS L0 | Cos sim | Recon cos sim | CE | Recon CE | Zero abl CE | %CErecovered | %Alive features | Model |
| - | - | - | - | - | - | - | - | - | - | - | - |
| 0 | resid post | 90 | 64 | 0.98764 | 0.99998 | 6.7622 | 6.7622 | 6.9084 | 99.995 | 46.80 | link |
| 1 | resid post | 96 | 64 | 0.99429 | 0.99999 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 4.86 | link |
| 2 | resid post | 96 | 64 | 0.99000 | 0.99998 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 5.50 | link |
| 3 | resid post | 95 | 64 | 0.98403 | 0.99995 | 6.7622 | 6.7622 | 6.9085 | 100.00 | 5.21 | link |
| 4 | resid post | 94 | 64 | 0.97485 | 0.99986 | 6.7621 | 6.7622 | 6.9082 | 99.998 | 6.81 | link |
| 5 | resid post | 93 | 64 | 0.96985 | 0.99962 | 6.7622 | 6.7622 | 6.9081 | 99.997 | 21.89 | link |
| 6 | resid post | 92 | 64 | 0.96401 | 0.99912 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 28.81 | link |
| 7 | resid post | 90 | 64 | 0.95057 | 0.99797 | 6.7622 | 6.7621 | 6.9082 | 100.03 | 65.84 | link |
| 8 | resid post | 87 | 64 | 0.93029 | 0.99475 | 6.7622 | 6.7620 | 6.9087 | 100.11 | 93.75 | link |
| 9 | resid post | 85 | 64 | 0.91814 | 0.98865 | 6.7622 | 6.7616 | 6.9083 | 100.43 | 98.90 | link |
| 10 | resid post | 86 | 64 | 0.93072 | 0.97929 | 6.7622 | 6.7604 | 6.9082 | 101.19 | 94.55 | link |
| 11 | resid post | 84 | 64 | 0.91880 | 0.94856 | 6.7622 | 6.7578 | 6.9086 | 102.97 | 97.99 | link |

## Vanilla SAEs (Spatial Patches)

Table 7. CLIP-ViT-B-32 vanilla sparse autoencoder performance metrics for spatial patches.

Top K Transcoders (All Patches)

| Layer | Sublayer | l1 coeff. | %Explained var. | Avg L0 | Cos sim | Recon cos sim | CE | Recon CE | Zero abl CE | %CErecovered | %Alive features | Model |
| - | - | - | - | - | - | - | - | - | - | - | - | - |
| 0 | resid post | 1e-12 | 99 | 989.19 | 0.99 | 0.99 | 6.7621 | 6.7621 | 6.9084 | 99.9981 | 100.00 | link |
| 1 | resid post | 3e-11 | 99 | 757.83 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9083 | 99.9969 | 45.39 | link |
| 2 | resid post | 4e-12 | 99 | 1007.89 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 97.93 | link |
| 3 | resid post | 2e-8 | 99 | 935.06 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9085 | 99.9882 | 100.00 | link |
| 4 | resid post | 3e-8 | 99 | 965.15 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9082 | 99.9842 | 100.00 | link |
| 5 | resid post | 1e-8 | 99 | 966.38 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9081 | 99.9961 | 100.00 | link |
| 6 | resid post | 1e-8 | 99 | 1006.62 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9083 | 100.00 | 99.97 | link |
| 7 | resid post | 1e-8 | 99 | 984.19 | 0.99 | 0.99 | 6.7622 | 6.7622 | 6.9082 | 100.00 | 100.00 | link |
| 8 | resid post | 3e-8 | 99 | 965.12 | 0.99 | 1.00 | 6.7622 | 6.7622 | 6.9087 | 100.00 | 92.37 | link |
| 9 | resid post | 9e-8 | 99 | 854.92 | 0.99 | 1.00 | 6.7622 | 6.7622 | 6.9083 | 99.9991 | 85.43 | link |
| 10 | resid post | 1e-4 | 72 | 88.80 | 0.84 | 0.97 | 6.7621 | 6.7638 | 6.9082 | 98.85 | 100.00 | link |
| 11 | resid post | 3e-7 | 99 | 829.09 | 0.99 | 1.00 | 6.7622 | 6.7622 | 6.9086 | 100.00 | 55.71 | link |

Table 8. CLIP Top-K transcoder performance metrics for all patches.

| Layer | Block | %Explained var. | k | Avg CLS L0 | Cos sim | CE | Recon CE | Zero abl CE | %CErecovered | Model |
| - | - | - | - | - | - | - | - | - | - | - |
| 0 | MLP | 96 | 768 | 767 | 0.9655 | 6.7621 | 6.7684 | 6.8804 | 94.68 | link |
| 1 | MLP | 94 | 256 | 255 | 0.9406 | 6.7621 | 6.7767 | 6.8816 | 87.78 | link |
| 2 | MLP | 93 | 1024 | 475 | 0.9758 | 6.7621 | 6.7681 | 6.7993 | 83.92 | link |
| 3 | MLP | 90 | 1024 | 825 | 0.9805 | 6.7621 | 6.7642 | 6.7999 | 94.42 | link |
| 4 | MLP | 76 | 512 | 29 | 0.9830 | 6.7621 | 6.7636 | 6.8080 | 96.76 | link |
| 5 | MLP | 91 | 1024 | 1017 | 0.9784 | 6.7621 | 6.7643 | 6.8296 | 96.82 | link |
| 6 | MLP | 94 | 1024 | 924 | 0.9756 | 6.7621 | 6.7630 | 6.8201 | 98.40 | link |
| 7 | MLP | 97 | 1024 | 1010 | 0.9629 | 6.7621 | 6.7631 | 6.8056 | 97.68 | link |
| 8 | MLP | 98 | 1024 | 1023 | 0.9460 | 6.7621 | 6.7630 | 6.8017 | 97.70 | link |
| 9 | MLP | 98 | 1024 | 1023 | 0.9221 | 6.7621 | 6.7630 | 6.7875 | 96.50 | link |
| 10 | MLP | 97 | 1024 | 1019 | 0.9334 | 6.7621 | 6.7636 | 6.7860 | 93.95 | link |

## 10.2. DINO-B SAEs

Vanilla (All patches)

Table 9. DINO Vanilla sparse autoencoder performance metrics for all patches (CLS + spatial).

| Layer | Sublayer | Avg L0. | %Explained var. | Avg CLS L0 | Cos sim | CE | Recon CE | Zero abl CE | %CERecovered | Model |
| - | - | - | - | - | - | - | - | - | - | - |
| 0 | resid post | 507 | 98 | 347 | 0.95009 | 1.885033 | 1.936518 | 7.2714 | 99.04 | link |
| 1 | resid post | 549 | 95 | 959 | 0.93071 | 1.885100 | 1.998274 | 7.2154 | 97.88 | link |
| 2 | resid post | 812 | 95 | 696 | 0.9560 | 1.885134 | 2.006115 | 7.201461 | 97.72 | link |
| 3 | resid post | 989 | 95 | 616 | 0.96315 | 1.885131 | 1.961913 | 7.2068 | 98.56 | link |
| 4 | resid post | 876 | 99 | 845 | 0.99856 | 1.885224 | 1.883169 | 7.1636 | 100.04 | link |
| 5 | resid post | 1001 | 98 | 889 | 0.99129 | 1.885353 | 1.875520 | 7.1412 | 100.19 | link |
| 6 | resid post | 962 | 99 | 950 | 0.99945 | 1.885239 | 1.872594 | 7.1480 | 100.24 | link |
| 7 | resid post | 1086 | 98 | 1041 | 0.99341 | 1.885371 | 1.869443 | 7.1694 | 100.30 | link |
| 8 | resid post | 530 | 90 | 529 | 0.9475 | 1.885511 | 1.978638 | 7.1315 | 98.22 | link |
| 9 | resid post | 1105 | 99 | 1090 | 0.99541 | 1.885341 | 1.894026 | 7.0781 | 99.83 | link |
| 10 | resid post | 835 | 99 | 839 | 0.99987 | 1.885371 | 1.884487 | 7.3606 | 100.02 | link |
| 11 | resid post | 1085 | 99 | 1084 | 0.99673 | 1.885370 | 1.911608 | 6.9078 | 99.48 | link |
