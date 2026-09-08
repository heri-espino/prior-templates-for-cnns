---
id: "gavrikov_2023_random-convolutions"
source_pdf: "../pdf/gavrikov_2023_random-convolutions.pdf"
source_filename: "gavrikov_2023_random-convolutions.pdf"
format: "academic-paper"
---

## The Power of Linear Combinations: Learning with Random Convolutions

Paul Gavrikov 1

Janis Keuper 1 , 2

1 IMLA, Offenburg University, Germany 2 Fraunhofer ITWM, Kaiserslautern, Germany {paul.gavrikov, janis.keuper}@hs-offenburg.de

## Abstract

Following the traditional paradigm of convolutional neural networks (CNNs), modern CNNs manage to keep pace with more recent, for example transformerbased, models by not only increasing model depth and width but also the kernel size. This results in large amounts of learnable model parameters that need to be handled during training. While following the convolutional paradigm with the according spatial inductive bias, we question the significance of learned convolution filters. In fact, our findings demonstrate that many contemporary CNN architectures can achieve high test accuracies without ever updating randomly initialized (spatial) convolution filters. Instead, simple linear combinations (implemented through efficient 1 × 1 convolutions) suffice to effectively recombine even random filters into expressive network operators. Furthermore, these combinations of random filters can implicitly regularize the resulting operations, mitigating overfitting and enhancing overall performance and robustness. Conversely, retaining the ability to learn filter updates can impair network performance. Lastly, although we only observe relatively small gains from learning 3 × 3 convolutions, the learning gains increase proportionally with kernel size, owing to the non-idealities of the independent and identically distributed ( i.i.d. ) nature of default initialization techniques.

## 1 Introduction

Convolutional Neural Networks (CNN) are building the backbone of state-of-the-art neural architectures in a wide range of learning applications on n -dimensional array data, such as standard computer vision problems like 2D image classification [3, 4, 32], semantic segmentation [4, 63], or scene understanding [2]. In order to solve these tasks, modern CNN architectures are learning the weights of millions of convolutional filter kernels. This process is not only very compute and data-intensive, but apparently also mostly redundant as CNNs are learning kernels that are bound to the same distribution, even when training different architectures on different datasets for different tasks [14], or can be replaced by random substitutes [71]. Yet if - in oversimplified terms - all CNNs are learning the 'same' filters, one could raise the fundamental question if we actually need to learn them at all. In this realm, several works have attempted to initialize convolution filters with better resemblance to converged filters, e. g. [55, 68].

Contrarily, in order to investigate if and how the training of a CNN with non-learnable filters is possible, we retreat to a simpler setup that eliminates any possible bias in the choice of the filters: we simply set random filters. This is not only practically feasible since random initializations [19, 68] of kernel weights are part of the standard training procedure, but also theoretically justified by a long line of prior work investigating the utilization of random feature extraction (e.g. see [43] for a prominent example) prior to the deep learning era.

Preprint. Under review.

One cornerstone of our analysis is the importance of the pointwise ( 1 × 1 ) convolution [29], which is increasingly used in modern CNNs. Despite its name and similarities in the implementation details, we will argue that this learnable operator differs significantly from spatial k × k convolutions and learns linear combinations of (non-learnable random) spatial filters.

In this paper, we deliberately avoid introducing new architectures. Instead, we take a step back and analyze an important operation in CNNs: linear combinations . We summarize our key contributions as follows:

- We show that modern CNNs (with specific 1 × 1 configurations serving as linear combinations) can be trained to high validation accuracies on 2D image classification tasks without ever updating weights of randomly initialized spatial convolutions, and, we provide a theoretical explanation for this phenomenon.
- By disentangling the linear combinations from intermediate operations, we empirically show that at a sufficiently high rate of those, training with random filters can outperform the accuracy and robustness of fully-learnable convolutions due to implicit regularization in the weight space. Alternatively, training with learnable convolutions and a high rate of linear combinations decreases accuracy.
- Based on our observations, we conclude that methods that seek to initialize convolutions filters to accelerate training must consider linear combinations during benchmarks. For the common 3 × 3 fi lters, there are only small margins for optimization compared to random baselines. Yet, current random methods struggle with larger convolution kernel sizes, due to their uniform distribution in the spatial space. As such, it is to be expected that better initializations can be found there.

## 2 Preliminaries

We define a 2D convolution layer by a function F ( X ; W ) , F transforming an input tensor X with c in input-channels into a tensor with c out output-channels using convolution filters with a size of k 0 × k 1 . Without loss of generality, we assume square kernels with k = k 0 = k 1 in this paper. Further, we denote the learned weights by W ∈ R c out × c in × k × k . The outputs Y i are then defined as:

$$Y _ { i } = W _ { i } * X = \sum _ { j = 1 } ^ { c _ { i n } } W _ { i , j } * X _ { j } , \quad \text {for } i \in \{ 1 , \dots , c _ { o u t } \} .$$

Note how the result of the convolution is reduced to a linear combination of inputs with a now scalar W i,j for the special case of k = 1 (pointwise convolution):

$$Y _ { i } = \sum _ { j = 1 } ^ { c _ { i n } } W _ { i , j } * X _ { j } = \sum _ { j = 1 } ^ { c _ { i n } } W _ { i , j } \cdot X _ { j } . \\$$

We assume that the default initialization of model weights is Kaiming Uniform [19] (default in PyTorch [40]). Here, every kernel weight w ∈ W is drawn i.i.d. from a uniform distribution bounded by a heuristic derived from the input fan (inputs c in × kernel area k 2 ). At default values, this is equivalent to

$$w \sim \mathcal { U } _ { [ - a , a ] } \quad \text {with} \quad a = \frac { 1 } { \sqrt { c _ { \text {in} } k ^ { 2 } } } .$$

## 3 Initial observation: Some CNNs perform well without learning filters

First, we analyze the performance of different CNNs that vary in depth, width, and implementation of the convolution layer, when the spatial convolution weights are fixed to their random initialization. For simplicity, we will refer to such models as frozen random through the remainder of the paper. Pointwise convolutions and all other operations always remain learnable.

After training common off-the-shelf architectures 1 such as ResNet-14/18/34/50/101 [20], (CIFAR)- ResNet-20 [20], Wide-ResNet-50x2 [69], and MobileNet v2 [47] on CIFAR-10 [28] we notice an interesting difference in the performance (Figure 1a). Although all models achieve an approximately similar validation accuracy when trained normally, we observe two kinds of frozen random behavior: ResNet-50/101, Wide-ResNet-50x2, and MobileNet v2 show only minor drops in accuracy (1.61.9% difference), while the other models show heavy drops of at least 16%. We obtain similar observations on ImageNet [8] training ResNet-18/50 [20], ResNet-50d [21], ResNeXt-50-32x4d [67], Wide-ResNet-50x2 [69], and MobileNet v2 S/v3 L [23] (Figure 1b). Accordingly, all models except ResNet-18 converge with 'just' 3.21-8.19% difference in validation accuracy. Instead, ResNet-18 shows a gap of 35.34%. We also find that increasing depth [ResNet-18 vs. ResNet-34] or width [ResNet-50 vs. Wide-ResNet-50x2], as well as reducing the kernel size in the stem [ResNet-50 vs. ResNet-50d] decreases the gap between frozen random and normal training.

1 Some are slightly modified to operate on low-resolution images. We will release these architectures with the rest of the code.

Figure 1: Validation accuracy of different off-the-shelf models trained on (a) CIFAR-10 and (b) ImageNet with random frozen random vs. learnable spatial convolutions. Models right of the vertical divider use blocks that integrate 1 × 1 convolutions after spatial convolutions and are, therefore, able to construct expressive filters from linear combinations of random filters. CIFAR-10 and ImageNet results are reported over 4 and 1 run(s), respectively.

<!-- image -->

An explanation for the performance differences can be found in the architectures: models, where we observe smaller gaps, use Bottleneck-Blocks [20] (or variants thereof [47]) which place pointwise ( 1 × 1 ) convolutions after spatial convolution layers. In these settings, the linear nature of convolutions lets us reformulate the operations by a linear combination of weights:

Lemma3.1. Apointwise convolution applied to the outputs of a spatial convolution layer is equivalent to a convolution with linear combinations (LCs) of previous filters with the same coefficients.

Proof. Assume that the l -th layer is a spatial convolution with k &gt; 1 , and inputs into a k = 1 pointwise convolution layer ( l +1 ) with X ( l ) being the input to the l -th layer. Then setting Equation (1) as input for Equation (2) results in:

$$Y _ { i } ^ { ( l + 1 ) } = \sum _ { j = 1 } ^ { c _ { i n } ^ { ( l + 1 ) } } W _ { i , j } ^ { ( l + 1 ) } \cdot X _ { j } ^ { ( l + 1 ) } = \sum _ { j = 1 } ^ { c _ { i n } ^ { ( l + 1 ) } } W _ { i , j } ^ { ( l + 1 ) } \cdot \left ( W _ { i } ^ { ( l ) } * X ^ { ( l ) } \right ) = X ^ { ( l ) } * \sum _ { j = 1 } ^ { c _ { i n } ^ { ( l + 1 ) } } \left ( W _ { i , j } ^ { ( l + 1 ) } \cdot W _ { i } ^ { ( l ) } \right )$$

As such, a set of (sufficiently many) random filters can be transformed into any kind of filter by learnable LCs (for an example see Figure 2). We hypothesize that architectures exist, where CNNs can be trained with frozen random filters to equal accuracy as learnable ones. In real implementations, however, intermediate operations such as normalization layers or activations will interfere with the LC in various ways. Contrary, models without linear combinations are restricted to the learning capacity that random filters provide, and, therefore will show lower accuracies.

Figure 2: Example of linear combinations of random filters resulting in more expressive filters.

<!-- image -->

Figure 3: LC-Block : Appending a pointwise convolution layer to all spatial convolution layers in the networks allows controlling the linear combination rate without altering the number of outputs via an expansion factor E . The LC-Block is equivalent to the original spatial convolution layer in the forward-pass.

<!-- image -->

Figure 4: Validation accuracy of ResNet models trained on CIFAR-10 with frozen random or learnable spatial convolutions under increasing LC expansion ( D = 20 , W = 16 ). (a) clean accuracy vs. expansion (b) clean accuracy vs. learnable parameters (c) robust accuracy against light adversarial attacks vs. expansion. After sufficiently many linear combinations, frozen random models outperform the baseline and learnable LC models . Results are reported over 4 runs.

<!-- image -->

## 4 Exploration of Linear Combinations

As discussed in the previous section, in practical implementations LCs are often affected by intermediate operations, such as activations and normalization layers. Thus, we experiment with specialized architectures to systematically study the effect of linear combinations of random filters free of the interference of such operations. Therein, we replace every spatial convolution layer with an LC-Block - which introduces linear combinations to networks that didn't include them previously (such as Basic-Block ResNets). This block is specifically designed to better control the linear combination rate of convolution filters without affecting other layers. We implemented this by replacing a spatial convolution with c in input channels and c out output channels with a combination of a spatial convolution with c out × E fi lters which are fed into a pointwise convolution with c out outputs. Intermediate operations such as activations or normalization layers between the two layers are deliberately omitted. Following Equation (4), the LC-Block is a reparameterization of the original convolution layer that can be folded back into a single spatial convolution operation during the forward-pass (Figure 3). We will refer to this reparameterization as combined filters in this paper. Note that in general, LC-Blocks represent an overparameterization and do not increase the expressiveness of the network.

Exemplarily, we study this modification to the (Basic-Block) CIFAR-ResNet as introduced in [20]. Compared to regular ResNets, CIFAR-ResNets have a drastically lower number of parameters and are, therefore, more suitable for large-scale studies such as the one presented in the following sections. We denote modified models by ResNet-LC - { D } - { W } x { E } , where D is the network depth i. e. the number of spatial convolution and fully-connected layers, W the network width (default 16) i. e. the initial number of channels communicated between Basic-Blocks, and E the LC expansion factor (default 1). In principle, LC-Blocks can be applied to any CNN architecture, yet they are unlikely to be relevant outside this study and just serve as a tool to analyze LCs in a clean and controlled way.

Figure 5: Filter variance entropy normalized by the randomness threshold as a metric of diversity in filter patterns of the (a) fi rst and (b) last layer. Measured on ResNet-LC-20-16x { E } trained on CIFAR-10 with frozen random or learnable spatial convolutions under increasing LC expansion E . Values ≥ 1 indicate a random distribution of kernel patterns, while values of 0 indicate a collapse to one specific pattern. Results are reported over 4 runs.

<!-- image -->

## 4.1 Increasing the rate of linear combinations

As per our assumption in Section 3, an increase in LCs should eventually close the gap between frozen random and learnable spatial convolutions. We test this by exponentially increasing the expansion factor (i.e. the number of LCs per LC-Block) of ResNet-LC-20-16 trained on CIFAR-10 and benchmark the performance of frozen random and learnable ResNet-LC against a fully learnable and unmodified ResNet-20-16 baseline. We report the mean and error over at least 4 runs with different random seeds.

We find three important observations based on the results in Figure 4: 1 ⃝ Although the LC-Block increases the number of learnable parameters, trainable ResNet-LCs perform worse than the original baseline . The gap diminishes with expansion but even at the highest tested expansion, the performance remains at the lower end of the baseline; 2 ⃝ The accuracy gap in frozen random models constantly decreases. Surprisingly, at E = 8 they outperform learnable ResNet-LCs, and, at E = 128 , they even outperform the baseline . Beyond that, the accuracy appears to slightly decline again but remains above the learnable ResNet-LCs; 3 ⃝ Similarly, we observe an increase in robustness to light adversarial attacks [51] ( ℓ ∞ -FGSM [18] with ϵ = 1 / 255 ) of random frozen models with expansion. Eventually, they again outperform the learnable and baseline counterparts. However, unlike clean validation accuracy, the robust accuracy does not appear to saturate under the tested expansion rates.

What differences can be observed in representations? Due to the clean and robust accuracy gap, it seems viable to conclude that frozen random models learn different representations. Compared to the baseline, random frozen models are intrinsically limited in the patterns that the combined filters can learn but this space increases with the number of linear combinations and matches the baseline at infinite combinations. At the same time, this limitation also serves as regularization and prevents overfitting which may explain why random frozen models generalize better than the baselines.

To further investigate this we measure the fi lter variance entropy [14] of the initial and final convolution layers (Figure 5). This singular value decomposition-based metric measures the diversity of filter kernel patterns, by providing a measurement in an interval between entirely random patterns (as seen in just initialized weights; a value ≥ 1 ) and a singular pattern repeated throughout all kernels (a value of 0). In the first layer of the baseline model, we observe an expected balanced diversity of patterns (e. g. compare to [68]) that is not random but neither highly repetitive. Whereas, the learnable LC models show a significantly lower diversity there that remains relatively stable independent of the expansion rate. Random frozen LC models are initially very close to random distributions but the diversity decreases with expansion and eventually converges slightly below the baseline but well above the diversity of learnable LC models.

Figure 6: Validation accuracy of ResNet-LC models trained on CIFAR-10 with frozen random or learnable spatial convolutions under increasing (a) network width ( D = 20 , E = 1 ), and (b) network depth ( W = 16 , E = 1 ). After sufficiently many linear combinations, frozen random models again outperform learnable ones. Results are reported over 4 runs.

<!-- image -->

In the last layer, we observe a significantly lower baseline due to degenerated convolution filters that collapse into a few patterns that are repeated throughout the layer, as shown in [14]. Learnable LC models show an even lower diversity and thus a higher degree of degeneration. In this layer, however, the diversity increases with expansion but still remains well below the baseline. We attribute the improvements with respect to the expansion to the smoothing/averaging effect of large numbers of linear combinations (analogous to the findings about ensemble averaging in [1]) which prevent collapsing into one specific pattern but still cannot avoid collapsing in general. For random frozen LC models, we again observe initially highly random filter patterns that decrease fast in diversity with increasing expansion but remain well above the baseline and learnable LC models.

Previous work [14] has already linked poor filter diversity to overfitting. Based on the diversity metrics it is thus not surprising that the learnable LC models underperform the baseline. In contrast to that, random frozen LC models cannot collapse that easily and remain even more diverse than the baseline. Gavrikov and Keuper [15] have attributed (adversarial) robustness with a higher diversity of filter patterns which correlates with our findings of higher robustness in random frozen LC models. Apparently, filter diversity only partially explains robustness, as for E ∈ { 16 , 32 } we see an increased diversity of filter pattern in random frozen models against the baseline, and although both perform relatively similarly on clean data, random frozen models still perform slightly worse against adversarial attacks.

It is also worth noting, that the improvements in robustness of random frozen networks are more of academic nature and are no match against special techniques such as adversarial training [34] and are also unlikely to withstand high-budget attacks.

Other options to increase the LC rate. An explicit LC expansion is not the only way to increase the rate of linear combinations in a network. Generally, increasing the network width naturally increases the number of LCs and closes the accuracy gap (Figure 6a; break-even at approx. W = 128 ). In addition, the gap diminishes with increasing depth due to the compositional nature of deep neural networks (Figure 6b; break-even at approx. D = 260 ).

## 4.2 Scaling to other datasets

In this section, we aim to demonstrate that our results also scale to other datasets such as CIFAR-100 [28], SVHN [36], Fashion-MNIST [66]. The results in Table 1 confirm our previous findings. At E = 128 random frozen ResNet-LC-20-16 models perform better or on par with the baseline, while the learnable LC variants perform worse. Additionally, we also perform experiments on ImageNet [8]. Since the ResNet-20 architecture [20] is under-parameterized for this problem we switch to the more powerful ResNet-18d architecture [21] which also avoids kernels larger than 3 × 3 . Again, we observe a similar behavior (Table 2). We were not able to outperform the baseline, but we attribute this to the granularity of tested expansion rates and the fluctuations in the measurements of a single run.

Table 1: Validation accuracy of a ResNet-20-16 trained on multiple other datasets as baseline and as learnable or random frozen LC with an expansion rate of 128. Results are reported over 4 runs. Best , second best.

|  | Validation Accuracy [%] | Validation Accuracy [%] | Validation Accuracy [%] |
| - | - | - | - |
| Dataset | Base.-Learn. | LC-Learn. | LC-Frzn.Rnd. |
| CIFAR-10 [28] | 91.60 ± 0.23 | 91.39 ± 0.29 | 91.89 ± 0.10 |
| CIFAR-100 [28] | 60.84 ± 0.52 | 59.44 ± 0.55 | 61.31 ± 0.74 |
| Fashion-MNIST [66] | 93.65 ± 0.18 | 93.44 ± 0.18 | 93.65 ± 0.06 |
| SVHN [36] | 96.36 ± 0.04 | 96.31 ± 0.08 | 96.34 ± 0.11 |

## 4.3 Increasing the kernel size

Our networks use the default 3 × 3 kernel size, which was dominant for the past years. However, recently proposed CNNs often increase the kernel size e. g. [32, 52, 54], sometimes to as large as 51 × 51 [31]. To verify that our observations hold on larger kernels, we increase the convolution sizes in a ResNet-LC-20-16 to k ∈ { 5 , 7 , 9 } (with a respective increase of input padding) and measure the performance gap between random frozen and learnable spatial convolutions on CIFAR-10 [28]. Our results (Figure 8a) show that the gap between frozen random and regular models significantly increases with kernel size, but steadily diminishes with increasing expansion and eventually breaks even for all our tested expansions.

Obviously, from a combinatorial perspective, more linear combinations are necessary to learn a specific kernel as the kernel size becomes larger. To understand further differences we

Table 2: Validation accuracy of a ResNet-18d trained on ImageNet . Learnable LCx128 exceeded 4x A100 VRAM.Results are reported over 1 run.

|  | Frozen Random | Frozen Random | Learnable | Learnable |
| - | - | - | - | - |
| Model | Top-1 | Top-5 | Top-1 | Top-5 |
| Baseline | 37.85 | 61.37 | 72.60 | 90.69 |
| LCx1 | 61.28 | 83.11 | 71.82 | 90.44 |
| LCx8 | 71.30 | 90.03 | 71.88 | 90.40 |
| LCx64 | 72.46 | 90.57 | 71.94 | 90.41 |
| LCx128 | 72.31 | 90.64 | n/a | n/a |

Figure 7: Visualization of the combined 9 × 9 convolution filters in the first convolution layer of frozen random ResNet-LC-20-16x { E } under increasing expansion E . Compared to random and learned filters.

<!-- image -->

take a closer look at the combined filters and compare random frozen against learnable models. We find an important difference there: (large) learnable filters primarily learn the weights in the center of the filter. Outer regions remain largely constant. This effect is barely or not at all visible for 3 × 3 kernels but gradually manifests with increasing kernel size. To better capture this effect we visualize these findings in the form of heatmaps highlighting the variance for each filter weight (Figure 8b). We compute the variance over all convolution kernels (total number N ) in a model. First, the kernels are normalized by the standard deviation of the entire convolution weight in their respective layer and finally stacked into a 3D tensor of shape k × k × N on which we compute the variance over the last axis. In the resulting heatmaps, we notice that random frozen models do not match this spatial distribution - their variance heatmaps are uniformly distributed independently of the kernel size. As such, the differences between learnable and random frozen models increase with kernel size due to poor reconstruction ability which correlates with the increasing accuracy gap. The cause for the uniform variance distribution can be found in the initialization. Initial kernel weights are drawn from i.i.d. initializations [16, 19] without consideration of the weight location in the filter. Linear combinations of these filters thus remain uniformly distributed and cannot manage to learn sharp patterns. For an example refer to Figure 7 where random frozen filters close in similarity to learn filters but never manage to reproduce the salient sharpness of learned 9 × 9 fi lters. The supplementary materials contain more examples at higher resolution.

## 5 Related Work

Random model parameters. Modern neural network weights are commonly initialized with values drawn i.i.d. from uniform or normal distributions with the standard deviation adjusted according to the channel fan, based on proposed heuristics by Glorot and Bengio [16], He et al. [19] to improve the gradient flow [22, 27]. Rudi and Rosasco [46] provided an analysis of generalization properties of such random neural networks and conclude that many problems exist, where exploiting random features can reach a significant accuracy, at a significant reduction in computation cost. Indeed, Ulyanov et al. [57] demonstrated that randomly weighted CNNs can provide good priors for standard inverse problems such as super-resolution, inpainting, or denoising. Additionally, Frankle et al. [13] showed that only training β and γ parameters of Batch-Normalization layers [25] results in a highly non-trivial performance of CNN image classifiers, although only affine transformations of random features are learned. But even when training all parameters, certain layers seem to learn negligible representations: Zhang et al. [70] show that entire weights of specific convolution layers can be reset to i.i.d. initializations after training without significantly hurting the accuracy. The number of affected layers depends on the specific architecture, parameterization, and problem complexity. Finally, both Ramanujan et al. [44], Zhou et al. [71] demonstrated that sufficiently large randomly-initialized CNNs contain subnetworks that achieve good (albeit well below trained) performance on complex problems such as ImageNet. Both approaches apply unstructured weight pruning to find these subnetworks and are based on the Lottery Ticket Hypothesis (LTH) [12] which suggests that deep neural networks contain extremely small subgraphs that can be trained to the same accuracy as the entire network.

Figure 8: Experiments with larger kernel sizes. (a) Gap in validation accuracy between frozen random and learnable ResNet-LC-20-16x { E } on CIFAR-10 with different convolution kernel sizes under increasing LC expansion E . Results are reported over 4 runs. (b) Spatial variance in the weights of combined filters of learned (top row) and frozen random (bottom row) models.

<!-- image -->

Learning convolution filters from base functions. A different line of work explores learning filters as linear combinations as different (frozen) bases [24] such as DCT [56], Wavelets [30], Fourier-Bessel [42], eigenimages of pretrained weights [53], or low-rank approximations [26]. Our analysis can be seen as a baseline for these works. However, most bases-approaches enforce the same amount of filters in every layer, whereas, naturally, the amount of filters varies per layer (as defined by the architecture). Furthermore, the number of bases is finite, which limits the amount of possible linear combinations. Contrary, there are infinitely many random filters. This 'overcompleteness' may in fact be necessary to train high-performance networks as suggested by the LTH [12].

Analysis of convolution filters. Multiple works have studied learned convolution filters, e. g. Yosinski et al. [68] studied their transferability and [15, 34] analyzed the impact of adversarial training on those. A long thread of connected research [5, 6, 37-39, 41, 48, 61, 62] extensively analyzed the features, connections, and their organization of a trained InceptionV1 [50] model. Among others, the authors claim that different CNNs will form similar features and circuits even when trained for different tasks. The findings are replicated in a large-scale analysis of learned 3 × 3 convolution kernels [14], which additionally reveals that CNNs generally seem to learn highly similar convolution kernel pattern distributions, independent of training data or task. Further, the authors find that the majority of kernels seem to be randomly distributed or defunct, and only a small rate seems to be performing useful transformations.

Pointwise convolutions. Lin et al. [29] first introduced the concept of network in network in which pointwise ( 1 × 1 ) convolutions are used to 'enhance the model discriminability for local receptive fields'. Although implemented similarly to spatial convolutions, pointwise convolutions do not aggregate the local neighborhood but instead compute linear combinations of the inputs and can be seen as a kind of fully-connected layer rather than a traditional convolution. Modern CNNs often use pointwise convolutions (e. g. [20, 32, 47]) to reduce the number of channels before computationally expensive operations such as spatial convolutions or to approximate the computation of regular convolutions using depthwise filters ( depthwise separable convolutions [7]). Alternatively, pointwise convolutions can also be utilized to fully reparameterize spatial convolutions [65].

## 6 Conclusion and Outlook

In our controlled experiments we have shown that random frozen CNNs can outperform fullytrainable baselines whenever a sufficient amount of linear combinations is present. The same findings apply to many modern real-world architectures which implicitly compute linear combinations (albeit not as clean due to intermediate operations).

In such networks, learned spatial convolution filters only marginally improve the performance compared to random baseline , and, in the settings of very wide, deep, or networks with an otherwise large number of linear combinations, learning spatial filters does not only result in no improvements but may even hurt the performance and robustness while wasting compute resources on training these parameters. This is in line with the findings [70] that CNNs contain (spatial) convolution layer that can be reset to their i.i.d. initialization without affecting the accuracy and the number of such layers increases in deeper networks. In contrast, training with random frozen filters regularizes training and prevents overfitting .

Additionally, this implies that works seeking better convolution filter initializations are limited by narrow optimization opportunities in modern architectures and must consider the presence of linear combinations when reporting performance. Only, under increasing kernel size, learned convolution filters begin to increase in importance - however, most likely not due to highly specific patterns, but rather due to a different spatial distribution of weights that cannot be reflected by i.i.d. initializations . It remains to be shown in future work, whether simply integrating this observation into current initialization techniques is sufficient to bridge the gap, though being likely as some proposed alternative initializations (e. g. [55]) only show improvements for larger kernel sizes.

We also see some cause for concern based on our results. There is a trend [23, 32, 47] to replace spatial convolution operations via pointwise ones whenever possible due to cheaper cost. Inversely, only a few works focus on spatial convolutions e.g. promising directions are the strengthening kernel skeletons [9, 11, 58], or (very) large kernel sizes [11, 31]. However, we have seen that adding learnable pointwise convolutions immediately after spatial convolutions decreased performance and robustness . This raises the question of whether this applies to (and limits) existing architectures. One indication for the imperfection of excessive LC architectures may be that VGG-style [49] networks (not containing LCs) can be trained to outperform ResNets [10]. Ultimately, answering this question will require a better understanding of the difference in learning behavior. While we have seen the outcome (filters are becoming less diverse which correlates with overfitting [14] and decreased robustness [15]), it remains unclear as to what the actual cause is. Apparently, it must be linked to the backward-pass, as the forward-pass is identical, e. g. consider a ResNet and an identical twin that inserts LC-Blocks and retains the spatial convolution weights. Setting the pointwise weights in the LC-Block with identity matrices will result in the same representations. Yet, although the network could learn this, it learns an arguably worse representation.

Limitations of this study. We have only studied the problem of image classification, but as random filters have been successfully used in other image problems [57, 60], we are not concerned about generalization. Still, it remains to be shown whether our findings hold for other data modalities. Secondly, we have only trained ResNet architectures, but given our theoretical proofs, we are confident that the observations will transfer to other neural network architectures. Further, all our models have been trained using (stochastic) gradient descent-based optimization. It is unclear if our findings - in particular the degeneration due to learnable LCs - will transfer to other solvers, such as evolutionary algorithms. Lastly, we have only examined linear combinations through pointwise convolutions. Yet, the same effect is obtainable by spatial kernels that only train the center element as observed in adversarially-trained models [15], fully-connected layers [45], attention layers [59], and potentially other operations. We aim to close this knowledge gap in future studies.

## Acknowledgements

Funded by the Ministry for Science, Research and Arts, Baden-Wuerttemberg, Germany under Grant 32-7545.20/45/1 (Q-AMeLiA).

## References

- [1] Zeyuan Allen-Zhu and Yuanzhi Li. Towards understanding ensemble, knowledge distillation and self-distillation in deep learning. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=Uuf2q9TfXGA . 6
- [2] Bruno Berenguel-Baeta, Jesus Bermudez-Cameo, and Jose J. Guerrero. Fredsnet: Joint monocular depth and semantic segmentation with fast fourier convolutions, 2022. 1
- [3] Andy Brock, Soham De, Samuel L Smith, and Karen Simonyan. High-performance large-scale image recognition without normalization. In Marina Meila and Tong Zhang, editors, Proceedings of the 38th International Conference on Machine Learning , volume 139 of Proceedings of Machine Learning Research , pages 1059-1071. PMLR, 18-24 Jul 2021. URL https:// proceedings.mlr.press/v139/brock21a.html . 1
- [4] Yuxuan Cai, Yizhuang Zhou, Qi Han, Jianjian Sun, Xiangwen Kong, Jun Li, and Xiangyu Zhang. Reversible column networks. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=Oc2vlWU0jFY . 1
- [5] Nick Cammarata, Gabriel Goh, Shan Carter, Ludwig Schubert, Michael Petrov, and Chris Olah. Curve detectors. Distill , 2020. doi: 10.23915/distill.00024.003. URL https://distill. pub/2020/circuits/curve-detectors . 8
- [6] Nick Cammarata, Gabriel Goh, Shan Carter, Chelsea Voss, Ludwig Schubert, and Chris Olah. Curve circuits. Distill , 2021. doi: 10.23915/distill.00024.006. URL https://distill.pub/ 2020/circuits/curve-circuits . 8
- [7] Francois Chollet. Xception: Deep learning with depthwise separable convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , July 2017. 9
- [8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A largescale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition , pages 248-255, 2009. doi: 10.1109/CVPR.2009.5206848. 3, 6
- [9] Xiaohan Ding, Yuchen Guo, Guiguang Ding, and Jungong Han. Acnet: Strengthening the kernel skeletons for powerful cnn via asymmetric convolution blocks. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , October 2019. 9, 2
- [10] Xiaohan Ding, Xiangyu Zhang, Ningning Ma, Jungong Han, Guiguang Ding, and Jian Sun. Repvgg: Making vgg-style convnets great again. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 13733-13742, June 2021. 9
- [11] Xiaohan Ding, Xiangyu Zhang, Jungong Han, and Guiguang Ding. Scaling up your kernels to 31x31: Revisiting large kernel design in cnns. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 11963-11975, June 2022. 9, 2
- [12] Jonathan Frankle and Michael Carbin. The lottery ticket hypothesis: Finding sparse, trainable neural networks. In International Conference on Learning Representations , 2019. URL https://openreview.net/forum?id=rJl-b3RcF7 . 8
- [13] Jonathan Frankle, David J. Schwab, and Ari S. Morcos. Training batchnorm and only batchnorm: On the expressive power of random features in CNNs. In International Conference on Learning Representations , 2021. URL https://openreview.net/forum?id=vYeQQ29Tbvx . 8
- [14] Paul Gavrikov and Janis Keuper. CNN Filter DB: An empirical investigation of trained convolutional filters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 19066-19076, June 2022. 1, 5, 6, 8, 9

- [15] Paul Gavrikov and Janis Keuper. Adversarial robustness through the lens of convolutional filters. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops , pages 139-147, June 2022. 6, 8, 9
- [16] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Yee Whye Teh and Mike Titterington, editors, Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics , volume 9 of Proceedings of Machine Learning Research , pages 249-256, Chia Laguna Resort, Sardinia, Italy, 13-15 May 2010. PMLR. URL https://proceedings.mlr.press/v9/glorot10a.html . 7, 8
- [17] Ian Goodfellow, Yoshua Bengio, and Aaron Courville. Deep Learning . MIT Press, 2016. http://www.deeplearningbook.org . 1
- [18] Ian J. Goodfellow, Jonathon Shlens, and Christian Szegedy. Explaining and harnessing adversarial examples. In Yoshua Bengio and Yann LeCun, editors, 3rd International Conference on Learning Representations, ICLR 2015, San Diego, CA, USA, May 7-9, 2015, Conference Track Proceedings , 2015. URL http://arxiv.org/abs/1412.6572 . 5
- [19] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE International Conference on Computer Vision (ICCV) , December 2015. 1, 2, 7, 8
- [20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition, 2015. 2, 3, 4, 6, 9
- [21] Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classification with convolutional neural networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , June 2019. 3, 6
- [22] Sepp Hochreiter. Untersuchungen zu dynamischen neuronalen netzen [in german]. Technical report, 1991. 8
- [23] Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, and Hartwig Adam. Searching for mobilenetv3. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) , October 2019. 3, 9
- [24] Robert A. Hummel. Feature detection using basis functions. Computer Graphics and Image Processing , 9(1):40-55, 1979. ISSN 0146-664X. doi: https://doi.org/10.1016/ 0146-664X(79)90081-9. URL https://www.sciencedirect.com/science/article/ pii/0146664X79900819 . 8
- [25] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In Francis Bach and David Blei, editors, Proceedings of the 32nd International Conference on Machine Learning , volume 37 of Proceedings of Machine Learning Research , pages 448-456, Lille, France, 07-09 Jul 2015. PMLR. URL https://proceedings.mlr.press/v37/ioffe15.html . 8
- [26] Max Jaderberg, Andrea Vedaldi, and Andrew Zisserman. Speeding up convolutional neural networks with low rank expansions. In Proceedings of the British Machine Vision Conference . BMVA Press, 2014. doi: http://dx.doi.org/10.5244/C.28.88. 8
- [27] John F. Kolen and Stefan C. Kremer. Gradient Flow in Recurrent Nets: The Difficulty of Learning LongTerm Dependencies , pages 237-243. 2001. doi: 10.1109/9780470544037.ch14. 8
- [28] Alex Krizhevsky. Learning multiple layers of features from tiny images. 2009. 2, 6, 7
- [29] Min Lin, Qiang Chen, and Shuicheng Yan. Network in network. In Yoshua Bengio and Yann LeCun, editors, 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings , 2014. URL http: //arxiv.org/abs/1312.4400 . 2, 8

- [30] Pengju Liu, Hongzhi Zhang, Wei Lian, and Wangmeng Zuo. Multi-level wavelet convolutional neural networks. IEEE Access , 7:74973-74985, 2019. doi: 10.1109/ACCESS.2019.2921451. 8
- [31] Shiwei Liu, Tianlong Chen, Xiaohan Chen, Xuxi Chen, Qiao Xiao, Boqian Wu, Tommi Kärkkäinen, Mykola Pechenizkiy, Decebal Constantin Mocanu, and Zhangyang Wang. More convnets in the 2020s: Scaling up kernels beyond 51x51 using sparsity. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview. net/forum?id=bXNl-myZkJl . 7, 9
- [32] Zhuang Liu, Hanzi Mao, Chao-Yuan Wu, Christoph Feichtenhofer, Trevor Darrell, and Saining Xie. A convnet for the 2020s. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 11976-11986, June 2022. 1, 7, 9
- [33] Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations , 2017. URL https://openreview. net/forum?id=Skq89Scxx . 1
- [34] Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations , 2018. URL https://openreview.net/forum?id=rJzIBfZAb . 6, 8
- [35] Paulius Micikevicius, Sharan Narang, Jonah Alben, Gregory Diamos, Erich Elsen, David Garcia, Boris Ginsburg, Michael Houston, Oleksii Kuchaiev, Ganesh Venkatesh, and Hao Wu. Mixed precision training. In International Conference on Learning Representations , 2018. URL https://openreview.net/forum?id=r1gs9JgRZ . 1
- [36] Yuval Netzer, Tao Wang, Adam Coates, A. Bissacco, Bo Wu, and Andrew Y. Ng. Reading digits in natural images with unsupervised feature learning. In NIPS Workshop on Deep Learning and Unsupervised Feature Learning , 2011. URL http://ufldl.stanford.edu/housenumbers . 6, 7
- [37] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill , 5, 2020. doi: 10.23915/distill.00024.001. URL https://distill.pub/2020/circuits/zoom-in . 8
- [38] Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. An overview of early vision in inceptionv1. Distill , 2020. doi: 10.23915/distill.00024.002. URL https://distill.pub/2020/circuits/early-vision .
- [39] Chris Olah, Nick Cammarata, Chelsea Voss, Ludwig Schubert, and Gabriel Goh. Naturally occurring equivariance in neural networks. Distill , 2020. doi: 10.23915/distill.00024.004. URL https://distill.pub/2020/circuits/equivariance . 8
- [40] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, highperformance deep learning library. In Advances in Neural Information Processing Systems 32 , pages 8024-8035. Curran Associates, Inc., 2019. URL http://papers.neurips.cc/paper/ 9015-pytorch-an-imperative-style-high-performance-deep-learning-library. pdf . 2, 1
- [41] Michael Petrov, Chelsea Voss, Ludwig Schubert, Nick Cammarata, Gabriel Goh, and Chris Olah. Weight banding. Distill , 2021. doi: 10.23915/distill.00024.009. URL https://distill. pub/2020/circuits/weight-banding . 8
- [42] Qiang Qiu, Xiuyuan Cheng, Robert Calderbank, and Guillermo Sapiro. DCFNet: Deep neural network with decomposed convolutional filters. International Conference on Machine Learning , 2018. 8

- [43] Ali Rahimi and Benjamin Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems , volume 20. Curran Associates, Inc., 2007. URL https://proceedings.neurips. cc/paper/2007/file/013a006f03dbc5392effeb8f18fda755-Paper.pdf . 1
- [44] Vivek Ramanujan, Mitchell Wortsman, Aniruddha Kembhavi, Ali Farhadi, and Mohammad Rastegari. What's hidden in a randomly weighted neural network? 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) , pages 11890-11899, 2019. 8
- [45] Frank Rosenblatt. The perceptron: a probabilistic model for information storage and organization in the brain. Psychological review , 65 6:386-408, 1958. 9
- [46] Alessandro Rudi and Lorenzo Rosasco. Generalization properties of learning with random features. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper/2017/ file/61b1fb3f59e28c67f3925f3c79be81a1-Paper.pdf . 8
- [47] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2018. 2, 3, 9
- [48] Ludwig Schubert, Chelsea Voss, Nick Cammarata, Gabriel Goh, and Chris Olah. High-low frequency detectors. Distill , 2021. doi: 10.23915/distill.00024.005. URL https://distill. pub/2020/circuits/frequency-edges . 8
- [49] Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition, 2015. 9
- [50] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions, 2014. 8
- [51] Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks, 2014. 5
- [52] Mingxing Tan and Quoc V. Le. Efficientnet: Rethinking model scaling for convolutional neural networks, 2020. 7
- [53] Muhammad Tayyab and Abhijit Mahalanobis. Basisconv: A method for compressed representation and learning in cnns. CoRR , abs/1906.04509, 2019. 8
- [54] Asher Trockman and J Zico Kolter. Patches are all you need? Transactions on Machine Learning Research , 2023. ISSN 2835-8856. URL https://openreview.net/forum?id= rAnB7JSMXL . Featured Certification. 7
- [55] Asher Trockman, Devin Willmott, and J Zico Kolter. Understanding the covariance structure of convolutional filters. In The Eleventh International Conference on Learning Representations , 2023. URL https://openreview.net/forum?id=WGApODQvwRg . 1, 9
- [56] Matej Ulicny, Vladimir A. Krylov, and Rozenn Dahyot. Harmonic convolutional networks based on discrete cosine transform. Pattern Recognition , 129:108707, 2022. ISSN 0031-3203. 8
- [57] Dmitry Ulyanov, Andrea Vedaldi, and Victor Lempitsky. Deep image prior. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2018. 8, 9
- [58] Pavan Kumar Anasosalu Vasu, James Gabriel, Jeff Zhu, Oncel Tuzel, and Anurag Ranjan. Fastvit: A fast hybrid vision transformer using structural reparameterization, 2023. 9, 2
- [59] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Ł ukasz Kaiser, and Illia Polosukhin. Attention is all you need. In I. Guyon, U. Von Luxburg, S. Bengio, H. Wallach, R. Fergus, S. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 30. Curran Associates, Inc., 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/file/ 3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf . 9

- [60] Roberto Antonio Vázquez and Juan Humberto Sossa Azuela. Random features applied to face recognition. Eighth Mexican International Conference on Current Trends in Computer Science (ENC 2007) , pages 47-51, 2007. 9
- [61] Chelsea Voss, Nick Cammarata, Gabriel Goh, Michael Petrov, Ludwig Schubert, Ben Egan, Swee Kiat Lim, and Chris Olah. Visualizing weights. Distill , 2021. doi: 10.23915/distill.00024. 007. URL https://distill.pub/2020/circuits/visualizing-weights . 8
- [62] Chelsea Voss, Gabriel Goh, Nick Cammarata, Michael Petrov, Ludwig Schubert, and Chris Olah. Branch specialization. Distill , 2021. doi: 10.23915/distill.00024.008. URL https: //distill.pub/2020/circuits/branch-specialization . 8
- [63] Wenhai Wang, Jifeng Dai, Zhe Chen, Zhenhang Huang, Zhiqi Li, Xizhou Zhu, Xiaowei Hu, Tong Lu, Lewei Lu, Hongsheng Li, et al. Internimage: Exploring large-scale vision foundation models with deformable convolutions. arXiv preprint arXiv:2211.05778 , 2022. 1
- [64] Ross Wightman, Hugo Touvron, and Herve Jegou. Resnet strikes back: An improved training procedure in timm. In NeurIPS 2021 Workshop on ImageNet: Past, Present, and Future , 2021. URL https://openreview.net/forum?id=NG6MJnVl6M5 . 1
- [65] Bichen Wu, Alvin Wan, Xiangyu Yue, Peter Jin, Sicheng Zhao, Noah Golmant, Amir Gholaminejad, Joseph Gonzalez, and Kurt Keutzer. Shift: A zero flop, zero parameter alternative to spatial convolutions. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , June 2018. 9
- [66] Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms, 2017. 6, 7
- [67] Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, and Kaiming He. Aggregated residual transformations for deep neural networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , July 2017. 3
- [68] Jason Yosinski, Jeff Clune, Yoshua Bengio, and Hod Lipson. How transferable are features in deep neural networks? In Z. Ghahramani, M. Welling, C. Cortes, N.D. Lawrence, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems 27 (NIPS '14) , pages 3320-3328. Curran Associates, Inc., 2014. 1, 5, 8
- [69] Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. In Edwin R. Hancock Richard C. Wilson and William A. P. Smith, editors, Proceedings of the British Machine Vision Conference (BMVC) , pages 87.1-87.12. BMVA Press, September 2016. ISBN 1-901725-59-6. doi: 10.5244/C.30.87. URL https://dx.doi.org/10.5244/C.30.87 . 2, 3
- [70] Chiyuan Zhang, Samy Bengio, and Yoram Singer. Are all layers created equal? Journal of Machine Learning Research , 23(67):1-28, 2022. URL http://jmlr.org/papers/v23/ 20-069.html . 8, 9
- [71] Hattie Zhou, Janice Lan, Rosanne Liu, and Jason Yosinski. Deconstructing lottery tickets: Zeros, signs, and the supermask. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems , volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper/2019/ file/1113d7a76ffceca1bb350bfe145467c6-Paper.pdf . 1, 8

## Contents

| A | Training Details | 1 |
| - | - | - |
| B | Derivation of the Layer Scale Coefficient | 1 |
| C | Ablation of Intermediate Operations in LC-Blocks | 2 |
| D | Combined Filters | 2 |
| E | Potential negative Societal Impacts | 3 |
| F | Computational Resources | 3 |

## A Training Details

Training scripts use PyTorch [40] 1.12.1 and CUDA 11.3.

Low-resolution datasets. We train all models for 75 epochs. We use an SGD optimizer (with Nesterov momentum of 0.9) with an initial learning rate of 1e-2 following a cosine annealing schedule [33], a weight decay of 1e-2, a batch size of 256, and Categorical Cross Entropy with a label smoothing [17] of 1e-1. We pick the last checkpoint for evaluation.

We use the following augmentations:

- CIFAR-10/100 : For training, images are zero-padded by 4 px along each dimension, apply random horizontal flips, and proceed with 32 2 px random crops. Test images are not modified.
- SVHN : No augmentations.
- Fashion-MNIST : Both, train and test images are upscaled to 32 2 px.

In all cases, the data is normalized by the channel mean and standard deviation.

ImageNet. We train all models following [64] (A2) with automatic mixed precision training [35] for 300 epochs at 224 2 px resolution without any pre-training and report top-1 and top-5 accuracy for both, learnable and frozen random training. We pick the checkpoint with the highest top-1 validation accuracy for evaluation.

## B Derivation of the Layer Scale Coefficient

We use the default PyTorch initialization of convolution layers: Weights are drawn from a uniform distribution and scaled according to [19]. PyTorch uses a default gain of √ 2 1+ α 2 with α = √ 5 . Which results in gain = √ 1 / 3 . Further, the channel input fan is used for normalization.

The standard deviation for weights drawn from normal distributions is given by:

$$\sigma _ { \text {he} } = \frac { \text {gain} } { \sqrt { \text {fan} } } = \frac { \text {gain} } { \sqrt { c _ { \text {in} } k ^ { 2 } } }$$

## The Power of Linear Combinations: Learning with Random Convolutions

## Supplementary Materials

And the standard deviation of a symmetric uniform distribution U [ - a,a ] is given by:

$$\sigma = a / \sqrt { 3 }$$

To retain the standard deviation we, therefore, compute the scaling coefficient as follow:

$$s = \sqrt { 3 } \sigma _ { h e } = \sqrt { 3 } \frac { \ g a i n } { \sqrt { c _ { i n } k ^ { 2 } } } = \sqrt { 3 } \frac { \sqrt { 1 / 3 } } { \sqrt { c _ { i n } k ^ { 2 } } } = \frac { 1 } { \sqrt { c _ { i n } k ^ { 2 } } }$$

## C Ablation of Intermediate Operations in LC-Blocks

Practical CNN architectures often include intermediate operations that influence linear combinations. Exemplarily, we study this in an experiment on ResNet-LC-20-16x64 trained on CIFAR-10 and insert ReLU, an (affine) BatchNorm operation, and a combination of both between the two convolution layers in an LC-Block. We compare frozen random against learnable models in Table 3. Just adding a BatchNorm layer lowers the performance in both cases. This is somewhat in line with our observations that adding learnable LCs lowered the accuracy as the performed affine transformation is an overparameterization that does not increase expressiveness. Adding a ReLU activation, however, increases the performance due to the additional non-linearity that can be exploited in the combination of filters. In this example, learnable LCs benefit from this and outperform random frozen models, although the random frozen baseline was superior. The combination of both operations performs best in, both, random frozen and learnable models.

Table 3: Influence of intermediate operations in LC-Blocks.

|  | Validation Accuracy [%] | Validation Accuracy [%] |
| - | - | - |
| Intermediate Operation | Frozen Random | Learnable |
| None | 91 . 71 ± 0 . 08 | 91 . 18 ± 0 . 10 |
| ReLU | 92 . 78 ± 0 . 31 | 93 . 38 ± 0 . 15 |
| BatchNorm | 91 . 50 ± 0 . 15 | 91 . 06 ± 0 . 23 |
| BatchNorm + ReLU | 93 . 26 ± 0 . 18 | 94 . 72 ± 0 . 16 |

## D Combined Filters

Figure 9 shows the combined filters (i. e. the convolutions filters obtained by the linear combination in LC-Blocks) in the first convolutions layers of frozen random and learnable filters at different rates of expansion and for different kernel sizes. The filters of learnable ResNet-LCs remain fairly similar independent of expansion, while the frozen random filters become less random with increasing depth. A well-traceable filter is a green color blob, that evolves from noise to a square blob and eventually to the Gaussian-like filter. Also visible is that larger filters concentrate more of their weights in the center of the filters.

Figure 10 shows the filter variance entropy (FVE) for the same combined filters. Note that contrary to Section 4 we do not normalize the FVE by the randomness threshold, as it was only derived for 3 × 3 convolutions by the original authors. Using the non-normalized values, however, allows a comparison independent of kernel size. Once again, we can see that the FVE of learnable LC models remains constant throughout different expansion rates and only marginally decreases with increasing kernel size. For all kernel sizes, we see that frozen random models decrease in FVE at increased expansion. Yet, they are increasingly more diverse with kernel size. Hence, the gap between learnable and frozen random weights significantly increases with increasing kernel size.

We repeat this measurement for the last convolution layer in Figure 11 which shows even larger differences between frozen random and learnable LC models with increasing kernel size. Although we generally observe similar trends, there is one salient difference to the first layer: the diversity of learnable LC models collapses for non3 × 3 layers. This highlights the importance of kernel strengthening [9, 11, 58] for larger kernels.

## E Potential negative Societal Impacts

We do not believe that our analysis causes any negative societal impacts. As with most publications in this field, our experiments consumed a lot of energy and caused the emission of CO 2 . However, by exposing non-idealities of current approaches we hope to inspire future researchers to reconsider their network designs to reduce emissions during training.

## F Computational Resources

The training was executed on internal clusters with NVIDIA A100-SXM4-40GB GPUs for a cumulative total of approximately 2901 . 6 GPU hours. Detailed budgets spent on evaluating the baseline models in Section 3, the linear combination experiments in Section 4 including adversarial evaluation in Section 4.1, ablation of intermediate operations, and abandoned experiments can be found in Table 4.

Table 4: Detailed compute resources spent for experiments in this paper. Cumulative hours refer to the number of GPUs ( NVIDIA A100-SXM4-40GB GPUs) used per experiment times the runtime.

| Experiment | Cumulative GPU hours |
| - | - |
| Baseline ImageNet | 870 |
| Baseline CIFAR-10 | 183 |
| LC CIFAR-10 | 787 . 3 |
| LC CIFAR-100 | 2 . 8 |
| LC SVHN | 3 . 9 |
| LC FashionMNIST | 3 . 0 |
| LC ImageNet | 869 . 2 |
| Adversarial Evaluation | 1 . 3 |
| Ablation | 20 . 2 |
| Abandoned Experiments | 160 . 9 |
| TOTAL | 2901 . 6 |

Figure 9: Visualization of the combined filters of the first convolution layer in ResNet-LCs-2016x { E } with increasing expansion E of frozen random (left) or learnable (right) models under different kernel size k .

<!-- image -->

Figure 10: Variance entropy (not normalized for comparability) as a measure of diversity in filter patterns of the combined filters in the fi rst convolution layer in ResNet-LC-20-16x { E } on CIFAR10. We compare random frozen to learnable models under increasing LC expansion E and kernel sizes k .

<!-- image -->

Figure 11: Variance entropy (not normalized for comparability) as a measure of diversity in filter patterns of the combined filters in the last convolution layer in ResNet-LC-20-16x { E } on CIFAR-10. We compare random frozen to learnable models under increasing LC expansion E and kernel sizes k .

<!-- image -->
