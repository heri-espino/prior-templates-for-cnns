---
id: "gaudio_2023_fixed-filters"
source_pdf: "../pdf/gaudio_2023_fixed-filters.pdf"
source_filename: "gaudio_2023_fixed-filters.pdf"
format: "academic-paper"
---

## ExplainFix: Explainable Spatially Fixed Deep Networks

Alex Gaudio 123* , Christos Faloutsos 1 , Asim Smailagic 1 , Pedro Costa 23 , and Aurélio Campilho 23

1 Carnegie Mellon University, Pittsburgh, PA, USA

2 Faculdade de Engenharia da Universidade do Porto, Portugal

3 INESC TEC, Portugal

## Abstract

Is there an initialization for deep networks that requires no learning? ExplainFix adopts two design principles: the 'fixed filters' principle that all spatial filter weights of convolutional neural networks can be fixed at initialization and never learned, and the 'nimbleness' principle that only few network parameters suffice. We contribute (a) visual model-based explanations , (b) speed and accuracy gains , and (c) novel tools for deep convolutional neural networks. ExplainFix gives key insights that spatially fixed networks should have a steered initialization, that spatial convolution layers tend to prioritize low frequencies, and that most network parameters are not necessary in spatially fixed models. ExplainFix models have up to 100x fewer spatial filter kernels than fully learned models and matching or improved accuracy. Our extensive empirical analysis confirms that ExplainFix guarantees nimbler models (train up to 17% faster with channel pruning), matching or improved predictive performance (spanning 13 distinct baseline models, four architectures and two medical image datasets), improved robustness to larger learning rate, and robustness to varying model size. We are first to demonstrate that all spatial filters in state-of-the-art convolutional deep networks can be fixed at initialization, not learned.

Keywords Deep Learning, Computer Vision, Fixed-Weight Networks, Explainability, Pruning, Medical Image Analysis

## 1 Introduction

Do all weights of deep convolutional neural network (CNN) need to be learned? On medical image data, the answer is 'no.' We analyze and explain the spatial convolution layers internal to CNNs in order to understand how to initialize and prune spatially fixed networks. ExplainFix adopts two design principles for fixed weight networks. The first proposed principle, 'fixed filters', dictates that we do not learn the spatial convolution filters in the CNN. The second principle, 'nimbleness', dictates that only a small fraction of spatial filters suffices. ExplainFix has the following main contributions:

1. Model-based Explanations: ExplainFix visualizes a model's internal properties, namely spatial filter steering and weight saliency, to show how to improve models according to our design principles.
2. Speed and Accuracy Gains : ExplainFix models are faster and smaller, with matching or improved accuracy, thanks to our initialization and pruning methods for spatially fixed networks.
3. Novel Tools : The ExplainFix system comprises open source tools for fixed filter networks, including ExplainSteer, ChannelPrune, and fixed kernel initialization methods (GuidedSteer, GHaar, Psine, Unchanged, DCT2).

Fig. 1 demonstrates the value of our ExplainFix system. Fig. 1a presents our visual explanation of a CNN to highlight that nearly all spatial layers can be pruned. Fig. 1b validates the visual explanation to show that we can use 100x fewer spatial filters while preserving predictive performance. We demonstrate in Sec. 4.5 that most spatial filters are indeed unnecessary for both training and inference. Fig. 1c illustrates that a spatially fixed architecture pruned by our method is faster and smaller than the baseline, and Fig. 1d shows the model is nearly equal in accuracy. Similar results hold for other models in Sec. 4.2.

Figure 1: ExplainFix checks all goals: explainable (a,b), nimble (c), and accurate (d). (a) Our ExplainSteer visual explanation exposes deep network inefficiencies. (b) Only 1 in 100 filters needs to be kept on ResNet50. We have similar results on other architectures. (c,d) Our ChannelPrune method makes models smaller and faster (c) and accurate (d).

<!-- image -->

Our hypothesis and conclusion that all spatial 2D convolution kernels can be fixed with a steered initialization, not learned, and mostly eliminated is strongly supported by our insights, explanations and results. In Sec. 2, we review related work and provide intuitions about relationships between deep networks and steerability. We describe our initialization, explanation and pruning methods in Sec. 3, and the experiments and resulting analyses in Sec. 4. In Sec. 5, we discuss applications and possible future research directions for fixed filter networks. To our knowledge, we are first to consider entirely fixed spatial convolution filters and their initialization methods without changing deep network architecture.

## 2 Background and Related Work

The proposed ExplainFix is the only method with all properties in Table 1. We highlight current challenges in deep learning and benefits of a fixed weight initialization. First, deep networks are increasingly larger (Hestness et al., 2017), they require large datasets that are costly to acquire and annotate (Segebarth et al., 2020; Tajbakhsh et al., 2020), and the models are difficult to train. The unsustainable increasing trends call for dramatically more efficient methods (Thompson et al., 2020). A fixed weight initialization minimizes the need for training, computational resources and data.

A second major challenge in deep learning is lack of interpretability. Despite gains, deep networks are still considered black boxes (Adadi &amp; Berrada, 2018; Samek et al., 2017; Tjoa &amp; Guan, 2020). Ante-hoc methods create novel deep network architectures that are explainable by their design (Angelov &amp; Soares, 2020; Bruna, 2013; Buhrmester et al., 2021; Gaudio et al., 2020; Patrick et al., 2019), yet these methods require up-front knowledge of how to design the network. We propose to develop a model-based explanation of spatial convolution layers that is compatible with many ante-hoc methods. Ante-hoc explainable fixed filter methods have been rigorously proposed, for example, in wavelet scattering networks (Bruna, 2013; Cotter &amp; Kingsbury, 2019; Mallat, 2012; Oyallon et al., 2018) and in group equivariant convolution networks (Cohen &amp; Welling, 2016). They yield interpretable, state-of-the-art networks and can help to understand how deep networks learn (Cotter &amp; Kingsbury, 2017; Mallat, 2016), but these models are hard to use. The co/in/equi-variance properties must be fully specified in advance, and they also introduce custom network architecture, thus discouraging wider adoption. We therefore contribute methods that start with a standard convolutional network architecture and do not need advance knowledge of the co/in/equi-variance properties or the dataset. Finally, a recent empirical study found that fixing as many as 90% of weights across the network, without discerning between types of convolutions or layers, had slightly decreased performance on common datasets (Rosenfeld &amp; Tsotsos, 2019). Despite showing reduced performance, the results justify our work. We propose to fix all spatial convolution layers and we develop explanations and methods for how to initialize the weights and prune channels without performance loss.

Redundancy: Some weights are unnecessary. Neural networks are highly compressible (Cheng et al., 2018). Pruning 99% of nodes on a trained multi-class model can yield no loss in single-class performance (Leino et al., 2018). Pruning the connections between nodes by 9-13x (Han et al., 2016) is possible without performance loss. Temporarily removing nodes during training with Dropout can improve performance and prevent over-fitting (Srivastava et al., 2014). The recently popular Lottery Ticket Hypothesis states that over-parameterized and untrained neural networks contain subnetworks, or winning tickets, that enjoy roughly equal accuracy to the original trained baseline network, either with training (Frankle &amp; Carbin, 2019) or without any training (Malach et al., 2020; Pensia et al., 2020). Moreover, over-parameterized deep networks contain a frozen subspace of weights that are not changed by learning with gradient descent (Advani et al., 2020), and fixing as much as 90% of randomly sampled weights in a deep network was shown to give small performance loss (Rosenfeld &amp; Tsotsos, 2019). Thus, deep networks are highly redundant, with a subset of weights relevant to the predictive task, and a subset of weights unchanged by learning. In this paper, we offer empirical evidence that all spatial filters can be entirely unchanged by learning and that as few as only 1% of spatial filters are necessary for the inference task.

Steerability: A property of convolution layers. A steered filter is a linear combination of basis vectors (Freeman, Adelson, et al., 1991). The convolution layer in deep networks, as in Eq. 1, defines each output channel as a sum of linearly transformed input channels, where O o is an output channel, I i is an input channel, f o,i are the weights of a convolution filter kernel, and ∗ is the cross-correlation operator. By a change of variables inside the sum, each output channel O o can be seen as a steered combination of either the normalized filters ˜ f o,i , the feature maps ˜ f o,i ∗ I i , or the input channels I i , via the steering weights w o,i = ∥ ∥ f o,i ∥ ∥ . Thus, convolution layers have a steered representation by design. Redundancy by steering is also an essential property of wavelet scattering networks (Bruna, 2013). Our work provides empirical evidence that steered representation is an important property of spatially fixed CNNs.

$$0 _ { o } = \sum _ { i } \mathbf f _ { o , i } * \mathbf I _ { i } = \sum _ { i } \left ( \| \mathbf f _ { o , i } \| \frac { \mathbf f _ { o , i } } { \| \mathbf f _ { o , i } \| } \right ) * \mathbf I _ { i } = \sum _ { i } w _ { o , i } ( \tilde { \mathbf f } _ { o , i } * \mathbf I _ { i } ) = \sum _ { i } \tilde { \mathbf f } _ { o , i } * ( w _ { o , i } \mathbf I _ { i } )$$

The pointwise 1 × 1 convolution was introduced and popularized in (He et al., 2016; Howard et al., 2017; Lin et al., 2014; Szegedy et al., 2016). A 1 × 1 convolution reduces the filter kernel in Eq. 1 to the scalar ˜ f = 1 . Each output channel O o is a linear mixture of the input channels, via mixing weights w o,i . We do not consider the 1 × 1 convolution a spatial convolution. Eq. 1 implies that 1 × 1 convolutions can also steer the inputs or outputs of a nearby spatial convolution. Thus, both network architectures and spatial convolution layers can jointly learn an optimal steering of spatial features. Our results suggest that relying on architecture alone is insufficient. Initializing fixed spatial convolution layers with redundant, steered fixed initializations can facilitate training of non-fixed weights and give better prediction performance.

Differences between Deep Networks and Wavelets: A discrete wavelet transform (DWT) (Daubechies, 1992) can be described as a sequence of convolution operations with fixed wavelet filters that map a single input channel to multiple output channels. The convolution layer in deep networks has minor differences; it uses a cross-correlation operator (filters are flipped) rather than a convolution operator, the filters are learned, and each output channel is a sum of the linearly transformed input channels. The sum over multiple input channels is entirely captured by steerability in Eq. 1. The other difference, that filters are learned, is the main subject of our hypothesis.

Table 1: ExplainFix Wins. Fixed Spatial Filters are preferable to existing approaches.

| Property / Method | Wavelet Scattering Bruna and Mallat, 2013 Oyallon et al., 2018 Cotter and Kingsbury, 2019 | Deep Wavelet Networks de Freitas Barbosa et al., 2020 Luan et al., 2018 Pérez et al., 2020 | Network Pruning Rosenfeld and Tsotsos, 2019 Han et al., 2016 Srivastava et al., 2014 Frankle and Carbin, 2019 | Deep Networks He et al., 2016 Tan and Le, 2019 Huang et al., 2017 | Explainable Fixed Filter Networks |
| - | - | - | - | - | - |
| Model Interpretability Faster Training Lower RAM Footprint No Performance Loss | " | " |  | " " " |  |
|  | " |  | ? |  | = |
|  | " |  | " |  | = |
| Compatible with ExplainFix | " |  | " |  |  |
|  |  | " |  |  |  |
|  |  |  |  | " |  |
|  |  |  |  |  | = |
|  |  |  | ? |  |  |
|  |  | " |  |  |  |
|  |  |  |  | " |  |
|  |  |  |  |  | " |

Similarities to Discrete Wavelet Transforms: Many saliency based explanations of deep networks, or methods that use the deep network's own parameters and activations to explain the network itself, behave analogously to edge detectors (Adebayo et al., 2018). The association of convolutions to edge detection was well known even thirty years ago (Caelli et al., 1988) for computing wavelet transforms (Grossmann, 1988), and wavelets have excellent edge detection properties (Hanov, 2006). Deep networks and wavelets both perform edge detection. Fig. 2 shows that activations randomly sampled from a pre-trained ResNet50 (He et al., 2016) and vertical coefficients of a Haar DWT both generate edge features. Gabor wavelets have been studied in deep networks at all layers (Luan et al., 2018), and applied in early layers to enhance a network's robustness to adversarial attack (Pérez et al., 2020). Wavelets have been used to help train autoencoders (Said et al., 2016) or sparse autoencoders (Mallick et al., 2019), where the encoder is subsequently used for classification. Depthwise separable convolutions (Sandler et al., 2018) apply a sequence of layers: 1 × 1 for channel expansion, 3 × 3 grouped convolution, 1 × 1 . Analogously to a DWT, the first two convolutions also map each input channel to multiple outputs. The authors of (Sandler et al., 2018) proposed no ReLU non-linearity after the last convolution to preserve information from negative values, a fact in harmony with the wavelet admissibility constraint that wavelet filters have zero mean (and therefore output negative values). Additionally, dilated convolution networks (Yu &amp; Koltun, 2016) and the â trous wavelet transform (AWT) (Shensa et al., 1992) both use dilated convolutions, and the former even proposes a nearly fixed initialization. Atrous Spatial Pyramid Pooling (ASPP) from DeepLabV3 (Chen et al., 2017) and a multi-scale AWT both apply dilated convolutions in parallel. ASPP has an analog to a first order wavelet scattering network due to its use of non-linearities. Wavelet scattering networks have been proposed as an interpretation of the basic structure of a convolutional network (Bruna &amp; Mallat, 2013). They place a custom structure in early (Oyallon et al., 2018) and middle layers (Cotter &amp; Kingsbury, 2019), while ASPP is placed at the end. The deep wavelet neural network of (de Freitas Barbosa et al., 2020) is essentially a one-level wavelet scattering transform with a multi-scale DWT but lacking explainability. Namely, it recursively applies a set of orthogonal wavelet filters and a downsampling step to an input image; the output is pooled to extract features for use with a classifier like a support vector machine or extreme learning machine. Moreover, deep convolutional networks have been interpreted as a multilayer implementation of a convolutional wavelet frames (Kang et al., 2018). These many similarities suggest deep network spatial filters can be initialized and fixed with wavelet-like filters.

Figure 2: Interpreting Deep Networks and Wavelets as edge detectors. Compare randomly sampled activations from ResNet (left) to a discrete wavelet transform (right).

<!-- image -->

Fixed spatial filter networks are preferable to existing related works. Table 1 highlights four criteria where spatially fixed methods improve deep networks more effectively than the existing approaches. ExplainFix is the only approach that satisfies all criteria. Wavelet Scattering Networks underperform on medium and large datasets and impose specific architectures that significantly limit their incorporation into standard deep networks. Deep wavelet networks add computational overhead by introducing new architecture, they are often not helpful in later layers, and spatial convolution filters are typically learned. Network Pruning methods prioritize efficiency over explainability and they do not necessarily have faster training or preserve predictive performance. Deep Networks are the reference baseline; they are fully learned blackbox models. We denote an equals sign where the baseline compares to itself by definition. Our proposed ExplainFix method satisfies all criteria, and ExplainFix visual explanations are mutually compatible with all methods.

## 3 Proposed Methods

The proposed methods are ordered into three subsections corresponding to the organization and development of our experiments: steerability, visual model-based explainability, and pruning. Sec. 3.1 introduces methods for showing that steered initialization is important to good performance of spatially fixed networks. We propose six spatial convolution initialization methods. The methods Ones and DCT2 are not steered, while Unchanged, GHaar, Psine, and GuidedSteer are steered. GHaar, Psine and GuidedSteer are novel contributions, and comprise three distinct ways to steer a network. GHaar linearly steers a sinusoid basis. Psine uses a polynomial steering of the basis. GuidedSteer initializes to the steering properties of a guide model. In order to visualize and verify our proposed initializations, we develop ExplainSteer in Sec. 3.2 as a novel visual model-based explanation tool. The ExplainSteer visualization shows at a glance the initialization of a network to verify the properties we designed for, and to enable sanity checks that spatial weights are indeed fixed before and after training. We extend ExplainSteer by incorporating a saliency weight, and we show how the explanation with saliency usefully highlights major inefficiencies in the design of deep network architectures. Finally, we exploit our saliency weighted visual model-based explanations by developing a saliency-based pruning method ChannelPrune in Sec. 3.3 to systematically remove unnecessary channels across a convolutional network, resulting in a smaller and faster deep network. The ExplainFix system comprises three components: fixed initialization methods, model-based visual explanation, and pruning.

## 3.1 Proposed Spatial Filter Initialization for Steerability Testing

Ones sets all spatial filter weights to one. The ones filter, also called a box filter, approximates a Gaussian blur. Wavelet transforms use box filters to obtain approximation coefficients. Ones is not steered. It tests our assumption that spatial filters need more complex initialization. Ones is visualized in Appendix F as a matrix with label '0'.

DCT2 initializes each spatial filter to a randomly selected orthonormal basis filter of the Discrete Cosine Transform Type II (DCT-II), defined in Sec. 3.2 and visualized in Appendix F. DCT2 is not steered. It enables testing the hypothesis that fixed spatial filter kernels should be steered.

Unchanged freezes the spatial filters of the model. We define Random Unchanged as spatial filters from a Kaiming Uniform random, never trained network. ImageNet Unchanged fi xes spatial filter weights to their

Figure 3: Proposed 2D GHaar Basis.

<!-- image -->

Top row: 2D Haar basis.

Bottom row: 2D GHaar basis for 3 × 3 fi lters.

'Ones' matrix aa ⊤ not shown.

## Algorithm 1 Layer-wise GuidedSteer Algorithm.

<!-- image -->

pre-trained values. We analyze how both initializations are steered in Sec. 4.4 and Appendix G. Visual examples of the Unchanged filter are shown in Appendix J.

GHaar is our first steered initialization. It generalizes and steers Haar wavelet basis filters. A multi-scale Haar transform iteratively downsamples an input image with a box filter to attain approximation images at different scales. It convolves the image at each scale with each of the three basis filters shown in the top row of Fig. 3. We highlight three challenges for a steerable Haar-like initialization: the initialization should have a multi-scale representation; it should work with kernels of shape 3 × 3 , 5 × 5 or h × w ; the initialization should steer a basis. Our approach addresses these challenges.

We generalize Haar wavelets by converting the 1-D step functions to sinusoid: g f ( x ) = cos ( f x ) , for a frequency f . The basis is 1-D separable, as shown in Fig. 3 and this construction generalizes to any dimensions. The 1-D vectors a and b in Fig. 3 are a = g f =0 ( x ) = cos(0 x ) and b = g f =1 ( x ) = cos(1 x ) , where x = linspace (0 , π, m ) and m = 2 (top row) or m = 3 (bottom row). The basis filters are periodic outside range f ∈ [0 , 2( m - 1)) . Our implementation addresses the three challenges: varying the frequency, f ∈ U [0 , 2( m - 1)] , enables a multi-scale representation; an outer product with m = h rows and m = w columns constructs ( h, w ) fi lters; third, the basis is guaranteed undercomplete. We steer with only three random frequencies, and the three may not be orthogonal. Example 2D GHaar filters are visualized in Fig. 4 for varying frequencies and steering weights, in both high resolution 20 × 20 filters to give a sense of the filter, and in 3 × 3 filters.

Psine steers an undercomplete basis with a polynomial combination of outer products, defined by the polynomial equation F = ∑ ℓ i =1 w i ( g ( x i ) g ( y i ) ⊤ ) p i . Psine is a weighted sum of ℓ outer products g ( x i ) g ( y i ) ⊤ , where each is raised element-wise to a positive integer power p i , and where x i and y i are vectors. We define g as a sinusoid function. 1 As the number ℓ of outer products increases, so does the capacity to represent higher order polynomials. To relate ℓ and p , we note that any N th order polynomial is steerable with 2 N +1 basis functions, and it is steerable with as few as N +1 basis functions if the function has entirely even or odd order terms (i.e. all terms x n y m satisfy n + m is even or odd) (Freeman, Adelson, et al., 1991). Thus, when N is too small, the filter cannot be steered in all possible directions. We will assume ℓ = N and secondly that p contains even and odd positive integers. To steer anywhere in the full space, we define ℓ ≥ 2 max( p ) + 1 , and we form an undercomplete basis for any given filter by randomly sampling ℓ and p . The filters are whitened to zero mean and unit norm. Example Psine filters are visualized in Fig. 5 for varying ℓ and max( p ) , where the top row satisfies and the bottom row does not satisfy the constraint ℓ ≥ 2 max( p ) + 1 . Fig. 6 shows example 3 × 3 filters as well as their higher resolution counterparts.

GuidedSteer initializes spatial filters with the same steering properties as a 'guide' set. Without loss of generality of the method, we apply GuidedSteer initialization layerwise to each each spatial convolution layer, where the guide set is the spatial filters of the corresponding layer of an ImageNet Unchanged model with identical architecture. The primary motivation for layer-wise GuidedSteer is to preserve steering properties of ImageNet trained models in case they contain predictively useful information not captured by GHaar or Psine initializations. Layer-wise GuidedSteer resets the connections between neighboring pointwise and spatial convolutions while preserving the steering properties within the spatial layer. More generally, the only restriction on the guide set is the filters have the same kernel shape. An interesting future application we do not explore but leave to future work is using GuidedSteer to initialize large or wide models from smaller or narrower pre-trained guide models. Example GuidedSteer filters are visualized in Fig. 7.

1 Our implementation adopted the GHaar sinusoid with x = linspace (0 , π ) and frequency f ∈ U [1 , 5] .

Figure 4: GHaar Filters visualized as 3 × 3 filters (top row) and in high resolution (bottom row) to give an intuitive sense of the filter. The frequencies and steering weights are randomly chosen.

<!-- image -->

Figure 6: Psine filters visualized as 3 × 3 filters (top row) and in higher resolution (bottom row).

<!-- image -->

Figure 5: Psine: Full and degenerate steering. The top rows satisfies ℓ ≥ 2max( p ) + 1 and steers the full space. The bottom row does not, resulting in degenerate filters.

<!-- image -->

Figure 7: GuidedSteer Visualization using as guide filters a DenseNet121 pre-trained on ImageNet. The filters do not have a high resolution counterpart.

<!-- image -->

Preliminary Definitions and Assumptions: Assume a guide collection of c spatial filter kernels of any given shape ( h, w, ... ) . The matrix G ∈ R c,m flattens each filter as a row vector. Without loss of generality, we assume the matrix G represents all same-sized spatial filter kernels in a pre-trained guide deep network. We wish to obtain a matrix F ∈ R n,m of n spatial filters similar to those in G . The number n can be chosen independently of c . Assume existence of a certain orthonormal basis, B ∈ R m,m , with a flattened basis filter per row. The projection GB ⊤ represents each filter as a row vector of 'steering' weights over the basis filters, and each i th column describes a distribution p ( w i | b i ) over possible steering weights w i given basis vector b i . A particular choice of basis B allows generating each row of F by independently sampling from the m column distributions p ( w i | b i ) . Then, F is a sampled approximation of the guide filters G .

We next consider simplifying assumptions to generate the rows of F . First, if we assume that the columns of GB ⊤ are linearly independent, then each column can be sampled independently. This is possible via the Singular Value Decomposition (SVD) of G = USV ⊤ , by choosing B = V ⊤ , since the covariance V ⊤ G ⊤ GV is diagonalized 2 and the columns of GV are linearly independent. 3 In implementation, SVD is memory intensive if G has many rows, so we compute V from the SVD of the covariance G ⊤ G = V S 2 V ⊤ . Second, we aim to capture the fact that different spatial layers of the network may steer differently. Therefore, we define a subset of rows G ℓ ⊂ G in order to define the guide spatial filters of a particular layer ℓ . While B is still obtained by SVD of G , we define each distribution p ( w i | b i ) from columns of G ℓ B ⊤ rather than GB ⊤ . Third, we can sample from each p ( w i | b i ) independently by assuming a normal distribution p ( w i | b i ) = N ( µ i , σ 2 i ) or by Gaussian Kernel Density Estimation (KDE) 4 . Finally, to sample a spatial filter: obtain samples w i ∼ p ( w i | b i ) ; concatenate into row vector w = [ w 1 , . . . , w m ] ; project w back to the spatial filter domain f i = w B + ¯ G where ¯ G = 0 if using non-centered SVD. Obtain F ℓ by generating rows f i . Without loss of generality, the process repeats for all ℓ layers. Algo. 1 is a pseudo-code implementation of layerwise non-centered GuidedSteer.

Table 2: ExplainSteer Notation.

| Algorithm 2 ExplainSteer Algorithm. | Algorithm 2 ExplainSteer Algorithm. | Variable | Description |
| - | - | - | - |
| 1: | procedure EXPLAINSTEER( F,B, w ) m,m , w ∈ R n ≥ | d ≥ 1 | Dimensionality of the basis and spatial filter kernels. |
| 2: 3: | Inputs: F ∈ R n,m ,B ∈ R 0 e ( d ) = w ∣ ∣ ∣ FB ⊤ ∣ ∣ ∣ | m ≥ 1 | Size of the spatial filter kernel and equivalently the basis size. |
| 4: | e (1) = [ e (1) h , e (1) w , . . . ] = zeros ( h ) , zeros ( w ) , . . . | F ∈ R n,m | Matrix of spatial filter kernels. |
| 5: | for e j ∈ e ( d ) do h idx ,w idx , · · · = get1dBasisVectorIdx ( j ) | B ( d ) ∈ R m,m | Matrix describing the basis, one basis vector per row. |
| 6: 7: 8: | e (1) h [ h idx ] += e 1 d j ; e (1) w [ w idx ] += e 1 d j ; . . . end for | w ∈ R n | Saliency weights, one per spatial filter kernel. |
| 9: 10: | E = asMatrix ( e (1) , shape= ( d, max( h, w,. . . ))) e (0) = E .mean(0) | f ∈ R m | Flattened spatial filter kernel (a row of F ). |
| 11: | return e ( d ) , e (1) , e (0) | b ( d ) ∈ R | Flattened basis filter ( i th row vector of B ( d ) |
| 12: | end procedure | i m ⊗ ◦ | ). Tensor outer product operator. Element-wise multiplication. |

## 3.2 Proposed ExplainSteer Method for Visual Model-based Explanations

To better understand how spatial filters are steered, sanity check our initializations, and expose inefficiencies in deep networks, we develop visual model-based explanations. Our explanations differ from the typical usage of explainability in that we do not explain model outputs but rather the internal properties of the model itself. The ExplainSteer method offers a spectrum characterizing relevant frequencies and edge orientations of a given set of spatial filters, two human-interpretable dimension reductions of the spectrum, and a saliency method. The explanation utilizes linear projection with any interpretable basis. We primarily adopt the DCT-II basis because it is 1-D separable and relates to the DCT2, GHaar and Psine initializations. We interpret ExplainSteer results in Sec. 4.4 to show how the visual explanations verify an initialization and expose deep network inefficiencies.

Preliminary Definitions: Table 2 summarizes most important notation. Assume no notation from previous sections is used in this section. A d -dimensional spatial filter is a tensor. Given a set of n spatial filters of dimension d , we define each row of a matrix F ∈ R n,m as a spatial filter f ′ ∈ R h,w,... flattened to a row vector f ∈ R m . We similarly define an orthonormal DCT-II basis B ( d ) ∈ R m,m with flattened basis filters b ( d ) i on rows. For any d ≥ 1 , the basis decomposes as a tensor outer product of the 1-d basis vectors, b ( d ) i = fl atten ( b (1) 1 ⊗··· ⊗ b (1) j ⊗··· ⊗ b (1) d ) where each b (1) j ∈ B (1) N j , and

2 We tested SVD with non-centered G and centered G - ¯ G and found no clear winner.

3 Linearly independent columns (LIC) follow from the definition of SVD, where GV = US and U and S have LIC.

4 We found no clear winner between KDE and assuming a normal distribution.

̸

$$\mathcal { B } _ { N _ { j } } ^ { ( 1 ) } = \left \{ s \cos \left ( \frac { 2 \pi } { N _ { j } } ( k + 0 ) ( x + . 5 ) \right ) \Big | s = \begin{cases} \sqrt { \frac { 1 } { N _ { j } } } & \text {if } k = 0 \\ \sqrt { \frac { 2 } { N _ { j } } } & \text {if } k \neq 0 \end{cases} , \quad k \in [ 0 \dots N _ { j } - 1 ] , x = [ 0 \dots N _ { j } - 1 ] \Big \} \quad ( 2 )$$

where j indexes the d spatial dimensions, N j is the size of the spatial filter and size of the basis in dimension j , k determines the frequency of a basis vector, and s ensures orthonormality. In this study we consider 2d basis filters. It is a matrix with spatial dimensions of h rows and w columns, and it has m = hw elements. It is an outer product of 1d basis vectors b (1) 1 ∈ B (1) h ⊂ R h and b (1) 2 ∈ B (1) w ⊂ R w . See Fig. 3 for 2d visual examples when m = 2 2 (top row) or m = 3 2 (bottom row); they are correct up to scale s . Appendix F visualizes DCT-II bases for d = 2 .

Columns of the linear projection P = FB ⊤ have nice interpretation. Each column corresponds to a basis filter b i . The total column magnitude characterizes how much energy each basis filter b i contributes to F . We define an energy spectrum e ( d ) = [ e ( d ) 1 . . . e ( d ) m ] , where each i th column of P is reduced via a weighted sum over its rows j (note that j in f j below and b j above have different meaning):

$$e _ { i } ^ { ( d ) } = \sum _ { j } w _ { j } \left | f _ { j } \left ( b _ { i } ^ { ( d ) } \right ) ^ { \top } \right | \quad \Longrightarrow \quad e ^ { ( d ) } = w \left | F B ^ { \top } \right |$$

Larger e ( d ) i corresponds to larger importance of basis filter b ( d ) i to the filters F . For example, when F represents 2 × 2 spatial filters (i.e. m = 2 2 , d = 2 ), it has an energy spectrum e ( d ) of four values corresponding to the importance of vertical, horizontal, diagonal and bias components in Fig. 3. The weight w j can introduce selective bias towards important spatial filters, or it can be w j = 1 n .

To visualize and verify an initialization method, we use the default weight, and we compute and visualize the energy spectrum for each layer of a model. In order to expose inefficiencies in deep networks, we also visualize the spectra by setting w j with weight saliency. We visualize the energy spectra of various deep networks, both with and without saliency in Appendix G. Detail describing the visualization is in Sec. 4.4. We observe that saliency makes the explanations more similar for a given architecture, thus making the explanation less dependent on initialization and more descriptive of the architecture. We define each saliency weight similarly to an input times gradient method method, except we use weights rather than pre-activations, and we sum over several examples:

$$w _ { j } = \sum _ { m } ^ { M } 1 ^ { \top } \left | \sum _ { ( x _ { i } , y _ { i } ) \in \mathcal { M } _ { m } } \frac { d \left ( y _ { i } ^ { \top } \frac { \xi _ { i } } { y _ { i } ^ { \prime } } \right ) } { d f _ { j } } \circ f _ { j } \right | ,$$

where M m is a minibatch containing labeled datapoints ( x i , y i ) with the ground truth labels y i in [0 , 1] ; M is the total number of minibatches 5 ; ˆ y i is the model output and ˆ y ′ i ≜ ˆ y i is a constant that rescales gradients so each output has even contribution; the absolute value is element-wise over the m weights in the spatial filter f j ; ◦ is element-wise multiplication; and 1 ∈ R m sums across all m spatial positions in f j .

Back-projecting e ( d ) ∈ R m onto the 1-D basis to obtain e (1) ∈ R h + w + ... or e (0) ∈ R max( h,w,... ) is an interpretable dimension reduction. Each scalar e (1) ℓ denotes an energy of a frequency in a spatial dimension (e.g. 'vertical components have low frequency'), and e (0) q describes the energy of the q th frequency generally (e.g. 'the spatial filters have low frequency components'). Therefore, the dimension reduction makes the spectrum easier to explain. Consider that 5 × 5 spatial filters have m = 25 elements, m = 25 basis filters, and correspondingly e ( d ) , e (1) and e (0) give 25 , 10 and 5 values.

5 We set M = 15 or M = 50 with a minibatch size of 4.

We next describe how to perform the reduction. Consider a 2-d spatial filter of m elements, so we have m basis filters, d = 2 , and f = w 1 b ( d ) 1 + · · · + w m b ( d ) m . A clarifying note on notation: w 1 ...m are steering weights; the saliency weight variables w j , defined above, are not explicitly used here. Each 2-d basis filter in the sum can be decomposed as a tensor outer product of 1-d basis vectors w i b (2) i = w i b ( d ) i = w i ( b (1) a ⊗ b (1) b ) = d √ w i b (1) a ⊗ d √ w i b (1) b , where the steering weight w i is equally distributed. We next substitute Eq. 3, w i ≜ e ( d ) i . Finally, for each distinct 1-d basis vector in the tensor outer product, we obtain the corresponding back-projected energy e (1) ℓ in Eq. 5. The expression attributes part of an energy e ( d ) i to a particular 1-d basis vector b (1) ℓ only if it was used to construct the corresponding basis filter, b ( d ) i .

$$e _ { \ell } ^ { ( 1 ) } = \sum _ { i } ^ { m } \delta _ { i , \ell } \, \mathcal { V } _ { i } ^ { \, \widehat { w } _ { i } } = \sum _ { i } ^ { m } \delta _ { i , \ell } \, \sqrt { e _ { i } ^ { ( d ) } } \quad \text {where} \quad \delta _ { i , \ell } = \begin{cases} 1 & \text {if } b _ { i } ^ { ( d ) } \text { is constructed with } b _ { \ell } ^ { ( 1 ) } \\ 0 & \text {otherwise} \end{cases}$$

Eq. 6 obtains e (0) by reshaping e (1) into a matrix E (1) ∈ R d, max( h,w,... ) . Each column of E (1) corresponds to a frequency of the DCT-II 1-d basis B (1) N j . The reshaping to E (1) and reduction of e (1) to e (0) is straightforward when all j spatial dimensions have equal shape (i.e. h = w = . . . ). In this case, B (1) N j = B (1) h = B (1) w = . . . since N j would be the same for all j , and a column sum or column mean of E (1) gives per-frequency energies e (0) ∈ R d . We do not analyze e (0) when spatial dimensions have different sizes N j , since (a) it would be unclear how to construct E (1) , and (b) deep networks typically use square kernels (such as 3 × 3 or 5 × 5 ).

$$e ^ { ( 0 ) } = \frac { 1 E ^ { ( 1 ) } } { d } \quad \text {where} \quad E ^ { ( 1 ) } = \begin{bmatrix} ( h \ e n g i e s \, in \, e ^ { ( 1 ) } \, \text { for dimension } 1 ) \\ ( w \ e n g i e s \, in \, e ^ { ( 1 ) } \, \text { for dimension } 2 ) \\ \dots \end{bmatrix}$$

Algo. 2 summarizes this explanation method. In Sec. 4.4, we analyze ExplainSteer explanations of different network architectures. Our analysis leads to saliency-based pruning of fixed filter networks in Sec. 4.5.

## 3.3 Proposed ChannelPrune Method for Smaller and Faster Models

Motivated by our analysis of ExplainSteer visual explanations in Sec. 4.4, we developed ChannelPrune to remove unnecessary spatial convolution filters from deep networks. We treat a deep network as a directed graph and then also carefully remove parameters throughout the network that are connected to unnecessary spatial convolution filters. The result is smaller, more efficient networks. The method has five steps: (1) obtain a saliency score for each spatial filter kernel; (2) zero out the least salient spatial filter kernels globally across the network; (3) prune entirely zeroed input and output channels of the spatial convolutions layers; (4) remove channels from neighboring layers by a graph traversal to correct inconsistencies in the network; (5) re-initialize any remaining zeroed kernel values with our FillZero method (described below). ChannelPrune preserves the most salient steering properties of the input model.

The saliency score in Eq. 4 can serve for step (1). We found that even using just the absolute value of the gradients is sufficient (i.e. assume f is a ones vector after computing the gradient). Step (2) uses saliency to zero out least salient spatial kernels instead of randomly choosing them. The results of our experiment to sanity check the saliency method in Appendix I and Fig. I.1b validate this choice, since the results on spatially fixed networks imply that randomly removing important fixed filters will strongly and negatively impact performance. Step (3) removes input and output channels from sparse spatial convolution layers by using the definition from Eq. 1 that each output channel of a spatial convolution is a sum of multiple cross-correlated input channels. Given a particular output channel O o , if all corresponding kernels f o,i were zeroed in step two, then that output channel O o and all connected kernels f o,i can be removed. Similarly, given an input I i , if all corresponding kernels f o,i were zeroed, then that input channel I i and its connected f o,i can be removed. Appendix M details why saliency-based channel pruning cannot typically remove all zeroed filters. After pruning a spatial layer, the network is inconsistent and needs correction. Step (4) ensures neighboring convolution, BatchNorm and Linear layers are channel pruned to precisely match the pruned spatial convolution layers. Step (4) represents the network as a graph and utilizes a graph traversal. We keep track of which specific channels were pruned, the direction traveled, the previous node, and we implement specific handling of various kinds of connections between layers. We refer to our source code for a complete reference (Gaudio, 2022). Finally, in step (5), we re-initialize all spatial weights that are close to zero with FillZero: entirely zero spatial kernels are re-initialized with Kaiming uniform random values; remaining zero weights are re-initialized with Gaussian noise of mean and variance equal to the spatial filter's mean and variance. We found it necessary to double the learning rate of pruned networks in order to fine-tune the non-fixed (non-spatial) weights. Sec. 4.5 and Appendix L show that fixed pruned models generally have matching predictive performance to the fully learned baseline while being both smaller and faster.

## 4 Experiments and Explanations

Experiment Design: We conduct ablative experiments in Sections 4.1, 4.2, and 4.3 verifying our 'fixed filters' design principle that all spatial convolution kernels can be fixed at initialization, that the fixed spatial initialization should have a steered representation, and that fixed models give speed and accuracy gains. In Sec. 4.4, we analyze ExplainSteer visual explanations, and we exploit these explanations in Sec. 4.5 to verify, in Sec. 4.5.3, our 'nimbleness' design principle. We show that most spatial weights, and indeed most weights generally, are not necessary for inference nor training. ExplainFix models have fixed spatial weights. They are always faster, smaller with ChannelPrune pruning, and as accurate as their corresponding fully learned baselines.

Datasets: CheXpert (Irvin et al., 2019) is a chest x-ray dataset of 223k:235 images for classification of five pathologies. BBBC038v1 (Caicedo et al., 2019) contains 670:65 annotated microscopy images for semantic segmentation. Appendix A details the datasets and how we split, pre-process and evaluate them.

Models: We implement 13 baseline models using four distinct kinds of deep network architectures. The DenseNet architecture (Huang et al., 2017) uses 1 × 1 and 3 × 3 convolutions, ResNet (He et al., 2016) uses the bottleneck ( 1 × 1 , 3 × 3 , 1 × 1 ), EfficientNet (Tan &amp; Le, 2019) uses depthwise separable bottlenecks where most weights are pointwise convolutions, and U-NetD is our U-Net (Ronneberger et al., 2015) implementation with entirely depthwise separable bottlenecks and channel expansion from MobileNetV2 (Sandler et al., 2018). We evaluate each architecture with non-spatial weights initialized to random values ('FromScratch') or ImageNet values ('PreTrained'). Table 3 and Table B.1 highlight how many spatial filter parameters each architecture has. See Appendix B for extensive details.

Reproducibility: Hyper-parameter configuration is defined in Appendix C. Source code at (Gaudio, 2022).

## 4.1 Steered Fixed Spatial Initializations Improve Predictive Performance

BBBC038v1: Steered initializations improve performance of spatially fixed models. Fig. 8 shows that the steered initializations significantly outperform the non-steered initializations. It also shows the GHaar initialization significantly outperforms the learned baseline. The figure represents 36 independently trained models, with six models per method. For each method on the x axis, we select the best of the six corresponding models and show its average test set performance for epochs 151-300 with a 95% confidence interval. Each model was trained for 300 epochs and evaluated on the out-of-distribution dataset at each epoch. We show average performance per epoch of all 36 models in Appendix D, and we compare it to a replicate the study with another set of 36 models evaluated the in-distribution dataset. Both appendix figures confirm the results in Fig. 8 to show that, in general, all steered spatially fixed models approximately equal or outperform the learned baseline with statistical significance.

CheXpert: Results show that (a) steered fixed filter methods significantly outperform the non-steered DCT2, (b) steered methods outperform the fully learned baselines on specific architectures, and (c) across all types of models generally, the steered fixed filter models approximately equal the baseline. Fig. 10a represents 36 independently trained models, each number represents the test set average ROC AUC over five tasks, and each column represents 6 trained models. The table in Fig. 10b reports the two-tailed p-values of paired-sample Wilcoxon tests to evaluate each column of the corresponding figure. The test supports our findings (a) and (c). On specific architectures, the figure suggests that Psine initialization is best for DenseNet, GHaar and GuidedSteer are best for EfficientNet, and in Appendix H we show that ImageNet Unchanged outperforms the pre-trained ResNet50 baseline on our hold out set of 56k validation images consistently for all epochs.

Figure 8: Steering Helps. Steered fixed models significantly outperform the non-steered models. GHaar significantly outperforms the learned baseline. Evaluated on BBBC038v1.

<!-- image -->

Figure 9: ExplainFix is Robust to training larger models.

<!-- image -->

Figure 10: Matching Accuracy. Steered fixed filter models approximately equal the baseline. Only DCT2 is not steered, and it has lower performance. Evaluated on CheXpert.

<!-- image -->

## 4.2 Computational Savings:

By design, fixed spatial filter models use less computation during training because the spatial filter weights are not updated during backpropagation. Pruning fixed filter models results in further savings. Table 3 shows how our ExplainFix models reduce per-epoch training time. The ResNet50 architecture has the largest percent of spatial parameters and shows the best savings from ExplainFix . The ExplainFix ResNet50 model is 5x smaller (only 21% of its parameters were kept after channel pruning) and 17% faster when training non-spatial weights. The DenseNet121 model is similarly pruned and fixed, while the EfficientNet model is fixed. We show expanded results in Fig. 11 that ExplainFix models are always faster than the fully learned baseline, even when the baseline is also pruned. Finally, to ensure integrity of our results, we sanity check the predictive performance of our methods: Sec. 4.1 verified that spatially fixed models approximately equal or surpass baseline performance; Appendix L shows that fixed and pruned ExplainFix models match baseline predictive performance while always being faster and smaller.

All SPE measurements in the Table 3 and Fig. 11 were obtained by training each model on the same isolated server and NVIDIA RTX2080 Ti GPU for 20 epochs, with no other sources of computational or input-output

Table 3: ExplainFix Saves Time.

| Architecture | Spatial Params | SPE * | SPE * |
| - | - | - | - |
|  | Num (% total) | ExplainFix | Baseline |
| ResNet50 | 11.3e6 (48%) | 191 ( 17% ) | 231 |
| DenseNet121 | 2.1e6 (31%) | 347 ( 9% ) | 383 |
| EfficientNet-b0 | 0.2e6 (5%) | 259 ( 3% ) | 267 |

(*) SPE compares Mean Seconds Per Training Epoch of the ExplainFix and Baseline models and reports (% savings) with respect to the baseline. SPE of ResNet and DenseNet also analyzed in Fig. 11.

Figure 11: Always Faster And Smaller than fully learned baselines.

<!-- image -->

bottlenecks. ExplainFix models are always faster than their fully learned counterparts.

## 4.3 Steered and Fixed Initializations Improve Robustness

Steering is important to spatial filter initialization . We observed on both datasets that steered initializations significantly improve performance over non-steered initializations. The results suggest that relying on architecture and learning dynamics alone to steer fixed spatial feature maps causes lower performance. When the fixed spatial filter kernels are initialized with a redundant and steered representation, performance typically increases.

Better robustness to larger learning rate: On BBBC038v1, we evaluate whether the fixed filter models are sensitive to an increase in learning rate. In Appendix E, we train and evaluate 36 models (six models per initialization method) with learning rate 0.008, and we replicate the experiment with learning rate 0.02. We observe the larger learning rate models more clearly outperform the baseline early in training, and that the non-steered DCT2 method performs significantly worse. Non-steered spatial initializations rely on the learning dynamics to build steerability into the network architecture. When these dynamics are dysfunctional, such as induced by larger learning rate, the model struggles to learn. These results suggest steered fixed initialization improves robustness to an increased learning rate, and further that steered initialization facilitates learning.

Robust to varying model capacity: On CheXpert, we evaluate how performance changes with varying model size, all other parameters constant. In Fig. 9 we observe that fixed filter models do not underperform when the model capacity increases. The x-axis corresponds to models of increasing capacity via compound scaling; b0 has smallest depth and channel width, and fewest number of learned parameters while b5 has 7x more parameters. Each point represents the test set performance of a model trained for 80 epochs on CheXpert. We also evaluate smaller models by pruning the baseline to as little as 10% of its original size. In Fig. L.1b, we observe that pruned fixed models have approximately equal performance to pruned learned models. In Sec. 3.3, we describe the pruning method. The results suggest that fixed spatial filter models behave approximately equally to the learned model as capacity increases or decreases.

Figure 12: Low Frequencies and Early Layers Suffice. (a) The e (2) spectra for DenseNet121 exposes its inefficiencies as vertical, horizontal and repeating trends. (b) The e (0) spectra shows that spatial filters across all models tested prioritize low frequencies (green) over high frequencies. More examples in Appendix G.

<!-- image -->

## 4.4 ExplainSteer: Explanations of Spatial Filter Steering

Our ExplainSteer explanations are primarily visual, applied globally to all spatial filters for e (0) and layerwise as a heatmap for e (2) . The explanations highlight inefficiencies of deep networks and led us to the 'nimbleness' principle that nearly all spatial filters are unnecessary for both inference and training.

Observation 1: Spatial kernels prioritize low frequencies. Fig. 12b shows e (0) spectra globally across different models. The models show a preference for lower frequencies, especially in ResNet50 and DenseNet121. Concurrent with our work, (Tancik et al., 2020) showed that multi-layer perceptrons have a natural bias towards low frequencies. Our saliency-based visual explanations suggest spatial convolution layers in CNNs may share this bias. Since saliency is dependent on a dataset, we also note a relation to the dataset domain. In chest x-ray data, low frequencies correspond to gradients and lines of structural features, like the outlines of the clavicle, diaphram, heart and ribs. These strong features anchor the key focus areas and are necessary anatomical landmarks to address the primary tasks in the CheXpert dataset. Thus, it also makes sense on CheXpert that low frequencies would be most salient.

Observation 2: Nimbleness Insight. Deep Networks tend to prioritize low frequency spatial convolution filter kernels located at early layers of the model. High frequency kernels and other spatial layers are largely unnecessary. Fig. 12a visualizes an example e (2) explanation. Appendix G interprets many of these plots with varying model architecture and initialization. Each row of the heatmap visualizes energies of a single basis filter over all spatial convolution layers of the network. Each column shows the e (2) energy spectrum of the spatial filter kernels in a given spatial convolution layer. The bar plots show the respective row and column sums of the heatmap. The basis vector index on y-axis corresponds to basis filter index numbers shown in Appendix F; indices sort the filters from lowest to highest frequency. In the plots, a column of 9 and 25 values correspond, respectively, to the 3 × 3 and 5 × 5 basis vectors shown and numbered in Appendix Fig. F .1a and F .1b.

The energy spectra across layers of the heatmap have intuitive interpretation. For instance, the relative contribution of bias, vertical gradients, and horizontal gradients are visualized as the first three heatmap rows (basis filter indices zero, one, and two). For any given heatmap column, a vertical trend from dark to bright shows that the filters prioritize low frequency components more than high frequency components. For any given row, a horizontal trend from dark to bright shows that the earlier filters have higher energy than later filters. Repeating horizontal trends in Fig. 12a of a DenseNet121 model suggest the spatial convolution layers towards the end of dense blocks are unnecessary. The additional ExplainSteer heatmaps in Appendix G show these trends consistently across architectures and initializations. These explanations also enable comparison of pre-trained versus random initialization in fully learned baselines. We observe that ImageNet pre-training concentrates, or sparsifies, energy onto low frequency components and fewer spatial layers better than learning from random networks. The main observations of the nimbleness insight are that the deep networks analyzed are inefficient by design.

Observation 3: ExplainSteer enables sanity checks on the initialization method. The heatmaps without saliency describe how a model is steered. The DCT2 initialization, for example, should appear as a flat (green) heatmap. The GuidedSteer initialization should look exactly like its guide model, ImageNet Unchanged. In Appendix G, we present these sanity checks and we observe that all sanity checks pass our visual inspection. The visual is also useful to verify a fixed initialization was indeed fixed during training.

## 4.5 Exploiting ExplainSteer Explanations

The ExplainSteer explanations suggest that all models we analyzed are inefficient, and that different fixed initializations can concentrate the ExplainSteer energy spectra in sparse locations. To test and exploit this explanation, we propose to eliminate nearly all kernels in spatial convolution layers with three sets of experiments. The first two experiments zero out spatial kernels. The third experiment prunes networks by removing zeroed input and output channels systemically across the network. Sec. 4.5.1 shows that nearly all spatial kernels are unnecessary for inference. Sec. 4.5.2 shows that most spatial kernels are unnecessary for training non-spatial weights. Sec. 4.5.3 shows that deep networks can be fixed and pruned to a fifth their size with approximately equal predictive performance to a fully learned baseline.

## 4.5.1 Few Spatial Filters Suffice for Inference

Least salient kernels are unnecessary for inference and for training. We conduct two sub-experiments to verify this hypothesis. All experiments are done on CheXpert for ResNet50, DenseNet121 and EfficientNet-b0.

The first experiment is a sanity check of our saliency metric. We evaluate the effect of progressively zeroing out least (Fig. I.1a) and most (Fig. I.1b) salient spatial convolution filter kernels. The results and extra details on experiment configuration are in Appendix I. We make two observations. First, removing least salient spatial filters has no performance drop in some cases even when 50% of kernels are zeroed. Second, removing less than 3% of the most salient filters makes the models have random output. We conclude that (a) a very small portion of most salient spatial kernels are essential for good performance, (b) the majority of least salient kernels are unnecessary, and (c), most importantly, the saliency metric works.

Our next experiment verifies that few spatial filters are necessary for inference. Leveraging our explanation from Sec. 4.4 that ImageNet pre-training sparsifies the ExplainSteer energy, we initialize 18 different networks with ImageNet Unchanged initialization, zero out the least salient filters, and fix all spatial layers. We then independently train each network on CheXpert and compare it to a corresponding fully learned baseline. We provide details in Appendix K and results in Fig. K.1b. We observe that 99%, 95% and 85% of spatial filters are unnecessary for inference on CheXpert data in ResNet50, DenseNet121 and EfficientNet-b0, respectively. Nearly all spatial filters are unnecessary for inference in spatially fixed models.

## 4.5.2 Few Spatial Filters Suffice for Training

In this experiment, we zero the fixed network before it is trained on CheXpert. We start with 18 completely random (never trained) networks, fix all spatial weights and zero out the least salient filters. Results and additional details are in Appendix K and in Fig. K.1a. We observe that the fixed and zeroed models are nearly equal to the learned baseline and that 95%, 90% and 85% of spatial filters are unnecessary for training non-spatial weights in ResNet50, DenseNet121 and EfficientNet-b0, respectively. Most spatial filters are unnecessary for training non-spatial weights.

## 4.5.3 Pruned Networks

Our conclusions that most spatial filters are unnecessary for training imply that the several input and output channels across the network, including channels in non-spatial layers, can be removed from deep networks. To verify this implication, we designed ChannelPrune to systemically remove input and output channels connected to the spatially zeroed kernels.

Channel pruning removes channels from several connected layers. The ImageNet pre-trained networks we prune have learned BatchNorm layers or bias terms that contribute predictive information regardless of whether the input activations have been zeroed at a spatial convolution layer. It is thus possible that pruning these bias terms results in some performance loss, though we note three facts. First, this bias can be approximately incorporated into the spatial layers by steering with the lowest frequency basis filter (an all ones matrix). Second, the FillZero fixed initialization in Methods Sec. 3.3 is our solution to mitigate performance loss caused by this bias. Third, random networks that are fixed and spatially sparse (as done in Sec. 4.5.2), guarantee exactly identical unchanged predictive performance before and after pruning because, in channels that are zeroed out, the corresponding 'ghost' bias terms also remain zero during training (they cannot receive non-zero gradient updates).

We conduct a sanity check on our pruning to verify that the pruned networks indeed have matching predictive performance to the fully learned baseline in Appendix L. The results of the sanity check agree with our results in Sec. 4.5.1 and Sec. 4.5.2, and further reinforce our overall hypothesis the spatial weights of a deep network, even a pruned deep network, can be fixed. Fixed and pruned ExplainFix models, as shown in Fig. 1c, train faster than the fully learned baseline with matching accuracy.

## 5 Discussion and Future Work

The current paradigm of deep learning is limited by its strong emphasis on weight optimization. Weight optimization is over-emphasized. ExplainFix shows that most deep network weights can be removed before training. All spatial convolution weights can be fixed. Deep networks contain millions of weights, and yet most of these weights do not need to be learned or even used. ExplainFix is a significant step towards realization of fully fixed networks. We next emphasize additional benefits of fixed weight deep networks in context of future research.

Standard deep networks define a fixed architecture at initialization. The fixed architecture does not adapt to the input. Architecture search is also computationally costly because the optimization occurs over both architecture and weights. Fixed weight architectures eliminate optimization over weights. They enable dynamically generated or adaptive architectures and more efficient architecture search. For instance, fixed weight networks can have essentially infinite size by dynamically expanding along selective paths, such as by the use of fixed weight attention mechanisms and a fixed weight operator such as a wavelet transform. Fixed weight networks enable dynamic or learned architectures.

Towards explainability, development of fully fixed networks requires and encourages strong understanding of how deep networks work. An explanation should affect decision making. Simply explaining predictions of single examples, for instance via interpretation of GradCam (Selvaraju et al., 2017) or Lime (Ribeiro et al., 2016), may help a physician trust a model's output given a particular input, but it does not sufficiently explain the model itself or suggest ways to improve it. The research literature needs model-based explanations that lead to concrete suggestions on how to improve a model. ExplainFix offers such model-based explanations. Our ExplainSteer visual explanations show that spatial weights are mostly unnecessary for training and inference especially in later layers and in higher frequency kernels. We also explain that steered spatial filter initialization facilitating learning dynamics, and resulting in better performance. ExplainFix results in nimble, higher performing and better explained models.

Fully fixed deep networks would need no weight training. They would enable development and democratization of larger and deeper networks at greatly reduced computational and data acquisition needs. ExplainFix 's focus on spatial convolution filters marks a significant step towards fully fixed networks. Future work should to consider fixing pointwise convolutions and ways to exploit trade-offs between fixed architectures and fixed weights.

## 6 Conclusion

ExplainFix is adopts two design principles, the 'fixed filters' principle that the weights of deep networks can be entirely fixed at initialization and never learned, and the 'nimbleness' principle that few parameters are necessary for training and inference. We give three main contributions:

- Model-based Explanations: We develop a saliency-based visualization tool to interpret spatial filter kernels in spatially fixed CNNS. We show that fixed CNNS should have a steered representation to attain good performance, all spatial filters can be fixed, and most can be eliminated (up to 100x) from common deep network architectures on medical image data.
- Speed and Accuracy Gains: ExplainFix models guarantee faster training (up to 17% savings), smaller models with channel pruning (up to 5x fewer parameters), and they have matching or improved predictive performance. Our novel steered initializations give winning ticket models that can even outperform fully learned networks.
- Novel Tools: We contribute open sourced tools dedicated to fixed spatial filter networks, including three novel spatial filter initialization methods (GHaar, Psine, GuidedSteer), a novel explanation method (ExplainSteer), and a novel deep network pruning method (ChannelPrune).

ExplainFix spans an extensive empirical analysis of four architectures, two medical image datasets, 13 baseline models and more than 330 distinct trained models.

## Acknowledgments

The project funding this work, Transparent Artificial Medical Intelligence (NORTE-01-0247-FEDER045905), is co-financed under the CMU-Portugal International Partnership by European Regional Fund through the North Portugal Regional Operational Program-NORTE 2020 (ERDF) and by the Portuguese Foundation for Science and Technology (FCT).

Special thanks to Shreshta Mohan for her review of the CheXpert loss function and correction of typographical errors.

## References

- Adadi, A., &amp; Berrada, M. (2018). Peeking inside the black-box: a survey on explainable artificial intelligence (XAI). IEEE access , 6 , 52138-52160.
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., &amp; Kim, B. (2018). Sanity checks for saliency maps. Advances in Neural Information Processing Systems , 9505-9515.

Advani, M. S., Saxe, A. M., &amp; Sompolinsky, H. (2020). High-dimensional dynamics of generalization error in neural networks. Neural Networks , 132 , 428-446.

Angelov, P ., &amp; Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks , 130 , 185-194.

Bruna, J. (2013). Scattering representations for recognition (Doctoral dissertation).

- Bruna, J., &amp; Mallat, S. (2013). Invariant scattering convolution networks. IEEE transactions on pattern analysis and machine intelligence , 35 (8), 1872-1886.

Buhrmester, V., Münch, D., &amp; Arens, M. (2021). Analysis of Explainers of Black Box Deep Neural Networks for Computer Vision: A Survey. Machine Learning and Knowledge Extraction , 3 (4), 966-989. https: //doi.org/10.3390/make3040048

- Caelli, T. M., Bischof, W. F ., &amp; Liu, Z.-Q. (1988). Filter-based models for pattern classification. Pattern recognition , 21 (6), 639-650.
- Caicedo, J. C., Goodman, A., Karhohs, K. W., Cimini, B. A., Ackerman, J., Haghighi, M., Heng, C., Becker, T., Doan, M., McQuin, C., et al. (2019). Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nature methods , 16 (12), 1247-1253.
- Chen, L.-C., Papandreou, G., Schroff, F., &amp; Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. CoRR , abs/1706.05587 . http://arxiv.org/abs/1706.05587
- Cheng, Y., Wang, D., Zhou, P ., &amp; Zhang, T. (2018). Model compression and acceleration for deep neural networks: The principles, progress, and challenges. IEEE Signal Processing Magazine , 35 (1), 126136.

- Cohen, T., &amp; Welling, M. (2016). Group equivariant convolutional networks. International conference on machine learning , 2990-2999.
- Cotter, F ., &amp; Kingsbury, N. (2017). Visualizing and improving scattering networks. 2017 IEEE 27th International Workshop on Machine Learning for Signal Processing (MLSP) , 1-6.
- Cotter, F ., &amp; Kingsbury, N. (2019). A learnable scatternet: Locally invariant convolutional layers. 2019 IEEE International Conference on Image Processing (ICIP) , 350-354.
- Daubechies, I. (1992). Ten lectures on wavelets . SIAM.
- de Freitas Barbosa, V. A., de Santana, M. A., Andrade, M. K. S., de Lima, R. d. C. F., &amp; dos Santos, W. P. (2020). Deep-wavelet neural networks for breast cancer early diagnosis using mammary termographies. Deep learning for data analytics (pp. 99-124). Elsevier.
- Frankle, J., &amp; Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. https://openreview.net/forum?id=rJl-b3RcF7
- Freeman, W. T., Adelson, E. H. et al. (1991). The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence , 13 (9), 891-906.
- [Gaudio, A. (2022). Open Source Code. https://github.com/adgaudio/ExplainFix](https://github.com/adgaudio/ExplainFix)
- Gaudio, A., Smailagic, A., &amp; Campilho, A. (2020). Enhancement of Retinal Fundus Images via Pixel Color Amplification. International Conference on Image Analysis and Recognition , 299-312.
- Grossmann, A. (1988). Wavelet transforms and edge detection. Stochastic processes in physics and engineering (pp. 149-157). Springer.
- Han, S., Mao, H., &amp; Dally, W. J. (2016). Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding. In Y. Bengio &amp; Y. LeCun (Eds.), 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings . http://arxiv.org/abs/1510.00149
- [Hanov, S. (2006). Wavelets and Edge Detection . http://stevehanov.ca/cs698\_wavelet\_project.pdf](http://stevehanov.ca/cs698_wavelet_project.pdf)
- He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition , 770-778.
- Hestness, J., Narang, S., Ardalani, N., Diamos, G. F., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y., &amp; Zhou, Y. (2017). Deep Learning Scaling is Predictable, Empirically. CoRR , abs/1712.00409 . http://arxiv.org/abs/1712.00409
- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., &amp; Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CoRR , abs/1704.04861 . http://arxiv.org/abs/1704.04861
- Huang, G., Liu, Z., Van Der Maaten, L., &amp; Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition , 47004708.
- Irvin, J., Rajpurkar, P ., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al. (2019). Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. Proceedings of the AAAI Conference on Artificial Intelligence , 33 , 590-597.
- Kang, E., Chang, W., Yoo, J., &amp; Ye, J. C. (2018). Deep convolutional framelet denosing for low-dose CT via wavelet residual network. IEEE transactions on medical imaging , 37 (6), 1358-1369.
- Leino, K., Sen, S., Datta, A., Fredrikson, M., &amp; Li, L. (2018). Influence-directed explanations for deep convolutional networks. 2018 IEEE International Test Conference (ITC) , 1-8.
- Lin, M., Chen, Q., &amp; Yan, S. (2014). Network In Network. In Y. Bengio &amp; Y. LeCun (Eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings . http://arxiv.org/abs/1312.4400
- Luan, S., Chen, C., Zhang, B., Han, J., &amp; Liu, J. (2018). Gabor convolutional networks. IEEE Transactions on Image Processing , 27 (9), 4357-4366.
- Malach, E., Yehudai, G., Shalev-Schwartz, S., &amp; Shamir, O. (2020). Proving the lottery ticket hypothesis: Pruning is all you need. International Conference on Machine Learning , 6682-6691.
- Mallat, S. (2012). Group invariant scattering. Communications on Pure and Applied Mathematics , 65 (10), 1331-1398.

- Mallat, S. (2016). Understanding deep convolutional networks. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences , 374 (2065), 20150203.
- Mallick, P . K., Ryu, S. H., Satapathy, S. K., Mishra, S., Nguyen, G. N., &amp; Tiwari, P . (2019). Brain MRI image classification for cancer detection using deep wavelet autoencoder-based deep neural network. IEEE Access , 7 , 46278-46287.
- Oyallon, E., Zagoruyko, S., Huang, G., Komodakis, N., Lacoste-Julien, S., Blaschko, M., &amp; Belilovsky, E. (2018). Scattering networks for hybrid representation learning. IEEE transactions on pattern analysis and machine intelligence , 41 (9), 2208-2221.
- Patrick, M. K., Adekoya, A. F ., Mighty, A. A., &amp; Edward, B. Y . (2019). Capsule networks-a survey. Journal of King Saud University-Computer and Information Sciences .
- Pensia, A., Rajput, S., Nagle, A., Vishwakarma, H., &amp; Papailiopoulos, D. (2020). Optimal Lottery Tickets via SUBSETSUM: Logarithmic Over-Parameterization is Sufficient. Advances in neural information processing systems .
- Pérez, J. C., Alfarra, M., Jeanneret, G., Bibi, A., Thabet, A., Ghanem, B., &amp; Arbeláez, P . (2020). Gabor Layers Enhance Network Robustness. European Conference on Computer Vision , 450-466.
- Pham, H. H., Le, T. T., Tran, D. Q., Ngo, D. T., &amp; Nguyen, H. Q. (2021). Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels. Neurocomputing , 437 , 186-194.
- Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). " Why should i trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining , 1135-1144.
- Ronneberger, O., Fischer, P ., &amp; Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer-assisted intervention , 234-241.
- Rosenfeld, A., &amp; Tsotsos, J. K. (2019). Intriguing properties of randomly weighted networks: Generalizing while learning next to nothing. 2019 16th Conference on Computer and Robot Vision (CRV) , 9-16.
- Said, S., Jemai, O., Hassairi, S., Ejbali, R., Zaied, M., &amp; Amar, C. B. (2016). Deep wavelet network for image classification. 2016 IEEE International conference on systems, man, and cybernetics (SMC) , 000922-000927.
- Samek, W., Wiegand, T., &amp; Müller, K.-R. (2017). Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models. CoRR , abs/1708.08296 . http://arxiv.org/abs/1708.08296
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., &amp; Chen, L.-C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE conference on computer vision and pattern recognition , 4510-4520.
- Segebarth, D., Griebel, M., Stein, N., von Collenberg, C. R., Martin, C., Fiedler, D., Comeras, L. B., Sah, A., Schoeffler, V., Lüffe, T., et al. (2020). On the objectivity, reliability, and validity of deep learning enabled bioimage analyses. Elife , 9 , e59780.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE international conference on computer vision , 618-626.
- Shensa, M. J. et al. (1992). The discrete wavelet transform: wedding the a trous and Mallat algorithms. IEEE Transactions on signal processing , 40 (10), 2464-2482.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &amp; Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research , 15 (1), 1929-1958.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., &amp; Wojna, Z. (2016). Rethinking the inception architecture for computer vision. Proceedings of the IEEE conference on computer vision and pattern recognition , 2818-2826.
- Tajbakhsh, N., Jeyaseelan, L., Li, Q., Chiang, J. N., Wu, Z., &amp; Ding, X. (2020). Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation. Medical Image Analysis , 101693.
- Tan, M., &amp; Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks, 6105-6114.

- Tancik, M., Srinivasan, P . P ., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., &amp; Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In H. Larochelle, M. Ranzato, R. Hadsell, M.-F . Balcan, &amp; H.-T. Lin (Eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual . https://proceedings. neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html
- Thompson, N. C., Greenewald, K. H., Lee, K., &amp; Manso, G. F. (2020). The Computational Limits of Deep Learning. CoRR , abs/2007.05558 . https://arxiv.org/abs/2007.05558
- Tjoa, E., &amp; Guan, C. (2020). A survey on explainable artificial intelligence (xai): Toward medical xai. IEEE Transactions on Neural Networks and Learning Systems .

Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. Biometrics Bulletin , 1 (6), 80-83. http: //www.jstor.org/stable/3001968

- Yu, F ., &amp; Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. CoRR , abs/1511.07122 .

Supplementary Appendices for 'ExplainFix: Explainable Spatially Fixed Deep Networks'

## Abstract

Is there an initialization for deep networks that requires no learning? ExplainFix adopts two design principles: the 'fixed filters' principle that all spatial filter weights of convolutional neural networks can be fixed at initialization and never learned, and the 'nimbleness' principle that only few network parameters suffice. We contribute (a) visual model-based explanations , (b) speed and accuracy gains , and (c) novel tools for deep convolutional neural networks. ExplainFix gives key insights that spatially fixed networks should have a steered initialization, that spatial convolution layers tend to prioritize low frequencies, and that most network parameters are not necessary in spatially fixed models. ExplainFix models have up to 100x fewer spatial filter kernels than fully learned models and matching or improved accuracy. Our extensive empirical analysis confirms that ExplainFix guarantees nimbler models (train up to 17% faster with channel pruning), matching or improved predictive performance (spanning 13 distinct baseline models, four architectures and two medical image datasets), improved robustness to larger learning rate, and robustness to varying model size. We are first to demonstrate that all spatial filters in state-of-the-art convolutional deep networks can be fixed at initialization, not learned.

Keywords Deep Learning, Computer Vision, Fixed-Weight Networks, Explainability, Pruning, Medical Image Analysis

## A Dataset Details

<!-- image -->

<!-- image -->

(a) BBBC038v1 Dataset

(b) CheXpert Dataset

Figure A.1: Example images from the datasets we used.

CheXpert: The CheXpert dataset Irvin et al., 2019 published by Stanford ML Group and Stanford Hospital contains 223,415:235 train:test chest x-rays for binary classification of 14 classes. Only five classes are emphasized by most literature, and we therefore consider just the five classes: Atelectasis, Cardiomegaly, Consolidation, Edema, and Pleural Effusion. The train labels are generated by an automatic software that mines text data in associated unpublished medical reports, and the test set labels are manually curated by three radiologists. Most labels are missing and some are marked uncertain. There are several possible baselines that attempt to incorporate the uncertainty labels. We compare against non-ensembled singlemodel U-Ignore baseline reported in the state of art literature Pham et al., 2021, and we provide details on this baseline and comparison to state of art in Appendix B. This baseline assumes missing labels are negative and excludes uncertainty labels from the analysis.

Regarding implementation, we use the CheXpert-v1.0-small dataset, which contains already down-sampled images. We permanently split the training set into 70%:30% (156389:67025) using a random seed of 138 and PyTorch's default random number generator. We train on the 70% split and evaluate the 30% validation set to find hyperparameters. All results we report are on the 200 image holdout test set from models trained on our 70% split of the training set. For pre-processing, we randomly crop to images of (320 , 320) pixels on the 70% training set, and on the validation or test sets we upsample images in a minibatch by padding zeros. The minibatch size is four images. Each epoch considers only 15k images sampled with replacement from the 70% split. To evaluate performance, we report the average test set ROC AUC across the five classes to enable comparison with most literature that uses the dataset and non-ensembled U-Ignore baseline.

BBBC038v1: The BBBC038v1 dataset Caicedo et al., 2019, hosted by the Broad Bioimage Benchmark Collection, is a diverse collection of microscopy images visualizing cell nuclei. The dataset is representative of the domain: images are aggregated from a variety of laboratories, they show cell nuclei of different animals in different states and contexts, and present varying lighting, magnification, staining, and background/foreground coloration. The dataset contains 670:65:106 images annotated with pixel-wise segmentation masks. The 106 hold-out images are out-of-distribution, as they come from entirely different laboratories and they present experimental conditions not in the 670:65 data. We treat both the 65 and 106 image datasets as test sets in order to have an in-distribution vs out-of-distribution analysis. We visualize example images in Fig. A.1a.

To choose hyperparameters, data pre-processing and loss, we temporarily split the training set into 468:202 training and cross validation sets using a different random seed each time we train a model. We report all results on the 106 image holdout set, except in Appendix D where we show results on the in-distribution vs out-of-distribution datasets. Regarding pre-processing, all images were normalized into [0,1] and then shifted to have a 0.5 mean. On the training set, the images were subjected to random flipping (even probability of no flip, horizontal, vertical, both), and then with 80% probability a pixel-wise noise function for each pixel x ′ = clip ( x + N (0 , 1) / U (5 , 10)) where clip puts the image into a [0,1] range. We use a batch size of ten on the training set, and one on the validation and test sets. We report results using the Sørensen-Dice Coefficient evaluated on the test set.

## B Model Details

We consider three standard architectures for the CheXpert dataset: DenseNet Huang et al., 2017, ResNet He et al., 2016 and EfficientNet Tan and Le, 2019. DenseNet121 is used by state-of-the-art literature on chest x-ray data, including CheXpert Pham et al., 2021. ResNet introduced the residual skip connection and the bottleneck sequence of convolutions 1 × 1 , 3 × 3 , 1 × 1 . Sec. 2 explains that this sequence enables the network architecture to perform steering of inputs and outputs to the spatial convolution. The DenseNet architecture builds on ResNet's skip connection with "dense" blocks and is largely defined as a sequence of 1 × 1 , 3 × 3 convolutions, thus steering spatial convolution inputs. The EfficientNet architecture uses the depthwise separable convolution with both 3 × 3 and 5 × 5 fi lters, though the network primarily contains of pointwise convolutions. Table 3 (middle column) compares the number and proportion of spatial parameters in each model. The pre-trained models use publicly available ImageNet weights, and for pre-trained EfficientNet we use the weights obtained via the AdvProp method. All models are modified to have 1 channel grayscale image input and output an unnormalized vector with five values corresponding to the five CheXpert tasks.

On the CheXpert U-Ignore baseline, current state-of-the-art literature reports a mean AUC ROC of 0.863, using DenseNet121 with a full training set Pham et al., 2021, meaning that each training image is considered five times in total (five epochs). All our baseline models are comparable to this baseline (see Fig. B.1a), and our models use significantly less training data (150k training images instead of 223k, and each image is used for training only four times on average). In Fig. B.1a, each dot represents an independent training of one of our baseline models. The gray dashed line is the 0.863 ROC AUC from existing literature.

On the microscopy dataset, we implement our own depthwise-separable U-Net Ronneberger et al., 2015 encoder-decoder architecture, called U-NetD. For clarity, the architecture is visualized with only 3 levels in Fig. B.1c. We use 5 levels, each with a different channel width: (3 , 8 , 16 , 32 , 64) . In the figure, green blocks are spatial 3 × 3 grouped convolutions with 6x channel expansion, blue blocks are pointwise convolutions, and red blocks are bilinear upsampling. Between convolutions, we insert CELU activations and then BatchNorm, though the pointwise convolution following every spatial convolution has no activation, as suggested in MobileNetV2 Sandler et al., 2018. We fuse the encoder output with previous decoder output using a weighted sum with two learned scalar weights. The final layer of U-NetD (not shown in figure) is a 2D spatial convolution that maps its 3 channel output to a 1 channel segmentation mask. All spatial convolutions are 3 × 3 kernels with no bias.

We compare this U-Net model against DeepLabv3+ with the ResNet50 and MobileNetV2 backbones in Fig. B.1b. Each architecture presented was trained three times independently and evaluated on the out-ofdistribution BBBC038v1 test set. The boxenplot shows test set segmentation performance after 150 epochs. The U-NetD model outperforms the MobileNetV2 backbone. It converges almost the same median value as the ResNet50 backbone. U-NetD has one and two orders of magnitude fewer parameters than the MobileNet and ResNet50 DeepLabV3+ architectures, respectively (see Table B.1). We adopt the U-NetD over the DeepLabv3+ models because (a) the ResNet backbone is already considered in the CheXpert models, (b) U-NetD follows MobileNetV2 recommendations in its design (MobileNetv2 is desirable to emulate for its steering and wavelet-like properties as described in Sec. 2), (c) U-NetD presents another common kind of architecture (the U-Net), and (c) U-NetD has orders of magnitude fewer parameters.

<!-- image -->

Table B.1: BBBC038v1 Baselines: U-NetD has fewest parameters

| Model | Num Params | Spatial Params (% total) |
| - | - | - |
| U-NetD | 116695 | 12744 (11%) |
| DeepLabV3+ MobileNetV2 | 5220577 | 2977344 (57%) |
| DeepLabV3+ ResNet50 | 39756705 | 26182848 (66%) |

## C Other Hyper-parameters and Loss Functions

Hyper-parameters: Dataset and model-specific configuration is defined in Appendices A and B. In fixed filter models, all spatial filter kernels are not modified by back propagation. Unless explicitly stated, we train all CheXpert models for 40 epochs using the default Adam optimizer with learning rate 0.0001, and we train all BBBC038v1 models for 300 epochs using the default Adam optimizer with learning rate 0.008. Pruned models have a 2x larger learning rate of 0.16.

The BBBC038v1 Loss function is a pixel-wise binary cross entropy loss with no class balancing weights.

CheXpert Loss function: We designed a multi-label focal binary cross entropy loss with class balancing weights. The unreduced loss is defined as a 4 × 5 matrix:

$$L = - \text {mw} _ { \text {tas} } ( \text {w} _ { \text {pos} } y \log ( \hat { y } ) + \text {w} _ { \text {neq} } ( 1 - y ) \log ( 1 - \hat { y } ) ) ,$$

where y ∈ [0 , 1] 4 × 5 is ground truth for each of the five classes across the minibatch of 4 images, ˆ y ∈ [0 , 1] 4 × 5 is the model output with each value normalized by a rescaled sigmoid function ˆ y = 0 . 99999 σ (ˆ y ′ ) + 0 . 000005) . The matrix m ∈ { 0 , 1 } 4 × 5 is a bitmask that ignores uncertainty labels, the multiplications are element-wise multiplication with broadcasting, and the class inter- and intra-class balancing weights: w task ∈ R 1 × 5 balances the total count of positive + negative labels across tasks; w pos ∈ R 1 × 5 and w neg ∈ R 1 × 5 balance the counts of positive and negative classes within any given class (specifically ignoring uncertainty labels and remapping missing labels to negative) and incorporate a focal term. For each of the classes, we have scalar weights:

$$w _ { \text {pos} } = a _ { \text {pos} } ( 1 - \hat { y } ) ^ { \gamma } & & a _ { \text {pos} } = \frac { 1 } { 1 + \frac { c _ { \text {pos} } } { c _ { \text {neg} } } } & & ( 8 )$$

$$w _ { \text {neg} } = a _ { \text {neg} } \hat { y } ^ { \gamma } & & a _ { \text {neg} } = \frac { 1 } { 1 + \frac { c _ { \text {neg} } } { c _ { \text {pos} } } } & & ( 9 ) \\ \text {where} \quad 1 \text { is the} \, \text {foo} \, \text {Level} \, \text {by} \, \text {non-zero} \, \text {operator} \, \text {that} \, \text {loos} \, \text { among} \, \text {boo} \, \text { in } \, \text {"bord"} \, \text {degree} \, \text {to} \, \text {the} \, \text {loos} \, \text { among} \, \text {the} \, \text {boo} \, \text { level} \, \text { }$$

pos

where γ = 1 is the focal loss hyperparameter that places emphasis on "hard" samples the model has low confidence for, where a pos and a neg are obtained by using standard class balancing weights of the vectorized form x = max( c ) c and subsequently normalizing them using f ( x ) = x ∑ i x i . The values c = [ c pos , c neg ] are counts of samples with positive and negative labels respectively over the entire training dataset for the given class. For the inter-class balancing weight, we have a similar definition:

$$c = \left [ c _ { t } \ \forall \text { tasks } t \ ; \ c _ { t } = ( c _ { p o s } + c _ { n e g } ) _ { ( t ) } \right ]$$

̸

$$w _ { \text {task} } = w _ { t } = \frac { 1 } { 1 + c _ { t } \sum _ { i \neq t } \frac { 1 } { c _ { i } } }$$

We aggregate the loss L by summing across columns (tasks) and then computing a mean across rows (minibatch samples).

## D Out-of-Distribution vs In-Distribution Analysis on Fixed Filters

Figure D.1: No Overfitting. Left: Microscopy results of 36 independently trained models on the out-ofdistribution test set. Right: Otherwise identical experiment, on the in-distribution test set (same as Fig. E left). We observe higher performance on the out-of-distribution dataset.

<!-- image -->

Each sub-figure represents 36 independently trained models. Blue and orange lines show the per-epoch average of the same fixed spatial filter model (varying only in initialization) trained six times independently, and the blue line is the same in all plots. The red and green background correspond to one-tailed pairedsample Wilcoxon significance tests Wilcoxon, 1945, one for each epoch, of whether the fixed filter models tended to outperform ( p &gt; . 999 ) or underperform ( p &lt; . 001 ) the baseline across all initializations. Each test evaluated six paired models in a bandwidth of ± 10 epochs and across six initializations.

## E Robustness to Learning Rate on Microscopy Dataset

Figure E.1: Robust to increased Learning Rate. Left: Showing microscopy results of 36 independently trained models and learning rate .008. Right: Otherwise identical experiment, with learning rate .02. With larger learning rate, steered methods outperform the baseline early in training, and the non-steered DCT2 method underperforms. The results suggests steered initialization improves robustness of the model. Evaluated on BBBC038v1 in-distribution test set.

<!-- image -->

## F DCT-II Basis

We visualize the 2-d DCT-II basis for 3 × 3 and 5 × 5 kernels in Fig. F .1. The 2-d DCT -II is 1-d separable, and the first row and column show which the 1-d basis vectors construct each 2-d vector. We also assign each basis filter an index value to sort filters from low to high frequency.

Figure F.1: Illustration of proposed basis for 3 × 3 (a) and 5 × 5 (b) matrices. The index numbers order filters from lowest frequency basis filters to highest frequency, in stable ordering.

<!-- image -->

## G ExplainSteer Explanations, Across Architectures and Initializations

Wevisualize the e ( d ) spectra for the spatial convolution layers of three architectures (DenseNet121, ResNet50, EfficientNet-b0), with varying initialization methods. The EfficientNet models have 3 × 3 and 5 × 5 fi lters and we visualize both kernel shapes in the same heatmap. In DenseNet models, we do not visualize the spectrum of first convolution layer because this layer, which uses a 7x7 kernel, would result in a single column with 49 rows while the remaining columns all have 9 rows. Below, Fig. G.1 compares across architectures and Fig. G.2 compares across initialization methods.

Figure G.1: Illustration of Nimbleness across architectures. Saliency heatmaps expose structural inefficiency in the network design, where the repeating horizontal trend from bright to dark in all three architectures suggests that later spatial layers of most stages in the networks are less useful to the network. All plots represent a fully learned baseline model initialized with random weights and trained on the CheXpert dataset.

<!-- image -->

Figure G.2: Illustration of Nimbleness across fixed initializations. Both horizontal and vertical trends reveal redundancy and inefficiency built into DenseNet121. Each sub-figure is a DenseNet121 model pre-trained on ImageNet, spatially re-initialized with one of our fixed spatial filter methods, and fine-tuned on the CheXpert dataset. All saliency heatmaps show that later layers are less influential (darker from left to right), and that higher frequency basis filters are less influential (darker from top to bottom). This visual explanation motivates pruning the network.

<!-- image -->

Sanity checks: The GuidedSteer and Unchanged methods should appear nearly identical in "without Saliency" heatmaps. The DCT2 heatmap without saliency should evenly distribute energy across basis vectors, resulting in a flat green heatmap. GHaar and Psine should prioritize lower frequencies. All sanity checks pass.

## H ResNet50 Model: A Winning Ticket on the 67k Image CheXpert Validation Set

Figure H.1: Illustration of a Winning Ticket Model. On ResNet50, the pre-trained Unchanged initialization outperforms the fully learned pre-trained baseline while random Unchanged initialization underperforms. This finding is also reflected in test set performance in Fig. 10a, suggesting the pre-trained model has a winning ticket initialization. The validation set has 67k images and is generally representative of test set performance.

<!-- image -->

## I Sanity Check on Spatial Filter Saliency

Figure I.1: Proposed Saliency Method Works Well. This sanity check experiment shows the effect of progressively replacing x percent of most salient (left) and least salient (right) spatial filter kernels with zeros. Bottom row: x-axis on log scale. In plots, each line represents six independently trained Learned Baseline models with a 95% confidence interval. The models were first trained on CheXpert, fixed and then evaluated while spatial kernels were progressively zeroed using our saliency metric in Eq. 4. The plots on right are a sanity check that the most salient filters are actually necessary for inference. Removing most important filters indeed makes the models random. All plots suggest that few spatial filters are relevant to the prediction task. All plots confirm that our saliency metric separates important from unimportant spatial filters.

<!-- image -->

## J Visualizing Unchanged Filters

Figure J.1: Visualization of Unchanged filters for a DenseNet121 model at the first 3 × 3 convolution layer.

<!-- image -->

## K Most Spatial Weights are Unnecessary for Inference and Training

<!-- image -->

Figure K.1: Spatial Filters are Unnecessary for Training and Inference . Nearly all spatial weights can be removed from the model by zeroing them out. Spatial filters are unnecessary for both training K.1a and inference K.1b. Results hold for three architectures: DenseNet121 (top), EfficientNet-b0 (middle), ResNet50 (bottom). In all six subplots, each line represents an average with 95% confidence interval of 6 independently trained models.

For Fig. K.1a, we start with completely random deep networks with weights that have never been trained. We fix the spatial weights and zero out 90%, 85% and 80% of least salient spatial filter kernels for ResNet50, DenseNet121 and EfficientNet-b0 architectures, respectively. We subsequently train the models on CheXpert. The baselines are randomly initialized (never learned) deep network trained on CheXpert. The results clearly show that only very few filters are necessary to the model.

For Fig. K.1b, we start with one of our ImageNet Unchanged ExplainFix models that was pretrained on CheXpert. We use the saliency method to zero out 99%, 95% and 85% of least salient spatial filter kernels for ResNet50, DenseNet121 and EfficientNet-b0 architectures, respectively. We then fine-tune the non-spatial weights on CheXpert. The baselines are corresponding fully learned models that were pre-trained and fine-tuned on CheXpert.

## L ChannelPrune: Sanity Check on Computational Savings

As a sanity check that pruned and fixed models have predictive performance nearly equal to the unpruned baseline, we display the computational savings alongside a corresponding predictive performance plot Fig.

L.1. Fig. L.1a is identical to Fig. 11. Each point in Fig. L.1b is a separately trained model trained for 40 epochs and then evaluated on the CheXpert test set. The horizontal dashed lines are the average of 5 independently trained fully learned and unpruned baselines. All ChannelPrune and ExplainFix models start from a spatially fixed Unchanged ImageNet model that has never been trained on CheXpert. The smallest ResNet50 model, at 86% pruned parameters, is another sanity check that zeroing out too many spatial filters (99.9% of them instead of 99%) can cause modest performance loss. In all other cases, the predictive performance of pruned fixed models is nearly equal to the unpruned baseline model, thus verifying the benefits of our proposed methods.

<!-- image -->

(b) Nearly Equal Predictive Performance

Figure L.1: Nimble and Accurate. ExplainFix models are smaller, faster and as accurate.

## M ChannelPrune: Percent Zeroed vs Percent Pruned

Given a starting spatial convolution layer, ChannelPrune removes input and output channels that are entirely zeroed. A simple way to conceptualize the task is to consider the spatial filters as a tensor ( O,I,. . . ) , where O and I are the number of output and input channels. If we reduce this to a boolean O,I matrix indicating True if the spatial kernel is entirely zero, then ChannelPrune removes an input channel if if the entire column is True, and an output channel if the entire row is True. Naturally, a zeroing process based on saliency weights will result in some rows that will no be pruned even though they are mostly zero. We report the discrepancy between the percent of spatial filters zeroed and percent of all weights in the network that were pruned in the table below. The analyzed models correspond to those in Appendix L.

Table M.1: Illustration of ChannelPrune . The relationship between ChannelPrune and Saliency-based Zeroing.

| Model | %Spatially Zeroed | %Pruned |
| - | - | - |
| DenseNet121 | 80.0 | 0.58 |
| DenseNet121 | 90.0 | 0.74 |
| ResNet50 | 90.0 | 0.52 |
| ResNet50 | 99.0 | 0.79 |
| ResNet50 | 99.9 | 0.86 |

## References

- Adadi, A., &amp; Berrada, M. (2018). Peeking inside the black-box: a survey on explainable artificial intelligence (XAI). IEEE access , 6 , 52138-52160.
- Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., &amp; Kim, B. (2018). Sanity checks for saliency maps. Advances in Neural Information Processing Systems , 9505-9515.
- Advani, M. S., Saxe, A. M., &amp; Sompolinsky, H. (2020). High-dimensional dynamics of generalization error in neural networks. Neural Networks , 132 , 428-446.
- Angelov, P ., &amp; Soares, E. (2020). Towards explainable deep neural networks (xDNN). Neural Networks , 130 , 185-194.
- Bruna, J. (2013). Scattering representations for recognition (Doctoral dissertation).
- Bruna, J., &amp; Mallat, S. (2013). Invariant scattering convolution networks. IEEE transactions on pattern analysis and machine intelligence , 35 (8), 1872-1886.
- Buhrmester, V., Münch, D., &amp; Arens, M. (2021). Analysis of Explainers of Black Box Deep Neural Networks for Computer Vision: A Survey. Machine Learning and Knowledge Extraction , 3 (4), 966-989. https: //doi.org/10.3390/make3040048
- Caelli, T. M., Bischof, W. F ., &amp; Liu, Z.-Q. (1988). Filter-based models for pattern classification. Pattern recognition , 21 (6), 639-650.
- Caicedo, J. C., Goodman, A., Karhohs, K. W., Cimini, B. A., Ackerman, J., Haghighi, M., Heng, C., Becker, T., Doan, M., McQuin, C., et al. (2019). Nucleus segmentation across imaging experiments: the 2018 Data Science Bowl. Nature methods , 16 (12), 1247-1253.
- Chen, L.-C., Papandreou, G., Schroff, F., &amp; Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. CoRR , abs/1706.05587 . http://arxiv.org/abs/1706.05587
- Cheng, Y., Wang, D., Zhou, P ., &amp; Zhang, T. (2018). Model compression and acceleration for deep neural networks: The principles, progress, and challenges. IEEE Signal Processing Magazine , 35 (1), 126136.
- Cohen, T., &amp; Welling, M. (2016). Group equivariant convolutional networks. International conference on machine learning , 2990-2999.
- Cotter, F ., &amp; Kingsbury, N. (2017). Visualizing and improving scattering networks. 2017 IEEE 27th International Workshop on Machine Learning for Signal Processing (MLSP) , 1-6.
- Cotter, F ., &amp; Kingsbury, N. (2019). A learnable scatternet: Locally invariant convolutional layers. 2019 IEEE International Conference on Image Processing (ICIP) , 350-354.
- Daubechies, I. (1992). Ten lectures on wavelets . SIAM.
- de Freitas Barbosa, V. A., de Santana, M. A., Andrade, M. K. S., de Lima, R. d. C. F., &amp; dos Santos, W. P. (2020). Deep-wavelet neural networks for breast cancer early diagnosis using mammary termographies. Deep learning for data analytics (pp. 99-124). Elsevier.
- Frankle, J., &amp; Carbin, M. (2019). The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks. https://openreview.net/forum?id=rJl-b3RcF7
- Freeman, W. T., Adelson, E. H. et al. (1991). The design and use of steerable filters. IEEE Transactions on Pattern analysis and machine intelligence , 13 (9), 891-906.

- [Gaudio, A. (2022). Open Source Code. https://github.com/adgaudio/ExplainFix](https://github.com/adgaudio/ExplainFix)
- Gaudio, A., Smailagic, A., &amp; Campilho, A. (2020). Enhancement of Retinal Fundus Images via Pixel Color Amplification. International Conference on Image Analysis and Recognition , 299-312.
- Grossmann, A. (1988). Wavelet transforms and edge detection. Stochastic processes in physics and engineering (pp. 149-157). Springer.
- Han, S., Mao, H., &amp; Dally, W. J. (2016). Deep Compression: Compressing Deep Neural Network with Pruning, Trained Quantization and Huffman Coding. In Y. Bengio &amp; Y. LeCun (Eds.), 4th International Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico, May 2-4, 2016, Conference Track Proceedings . http://arxiv.org/abs/1510.00149
- [Hanov, S. (2006). Wavelets and Edge Detection . http://stevehanov.ca/cs698\_wavelet\_project.pdf](http://stevehanov.ca/cs698_wavelet_project.pdf)
- He, K., Zhang, X., Ren, S., &amp; Sun, J. (2016). Deep residual learning for image recognition. Proceedings of the IEEE conference on computer vision and pattern recognition , 770-778.
- Hestness, J., Narang, S., Ardalani, N., Diamos, G. F., Jun, H., Kianinejad, H., Patwary, M. M. A., Yang, Y., &amp; Zhou, Y. (2017). Deep Learning Scaling is Predictable, Empirically. CoRR , abs/1712.00409 . http://arxiv.org/abs/1712.00409
- Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., &amp; Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. CoRR , abs/1704.04861 . http://arxiv.org/abs/1704.04861
- Huang, G., Liu, Z., Van Der Maaten, L., &amp; Weinberger, K. Q. (2017). Densely connected convolutional networks. Proceedings of the IEEE conference on computer vision and pattern recognition , 47004708.
- Irvin, J., Rajpurkar, P ., Ko, M., Yu, Y., Ciurea-Ilcus, S., Chute, C., Marklund, H., Haghgoo, B., Ball, R., Shpanskaya, K., et al. (2019). Chexpert: A large chest radiograph dataset with uncertainty labels and expert comparison. Proceedings of the AAAI Conference on Artificial Intelligence , 33 , 590-597.
- Kang, E., Chang, W., Yoo, J., &amp; Ye, J. C. (2018). Deep convolutional framelet denosing for low-dose CT via wavelet residual network. IEEE transactions on medical imaging , 37 (6), 1358-1369.
- Leino, K., Sen, S., Datta, A., Fredrikson, M., &amp; Li, L. (2018). Influence-directed explanations for deep convolutional networks. 2018 IEEE International Test Conference (ITC) , 1-8.
- Lin, M., Chen, Q., &amp; Yan, S. (2014). Network In Network. In Y. Bengio &amp; Y. LeCun (Eds.), 2nd International Conference on Learning Representations, ICLR 2014, Banff, AB, Canada, April 14-16, 2014, Conference Track Proceedings . http://arxiv.org/abs/1312.4400
- Luan, S., Chen, C., Zhang, B., Han, J., &amp; Liu, J. (2018). Gabor convolutional networks. IEEE Transactions on Image Processing , 27 (9), 4357-4366.
- Malach, E., Yehudai, G., Shalev-Schwartz, S., &amp; Shamir, O. (2020). Proving the lottery ticket hypothesis: Pruning is all you need. International Conference on Machine Learning , 6682-6691.
- Mallat, S. (2012). Group invariant scattering. Communications on Pure and Applied Mathematics , 65 (10), 1331-1398.
- Mallat, S. (2016). Understanding deep convolutional networks. Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences , 374 (2065), 20150203.
- Mallick, P . K., Ryu, S. H., Satapathy, S. K., Mishra, S., Nguyen, G. N., &amp; Tiwari, P . (2019). Brain MRI image classification for cancer detection using deep wavelet autoencoder-based deep neural network. IEEE Access , 7 , 46278-46287.
- Oyallon, E., Zagoruyko, S., Huang, G., Komodakis, N., Lacoste-Julien, S., Blaschko, M., &amp; Belilovsky, E. (2018). Scattering networks for hybrid representation learning. IEEE transactions on pattern analysis and machine intelligence , 41 (9), 2208-2221.
- Patrick, M. K., Adekoya, A. F ., Mighty, A. A., &amp; Edward, B. Y . (2019). Capsule networks-a survey. Journal of King Saud University-Computer and Information Sciences .
- Pensia, A., Rajput, S., Nagle, A., Vishwakarma, H., &amp; Papailiopoulos, D. (2020). Optimal Lottery Tickets via SUBSETSUM: Logarithmic Over-Parameterization is Sufficient. Advances in neural information processing systems .

- Pérez, J. C., Alfarra, M., Jeanneret, G., Bibi, A., Thabet, A., Ghanem, B., &amp; Arbeláez, P . (2020). Gabor Layers Enhance Network Robustness. European Conference on Computer Vision , 450-466.
- Pham, H. H., Le, T. T., Tran, D. Q., Ngo, D. T., &amp; Nguyen, H. Q. (2021). Interpreting chest X-rays via CNNs that exploit hierarchical disease dependencies and uncertainty labels. Neurocomputing , 437 , 186-194.
- Ribeiro, M. T., Singh, S., &amp; Guestrin, C. (2016). " Why should i trust you?" Explaining the predictions of any classifier. Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining , 1135-1144.
- Ronneberger, O., Fischer, P ., &amp; Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation. International Conference on Medical image computing and computer-assisted intervention , 234-241.
- Rosenfeld, A., &amp; Tsotsos, J. K. (2019). Intriguing properties of randomly weighted networks: Generalizing while learning next to nothing. 2019 16th Conference on Computer and Robot Vision (CRV) , 9-16.
- Said, S., Jemai, O., Hassairi, S., Ejbali, R., Zaied, M., &amp; Amar, C. B. (2016). Deep wavelet network for image classification. 2016 IEEE International conference on systems, man, and cybernetics (SMC) , 000922-000927.
- Samek, W., Wiegand, T., &amp; Müller, K.-R. (2017). Explainable Artificial Intelligence: Understanding, Visualizing and Interpreting Deep Learning Models. CoRR , abs/1708.08296 . http://arxiv.org/abs/1708.08296
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., &amp; Chen, L.-C. (2018). Mobilenetv2: Inverted residuals and linear bottlenecks. Proceedings of the IEEE conference on computer vision and pattern recognition , 4510-4520.
- Segebarth, D., Griebel, M., Stein, N., von Collenberg, C. R., Martin, C., Fiedler, D., Comeras, L. B., Sah, A., Schoeffler, V., Lüffe, T., et al. (2020). On the objectivity, reliability, and validity of deep learning enabled bioimage analyses. Elife , 9 , e59780.
- Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., &amp; Batra, D. (2017). Grad-cam: Visual explanations from deep networks via gradient-based localization. Proceedings of the IEEE international conference on computer vision , 618-626.
- Shensa, M. J. et al. (1992). The discrete wavelet transform: wedding the a trous and Mallat algorithms. IEEE Transactions on signal processing , 40 (10), 2464-2482.
- Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., &amp; Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research , 15 (1), 1929-1958.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., &amp; Wojna, Z. (2016). Rethinking the inception architecture for computer vision. Proceedings of the IEEE conference on computer vision and pattern recognition , 2818-2826.
- Tajbakhsh, N., Jeyaseelan, L., Li, Q., Chiang, J. N., Wu, Z., &amp; Ding, X. (2020). Embracing imperfect datasets: A review of deep learning solutions for medical image segmentation. Medical Image Analysis , 101693.
- Tan, M., &amp; Le, Q. (2019). Efficientnet: Rethinking model scaling for convolutional neural networks, 6105-6114. Tancik, M., Srinivasan, P . P ., Mildenhall, B., Fridovich-Keil, S., Raghavan, N., Singhal, U., Ramamoorthi, R., Barron, J. T., &amp; Ng, R. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. In H. Larochelle, M. Ranzato, R. Hadsell, M.-F . Balcan, &amp; H.-T. Lin (Eds.), Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual . https://proceedings. neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html
- Thompson, N. C., Greenewald, K. H., Lee, K., &amp; Manso, G. F. (2020). The Computational Limits of Deep Learning. CoRR , abs/2007.05558 . https://arxiv.org/abs/2007.05558
- Tjoa, E., &amp; Guan, C. (2020). A survey on explainable artificial intelligence (xai): Toward medical xai. IEEE Transactions on Neural Networks and Learning Systems .
- Wilcoxon, F. (1945). Individual Comparisons by Ranking Methods. Biometrics Bulletin , 1 (6), 80-83. http: //www.jstor.org/stable/3001968
- Yu, F ., &amp; Koltun, V. (2016). Multi-Scale Context Aggregation by Dilated Convolutions. CoRR , abs/1511.07122 .
