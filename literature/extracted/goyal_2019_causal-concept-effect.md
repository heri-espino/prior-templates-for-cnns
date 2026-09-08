---
id: "goyal_2019_causal-concept-effect"
source_pdf: "../pdf/goyal_2019_causal-concept-effect.pdf"
source_filename: "goyal_2019_causal-concept-effect.pdf"
format: "academic-paper"
---

## Explaining Classifiers with Causal Concept Effect (CaCE)

Yash Goyal 1 * Amir Feder 2 Uri Shalit 2 Been Kim 3

## Abstract

How can we understand classification decisions made by deep neural networks? Many existing explainability methods rely solely on correlations and fail to account for confounding, which may result in potentially misleading explanations. To overcome this problem, we define the Causal Concept Effect (CaCE) as the causal effect of (the presence or absence of) a human-interpretable concept on a deep neural net's predictions. We show that the CaCE measure can avoid errors stemming from confounding. Estimating CaCE is difficult in situations where we cannot easily simulate the do-operator. To mitigate this problem, we use a generative model, specifically a Variational AutoEncoder (VAE), to measure VAE-CaCE. In an extensive experimental analysis, we show that the VAE-CaCE is able to estimate the true concept causal effect, compared to baselines for a number of datasets including high dimensional images.

## 1. Introduction

The rise of machine learning use in many practical applications has brought up a new challenge: how to interpret and understand the reasons behind a model's prediction. Particularly in high-risk domains such as medicine, security, etc., it has been widely recognized that understanding the model's reasoning for a prediction is one of the crucial components for wide and safe adoption of the technology.

The machine learning community has been responding to this demand. Many approaches have been proposed to tackle the challenge: for example by developing a model with interpretable components built-in (Kim et al., 2014; Ustun &amp;Rudin, 2014) or by building post-training interpretability methods (Ribeiro et al., 2016; Smilkov et al., 2017; Fong &amp; Vedaldi, 2017; Dabkowski &amp; Gal, 2017; Chang et al., 2018; Kim et al., 2018). While these methods may be useful in showing features or concepts that are correlated with a

1 Georgia Tech 2 Technion 3 Google Brain. * Work partially done while the first author was an intern at Google Brain. Correspondence to: Yash Goyal &lt;ygoyal@gatech.edu&gt;.

model's prediction, their explanations might be confounded by correlations present in the data which are not causally relevant to the model, as we describe now.

Say we wish to explain what drives classification decisions for an entire class by a deep neural network, e.g. 'what drives the decision to classify an image as BICYCLE?'. Now consider the following case: within the training dataset, there is 0 . 8 correlation between the presence of cars and the presence of bicycles. However, the dataset is diverse enough and the classifier is powerful enough such that it does not rely on the presence of cars in order to classify bicycles: If we were to take images containing bicycles and edit out the cars, we will find that the classifier's output for the label BICYCLE is virtually unchanged. Even so, as we show below, the strong correlation between cars and bicycles can lead many interpretability methods to wrongfully give the concept CAR as an explanation for classifying bicycles. In this work, we attempt to tease out the causal aspect: does the presence of a concept like CAR actually change the classifier's output. The case of 'editing out the cars' is an example of what is known as the do -operator (Pearl, 2009): it formalizes the act of intervening in the world, an act which lies in the heart of defining and understanding causal effects.

In this paper we propose explaining classifiers with the Causal Concept Effect (CaCE) for high-level concepts whose presence or absence (everything else being equal) affect the model's prediction, as opposed to merely being correlated with the model's prediction. CaCE is a global explanation method, where the goal is to explain a model's prediction for an entire class, rather than individual data points (i.e., local explanation methods). As global methods aim to summarize all data points, they are much more vulnerable to confounding of concepts. By concept, we mean a higher level unit than low level, individual input features such as pixels. The example of the cars and bicycles above illustrates the issue. Concepts are often highly correlated with each other in datasets, and we want our explanations to zero-in on the concepts whose presence or absence in isolation causally affects the model's output.

One of the challenges in estimating CaCE for high dimensional data such as images is that there is often no easy way to directly perform the intervention of adding or removing a concept. In the example above regarding CARS and BI- CYCLES, we might consider ways to cut and/or paste the object in the image. For a more challenging scenario, consider the case where we want to know the causal effect of the concept MALE on a classifier for DOCTOR. Here, the ideal intervention would be to replace all male doctors in an image with female doctors who are dressed and positioned in the same way as the male doctor, and examine the change in the classifiers output. This is a difficult task, but we claim that it is worthwhile to try and approximate it using generative models. We show that this approach can successfully capture at least part of the actual causal effect of the concept on the classifier's decisions.

Specifically, we propose using conditional VAEs (Sohn et al., 2015; Lorberbom et al., 2019) trained on the training data of the classifier of interest to approximate the true generation process conditioned on concepts. Interventions in the generation process then translate to using different values of concepts as inputs to the generative network of the conditional VAE. We show our approach can approximate CaCE well for a number of datasets.

Our main contributions are the following:

- We propose a general framework to quantitatively measure the causal effect of concept explanations on a deep model's prediction.
- We propose an approach which uses conditional generative models to generate counterfactuals and approximate the causal effect of concept explanations.
- We demonstrate the effectiveness of our approach in estimating the true causal effect of concept explanations for a number of datasets including high dimensional images.

## 2. Related Work

The relationship between causality and explainability has a long history, see (Woodward, 2005) for a discussion from a philosophy of science point of view. Halpern &amp; Pearl (2005) give a formal causal theory of what constitutes an explanation, in terms of what is known as 'actual causality'. In this paper we use a more focussed notion of explanation, i.e., we use the causal effect of a concept on the output of a given trained model as a form of explanation in and of itself.

Recent interpretability methods typically fall into two categories: global or local (Kim &amp; Doshi-Velez, 2017). Global methods explain how a model classifies the predictions for an entire class. Local methods explain how a model classifies a single test instance and answer questions such as 'which part of the test input is most responsible for the classification output?'. While local explanations are important for investigating predictions for individual data points, global explanations are more informative in evaluating the overall robustness of the model and in making deployment decisions.

Our work aims to improve the limitation of global explanation methods: the problem of confounded concepts, and proposes a solution towards correcting it. Specifically, we focus on a recent work TCAV (Kim et al., 2018) that generates high-level concept-based global explanations. It may work well in a variety of applications, however, since various concepts (such as cars and roads) are often highly correlated with each other in the data, TCA V suffers from this confounding of concepts, often providing potentially misleading explanations. Ideally, we want our explanations to identify the concepts which causally affect the model's output. In this work, we take a step in this direction by formally defining the causal effect of a concept explanation on a model's output and proposing an approach to estimate it.

We note that this problem does not exist as such for most local interpretation methods: because for a given image, the pixels deterministically cause the output of a model, there is no notion of probability or confounding. However, confounding might affect local models where pixels are perturbed based on data-dependent models (e.g. (Dabkowski &amp; Gal, 2017; Fong &amp; Vedaldi, 2017; Chang et al., 2018)). We leave these cases for future work.

Many interpretability methods developed to have causalflavor are for local explanations, such as removing and adding pixels to generate counterfactual explanations for images (Goyal et al., 2019; Chang et al., 2018) or for texts (Hendricks et al., 2018). In particular, (Chang et al., 2018) used the language of counterfactuals to generate local explanations. In addition to local and global differences, our work and these prior works face different sets of challenges: performing the do -operation with pixels merely involves changing specific pixels. However, the space of possible operations (combinations of pixels) is huge, as there are millions of pixels, each attaining one of hundreds of values. On other hand, realizing do -operation on concepts is not trivial as it requires some form of data generation process; it is no longer just about changing specific pixels. However, the space of possible operations is much smaller than that on pixels. Our goal is to generate global concept-based explanations that can succinctly explain if the presence or absence of concepts cause the model's prediction or not.

## 3. CaCE: Causal Concept Effect

Denote by I an image, and let f : I → Y be a trained classification model whose output we wish to explain 1 . Let C 0 , . . . C k be concepts which are potential causes of an image: these may be objects such as 'cars' or 'bicycles', but also more abstract concepts such as 'night time' or 'brightness above some threshold'.

1 For a binary classifier, we typically have Y = [0 , 1] .

Figure 1. Causal graph relating high-level concepts (such as objects, background, lighting conditions, etc.), images and classifier f output. The dashed edge indicates possible confounding of the two concepts, by other concepts (not shown in the graph). The thick arrow from Image to the output of f indicates that this relation is mechanistic and we have direct access to it through our knowledge of f . This is different from the edges connecting the concepts to the Image, which correspond to the natural generation process of images. In Section 4, we propose using a conditionalVAE conditioned on concepts to approximate this relation.

<!-- image -->

Let's consider the process that gives rise to the pixels of a typical natural image. We build upon the idea noted in a previous work (Schölkopf et al., 2012) that the causal generation process for a digit image proceeds as the following: a person intends to write the digit 7 , say, and this intention causes a motor pattern producing an image of the digit 7 - in that sense the class label 7 causes the digit image. Generalizing this idea to a natural image, there are many objects in the world, different backgrounds, lighting and angle decisions, as well as properties of the camera and signal processing, all these concepts lead to an ordered set of pixels which is an image. These concepts are all considered as causes for the image, as shown in a simplified example in Figure 1. We say they cause the image since there is a mechanism at work that, given all these concepts, creates a distribution of images with the relevant concepts. Importantly, concepts are far from independent, as some objects typically occur together within a given set of images, as in the car and bicycle example above.

Note that for a fixed classifier f , the only dependence of its output on the concepts is through the image itself. Moreover, the mechanism leading from image to classifier output is in principle known to us, because we have access to the model f . On the other hand, our understanding of the generative process g leading from concepts to the image can vary. For example, if we take g to be the natural process giving rise to images of streets with cars and bicycles, it might be very complex. If we take g to be a controlled process whereby we paste a fixed image of a car onto an existing image, then g is relatively simpler. To make this distinction between f and g , in Figure 1, we represent the potentially complex image generative process g by thin arrows from the Concept nodes to the Image node, and the fixed classifier mechanism by a thick arrow from Image to Classifier Output.

Let C 0 , . . . , C k be a set of concepts representing the causes of images and g be the generative process giving rise to images. For simplicity, we assume the concepts to be binary corresponding to the presence or absence of the concept. We define the following structural causal model (SCM, Pearl (2009)) for the image I :

$$( C _ { 0 } , C _ { 1 } , \dots , C _ { k } ) & = h ( \epsilon _ { C } ) \\ I & = g ( C _ { 0 } , C _ { 1 } , \dots , C _ { k } , \epsilon _ { I } ) ,$$

where, as is standard in SCMs, ϵ C and ϵ I are independent 'noise' variables. The function h is the generation process of the concept variables from the random variable ϵ C and is not the focus of this work. The interventional SCM setting C 0 to a ∈ { 0 , 1 } is then

$$( C _ { 0 } , C _ { 1 } , \dots , C _ { k } ) & = h ( \epsilon _ { C } ) \\ C _ { 0 } & = a \\ I & = g ( C _ { 0 } , C _ { 1 } , \dots , C _ { k } , \epsilon _ { I } ) .$$

The SCMs in Eqs (1) and (2) make an important assumption that it is possible to intervene atomically on one concept while leaving all others the same. This might fail for example if some of the concepts are mutually exclusive. We denote expectations under the interventional distribution by the standard do-operator notation E g [ ·| do ( C 0 = a ) ] , where the subscript g indicates that this expectation also depends on the choice of the generative process g .

## Definition 1 (Causal Concept Effect, CaCE) .

The causal effect of a binary concept C 0 on the output of the classifier f under the generative process g is:

$$C a C E ( C _ { 0 } , f ) & = \\ \mathbb { E } _ { g } \left [ f ( I ) | d o ( C _ { 0 } = 1 ) \right ] - \mathbb { E } _ { g } \left [ f ( I ) | d o ( C _ { 0 } = 0 ) \right ] .$$

In causal inference literature this is simply known as the average treatment effect (ATE) or average causal effect (ACE) of the concept C 0 on the output f ( I ) . Weuse the term CaCE to focus the discussion on explaining an image classification model's outputs in terms of concepts.

## 3.1. CaCE for N -way categorical concepts

We can further generalize our definition for CaCE for N - way categorical concept variable. To account for more than 2 possible values of the concept of interest C 0 , we define CaCE for a pair of concept values C 0 = a and C 0 = b as the following:

$$C a C E ( C _ { 0 } , f , a , b ) & = \\ \mathbb { E } _ { g } \left [ f ( I ) | d o ( C _ { 0 } = a ) \right ] - \mathbb { E } _ { g } \left [ f ( I ) | d o ( C _ { 0 } = b ) \right ]$$

If C 0 is binary, a = 1 and b = 0 are the most obvious choices resulting in Definition 1 . If C 0 is N -way categorical, we treat b as the base value of the concept C 0 and marginalize a over all other possible values of C 0 . Hence 2 ,

$$C a C E ( C _ { 0 } , f , b ) = & \quad \text {score for} \quad \\ \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] - \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = b ) \right ] & \quad \text {between} \quad \\ \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] - \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = b ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] - \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = b ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] - \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = b ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \} \{ b \} } \mathbb { E } \left [ f ( I ) | d o ( C _ { 0 } = n ) \right ] & \quad \text {between} \quad \\ \intertext { \quad } \intertext { \frac { 1 } { N } \sum _ { n \in \{ N \}$$

where { N } = { 1 , 2 , ..., N } refers to the set of all possible values of C 0 . Now, for a given image I i , we can calculate CaCE using the above definition by fixing b to be the concept label of I i .

## 3.2. Why do we need CaCE?

The first question that comes to mind is - 'Why do we need CaCE?'. Let us address this question using a simple example where we can directly control the image generation process. Lets consider an image classification dataset where an image contains exactly one bar, as shown in Figure 2. The orientation of the bar can either be horizontal or vertical, which defines the class label of the image - 0 for horizontal and 1 for vertical. The bar can also take two different colors - red or green. In this case, we consider the color as a binary concept ( 0 for red and 1 for green) for which we measure our metric 'CaCE'. We train a network to classify the images according to the class labels, in two different scenarios. In both scenarios, we can calculate the CaCE exactly, by intervening directly on the images and changing the color of the bars while keeping all other things (location of the bar, its width, etc.) fixed. For each image in the test set, we compute the difference in the classifier's outputs for the original test image and for the counterfactual image with the flipped color. CaCE, by definition, is the average of these differences (each with appropriate + or - sign).

We first consider an unconfounded scenario when each concept (red or green) is equally balanced with the labels (horizontal and vertical). In other words, each class contains equal number of red and green colored bars. In this case, CaCE for the color concept turns out to be zero, as expected. The network learns to ignore the irrelevant concept (color) while making its predictions. TCAV (Kim et al., 2018), a correlation-based interpretability method, also scores the importance of color concept as zero, i.e., the color concept does not explain the model's decisions, as we would expect.

In a more interesting scenario, we consider a biased dataset. In this case, 90% of the horizontal bars are red in color (and 10% are green) and only 10% of the vertical bars are red in color (and 90% are green). Unlike the previous scenario, color and class are strongly confounded in this dataset. However, we find that with enough training data, the network still learns to ignore the color; it makes the same prediction for a pair of images which differ from each other only in terms of the color of the bar. Hence, the CaCE of the color concept is still zero. However, we find that TCA V score for the color concept in this binary classification is 1.0 probably because TCAV is fooled by the strong correlation between the class label and the concept, even though the network itself learned to ignore the color. Hence, there is a need for a mechanism to measure the causal effect of such explanations.

2 subscript g has been omitted for simplicity.

## 4. Measuring CaCE

Estimating CaCE in a real-world setting is an exceedingly challenging task: we do not have a good understanding of the complete causal graph underlying the probability distribution of natural images. This lack of understanding implies, for example, that we cannot confidently use standard causal inference methods such as backdoor adjustment. The reason is that in most cases we cannot hope to account for all the confounding factors between a concept and an image. Note also that the image pixels are causally downstream from the concepts, so we cannot simply condition on them.

However, we believe that this fact should not lead us to completely give up on trying to limit confounding in explanations. We focus on approaches that allow us to perform or approximate the do -operator directly on images. Thus, our major challenge is instantiating the intervention do ( C 0 = a ) for an image generative process g . For example, consider the quantity E g [ f ( I ) | do ( BICYCLE = 1 ) ] : how can we intervene on an image so that it has a bicycle?

We present the following two approaches for measuring CaCE: (1) Using controlled environments where we can directly intervene on parts of the image generation process and calculate the exact value of CaCE for this intervention, and (2) Using image generative models ((Lorberbom et al., 2019), (Kocaoglu et al., 2017)) - while we do not have a full causal graph underlying image generation, we can try and leverage the great strides made in recent years in generative modeling. This approach is not guaranteed to yield the true CaCE value, since the learned generative process is only an approximation of the true generative process tying concepts to images. We therefore conduct a wide set of experiments assessing if and when can the VAE-based approach yield explanations which are less confounded than existing methods. We do this by comparing the VAE-CaCE results to ground-truth CaCE values obtained via interventions in the generation process in controlled settings; by examining and comparing the VAE-CaCE results to other explanation methods; and by proposing and evaluating a set of diagnostic tests whose purpose is to boost our confidence in our results.

## 4.1. Ground Truth CaCE (GT-CaCE)

CaCE can be computed exactly if we can precisely intervene in the generation process of the data. We call this measure the Ground truth CaCE ( GT-CaCE in short). Under this ideal scenario we can generate the 'true' counterfactual image for any given image by only changing the concept of interest, while keeping all the other things in the image fixed. Then, GT-CaCE can be computed as the mean difference between the predictions of f on these pairs of an image and its counterfactual.

Since this is a limited scenario and most generation processes we wish to study do not allow such direct interventions, we propose below an approach that learns a generative model approximating the natural image generation process, allowing us to compute an approximation of the CaCE.

## 4.2. VAE-CaCE

We now describe how we use a conditional-VAE to approximate the image generation process and estimate CaCE. The challenge with the VAE approach is that the VAE model is also vulnerable to confounding between the concepts, for example, generating cars when conditioning on bicycles. Generally we cannot guarantee that the learned VAE correctly approximates the true causal generative process of images in the world; however we do believe it captures important parts of it, allowing us to alleviate confounding, while probably not removing it completely. In order to gain confidence in this approach we use extensive experimental evaluation to assess how well does the V AE approach work in scenarios where we do have ground truth. We also propose and conduct a series of diagnostic tests for increasing our confidence in the model's performance. We note that as usual in causal questions, complete confidence is hard to come by.

We choose a conditional generative model because it allows us to generate images for a given concept value C 0 = a , leaving all other concepts fixed. In addition to conditioning on the concept label, we condition on the class label to be able to generate class-conditional CaCE and to learn a better generative model in comparison to not conditioning on class label. Hence, our conditional VAE approximates the distribution p ( I | C 0 = a, L = l ) where L = l denotes the label.

In our experiments, we use the discrete conditional VAE (DC-VAE)(Lorberbom et al., 2019), specifically their mixture model variant, to allow for discrete latent space. The DC-VAE encoder is comprised of three parts. It has a shared encoder that takes in an image, the class label and the concept label as inputs. Its output is then fed into a discrete and a continuous encoders, resulting in a continuous and a discrete latent spaces. The decoder takes in samples from both the discrete and continuous latent spaces alongwith the class and the concept labels as inputs, and generates the image reconstruction. For more details, please refer to (Lorberbom et al., 2019).

Figure 2. Our Dec-CaCE approach generates pairs of images from the decoder with the same latent sample vector z but with different values of the concept C , resulting in pairs of counterfactual images which differ from each other in terms of the concept only.

<!-- image -->

Figure 3. Our EncDec-CaCE approach first infers the latent distribution for a given image and its corresponding concept label using the encoder and then generates a counterfactual image from the decoder using a sample from the inferred latent distribution and the flipped value of the concept C .

<!-- image -->

We propose two methods of calculating VAE-CaCE: 1) DecCaCE which only utilizes the generative network (DECoder) of the V AE, and 2) EncDec-CaCE which utilizes both the inference (ENCoder) and the generative networks (DECoder). We describe the two methods in detail below.

## 4.2.1. DEC-CACE

Analogous to the ideal case of intervening during the generation process (Sec. 4.1), the generative network of the VAE allows us to sample pairs of counterfactual images from p ( I | z, C 0 , L ) 3 which only differ from each other in the value of the concept C 0 (one image for C 0 = 0 and another image for C 0 = 1 ). This can be achieved by only changing the value a of the concept C 0 while keeping the class label L and the sampled latent vector z fi xed (see Fig. 2). Dec-CaCE is then computed by averaging the difference in the prediction scores f ( I ) of the two counterfactual images (one with C 0 = 1 and the other with C 0 = 0 ) for many random samples of z .

## 4.2.2. ENCDEC-CACE

Recall that CaCE is a global explanation method which explains a classifier's predictions for an entire class in terms of a human-interpretable concept. But often times, we might be interested in measuring CaCE for a particular image or a specific set of images, e.g., the set of images for which the classifier makes mistakes. This is not possible using the 'generative-net-only' approach (Dec-CaCE), but can be achieved by utilizing both the inference and the generative networks of the VAE.

3 Note that z here refers to the mixed latent space of the V AE for simplicity in notations.

In this approach, for each given image I i of interest with the class label l i and the concept label C 0 = c i , we can approximately infer the posterior distribution p ( z i | I i , L = l i , C 0 = c i ) using the inference network, and then sample a counterfactual image I ′ i from p ( I | z i , L = l i , C 0 = c i ) from the generative network using the flipped value c i of the concept C 0 (see Fig. 3). EncDec-CaCE is then computed as the difference in the prediction scores of the original image I i and the generated counterfactual image I ′ i , i.e., if the concept label c i for the original image I i is equal to 1 , EncDec-CaCE ( I i ) = f ( I i ) - f ( I ′ i ) . This approach then allows us to estimate CaCE for any set of images by averaging over their individual EncDec-CaCE values.

## 4.3. Diagnostic tests on VAE for measuring CaCE

Our method relies on the assumption that hidden confounding does not significantly impact the concepts and labels we use, and that the V AE successfully disentangles the concept of interest from the other concepts. As is usually the case in causal inference, this assumption is not statistically testable. We thus propose two simple diagnostic tests for our approach, with the goal of increasing the confidence in our estimates. Note that passing this check does not mean that the estimated CaCE is correct; failing it, however, suggests that the estimated CaCE is probably substantially wrong. We report results from running these tests in Section 5.5.

Diagnostic test I: positive effect We suggest estimating the CaCE of the label on the classification output of that same label, i.e. we take the concept to be the label itself. Assuming that the classifier f has reasonably good performance classifying the label l = C 0 , the presence or absence of l should have a strong causal effect on the output. Failure of this diagnostic might mean that the conditional-VAE is weak and does not capture the relation of labels and the image or image embedding.

Diagnostic test II: null effect We suggest estimating the CaCE of a concept which we know should have essentially no effect on the output of f (known as 'negative controls' in the causal inference literature). For example, we can add a random, independent dummy concept with probability 0 . 5 to each image in the dataset, and estimate its CaCE.

## 5. Results

We apply our approach on four datasets - a synthetically generated BARS dataset, colored-MNIST (Kim et al., 2019), COCO-Miniplaces (Yang &amp; Kim, 2019) and CelebA (Liu et al., 2015) - to demonstrate the generalizability of CaCE and our methods for estimating CaCE. The first three datasets are designed such that we have access to the full data generation process. With such, we can intervene on the concept label of each image to compute the GT-CaCE and evaluate how well our approach works as compared with GT-CaCE.

The definition of CaCE outputs a vector with the same dimension as the output of the classifier, i.e., the number of classes. For binary classification, it is a 2 -dim vector, the two values being equal in magnitude but with opposite signs. In that case, without any loss of information, we report the CaCE score for class 0 in our results for BARS, COCOMiniplaces and CelebA datasets. In the case of multiple classes (10-way classification for colored-MNIST dataset), we instead compute the mean absolute difference in the probability distributions, averaged over the classes.

We compare our results with two baselines: 1) a non-causal baseline with no intervention, which we call CONEXP, and 2) TCAV (Kim et al., 2018). The CONEXP simply computes conditional expectation of the prediction scores conditioned on the concept label i.e., same quantity as CaCE but without the do-operators:

$$\text { our } \quad \text {Conex} ( C , f ) = \mathbb { E } \left [ f ( I ) | C = 1 \right ] - \mathbb { E } \left [ f ( I ) | C = 0 \right ]$$

TCAV (Kim et al., 2018) computes a global explanation score for a given (concept, label) pair, purely relying on correlations. Note that TCAV requires access to internal representations of the image (i.e., embeddings) from the classifier while CaCE and our method VAE-CACE can be applied to any black-box classifier.

## 5.1. BARS: CaCE for a synthetic dataset

We first use a simple dataset of bar images as described in Section 3. Recall that this dataset is generated such that the orientation of the bar (e.g., vertical v.s. horizontal) indicates the class label, and the color of the bar represents the concept. We create a number of biased datasets by varying how often each concept (color) appears in each class (each row in Table 1) to create a spectrum of values of GT-CaCEs. For each dataset, we learn a simple binary classifier consisting of 3 fully-connected layers.

In a case when the correlation between color and class is low (first row in Table 1), GT-CaCE is zero, meaning that the classifier ignores the color. While our all methods measure close to zero, ConExp and TCAV incorrectly reports higher impact of the color. In contrast, when the correlation between color and class is high (second row in Table 1), GT-CaCE is 0 . 58 , meaning that the classifier does rely on colors to some extent (note that the highest CaCE is 1.0), Dec-CacE estimates this correctly, while ConExp and TCAV again only report pure correlations. Among our methods, we find that Dec-CaCE performs better than EncDec-CaCE.

Table 1. CaCE results for BARS dataset.

| % of red in class 0 (horz) | % of red in class 1 (vert) | GT- CaCE | Dec- CaCE | EncDec- CaCE | ConExp | TCAV |
| - | - | - | - | - | - | - |
| 60 | 40 | 0.00 | 0.02 | 0.00 | 0.21 | 0.76 |
| 99 | 01 | 0.58 | 0.59 | 0.30 | 1.00 | 1.00 |
| 98 | 02 | 0.47 | 0.49 | 0.26 | 0.97 | 0.96 |
| 99 | 50 | 0.39 | 0.44 | 0.04 | 0.69 | 0.96 |

## 5.2. Colored-MNIST: CaCE for N -way categorical concept

The colored-MNIST dataset (Kim et al., 2019) is a variant of the MNIST dataset (Deng, 2012) where the foreground digit is colored. The digit defines the class of the image, and the color defines the concept. Following (Kim et al., 2019), a color bias is introduced in the dataset in the following way: each digit class is assigned a mean color value. For each image, a color value is sampled from a normal distribution with the digit class's color value as mean and a fixed covariance ( σ , same for all classes). The value of covariance, σ , can be used to control the degree of the bias in the dataset (smaller σ means highly biased). We created a number of biased datasets by varying σ from 0.02 to 0.05 with an interval of 0.005. For each dataset, we train a ResNet-100 classifier for digit classification. Unlike the BARS dataset, the color concept is non-binary; it is defined as the closest out of 13 possible colors. Therefore, CaCE measures in this case are measured as described in Section 3.1.

Table 2. CaCE results for Colored-MNIST dataset.

| σ | Avg-GT- CaCE | Dec-CaCE | EncDec- CaCE | ConExp | TCAV |
| - | - | - | - | - | - |
| 0.02 | 0.094 | 0.097 | 0.099 | 0.154 | 0.16 |
| 0.025 | 0.089 | 0.092 | 0.093 | 0.147 | 0.155 |
| 0.03 | 0.076 | 0.079 | 0.08 | 0.135 | 0.152 |
| 0.035 | 0.068 | 0.07 | 0.071 | 0.133 | 0.152 |
| 0.04 | 0.061 | 0.063 | 0.065 | 0.121 | 0.139 |
| 0.045 | 0.062 | 0.065 | 0.066 | 0.131 | 0.13 |
| 0.05 | 0.058 | 0.061 | 0.062 | 0.118 | 0.117 |

As shown in Table 2, we can see that our methods - DecCaCE and EncDec-CaCE are able to estimate CaCE well while baselines method tend to overestimate the importance of the concept. Examples of the counterfactual images generated from our DC-VAE are shown in Fig. 4.

CaCE estimate as complexity of classifiers vary. How does CaCE estimate vary as the complexity of the classifier change? We repeat the experiments from Table 2 (which correspond to ResNet-100) for ResNet-1 and a simple-CNN model consisting of 2 convolutional layers and a fc layer. Wefind that a more complex classifier (ResNet-100) tends to be more affected by the correlation between class and color concept and results in higher CaCE values as compared to relatively simpler classifiers (such as simple-CNN).

Figure 4. Colored-MNIST images from the test set (leftmost column) alongside their counterfactuals generated from DC-VAE . Each of the 2-14 columns correspond to one possible value for the color concept. Each row corresponds to one image from each digit class.

<!-- image -->

## CaCE estimate as complexity of generative models vary.

We ask similar question now with generative models. We again repeat the experiments from Table 2 (which correspond to a DC-VAE encoder architecture consisting of convolutional layers) for a simpler DC-VAE encoder architecture consisting of only fc layers. We find that while both DC-VAEs can estimate CaCE pretty well, the estimates from 'convolutional' DC-VAE are 8 - 49% closer to GT-CaCE as compared to 'fc' DC-VAE.

## 5.3. COCO-Miniplaces: CaCE for high-dimensional images

This dataset, introduced in (Yang &amp; Kim, 2019), combines images from Miniplaces (Zhou et al., 2017) and COCO (Lin et al., 2014) datasets. Each image in this dataset is generated by pasting an object segmentation crop (e.g., a dog) from a COCO image on to a scene image. The Miniplaces scene category defines the class label for the modified image, while the presence or absence of the object crop is a binary concept.

Similar to the BARS dataset, this dataset allows us to control the value of the binary concept during the generation process. We vary how often the object is pasted on to the images for a class to create a range of GT-CaCE values.

We create this dataset to have a simple setup, potentially to control for unrelated randomness and focus on investigating the causal effect of the presence and absence of the objects. We use a single object and keep the size of the crop and its pasted location in the scene image to be fixed.

To encourage the classifier to use the presence or absence of the object as a signal for its prediction, we manually choose the two most confusing classes from the dataset - 'bathroom' and 'shower'. Note that if two classes are not confusing, we observe that classifiers lean to ignore the object, resulting GT-CaCE to be zero. For each setting of the dataset, we finetune a ResNet-50 model, pretrained for 365-way classification on Places dataset (Zhou et al., 2017), for binary classification between these 2 classes. To train VAEs, we augment the Miniplaces dataset with images from the Places dataset (Zhou et al., 2017) to increase the size of the training dataset (note that Miniplaces is a subset of Places dataset).

Table 3. CaCE results for COCO-Miniplaces dataset.

| % of obj in 'bathroom' | %of obj in 'shower' | GT- CaCE | Dec- CaCE | EncDec- CaCE | ConExp | TCAV |
| - | - | - | - | - | - | - |
| 60 | 40 | 0.13 | 0.154 | 0.078 | 0.23 | 0.723 |
| 99 | 01 | 0.694 | 0.651 | 0.345 | 0.841 | 1.000 |
| 95 | 05 | 0.604 | 0.543 | 0.262 | 0.791 | 0.988 |
| 99 | 50 | 0.328 | 0.31 | 0.291 | 0.49 | 0.944 |

Table 3 shows CaCE scores for class 0 ('bathroom'). For the first case where 60% of the 'bathroom' and 40% of the 'shower' images contain the object, the GT-CaCE and estimates from all our methods are small, while ConExp and TCAV incorrectly assign a large importance to the concept, consistent with results in Sec. 5.1.

In cases of high correlation (rows 2-4), we observe that our method Dec-CaCE estimates CaCE values close to the GTCaCE values in all cases. On the other hand, the baseline CONEXP and TCAV tend to overestimate the importance of the concept, as we would expect when the concept is strongly correlated with the label. As evident from these empirical results, we believe our CaCE estimates can help provide a better understanding of the degree to which correlated concepts in high-dimensional images actually impact the classifier's output.

## 5.4. CelebA: CaCE for naturally occurring confounding concepts

Lastly, we showcase our estimates in a real dataset, where confoundings may naturally occur. CelebA (Liu et al., 2015) dataset contains celebrities' face images. Each image is annotated with a number of attributes such as gender, smiling, wearing glasses, etc. We consider the binary task of classifying these face images into male or female and consider the attributes of 'eyeglasses' and 'blonde hair' as binary concepts. In order to create biased settings, we subsample from the training set to achieve a desired level of bias. We then train a ResNet-100 classification model for each of these biased datasets.

Figure 5. CelebA test images (with eyeglasses) along with their counterfactuals (without eyeglasses) generated from the DC-VAE .

<!-- image -->

Since we do not control the image generation process, it is not possible to calculate GT-CaCE. However, for one particular concept, hair color, StarGAN (Choi et al., 2018) has been shown to generate realistic counterfactual images for the blonde hair attribute. We report approximated GTCaCE using StarGAN by generating realistic counterfactual images for the blonde hair and calculating EncDec-CaCE for all test images (third column in Table 4).

Table 4. CaCE results for CelebA dataset for 'blonde hair' concept.

| % of blonde women | % of blonde men | StarGAN | Dec- CaCE | EncDec- CaCE | ConExp | TCAV |
| - | - | - | - | - | - | - |
| 60 | 40 | 0.049 | 0.057 | 0.05 | 0.152 | 0.755 |
| 99 | 1 | 0.523 | 0.585 | 0.513 | 0.752 | 1 |
| 98 | 2 | 0.448 | 0.537 | 0.44 | 0.706 | 1 |
| 95 | 5 | 0.468 | 0.479 | 0.474 | 0.642 | 0.978 |
| 99 | 50 | 0.176 | 0.209 | 0.174 | 0.376 | 0.953 |

As shown in Table 4, we observe consistently similar results for CelebA for the 'blonde hair' attribute as we observed for previous three datasets, i.e., we see that our estimates of CaCE are much closer to those obtained by StarGAN, while other baselines do not.

We also show qualitative examples from our trained VAE for the attribute 'eyeglasses' which are used to measure EncDec-CaCE approach in Fig. 5. For the first example (top row) where the classifier is trained on '95-05' biased dataset (95% of women images are wearing eyeglasses; 5% of men images are wearing eyeglasses), the classifier predicts p ( woman ) = 0 . 88 for the test image, perhaps because it has mostly seen eyeglasses in women images (because of the bias in the dataset). It predicts p ( woman ) = 0 . 4 for the generated counterfactual image without eyeglasses (top right). Hence, EncDec-CaCE for this example is 0 . 48 . The bottom row shows an example from '60-40' biased dataset where the effect of eyeglasses on the classifier's prediction is negligible. Note that EncDec-CaCE numbers reported in the tables are averages of such individual calculations over the entire test set.

## 5.5. Diagnostic tests results

In this section, we report results for the diagnostic tests proposed in Section 4.3 for colored-MNIST dataset. For test 1, we set the concept to be each class label and calculate the average CaCE for all classes. For each image in the test set, we generate counterfactual images that differ only in their concept value (in this case, class label), and compare the classifier's outputs. We observe an average GT-CaCE of 0 . 152 . Note that we expect this result to be close to the upper limit score ( 0 . 2 ).

For test 2, we created a binary dummy concept that appears on each digit image with a probability of 0 . 5 across all classes. Since it has no correlation with digit label, the classifier is expected to ignore it, resulting in a CaCE of zero. We observe the CaCE to be 0 . 003 . Hence, our DCVAE passes the diagnostics tests.

## 6. Conclusions

The goal of interpretability methods is to help humans make decisions about machine learning models, whether the decision is about deployment in high risk domains, or checking if the model is unfair to a subgroup of people. It is critical that the explanations correctly reflect how the model is making predictions, instead of merely reflecting correlations with predictions. We propose a simple metric CaCE, and show it captures more closely what we expect of explanations of models: the causal effect of the absence or presence of a concept on the classifier's output. We then show how we can estimate CaCE, leveraging the recent development of powerful conditional VAEs. We demonstrate that our method can closely match the GT-CaCE for a number of datasets. We hope that CaCE is a starting point towards targeting succinct and causal explanations to unveil the causal processes in classifiers.

## References

Chang, C.-H., Creager, E., Goldenberg, A., and Duvenaud, D. Explaining image classifiers by counterfactual generation. In Proceedings of the 3rd International Conference on Learning Representations (ICLR , 2018.

Choi, Y., Choi, M., Kim, M., Ha, J.-W., Kim, S., and Choo, J. Stargan: Unified generative adversarial networks for multi-domain image-to-image translation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , 2018.

Dabkowski, P. and Gal, Y. Real time image saliency for black box classifiers. In Advances in Neural Information Processing Systems , pp. 6967-6976, 2017.

Deng, L. The mnist database of handwritten digit images for machine learning research [best of the web]. IEEE Signal Processing Magazine , 29(6):141-142, 2012.

Fong, R. C. and Vedaldi, A. Interpretable explanations of black boxes by meaningful perturbation. In Proceedings of the IEEE International Conference on Computer Vision , pp. 3429-3437, 2017.

Goyal, Y., Wu, Z., Ernst, J., Batra, D., Parikh, D., and Lee, S. Counterfactual visual explanations. In International Conference on Machine Learning , 2019.

Halpern, J. Y. and Pearl, J. Causes and explanations: A structural-model approach. part ii: Explanations. pp. 889911, 2005.

Hendricks, L. A., Hu, R., Darrell, T., and Akata, Z. Generating counterfactual explanations with natural language. 2018.

Kim, B. and Doshi-Velez. Towards a rigorous science of interpretable machine learning. In eprint arXiv:1702.08608 , 2017.

Kim, B., Rudin, C., and Shah, J. A. The bayesian case model: A generative approach for case-based reasoning and prototype classification. In Advances in Neural Information Processing Systems , pp. 1952-1960, 2014.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., and Sayres, R. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). In International Conference on Machine Learning , pp. 2673-2682, 2018.

Kim, B., Kim, H., Kim, K., Kim, S., and Kim, J. Learning not to learn: Training deep neural networks with biased data. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition , pp. 9012-9020, 2019.

Kocaoglu, M., Snyder, C., Dimakis, A. G., and Vishwanath, S. Causalgan: Learning causal implicit generative models with adversarial training. In arXiv preprint arXiv:1709.02023 , 2017.

Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., and Zitnick, C. L. Microsoft coco: Common objects in context. In ECCV , 2014.

- Liu, Z., Luo, P., Wang, X., and Tang, X. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV) , December 2015.
- Lorberbom, G., Gane, A., Jaakkola, T., and Hazan, T. Direct optimization through arg max for discrete variational auto-encoder. In Advances in Neural Information Processing Systems , pp. 6200-6211, 2019.
- Pearl, J. Causality . 2009.
- Ribeiro, M. T., Singh, S., and Guestrin, C. Model-agnostic interpretability of machine learning. In arXiv preprint arXiv:1606.05386 , 2016.
- Schölkopf, B., Janzing, D., Peters, J., Sgouritsa, E., Zhang, K., and Mooij, J. On causal and anticausal learning. In Proceedings of the 29th International Conference on Machine Learning , 2012.
- Smilkov, D., Thorat, N., Kim, B., Viégas, F., and Wattenberg, M. Smoothgrad: removing noise by adding noise. In arXiv preprint arXiv:1706.03825 , 2017.
- Sohn, K., Lee, H., and Yan, X. Learning structured output representation using deep conditional generative models. In Advances in Neural Information Processing Systems , 2015.
- Ustun, B. and Rudin, C. Methods and models for interpretable linear classification. In ArXiv , 2014.
- Woodward, J. Making things happen: A theory of causal explanation. 2005.
- Yang, M. and Kim, B. Towards Quantitative Evaluation of Interpretability Methods with Ground Truth. In arXiv preprint arXiv:1907.09701 , 2019.
- Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. In IEEE Transactions on Pattern Analysis and Machine Intelligence , 2017.
