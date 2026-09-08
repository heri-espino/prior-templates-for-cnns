---
id: "koh_2020_concept-bottlenecks"
source_pdf: "../pdf/koh_2020_concept-bottlenecks.pdf"
source_filename: "koh_2020_concept-bottlenecks.pdf"
format: "academic-paper"
---

## Concept Bottleneck Models

Pang Wei Koh * 1 Thao Nguyen * 1 2 Yew Siang Tang * 1 Stephen Mussmann 1 Emma Pierson 1 Been Kim 2 Percy Liang 1

## Abstract

We seek to learn models that we can interact with using high-level concepts: if the model did not think there was a bone spur in the x-ray, would it still predict severe arthritis? State-of-the-art models today do not typically support the manipulation of concepts like 'the existence of bone spurs', as they are trained end-to-end to go directly from raw input (e.g., pixels) to output (e.g., arthritis severity). We revisit the classic idea of first predicting concepts that are provided at training time, and then using these concepts to predict the label. By construction, we can intervene on these concept bottleneck models by editing their predicted concept values and propagating these changes to the final prediction. On x-ray grading and bird identification, concept bottleneck models achieve competitive accuracy with standard end-to-end models, while enabling interpretation in terms of high-level clinical concepts ('bone spurs') or bird attributes ('wing color'). These models also allow for richer human-model interaction: accuracy improves significantly if we can correct model mistakes on concepts at test time.

## 1. Introduction

Suppose that a radiologist is collaborating with a machine learning model to grade the severity of knee osteoarthritis. She might ask why the model made its prediction-did it deem the space between the knee joints too narrow? Or she might seek to intervene on the model-if she told it that the x-ray showed a bone spur, would its prediction change?

State-of-the-art models today do not typically support such queries: they are end-to-end models that go directly from raw input x (e.g., pixels) to target y (e.g., arthritis severity),

* Equal contribution 1 Stanford University 2 Google Research. Correspondence to: Pang Wei Koh &lt; pangwei@cs.stanford.edu &gt; , Been Kim &lt; beenkim@google.com &gt; , Percy Liang &lt; pliang@cs.stanford.edu &gt; .

Proceedings of the 37 th International Conference on Machine Learning , Online, PMLR 119, 2020. Copyright 2020 by the author(s).

Figure 1. We study concept bottleneck models that first predict an intermediate set of human-specified concepts c , then use c to predict the final output y . We illustrate the two applications we consider: knee x-ray grading and bird identification.

<!-- image -->

and we cannot easily interact with them using the same high-level concepts that practitioners reason with, like 'joint space narrowing' or 'bone spurs'.

We approach this problem by revisiting the simple idea of first predicting an intermediate set of human-specified concepts c like 'joint space narrowing' and 'bone spurs', then using c to predict the target y . In this paper, we refer to such models as concept bottleneck models . These models are trained on data points ( x, c, y ) , where the input x is annotated with both concepts c and target y . At test time, they take in an input x , predict concepts ˆ c , and then use those concepts to predict the target ˆ y (Figure 1).

Earlier versions of concept bottleneck models were overtaken in predictive accuracy by end-to-end neural networks (e.g., Kumar et al. (2009) for face recognition and Lampert et al. (2009) for animal identification), leading to a perceived tradeoff between accuracy and interpretability in terms of concepts. Recently, concept bottleneck models have started to re-emerge as targeted tools for solving particular tasks (Fauw et al., 2018; Yi et al., 2018; Bucher et al., 2018; Losch et al., 2019; Chen et al., 2020).

In this paper, we propose a straightforward method for turning any end-to-end neural network into a concept bottleneck model, given concept annotations at training time: we simply resize one of the layers to match the number of concepts provided, and add an intermediate loss that encourages the neurons in that layer to align component-wise to the provided concepts. We show that concept bottleneck models trained in this manner can achieve task accuracies competitive with or even higher than standard models. We emphasize that concept annotations are not needed at test time; the model predicts the concepts, then uses the predicted concepts to make a final prediction.

Importantly-and unlike standard models-these bottleneck models allow us to intervene on concepts by editing the concept predictions ˆ c and propagating those changes to the target prediction ˆ y . Interventions enable richer humanmodel interaction: e.g., if the radiologist realizes that what the model thinks is a bone spur is actually an artifact, she can update the model's prediction by directly changing the corresponding value of ˆ c . Whenwesimulate this injection of human knowledge by partially correcting concept mistakes that the model makes at test time, we find that accuracy improves substantially beyond that of a standard model.

Interventions also make concept bottleneck models interpretable in terms of high-level concepts: by manipulating concepts ˆ c and observing the model's response, we can obtain counterfactual explanations like 'if the model did not think the joint space was too narrow for this patient, then it would not have predicted severe arthritis'. In contrast, prior work on explaining end-to-end models in terms of high-level concepts has been restricted to post-hoc interpretation of already-trained models: e.g., predicting concepts from hidden layers (Kim et al., 2018) or measuring the correlation of individual neurons with concepts (Bau et al., 2017).

The validity of interventions on a model depends on the alignment between its predicted concepts ˆ c and the true concepts c . We can estimate this alignment by measuring the model's concept accuracy on a held-out validation set (Fong &amp; Vedaldi, 2017). 1 A model with perfect concept accuracy across all possible inputs makes predictions ˆ c that align with the true concepts c . Conversely, if a model has low concept accuracy, then the model's predictions ˆ c need not match with the true concepts, and we would not expect interventions to lead to meaningful results.

Contributions. We systematically study variants of concept bottleneck models and contrast them with standard end-to-end models in different settings, with a focus on the previously-unexplored ability of concept bottleneck models to support concept interventions. Our goal is to characterize concept bottleneck models more fully: Is there a tradeoff between task accuracy and concept interpretability? Do interventions at test time help model accuracy, and is concept accuracy a good indicator of the ability to effectively intervene? Do different ways of training bottleneck models lead to significantly different outcomes in intervention?

1 With the usual caveats of measuring accuracy: in practice, the validation set might be skewed such that models that learn spurious correlations can still achieve high concept accuracy.

We evaluate concept bottleneck models on the two applications in Figure 1: the osteoarthritis grading task (Nevitt et al., 2006) and a fine-grained bird species identification task (Wah et al., 2011). On these, we show that bottleneck models are comparable to standard end-to-end models while also attaining high concept accuracies. In contrast, the concepts cannot be predicted with high accuracy from linear combinations of neurons in a standard black-box model, making it difficult to do post-hoc interpretation in terms of concepts like in Kim et al. (2018). We demonstrate that we can substantially improve model accuracy by intervening on these bottleneck models at test time to correct model mistakes on concepts, and we show that different methods of training bottleneck models lead to different trade-offs between task accuracy with and without interventions. Finally, we show that bottleneck models guided to learn the right concepts can also be more robust to covariate shifts.

## 2. Related work

Concept bottleneck models. Models that bottleneck on human-specified concepts-where the model first predicts the concepts, then uses only those predicted concepts to make a final prediction-have been previously used for specific applications (Kumar et al., 2009; Lampert et al., 2009). Early versions did not use end-to-end neural networks, which soon overtook them in predictive accuracy. Consequently, bottleneck models have historically been more popular for few-shot learning settings, where shared concepts might allow generalization to unseen contexts, rather than the standard supervised setting we consider here.

More recently, deep neural networks with concept bottlenecks have re-emerged as targeted tools for solving particular tasks, e.g., Fauw et al. (2018) for retinal disease diagnosis, Yi et al. (2018) for visual question-answering, and Bucher et al. (2018) for content-based image retrieval. Losch et al. (2019) and Chen et al. (2020) also explore learning concept-based models via auxiliary datasets.

Feature engineering. Constructing a concept bottleneck model is similar to traditional feature engineering (Lewis, 1992; Zheng &amp; Casari, 2018; Nixon &amp; Aguado, 2019) in that both require specifying intermediate concepts/features. However, they differ in an important way: in the former, we learn mappings from raw input to high-level concepts, whereas in the latter, we construct low-level features that can be computed from the raw input by handwritten functions.

Concepts as auxiliary losses or features. Non-bottleneck models that use human-specified concepts commonly use them in auxiliary objectives in a multi-task setup, or as auxiliary features; examples include using object parts (Huang et al., 2016; Zhou et al., 2018), parse trees (Zelenko et al., 2003; Bunescu &amp; Mooney, 2005), or natural language explanations (Murty et al., 2020). However, these models do not support intervention on concepts. For instance, consider a multi-task model c ← x → y , with the concepts c used in an auxiliary loss; simply intervening on ˆ c at test time will not affect the model's prediction of y . Interventions do affect models that use c as auxiliary features by first predicting x → c and then predicting ( x, c ) → y (e.g., Sutton &amp; McCallum (2005)), but we cannot intervene in isolation on a single concept because of the side channel from x → y .

Causal models. We emphasize that we study interventions on the value of a predicted concept within the model, not on that concept in reality. In other words, we are interested in how changing the model's predicted concept values ˆ c would affect its final prediction ˆ y , and not in whether intervening on the true concept value c in reality would actually affect the true label y . This is similar to the notion of causality explored by other concept-based interpretability methods, e.g., in Goyal et al. (2019) and O'Shaughnessy et al. (2020).

More broadly, many others have studied learning models of actual causal relationships in the world (Pearl, 2000). While concept bottleneck models can represent causal relationships between x → c → y if the set of concepts c is chosen appropriately, they have the advantage of being flexible and do not require c to cause y . For example, imagine that arthritis grade ( y ) is highly correlated with swelling ( c ). In this case, c does not cause y (hypothetically, if one could directly induce swelling in the patient, it would not affect whether they had osteoarthritis). However, concept bottleneck models can still exploit the fact that c is highly predictive for y , and intervening on the model by replacing the predicted concept value ˆ c with the true value c can still improve accuracy, even if c does not cause y .

Post-hoc concept analysis. Many methods have been developed to interpret models post-hoc, including recent work on using human-specified concepts to generate explanations (Bau et al., 2017; Kim et al., 2018; Zhou et al., 2018; Ghorbani et al., 2019). These techniques rely on models automatically learning those concepts despite not having explicit knowledge of them, and can be particularly useful when paired with models that attempt to learn more interpretable representations (Bengio et al., 2013; Chen et al., 2016; Higgins et al., 2017; Melis &amp; Jaakkola, 2018). However, post-hoc methods can fail when the models do not learn these concepts, and also do not admit straightforward interventions on concepts. In this work, we instead directly guide models to learn these concepts at training time.

## 3. Setup

Consider predicting a target y ∈ R from input x ∈ R d ; for simplicity, we present regression first and discuss classification later. We observe training points { ( x ( i ) , y ( i ) , c ( i ) ) } n i =1 , where c ∈ R k is a vector of k concepts. We consider bottleneck models of the form f ( g ( x )) , where g : R d → R k maps an input x into the concept space ('bone spurs', etc.), and f : R k → R maps concepts into a final prediction ('arthritis severity'). We call these concept bottleneck models because their prediction ˆ y = f ( g ( x )) relies on the input x entirely through the bottleneck ˆ c = g ( x ) , which we train to align component-wise to the concepts c . We define task accuracy as how accurately f ( g ( x )) predicts y , and concept accuracy as how accurately g ( x ) predicts c (averaged over each concept). We will refer to g ( · ) as predicting x → c , and to f ( · ) as predicting c → y .

In our work, we systematically study different ways of learning concept bottleneck models. Let L C j : R × R → R + be a loss function that measures the discrepancy between the predicted and true j -th concept, and let L Y : R × R → R + measure the discrepancy between predicted and true targets. We consider the following ways to learn a concept bottleneck model ( ˆ f, ˆ g ) :

1. The independent bottleneck learns ˆ f and ˆ g independently: ˆ f = arg min f ∑ i L Y ( f ( c ( i ) ); y ( i ) ) , and ˆ g = arg min g ∑ i,j L C j ( g j ( x ( i ) ); c ( i ) j ) . While ˆ f is trained using the true c , at test time it still takes ˆ g ( x ) as input.
2. The sequential bottleneck fi rst learns ˆ g in the same way as above. It then uses the concept predictions ˆ g ( x ) to learn ˆ f = arg min f ∑ i L Y ( f (ˆ g ( x ( i ) )); y ( i ) ) .
3. The joint bottleneck minimizes the weighted sum ˆ f, ˆ g = arg min f,g ∑ i [ L Y ( f ( g ( x ( i ) )); y ( i ) ) + ∑ j λL C j ( g ( x ( i ) ); c ( i ) ) ] for some λ &gt; 0 .

As a control, we also study the standard model , which ignores concepts and directly minimizes ˆ f, ˆ g = arg min f,g ∑ i L Y ( f ( g ( x ( i ) )); y ( i ) ) .

The hyperparameter λ in the joint bottleneck controls the tradeoff between concept vs. task loss. The standard model is equivalent to taking λ → 0 , while the sequential bottleneck can be viewed as taking λ → ∞ . Compared to independent bottlenecks, sequential bottlenecks allow the c → y part of the model to adapt to how well it can predict x → c ; and joint bottlenecks further allow the model's version of the concepts to be refined to improve predictive performance.

We propose a simple scheme to turn an end-to-end neural network into a concept bottleneck model: simply resize one of its layers to have k neurons to match the number of concepts k , then choose one of the training schemes above.

Table 1. Task errors with ± 2SD over random seeds. Overall, independent, sequential, and joint concept bottleneck models are comparable to standard end-to-end models on task error. Removing the bottleneck from the standard model ('no bottleneck') does not significantly affect task error. Multi-task learning on the standard models further improves task error, but does not allow for interventions.

| MODEL | y RMSE (OAI) | y ERROR (CUB) |
| - | - | - |
| INDEPENDENT | 0.435 ± 0.024 | 0.240 ± 0.012 |
| SEQUENTIAL | 0.418 ± 0.004 | 0.243 ± 0.006 |
| JOINT | 0.418 ± 0.004 | 0.199 ± 0.006 |
| STANDARD | 0.441 ± 0.006 | 0.175 ± 0.008 |
| NO BOTTLENECK | 0.443 ± 0.008 | 0.173 ± 0.003 |
| MULTITASK | 0.425 ± 0.010 | 0.162 ± 0.002 |

Classification. In classification, f and g compute realvalued scores (e.g., concept logits ˆ ℓ = ˆ g ( x ) ∈ R k ) that we then turn into a probabilistic prediction (e.g., P (ˆ c j = 1) = σ ( ˆ ℓ j ) for logistic regression). This does not change the independent bottleneck, since f is directly trained on the binary-valued c . For the sequential and joint bottlenecks, we connect f to the logits ˆ ℓ , i.e., we compute P (ˆ c j = 1) = σ (ˆ g j ( x )) and P (ˆ y = 1) = σ ( ˆ f (ˆ g ( x ))) .

## 4. Benchmarking bottleneck model accuracy

We start by showing that concept bottleneck models achieve both competitive task accuracy and high concept accuracy. While this is necessary for bottleneck models to be viable in practice, their strength is that we can interpret and intervene on them; we explore those aspects in Sections 5 and 6.

## 4.1. Applications

We consider an x-ray grading and a bird identification task. Their corresponding datasets are annotated with high-level concepts that practitioners (radiologists/birders) use to reason about their decisions. (Dataset details in Appendix A.)

X-ray grading (OAI). We use knee x-rays from the Osteoarthritis Initiative (OAI) (Nevitt et al., 2006), which compiles radiological and clinical data on patients at risk of knee osteoarthritis (Figure 1-Top; n = 36 , 369 data points). Given an x-ray, the task is to predict the Kellgren-Lawrence grade (KLG), a 4-level ordinal variable assessed by radiologists that measures the severity of osteoarthritis, with higher scores denoting more severe disease. 2 As concepts, we use k = 10 ordinal variables describing joint space narrowing, bone spurs, calcification, etc.; these clinical concepts are also assessed by radiologists and used directly in the assessment of KLG (Kellgren &amp; Lawrence, 1957).

2 Due to technicalities in the data collection protocol, we use a modified version of KLG where the first two grades are combined.

Table 2. Average concept errors. Bottleneck models have lower error than linear probes on standard and SENN models.

| MODEL | c RMSE (OAI) | c ERROR (CUB) |
| - | - | - |
| INDEPENDENT | 0.529 ± 0.004 | 0.034 ± 0.002 |
| SEQUENTIAL | 0.527 ± 0.004 | 0.034 ± 0.002 |
| JOINT | 0.543 ± 0.014 | 0.031 ± 0.000 |
| STANDARD [PROBE] | 0.680 ± 0.038 | 0.093 ± 0.004 |
| SENN [PROBE] | 0.676 ± 0.026 | - |

Bird identification (CUB). We use the Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011), which comprises n = 11 , 788 bird photographs (Figure 1-Bot). The task is to classify the correct bird species out of 200 possible options. As concepts, we use k = 112 binary bird attributes representing wing color, beak shape, etc. Because the provided concepts are noisy (see Appendix A), we denoise them by majority voting, e.g., if more than 50% of crows have black wings in the data, then we set all crows to have black wings. In other words, we use class-level concepts and assume that all birds of the same species in the training data share the same concept annotations. In contrast, the OAI dataset uses instance-level concepts: examples with the same y can have different concept annotations c .

Models. For each task, we construct concept bottleneck models by adopting model architectures and hyperparameters from previous high-performing approaches; see Appendix B for experimental details. For the joint bottleneck model, we search over the task-concept tradeoff hyperparameter λ and report results for the model that has the highest task accuracy while maintaining high concept accuracy on the validation set ( λ = 1 for OAI and λ = 0 . 01 for CUB). We model x-ray grading as a regression problem (minimizing mean squared error) on both the KLG target y and concepts c , following Pierson et al. (2019); we learn g , which goes from x → c , by fine-tuning a pretrained ResNet18 model (He et al., 2016), and we learn f , which goes from c → y , by training a small 3-layer multi-layer perceptron. We model bird identification as multi-class classification for the species y and binary classification for the concepts c . Following Cui et al. (2018), we learn g by fine-tuning an Inception-v3 network (Szegedy et al., 2016), and learn f by training a single linear layer (i.e., logistic regression).

## 4.2. Task and concept accuracies

Table 1 shows that concept bottleneck models achieve comparable task accuracy to standard black-box models on both tasks, despite the bottleneck constraint (all numbers reported are on a held-out test set). On OAI, joint and sequential bottlenecks are actually better in root mean square error

Figure 2. Left : The shaded regions show the optimal frontier between task vs. concept error. On OAI, we find little trade-off; models can do well on both task and concept prediction. On CUB, there is some trade-off, with standard models and joint models that prioritize task prediction (i.e., with sufficiently low λ ) having lower task error. For standard models, we plot the concept error of the mean predictor (OAI) or random predictor (CUB). Mid : Histograms of how accurate individual concepts are, averaged over multiple random seeds. In our tasks, each individual concept can be accurately predicted by bottleneck models. Right : Data efficiency curves. Especially on OAI, bottleneck models can achieve the same task accuracy as standard models with many fewer training points.

<!-- image -->

(RMSE) than the standard model, 3 and on CUB, sequential and independent bottlenecks are worse in 0-1 error than the standard model, though the joint model (which is allowed to modify the concepts to improve task performance) closes most of the gap.

At the same time, the bottleneck models are able to accurately predict each concept well (Figure 2), and they achieve low average error across all concepts (Table 2). As discussed in Section 1, low concept error suggests that the model's concepts are aligned with the true concepts, which in turn suggests that we might intervene effectively on them; we will explore this in Section 6.

Overall, we do not observe a tradeoff between high task accuracy and high concept accuracy: pulling the bottleneck layer towards the concepts c does not substantially affect the model's ability to predict y in our tasks, even when the bottleneck is trained jointly. We illustrate this in Figure 2Left, which plots the task vs. concept errors of each model.

Additional baselines. We ran two further baselines to determine if the bottleneck architecture impacted model performance. First, standard models in the literature generally do not use architectures that have a bottleneck layer with exactly k units, so we trained a variant of the standard model without that bottleneck layer (directly using a ResNet-18 or Inception-v3 model to predict x → y ); this performed similarly to the standard bottleneck model ('Standard, no bottleneck' in Table 1). Second, we tested a typical multitask setup using an auxiliary loss to encourage the activations of the last layer to be predictive of the concepts c , hyperparameter searching across different weightings of this auxiliary loss. These models also performed comparably ('Multitask' in Table 1), but since they do not support concept interventions, we focus on comparing standard vs. concept bottleneck models in the rest of the paper.

3 To contextualize RMSE, our modified KLG ranges from 0-3, and average Pearson correlations between each predicted and true concept are ≥ 0 . 87 for all bottleneck models.

Data efficiency. Another way to benchmark different models is by measuring data efficiency, i.e., how many training points they need for a desired level of accuracy. To study this, we subsampled the training and validation data and retrained each model (details in Appendix B.4). Concept bottleneck models are particularly effective on OAI: the sequential bottleneck model with ≈ 25% of the full dataset performs similarly to the standard model. On CUB, the joint bottleneck and standard models are more accurate throughout, with the joint model slightly more accurate in lower data regimes (Figure 2-Right).

## 5. Benchmarking post-hoc concept analysis

Concept bottleneck models are trained to have a bottleneck layer that aligns component-wise with the human-specified concepts c . For any test input x , we can read out predicted concepts directly from the bottleneck layer, as well as intervene on concepts by manipulating the predicted concepts ˆ c and inspecting how the final prediction ˆ y changes. This enables explanations like 'if the model did not think the joint space was too narrow for this patient, then it would not have predicted severe arthritis'. An alternative approach to interpreting models in terms of concepts is post-hoc analysis: take an existing model trained to directly predict x → y without any concepts, and use a probe to recover the known concepts from the model's activations. For example, Bau et al. (2017) measure the correlation of individual neurons with concepts, while Kim et al. (2018) use a linear probe to predict concepts with linear combinations of neurons.

A necessary condition for post-hoc interpretation is high concept accuracy. In this section, we therefore evaluate how accurately linear probes can predict concepts post-hoc. We emphasize that this is a necessary but not sufficient condition for accurate post-hoc interpretations, as post-hoc analysis does not enable interventions on concepts: even if we find a linear combination of neurons that predicts a concept well, it is unclear how to modify the model's activations to change what it thinks of that concept alone. Without this ability to intervene, interpretations in terms of concepts are suggestive but fraught: even if we can say that 'the model thinks the joint space is narrow', it is hard to test if that actually affects its final prediction. This is an important limitation of post-hoc interpretation, though we will set it aside for this section.

Following Kim et al. (2018), we trained a linear probe to predict each concept from the layers of the standard model (see Appendix B). We found that these linear probes have lower concept accuracy compared to simply reading concepts out from a bottleneck model (Table 2). On OAI, the best-performing linear probe achieved an average concept RMSE of 0 . 68 , vs. 0 . 53 in the bottleneck models; average Pearson correlation dropped to 0 . 72 from 0 . 84 . On CUB, the linear probe achieved an average concept error of 0 . 09 instead of 0 . 03 ; average F1 score dropped to 0 . 77 from 0 . 92 .

We also tested if we could predict concepts post-hoc from models designed to learn an interpretable mapping from x → y . Specifically, we evaluated self-explaining neural networks (SENN) (Melis &amp; Jaakkola, 2018). As with standard models, SENN does not use any pre-specified concepts; it learns an input representation encouraged to be interpretable through diversity and smoothness constraints. However, linear probes on SENN also had lower concept accuracy on OAI ( 0 . 68 concept RMSE; see Appendix B). 4

The comparative difficulty in predicting concepts post-hoc suggests that if we have prior knowledge of what concepts practitioners would use, then it helps to directly train models with these concepts instead of hoping to recover them from a model trained without knowledge of these concepts. See Chen et al. (2020) for a related discussion.

## 6. Test-time intervention

The ability to intervene on concept bottleneck models enables human users to have richer interactions with them. For example, if a radiologist disagrees with a model's prediction, she would not only be able to inspect the predicted concepts, but also simulate how the model would respond to changes in those predicted concepts. This kind of test-time intervention can be particularly useful in high-stakes settings like medicine, or in other settings where it is easier for users to identify the concepts c (e.g., wing color) than the target y (exact species of bird).

We envision that in practice, domain experts interacting with the model could intervene to 'fix' potentially incorrect concept values predicted by the model. To study this setting, we use an oracle that can query the true value of any concept for a test input. Figure 3 shows examples of interventions that lead to the model making a correct prediction.

## 6.1. Intervening on OAI

Recall that concept bottleneck models first predict concept values ˆ c from the input x , and then use those predicted concept values to predict the target ˆ y . On OAI, we define intervening on the j -th concept as replacing ˆ c j with its true value c j , as provided by the oracle, and then updating the prediction ˆ y after this replacement. Similarly, we can intervene on multiple concepts by simultaneously replacing all of their corresponding predicted concept values, and then updating the prediction. (Intervening on CUB, which we describe in the next subsection, is similar but slightly more complicated because of its regression setting.)

Concretely, we iteratively select concepts on which to intervene on, using an input-independent ordering over concepts computed from the held-out validation set. This means that we always intervene on the same concept c i 1 first, followed by intervening on both c i 1 and c i 2 , and so on (see Appendix B for more detail).

We found that test-time intervention in this manner significantly improved task accuracy on OAI: e.g., querying for just 2 concepts reduces task RMSE from &gt; 0 . 4 to ≈ 0 . 3

4 We were unable to run SENN on CUB because the default implementation was too memory-intensive; CUB has many more classes/concepts than the tasks SENN was originally used for.

Figure 3. Successful examples of test-time intervention, where intervening on a single concept corrects the model prediction. Here, we show examples from independent bottleneck models. Right : For CUB, we intervene on concept groups instead of individual binary concepts. The sample birds on the right illustrate how the intervened concept distinguishes between the original and new predictions.

<!-- image -->

Figure 4. Test-time intervention results. Left : Intervention substantially improves task accuracy, except for the control model, which is a joint model that heavily prioritizes label accuracy over concept accuracy. Mid : Replacing c → y with a linear model degrades effectiveness. Right : Intervention improves task accuracy except for the joint model. Connecting c → y to probabilities rescues intervention but degrades normal accuracy.

<!-- image -->

(Figure 4-Left). These results hint that a single radiologist collaborating with bottleneck models might be able to outperform either the radiologist or model alone, since the concept values used for intervention mostly come from a single radiologist instead of a consensus reading (see Appendix A), and neural networks similar to ours are comparable with individual radiologist performance in terms of agreement with the consensus grade (Tiulpin et al., 2018; Pierson et al., 2019). However, definitively showing this would require more careful human studies.

Furthermore, we found a trade-off between intervenability and task accuracy: the independent bottleneck achieved better test error when all k = 10 concepts are replaced than the sequential or joint bottlenecks, but performed slightly worse without any intervention (Figure 4-Left). This behavior is consistent with how these different bottleneck models are trained. Recall that in the independent bottleneck, c → y is trained using the true c , which is what we replace the predicted concepts ˆ c with. In contrast, in the sequential and joint models, c → y is trained using the predicted ˆ c , which in general will have a different distribution from the true c . Without any interventions, we might therefore expect the independent bottleneck to perform worse, as at test time, it receives the distribution over the predicted ˆ c instead of the distribution of the true c that it was trained with. However, when all concepts are replaced, the reverse is true.

To better understand what influences intervention effectiveness, we ran two ablations. First, we found that intervention can fail in joint models when we prioritize fitting y over c too much (i.e., when λ is too small). Specifically, the joint model with λ = 0 . 01 learned a concept representation that was not as well-aligned with the true concepts, and replacing ˆ c with the true c at test time slightly increased test error ('control' model in Figure 4-Left). Second, we changed the c → y model from the 3-layer multi-layer perceptron used throughout the paper to a single linear layer. Surprisingly, test-time intervention was less effective here compared to the non-linear counterparts (Figure 4-Mid), even though task and concept accuracies were similar before intervention (concept RMSEs of the sequential and independent models are not even affected by the change in c → y ). It is unclear to us why a linear c → y model should be worse at handling interventions, and this observation warrants further investigation in future work.

Altogether, these results suggest that task and concept accuracies alone are insufficient for determining how effective test-time intervention will be on a model. Different inductive biases in different models control how effectively they can handle distribution shifts from ˆ c → y (pre-intervention) to c → y (post-intervention). Even without this distribution shift, as in the case of the linear vs. non-linear independent bottlenecks, the expressivity of c → y has a large effect on intervention effectiveness. Moreover, it is possible that the average concept accuracy masks differences in individual concept accuracies that influence these results.

## 6.2. Intervening on CUB

Intervention on CUB is complicated by the fact that it is classification instead of regression. Recall from Section 3 that for sequential and joint bottleneck classifiers, the final predictor f takes in the predicted concept logits ˆ ℓ = ˆ g ( x ) instead of the predicted binary concepts ˆ c . To intervene on a concept ˆ c j , we therefore cannot directly copy over the true c j . Instead, we need to alter the logits ˆ ℓ j such that P (ˆ c j = 1) = σ ( ˆ ℓ j ) is close to the true c j . Concretely, we intervene on ˆ c j by setting ˆ ℓ j to the 5th (if c j = 0 ) or 95th (if c j = 1 ) percentile of ˆ ℓ j over the training distribution.

Another difference is that for CUB, we group related concepts and intervene on them together. This is because many of the concepts encode the same underlying property, e.g., c 1 = 1 if the wing is red, c 2 = 1 if the wing is black, etc. We assume that the human (oracle) returns the true wing color in a single query, instead of only answering yes/no questions about the wing color; see Figure 4-Right.

An important caveat is that we use denoised class-level concepts in the CUB dataset (Section 4.1). To avoid unrealistic scenarios where a bird part is not visible in the image but we still 'intervene' on it, we only replace a concept value with the true concept value if that concept is actually visible in the image (visibility information is included in the dataset). The results here are nonetheless still optimistic, because they assume that human experts do not make mistakes in identifying concepts and that birds of the same species always share the same concept values.

Test-time intervention substantially improved accuracy on CUB bottleneck models (Figure 4-Right), though it took intervention on several concept groups to see a large gain. For simplicity, we queried concept groups in random order, which means that many queries were probably irrelevant for any given test example. 5

Table 3. Task and concept error with background shifts. Bottleneck models have substantially lower task error than the standard model.

| MODEL | y ERROR | c ERROR |
| - | - | - |
| STANDARD | 0.627 ± 0.013 | - |
| JOINT | 0.482 ± 0.018 | 0.069 ± 0.002 |
| SEQUENTIAL | 0.496 ± 0.009 | 0.072 ± 0.002 |
| INDEPENDENT | 0.482 ± 0.008 | 0.072 ± 0.002 |

As with OAI, test-time intervention was more effective on independent bottleneck models than on the sequential and joint models (Figure 4-Right). We hypothesize that this is partially due to the ad-hoc fashion in which we set logits to the 5th or 95th percentiles for the latter models. To study this, we trained a joint bottleneck with the same taskconcept tradeoff λ but with the final predictor f connected to the probabilities P ( ˆ c j = 1) = σ ( ˆ ℓ j ) instead of the logits ˆ ℓ j . We show the performance of this model under test-time intervention as the 'Joint, from sigmoid' curve in Figure 4Right. It had a higher task error of 0 . 224 vs. 0 . 199 with the normal joint model; we suspect that the squashing from the sigmoid makes optimization harder. However, test-time intervention worked better, and it is more straightforward as we can directly edit ˆ c instead of having to arbitrarily choose a percentile for the logits ˆ ℓ . This raises the question of how to effectively intervene in the classification setting while maintaining the computational advantages of avoiding the sigmoid in the c → y connection.

## 7. Robustness to background shifts

Finally, we investigate if concept bottleneck models can be more robust than standard models to spurious correlations that hold in the training distribution but not the test distribution. For example, in the bird recognition task, a model might spuriously associate the image background with the label and therefore make more errors if the relationship between the background and label changes. However, if the relationship between the concepts and the label remains invariant, then one might hope that concept bottleneck models might still perform well under this change.

To test this, we constructed the TravelingBirds dataset, a variant of the CUB dataset where the target y is spuriously correlated with image background in the training set. Specifically, we cropped each bird out of its original background (using segmentation masks from the original dataset) and onto a new background from the Places dataset (Zhou et al., 2017), with each bird class (species) assigned to a unique and randomly-selected category of places. At test time, we shuffle this mapping, so each class is associated with a different category of places. For example, at training time, all robins might be pictured against the sky, but at test time they might all be on grassy plains (Figure 5). 6

5 This differs from OAI, which used a fixed ordering computed on a held-out validation set. For CUB, it is common to retrain the model on the training + validation set after the hyperparameter search, which makes it difficult to subsequently use the held-out validation set. See Appendix B for more details.

Figure 5. In the TravelingBirds dataset, we change the image backgrounds associated with each class from train to test time (illustrated above for a single class).

<!-- image -->

The results on TravelingBirds are shown in Table 3. Concept bottleneck models do better than standard models: they rely less on background features, since each concept is shared among multiple bird classes and thus appears in training data points that span multiple background types, reducing the correlation between the concept and the background. On the other hand, standard models that go straight from the input x to the label y leverage the spurious correlation between background and label, and consequently fail on the shifted test set.

This toy experiment shows that concept bottleneck models can be more robust to spurious correlations when the target y is more correlated with training data artifacts compared to the concepts c . We emphasize that whether bottleneck models are more robust depends on the choice of the set of concepts c and the shifts considered; a priori, we do not expect that an arbitrary set of concepts c will lead to a more robust model.

6 TravelingBirds is constructed in a similar manner to the Waterbirds dataset introduced in Sagawa et al. (2020). In Waterbirds, which was designed to benchmark methods for handling group shifts, the classes are collapsed into just two classes (waterbirds and landbirds), and only land or water backgrounds are used. In TravelingBirds, which is a more adversarial setting, we retain the multi-class labels, use a larger set of backgrounds, and change the test distribution so that the relationship between the backgrounds and labels are completely altered.

## 8. Discussion

Concept bottleneck models can compete on task accuracy while supporting intervention and interpretation, allowing practitioners to reason about these models in terms of highlevel concepts they are familiar with, and enabling more effective human-model collaboration through test-time intervention. We believe that these models can be promising in settings like medicine, where the high stakes incentivize human experts to collaborate with models at test time, and where the tasks are often normatively defined with respect to a set of standard concepts (e.g., 'osteoarthritis is marked by the presence of bone spurs'). A flurry of recent papers have used similar human concepts for post-hoc interpretation of medical and other scientific ML models, e.g., Graziani et al. (2018) for breast cancer histopathology; Clough et al. (2019) for cardiac MRIs; and Sprague et al. (2019) for meteorology (storm prediction). We expect that concept bottleneck models can be applied directly to similar settings.

A drawback of concept bottleneck models is that they require annotated concepts at training time. However, if the set of concepts are good enough, then fewer training examples might be required to achieve a desired accuracy level (as in OAI). This allows model developers to trade off the cost of acquiring more detailed annotations against the cost of acquiring new training examples, which can be helpful when new training examples are expensive to acquire, e.g., in medical settings where adding training points might entail invasive/expensive procedures on patients, but the incremental cost in asking a doctor to add annotations to data points that they already need to look at might be lower.

Below, we discuss several directions for future work.

Learning concepts. In tasks that are not normatively defined, we can learn the right concepts by interactively querying humans. For example, Cheng &amp; Bernstein (2015) asked crowdworkers to generate concepts to differentiate between adaptively-chosen pairs of examples, and used those concepts to train models to recognize the artist of a painting, tell honest from deceptive reviews, and identify popular jokes. Similar methods can also be used to refine existing concepts and make them more discriminative (Duan et al., 2012).

Side channel from x → y . We can also account for having an incomplete set of concepts by adding a direct side channel from x → y to a bottleneck model. This is equivalent to using the concepts as auxiliary features, and as discussed in Section 2, has the drawback that we cannot cleanly intervene on a single concept, since the x → y connection might also be implicitly reasoning about that concept. Devising approaches to mitigate this issue would allow concept models to have high task accuracy even with an incomplete set of concepts; for example, one might consider carefully regularizing the x → y connection or using some sort of adversarial loss to prevent it from using existing concepts.

Theoretically analyzing concept bottlenecks. In OAI, we found that concept bottleneck models outperform standard models on task accuracy. Better understanding when and why this happens can inform how we collect concepts or design the architecture of bottleneck models. As an example of what this could entail, we sketch an analysis of a simple well-specified linear regression setting, where we assume that the input x ∈ R d is normally distributed, and that the concepts c ∈ R k and the target y ∈ R are noisy linear transformations of x and c respectively. Our analysis suggests that in this setting, concept bottleneck models can be particularly effective when the number of concepts k is much smaller than the input dimension d and when the concepts have relatively low noise compared to the target.

Concretely, we compared an independent bottleneck model (two linear regression problems for x → c and c → y ) to a standard model (a single linear regression problem) by deriving the ratio of their excess mean-squared-errors as the number of training points n goes to infinity:

$$\frac { \text {Excess error for indp bottleneck model} } { \text {Excess error for standard model} } \leq \frac { \frac { k } { d } \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } { \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } ,$$

where σ 2 C and σ 2 Y are the variances of the noise in the concepts c and target y , respectively. See Appendix C for a formal statement and proof. The asymptotic relative excess error is small when k d is small and σ 2 Y ≫ σ 2 C , i.e., when the number of concepts is much smaller than the input dimension and the concepts are less noisy than the target.

Intervention effectiveness. Our exploration of the design space of concept bottleneck models showed that the training method (independent, sequential, joint) and choice of architecture influence not just the task and concept accuracies, but also how effective interventions are. This poses several open questions, for example: What factors drive the effectiveness of test-time interventions? Could adaptive strategies that query for the concepts that maximize expected information gain on a particular test example make interventions more effective? Finally, how might we have models learn from interventions to avoid making similar mistakes in the future?

## Reproducibility

The code for replicating our experiments is available on GitHub at https://github.com/yewsiang/ ConceptBottleneck . An executable version of the CUB experiments in this paper is on CodaLab at https: //worksheets.codalab.org/worksheets/ 0x362911581fcd4e048ddfd84f47203fd2 . The TravelingBirds dataset can also be downloaded at that link. While we are unable to release the OAI dataset publicly, an application to access the data can be made at https://nda.nih.gov/oai/ .

## Acknowledgements

We are grateful to Jesse Mu, Justin Cheng, Kensen Shi, Michael Bernstein, Rui Shu, Sendhil Mullainathan, Shyamal Buch, Yair Carmon, Ziad Obermeyer, and our anonymous reviewers for helpful advice. PWK was supported by a Facebook PhD Fellowship. YST was supposed by an IMDA Singapore Digital Scholarship. SM was supported by an NSF Graduate Fellowship. EP was supported by a Hertz Fellowship. Other funding came from the PECASE Award. Toyota Research Institute ('TRI') provided funds to assist the authors with their research but this article solely reflects the opinions and conclusions of its authors and not TRI or any other Toyota entity.

## References

- Bau, D., Zhou, B., Khosla, A., Oliva, A., and Torralba, A. Network dissection: Quantifying interpretability of deep visual representations. In Computer Vision and Pattern Recognition (CVPR) , pp. 6541-6549, 2017.
- Bengio, Y., Courville, A., and Vincent, P. Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence , 35 (8):1798-1828, 2013.
- Bucher, M., Herbin, S., and Jurie, F. Semantic bottleneck for computer vision tasks. In Asian Conference on Computer Vision , pp. 695-712, 2018.
- Bunescu, R. C. and Mooney, R. J. A shortest path dependency kernel for relation extraction. In Empirical Methods in Natural Language Processing (EMNLP) , pp. 724-731, 2005.
- Chen, X., Duan, Y., Houthooft, R., Schulman, J., Sutskever, I., and Abbeel, P. InfoGAN: Interpretable representation learning by information maximizing generative adversarial nets. In Advances in Neural Information Processing Systems (NeurIPS) , 2016.
- Chen, Z., Bei, Y., and Rudin, C. Concept whitening for interpretable image recognition. arXiv preprint arXiv:2002.01650 , 2020.
- Cheng, J. and Bernstein, M. S. Flock: Hybrid CrowdMachine learning classifiers. In Proceedings of the 18th ACM Conference on Computer Supported Cooperative Work &amp; Social Computing , pp. 600-611, 2015.
- Clough, J. R., Oksuz, I., Puyol-Ant´ on, E., Ruijsink, B., King, A. P., and Schnabel, J. A. Global and local interpretability for cardiac MRI classification. In International Conference on Medical Image Computing and ComputerAssisted Intervention , pp. 656-664, 2019.
- Cui, Y., Song, Y., Sun, C., Howard, A., and Belongie, S. Large scale fine-grained categorization and domainspecific transfer learning. In Computer Vision and Pattern Recognition (CVPR) , pp. 4109-4118, 2018.
- Duan, K., Parikh, D., Crandall, D., and Grauman, K. Discovering localized attributes for fine-grained recognition. In Computer Vision and Pattern Recognition (CVPR) , pp. 3474-3481, 2012.
- Fauw, J. D., Ledsam, J. R., Romera-Paredes, B., Nikolov, S., Tomasev, N., Blackwell, S., Askham, H., Glorot, X., O'Donoghue, B., Visentin, D., et al. Clinically applicable deep learning for diagnosis and referral in retinal disease. Nature Medicine , 24(9):1342-1350, 2018.
- Fong, R. C. and Vedaldi, A. Interpretable explanations of black boxes by meaningful perturbation. In International Conference on Computer Vision (ICCV) , pp. 3429-3437, 2017.
- Ghorbani, A., Wexler, J., Zou, J. Y., and Kim, B. Towards automatic concept-based explanations. In Advances in Neural Information Processing Systems (NeurIPS) , pp. 9277-9286, 2019.
- Goyal, Y., Shalit, U., and Kim, B. Explaining classifiers with causal concept effect (CaCE). arXiv preprint arXiv:1907.07165 , 2019.
- Graziani, M., Andrearczyk, V., and M¨ uller, H. Regression concept vectors for bidirectional explanations in histopathology. In Understanding and Interpreting Machine Learning in Medical Image Computing Applications , pp. 124-132, 2018.
- He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In Computer Vision and Pattern Recognition (CVPR) , 2016.
- Higgins, I., Matthey, L., Pal, A., Burgess, C., Glorot, X., Botvinick, M., Mohamed, S., and Lerchner, A. betavae: Learning basic visual concepts with a constrained variational framework. In International Conference on Learning Representations (ICLR) , 2017.
- Huang, S., Xu, Z., Tao, D., and Zhang, Y. Part-stacked CNN for fine-grained visual categorization. In Computer Vision and Pattern Recognition (CVPR) , pp. 1173-1182, 2016.
- Kellgren, J. and Lawrence, J. Radiological assessment of osteo-arthrosis. Annals of the Rheumatic Diseases , 16(4), 1957.
- Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors

- (tcav). In International Conference on Machine Learning (ICML) , pp. 2668-2677, 2018.
- Kumar, N., Berg, A. C., Belhumeur, P. N., and Nayar, S. K. Attribute and simile classifiers for face verification. In International Conference on Computer Vision (ICCV) , pp. 365-372, 2009.
- Lampert, C. H., Nickisch, H., and Harmeling, S. Learning to detect unseen object classes by between-class attribute transfer. In Computer Vision and Pattern Recognition (CVPR) , pp. 951-958, 2009.
- Lewis, D. D. Feature selection and feature extract ion for text categorization. In Speech and Natural Language: Proceedings of a Workshop Held at Harriman, New York, February 23-26, 1992 , 1992.
- Losch, M., Fritz, M., and Schiele, B. Interpretability beyond classification output: Semantic bottleneck networks. arXiv preprint arXiv:1907.10882 , 2019.
- Melis, D. A. and Jaakkola, T. Towards robust interpretability with self-explaining neural networks. In Advances in Neural Information Processing Systems (NeurIPS) , pp. 7775-7784, 2018.
- Murty, S., Koh, P. W., and Liang, P. ExpBERT: Representation engineering with natural language explanations. In Association for Computational Linguistics (ACL) , 2020.
- Nevitt, M., Felson, D. T., and Lester, G. The Osteoarthritis Initiative. Cohort study protocol , 2006.
- Nixon, M. and Aguado, A. Feature extraction and image processing for computer vision . Academic press, 2019.
- O'Shaughnessy, M., Canal, G., Connor, M., Davenport, M., and Rozell, C. Generative causal explanations of blackbox classifiers. arXiv preprint arXiv:2006.13913 , 2020.
- Pearl, J. Causality: Models, Reasoning and Inference , volume 29. Springer, 2000.
- Pierson, E., Cutler, D., Leskovec, J., Mullainathan, S., and Obermeyer, Z. Using machine learning to understand racial and socioeconomic differences in knee pain. NBER Machine Learning and Healthcare Conference , 2019.
- Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks for group shifts: On the importance of regularization for worst-case generalization. In International Conference on Learning Representations (ICLR) , 2020.
- Sprague, C., Wendoloski, E. B., and Guch, I. Interpretable AI for deep learning-based meteorological applications. In 99th American Meteorological Society Annual Meeting , 2019.
- Sutton, C. and McCallum, A. Joint parsing and semantic role labeling. In Computational Natural Language Learning (CoNLL) , 2005.
- Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., and Wojna, Z. Rethinking the Inception architecture for computer vision. In Computer Vision and Pattern Recognition (CVPR) , pp. 2818-2826, 2016.
- Tiulpin, A., Thevenot, J., Rahtu, E., Lehenkari, P., and Saarakkala, S. Automatic knee osteoarthritis diagnosis from plain radiographs: A deep learning-based approach. Scientific Reports , 8(1):1-10, 2018.
- Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The Caltech-UCSD Birds-200-2011 dataset. Technical report, California Institute of Technology, 2011.
- Yi, K., Wu, J., Gan, C., Torralba, A., Kohli, P., and Tenenbaum, J. Neural-symbolic vqa: Disentangling reasoning from vision and language understanding. In Advances in Neural Information Processing Systems (NeurIPS) , pp. 1031-1042, 2018.
- Zelenko, D., Aone, C., and Richardella, A. Kernel methods for relation extraction. Journal of Machine Learning Research , 3(0):1083-1106, 2003.
- Zheng, A. and Casari, A. Feature engineering for machine learning: principles and techniques for data scientists . ' O'Reilly Media, Inc.', 2018.
- Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence , 40(6):1452-1464, 2017.
- Zhou, B., Sun, Y., Bau, D., and Torralba, A. Interpretable basis decomposition for visual explanation. In European Conference on Computer Vision (ECCV) , pp. 119-134, 2018.

## A. Datasets

## A.1. Osteoarthritis Initiative (OAI)

Description and statistics. The source of the knee x-ray dataset is the Osteoarthritis Initiative 7 , which compiles radiological and clinical data on patients who have or are at high risk of developing knee osteoarthritis. We follow the dataset processing procedure used by Pierson et al. (2019) in their previous analysis. They analyzed data from the baseline visit and four follow-up timepoints (12-, 24-, 36-, and 48-month follow-ups). Two types of data from this dataset were used in our analysis: the knee x-rays, which served as the input to the neural network, and the clinical concepts associated with osteoarthritis, which were annotated by radiologists for each knee x-ray.

After filtering for observations which contain basic demographic and clinical data, the dataset contains 4,172 patients and 36,369 observations, where an observation is one knee for one patient at one timepoint. We randomly divided patients into training, validation, and test sets, with no overlap in the patient groups. Specifically, we have 21,340 observations from 2,456 people in the training set; 3,709 observations from 421 people in the validation set; and 11,320 observations from 1,295 people in the test set.

Image processing. To process the knee x-rays, each x-ray was downsampled to 512 x 512 pixels and normalized by dividing pixel values by the maximum pixel value (so all pixel values were in the range 0-1) and then z-scoring. Images were removed if they did not pass OAI x-ray image metadata quality control filters.

Clinical concept assessment and KLG merging. The primary clinical image feature used in analysis is Kellgren-Lawrence grade (KLG), a 5-level categorical variable (0 to 4) which is assessed by radiologists and used as a standard measure of radiographic osteoarthritis severity, with higher scores denoting more severe disease. In addition to KLG, each knee image is also assessed for 18 other clinical concepts (features) of osteoarthritis in various knee compartments, describing joint space narrowing (JSN), osteophytes, chondrocalcinosis, subchondral sclerosis, cysts, and attrition.

The Osteoarthritis Initiative only assessed these additional 18 clinical concepts (besides joint space narrowing, which is available for all participants) for participants with KLG ≥ 2 (a standard threshold for radiographic osteoarthritis) in at least one knee at any time point. Therefore, in their analysis (and in this paper), Pierson et al. (2019) set these clinical concepts to zero for other participants. This corresponds to assuming that that participants who were never assessed to have osteoarthritis, and thus were not assessed for other clinical concepts, did not display those features. This procedure also means it is impossible to use the clinical concepts to distinguish most x-rays with KLG = 0 from those with KLG = 1 in the dataset. To evaluate concept bottleneck models on this dataset, we therefore merged the KLG = 0 and KLG = 1 classes into a single level and translated the other KLG levels downwards by 1, leading to a 4-level categorical variable (0 to 3).

Concept processing. Some of the clinical concepts are very sparse, with almost all x-rays in the dataset showing an absence of the associated radiographic feature. We found that there were insufficient positive training examples to be able to accurately predict these concepts; moreover, including these sparse concepts in the bottleneck models lowered the accuracy of KLG prediction. We therefore filtered out the clinical concepts for which the dominant class (corresponding to an absence of the feature) represents ≥ 95% of the training data.

This procedure kept 10 clinical concepts: 'osteophytes femur medial', 'sclerosis femur medial', 'joint space narrowing medial', 'osteophytes tibia medial', 'sclerosis tibia medial', 'osteophytes femur lateral', 'sclerosis femur lateral', 'joint space narrowing lateral', 'osteophytes tibia lateral', and 'sclerosis tibia lateral'. It filtered 8 concepts: 'cysts femur medial', 'chondrocalcinosis medial', 'cysts tibia medial', 'attrition tibia medial', 'cysts femur lateral', 'chondrocalcinosis lateral', 'cysts tibia lateral', 'attrition tibia lateral'.

After filtering, we z-scored the remaining clinical concepts using the training set to bring them onto the same scale.

Some of the clinical concepts, such as joint space narrowing, are annotated with fractional grades (e.g., 1.2, 1.4, 1.6 etc.) in the dataset. These partial grades represent temporal progression and cannot be deduced by looking at a single timepoint, and they explicitly do not reflect fractional grades (e.g., 1.2 on one patient does not mean it is worse than 1.0 on another patient); we therefore truncate these fractional grades.

Reader disagreements and adjudication procedures. KLG was read by two expert readers (i.e., radiologists) for each x-ray. Discrepancies in these readings, if they met the adjudication criteria described below, were adjudicated by a third reader: if the third reading agreed with either of the existing readings, then that reading was taken to be final, and otherwise, the three readers attended an adjudication session to form a consensus reading. If discrepancies were not adjudicated, the final reading was taken to be the one from the more senior reader. KLG readings were adjudicated when they disagreed on whether KLG was within 0-1 or 2-4, or when they there was a difference in the direction of change of KLG between time points.

[7 https://nda.nih.gov/oai/](https://nda.nih.gov/oai/)

JSN was also read by two readers, with similar adjudication procedures. Discrepancies were adjudicated if the readers did not agree on the direction of change between time points.

All other clinical concepts in our dataset were read by a single reader. For more information on the adjudication procedures, please refer to the OAI documentation on Project 15.

## A.2. Caltech-UCSD Birds-200-2011 (CUB)

Description and statistics. The Caltech-UCSD Birds-200-2011 (CUB) dataset (Wah et al., 2011) comprises 11,788 photographs of birds from 200 species, with each image additionally annotated with 312 binary concepts (before processing) corresponding to bird attributes like wing color, beak shape, etc. Visibility information on each concept is also provided for each image (e.g., is the beak visible in this image?); we use this information to make our test-time intervention experiments more realistic, but not at training time. Since the original dataset only has train and test sets, we randomly split 20% of the data from the official train set to make a validation set.

Concept processing. The individual concept annotations are noisy: each annotation was provided by a single crowdworker (not a birding expert), and the concepts can be quite similar to each other, e.g., some crowdworkers might indicate that birds from some species have a red belly, while others might say that the belly is rufous (reddish-brown) instead.

To deal with this issue, we aggregate instance-level concept annotations into class-level concepts via majority voting: e.g., if more than 50% of crows have black wings in the data, then we set all crows to have black wings. This makes the approximation that all birds of the same species in the training data should share the same concept annotations. While this approximation is mostly true for this dataset, there are some exceptions due to visual occlusion, as well as sexual and age dimorphism.

After majority voting, we further filter out concepts that are too sparse, keeping only concepts (binary attributes) that are present after majority voting in at least 10 classes. After this filtering, we are left with 112 concepts.

## B. Experimental details

## B.1. OAI model architecture and training

The models we use to predict KLG from knee x-rays follow the hyperparameters and model setup used by Pierson et al. (2019), except for the learning rate and learning rate schedule, which we tune separately. Our models use a ResNet-18 (He et al., 2016) pretrained on ImageNet, with the last 12 convolutional layers fine-tuned on the OAI dataset.

For the bottleneck models, the ResNet-18 network extracts high-level features from the image that is used to regress to the concepts c with a single fully-connected layer. Subsequently, there is a 3-layer MLP, with a dimensionality of 50 for the first two layers, that is used to regress to the final KLG y . The standard model is similar, except without any loss term that encourages the bottleneck layer to align with the concepts.

For fine-tuning, we use a batch size of 8, with random horizontal and vertical translations as data augmentation. Network weights are optimized with Adam, with beta parameters of 0.9 and 0.999 and an initial learning rate determined by grid search over [0.00005, 0.0005, 0.005], which decays by a factor of 2 every 10 epochs. The network is trained for 30 epochs with early stopping; model weights are set at the conclusion of training to those after the epoch with lowest RMSE for KLG on the validation set.

## B.2. CUB model architecture and training

The main architecture for fine-grained bird classification is Inception V3, pretrained on ImageNet (except for the fullyconnected layers) and then finetuned end-to-end on the CUB dataset. We follow the preprocessing practices described in Cui et al. (2018). Each image used for training is augmented with random color jittering, random horizontal flip and random cropping with a resolution of 299. During inference, the original image is center-cropped and resized to 299.

For each model, we hyperparameter search on the validation set over a range of learning rates ([0.001, 0.01]), learning rate schedules (keeping learning rate constant or reducing learning rate by 10 times after every [10, 15, 20] epochs until it reaches 0.0001), and regularization strengths ([0.0004, 0.00004]), to find a good hyperparameter configuration. The best model is decided based on task accuracy (or concept accuracy for the x → c part of sequential models) on the validation set. Once we have found the best-performing hyperparameter configuration, we then retrain the model on both the train and validation sets until convergence, following Cui et al. (2018).

All training is done with a batch size of 64, and SGD with momentum of 0.9 as the optimizer. For bottleneck models, we weight each concept's contribution to the overall concept loss equally (which is in turn determined by λ for joint bottleneck models). However, the binary cross-entropy loss used for each individual concept prediction task is weighted by the ratio of class imbalance for that individual concept (which is about 1 : 9 on average) and normalized accordingly. This encourages the model to learn to predict positive concept labels, which are more rare, instead of mostly predicting negative labels.

## B.3. Test-time intervention

OAI. For OAI, we use the held-out validation set to determine an input-independent ordering for concept intervention. Specifically, we use the concept labels in the validation set to intervene separately on each concept, replacing a single value in our original concept predictions with that ground truth concept. We obtain the intervention ordering by sorting the concepts in descending order of the improvement in KLG accuracy gained from intervening separately on each concept.

CUB. For CUB, the concept groups are determined by having a common prefix in the list of concept names. For example, 'has back pattern::solid', 'has back pattern::spotted', 'has back pattern::striped', 'has back pattern::multi-colored' all describe the same group that concerns back-pattern. Since all models are retrained on both train and validation sets, as described above, we do not follow the OAI procedure of determining a fixed ordering. Instead, we randomly select concept groups to intervene on at test time, using the class-level labels for all concepts within that group to replace the predicted logits. To avoid intervening on concepts that are not even visible in the image, we use the concept visibility information that comes with the official CUB dataset: for all concepts that are not visible in a given test image, their corrected values are set to 0 regardless of what the corresponding class-level labels may be.

## B.4. Data efficiency

For OAI, we subsampled the training and validation data uniformly at random. For CUB, to ensure that each of the 200 classes had similar numbers of examples, we subsampled the images from each class uniformly at random. To avoid the computational load of hyperparameter searching for each model and degree of subsampling, we adopted the hyperparameters chosen for the best-performing models on the full dataset but did early stopping on the subsampled validation datasets.

## B.5. Linear probes

Standard (end-to-end) models. For OAI, we separately trained linear probes on the outputs after every ResNet block and the fully-connected layers of the MLP of the standard model. The best-performing linear probe was the one trained on the output of the final ResNet block. For CUB, we ran a linear probe on the fully-connected layer of the standard model, since the c → y part of the bottleneck models are linear.

SENN. To evaluate self-explaining neural networks (SENNs) (Melis &amp; Jaakkola, 2018), we first trained a SENN model to predict KLG on the OAI dataset and then trained linear probes on the concept layer in the SENN model. We used the open-source implementation from the authors of SENN, 8 and therefore used a classification objective for KLG prediction. To match the expressiveness of our bottleneck models, we swapped the small CNNs of the SENN concept encoder and relevance parameterizer with our ResNet-18 models. Similarly, for the decoder network in SENN, we used a more expressive decoder comprising 2 fully-connected layers with batch normalization, followed by 5 transposed convolutional layers with upsampling. The decoder was obtained by adapting a public auto-encoder implementation, 9 changing the dimensionalities of the fully-connected and transposed convolutional layers, and increasing upsampling layers to match our input image size. We set the number of concepts for SENN to 10, corresponding to the number of clinical features in OAI. The learning rate was set to 0.0005 and the batch size was set to 4, which was the maximum possible given the memory constraints. With the above settings, the experiments were ran with two different seeds.

[8 https://github.com/dmelis/SENN](https://github.com/dmelis/SENN)

[9 https://github.com/arnaghosh/Auto-Encoder](https://github.com/arnaghosh/Auto-Encoder)

## C. Excess errors of independent vs. standard models

We present an analysis of the independent bottleneck model, which uses concepts at training time, versus the standard model, which does not. For simplicity, we consider a well-specified linear regression setting with normally-distributed inputs X ∈ R d , concepts C ∈ R k , and target Y ∈ R :

$$X \sim N ( 0 , \sigma _ { X } ^ { 2 } I _ { d } )$$

$$C = X B + \epsilon _ { 1 } ,$$

$$Y = C b + \epsilon _ { 2 } ,$$

where ϵ 1 ∼ N (0 , σ 2 C I k ) and ϵ 2 ∼ N (0 , σ 2 Y ) . In contrast to the main text, we use capital letters for X , C , and Y here to emphasize the fact that the input, concepts, and target are random variables. In words, the input X is a normally distributed with dimension d ; the concepts C of dimension k are a linear transformation of X with additive Gaussian noise; and the output Y is a scalar-valued linear transformation with additive Gaussian noise. For analytical simplicity, we require ‖ b ‖ 2 = 1 and B ⊤ B = I k .

Independent bottleneck model. In this setting, the independent bottleneck model comprises two linear regression models: the first estimates the matrix B that takes X → C , and the second estimates the vector b that takes C → Y . For ease of analysis, we assume that the linear regression models are fit using least squares on separate datasets: the first dataset has n 1 training points in data matrices X ∈ R n 1 × d and C ∈ R n 1 × k , and the second dataset has n 2 points in data matrices C ∈ R n 2 × k and Y ∈ R n 2 . Concretely, we estimate

$$\hat { B } = \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \underline { C } & & ( 4 )$$

$$\hat { b } = \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { Y }$$

and then compose these estimators into the final prediction ˆ Y IB = X ˆ B ˆ b .

Standard model. In contrast, the standard model does not use concepts, and uses only one dataset with n points in X ∈ R n × d and Y ∈ R n . Concretely, we can express Y directly in terms of X as Y = Xv + ϵ , where v = Bb and ϵ ∼ N (0 , σ 2 C + σ 2 Y ) . This gives the least squares estimate

$$\hat { v } = ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } \underline { X } ^ { \top } \underline { Y }$$

and the resulting prediction ˆ Y SM = X ˆ v .

Excess errors. We compare these two models using their asymptotic excess error as the number of training points n 1 = n 2 = n goes to infinity, where a model's excess error is defined as how much higher its mean-squared-error is compared to the optimal estimator E [ Y | X ] .

Proposition 1 (Relative excess error of independent bottleneck models vs. standard models in linear regression) . Let n 1 = n 2 = n tend to infinity. Then the ratio of excess errors of the independent bottleneck model to the standard model in the well-specified linear regression setting above is

$$\lim _ { n \to \infty } \frac { \mathbb { E } [ ( Y - \hat { Y } _ { I B } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ] ) ^ { 2 } ] } { \mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ] ) ^ { 2 } ] } \leq \frac { \frac { k } { d } \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } { \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } .$$

Note that asymptotic relative excess error is small-i.e., the independent bottleneck has lower excess error than the standard model-when k d is small and σ 2 Y ≫ σ 2 C . This corresponds to low dimensional concepts (relative to the input dimension) and concepts with low noise (relative to the noise in the output).

To prove this proposition, we first derive the expected errors of the independent bottleneck model and the standard model.

Lemma 1 (Risk of the independent bottleneck model) .

$$\mathbb { E } [ ( Y - \hat { Y } _ { I B } ) ^ { 2 } ] = \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \sigma _ { Y } ^ { 2 } \frac { \sigma _ { X } ^ { 2 } } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { n _ { 2 } - k - 1 } + \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } + \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } .$$

Proof. A direct calculation gives

$$\mathbb { E } [ ( Y - \hat { Y } _ { I B } ) ^ { 2 } ] & = \mathbb { E } [ ( ( X B b + \epsilon _ { 1 } b + \epsilon _ { 2 } ) - X \hat { B } \hat { b } ) ^ { 2 } ]$$

$$= \mathbb { E } [ ( \epsilon _ { 1 } b + \epsilon _ { 2 } + X ( B b - \hat { B } \hat { b } ) ) ^ { 2 } ]$$

$$= \mathbb { E } [ ( \epsilon _ { 1 } b + \epsilon _ { 2 } ) ^ { 2 } ] + \mathbb { E } [ X ( B b - \hat { B } \hat { b } ) ( B b - \hat { B } \hat { b } ) ^ { \top } X ^ { \top } ]$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \text {tr} \left ( \mathbb { E } [ X ^ { \top } X ] \mathbb { E } [ ( B b - \hat { B } \hat { b } ) ( B b - \hat { B } \hat { b } ) ^ { \top } ] \right )$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \sigma _ { X } ^ { 2 } \text {tr} \left ( \mathbb { E } [ ( B b - \hat { B } \hat { b } ) ( B b - \hat { B } \hat { b } ) ^ { \top } ] \right ) ,$$

$$( \hat { B } \hat { b } - B b ) = \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } ( \underline { X } B + \underline { \epsilon _ { 1 } } ) \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } ( \overline { C } b + \overline { \epsilon _ { 2 } } ) - B b$$

$$= \left ( B + \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \epsilon _ { \underline { 1 } } \right ) \left ( b + \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { C } _ { \epsilon _ { 2 } } \right ) - B b$$

$$& = B \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { \epsilon _ { 2 } } + \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \underline { \epsilon _ { 1 } } b + \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \underline { \epsilon _ { 1 } } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { \epsilon _ { 2 } } .$$

We need to evaluate the expectation of this expression multiplied with itself, E [( Bb - ˆ B ˆ b )( Bb - ˆ B ˆ b ) ⊤ ] . Note that the cross terms will cancel since ϵ 1 and ϵ 2 are independent of other random variables and have mean 0 , E [ ϵ 1 ] = E [ ϵ 2 ] = 0 . This leaves three remaining direct (squared) terms, which we can evaluate separately since tr and E are linear operators.

The first term is

where

$$\text {tr} \left ( \mathbb { E } \left \lceil B \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { \epsilon _ { 2 } \epsilon _ { 2 } } ^ { \top } \overline { C } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } B ^ { \top } \right ] \right )$$

$$= \text {tr} \left ( \mathbb { E } \left \lceil \overline { C } ^ { \top } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } B ^ { \top } B \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \right ] \right ) \mathbb { E } \left [ \overline { \epsilon _ { 2 } \epsilon _ { 2 } } ^ { \top } \right ] \right )$$

$$= \text {tr} \left ( \mathbb { E } \left \lceil \overline { C } ^ { \top } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } I _ { k } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \right \rceil \sigma _ { Y } ^ { 2 } I _ { n _ { 2 } } \right )$$

$$= \sigma _ { Y } ^ { 2 } \text {tr} \left ( \mathbb { E } \left [ \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \right ) \right ] .$$

The expression within the above expectation is distributed as an inverse Wishart distribution, and therefore

$$\sigma _ { Y } ^ { 2 } \text {tr} \left ( \mathbb { E } \left [ \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \right ) \right ]$$

$$= \sigma _ { Y } ^ { 2 } \text {tr} \left ( \frac { \mathbb { E } \left [ C ^ { \top } C \right ] ^ { - 1 } } { n _ { 2 } - k - 1 } \right )$$

$$= \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { n _ { 2 } - k - 1 } ,$$

where the last equality comes from E [ C ⊤ C ] = σ 2 X B ⊤ B + σ 2 C I k = ( σ 2 X + σ 2 C ) I k .

The second term follows a similar calculation:

$$\text {tr} \left ( \mathbb { E } \left [ \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \epsilon _ { 1 } b b ^ { \top } \epsilon _ { 1 } \underline { ^ { \top } } \underline { X } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \right ] \right )$$

$$= \text {tr} \left ( \mathbb { E } \left [ \underline { X } ^ { \top } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \right ] \mathbb { E } \left [ \underline { \epsilon } _ { \underline { b } } b b ^ { \top } \epsilon _ { \underline { 1 } } ^ { \top } \right ] \right )$$

$$= \sigma _ { C } ^ { 2 } \text {tr} \left ( \mathbb { E } \left [ \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \right ] \right )$$

$$= \sigma _ { C } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } } \frac { d } { n _ { 1 } - d - 1 } ,$$

where the second equality follows because ϵ 1 b is normally distributed with mean 0 and covariance σ 2 C I n 1 .

The third term is

$$\text {tr} \left ( \mathbb { E } \left [ \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \epsilon _ { \underline { 1 } } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { c } _ { \epsilon _ { 2 } \epsilon _ { 2 } } \overline { C } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \epsilon _ { \underline { 1 } } ^ { \top } \underline { X } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \right ] \right )$$

$$= \text {tr} \left ( \mathbb { \mathbb { J } } \left [ \overline { C } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \epsilon _ { \underline { 1 } } \overset { \top } { X } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \epsilon _ { \underline { 1 } } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \overline { C } ^ { \top } \overline { \epsilon _ { 2 } \epsilon _ { 2 } } ^ { \top } \right ) \right )$$

$$= \sigma _ { Y } ^ { 2 } \text {tr} \left ( \mathbb { E } \left [ \epsilon _ { \underline { 1 } } \top \underline { X } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \epsilon _ { \underline { 1 } } \left ( \overline { C } ^ { \top } \overline { C } \right ) ^ { - 1 } \right ] \right )$$

$$= \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \text {tr} \left ( \mathbb { E } \left [ \frac { \tau } { \underline { \ell } } \, \overset { \top } { X } \, \underline { X } \right ) ^ { - 1 } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \, \underline { \epsilon } \right ] \right ) \\$$

$$= \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \text {tr} \left ( \mathbb { E } \left [ \frac { X \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \left ( \underline { X } ^ { \top } \underline { X } \right ) ^ { - 1 } \underline { X } ^ { \top } \right ] \right ) \\$$

$$= \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } } \frac { d } { n _ { 1 } - d - 1 } .$$

Putting the three terms together,

$$\text {tr} \left ( \mathbb { E } ( [ \hat { B } \hat { b } - B b ) ( \hat { B } \hat { b } - B b ) ^ { \top } \right ) \right )$$

$$= \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { n _ { 2 } - k - 1 } + \sigma _ { C } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } } \frac { d } { n _ { 1 } - d - 1 } + \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } } \frac { d } { n _ { 1 } - d - 1 } ,$$

so the expected squared error is

$$\mathbb { E } ( [ Y - \hat { Y } _ { I B } ) ^ { 2 } ]$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \sigma _ { Y } ^ { 2 } \frac { \sigma _ { X } ^ { 2 } } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { n _ { 2 } - k - 1 } + \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } + \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } .$$

Lemma 2 (Risk of the standard model) .

$$\mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] = \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \frac { d ( \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } ) } { n - d - 1 } .$$

Proof. A direct calculation gives

$$\mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] = \mathbb { E } [ ( ( X v + \epsilon ) - X \hat { v } ) ^ { 2 } ]$$

$$= \mathbb { E } [ ( \epsilon + X ( v - \hat { v } ) ) ^ { 2 } ]$$

$$= \mathbb { E } [ \epsilon ^ { 2 } ] + \mathbb { E } [ X ( v - \hat { v } ) ( v - \hat { v } ) ^ { \top } X ^ { \top } ]$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \text {tr} \left ( \mathbb { E } [ X ^ { \top } X ] \mathbb { E } [ ( v - \hat { v } ) ( v - \hat { v } ) ^ { \top } ] \right )$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \sigma _ { X } ^ { 2 } \text {tr} \left ( \mathbb { E } [ ( v - \hat { v } ) ( v - \hat { v } ) ^ { \top } ] \right ) .$$

$$\hat { v } - v = ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } \underline { X } ^ { \top } ( \underline { X } v + \underline { \epsilon } ) - v$$

$$= ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } \underline { X } ^ { \top } \epsilon ,$$

Since we have

$$\text {tr} \left ( \mathbb { E } [ ( v - \hat { v } ) ( v - \hat { v } ) ^ { \top } ] \right ) = \text {tr} \left ( \mathbb { E } [ ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } \underline { X } ^ { \top } \underline { \epsilon } ^ { \top } \underline { X } ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } ] \right )$$

$$= \text {tr} \left ( \mathbb { E } [ \underline { X } ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } \underline { X } ^ { \top } ] \mathbb { E } [ \underline { \epsilon } ^ { \top } ] \right )$$

$$= ( \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } ) \text {tr} \left ( \mathbb { E } [ ( \underline { X } ^ { \top } \underline { X } ) ^ { - 1 } ] \right )$$

$$= ( \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } ) \frac { 1 } { \sigma _ { X } ^ { 2 } } \frac { d } { n - d - 1 } .$$

Plugging this back into the expression for E [( Y - ˆ Y SM ) 2 ] yields

$$\mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] = \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } + \frac { d ( \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } ) } { n - d - 1 } .$$

Proof of Proposition 1. Note that the optimal estimator has risk

$$\mathbb { E } [ ( Y - \mathbb { E } [ Y | X ] ) ^ { 2 } ] = \mathbb { E } [ \epsilon ^ { 2 } ]$$

$$= \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } .$$

Thus, from Lemmas 1 and 2, the ratio of excess errors is

$$\frac { \mathbb { E } [ ( Y - \hat { Y } _ { I B } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ) ] ^ { 2 } ] } { \mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ) ] ^ { 2 } ] }$$

$$= \frac { n - d - 1 } { d ( \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } ) } \left ( \sigma _ { Y } ^ { 2 } \frac { \sigma _ { X } ^ { 2 } } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { n _ { 2 } - k - 1 } + \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } + \sigma _ { Y } ^ { 2 } \frac { 1 } { \sigma _ { X } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { 1 } { n _ { 2 } - k - 1 } \sigma _ { C } ^ { 2 } \frac { d } { n _ { 1 } - d - 1 } \right ) .$$

Taking the limit as n goes to infinity and letting n 1 = n 2 = n gives the desired result

$$\lim _ { n \to \infty } \frac { \mathbb { E } [ ( Y - \hat { Y } _ { I B } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ] ) ^ { 2 } ] } { \mathbb { E } [ ( Y - \hat { Y } _ { S M } ) ^ { 2 } ] - \mathbb { E } [ ( Y - \mathbb { E } [ Y | X ] ) ^ { 2 } ] } = \frac { \sigma _ { Y } ^ { 2 } } { \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } } \frac { \sigma _ { X } ^ { 2 } } { \sigma _ { C } ^ { 2 } + \sigma _ { C } ^ { 2 } } \frac { k } { d } + \frac { \sigma _ { C } ^ { 2 } } { \sigma _ { C } ^ { 2 } + \sigma _ { Y } ^ { 2 } }$$

$$\leq \frac { \frac { k } { d } \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } { \sigma _ { Y } ^ { 2 } + \sigma _ { C } ^ { 2 } } .$$
