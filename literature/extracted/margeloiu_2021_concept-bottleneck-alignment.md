---
id: "margeloiu_2021_concept-bottleneck-alignment"
source_pdf: "../pdf/margeloiu_2021_concept-bottleneck-alignment.pdf"
source_filename: "margeloiu_2021_concept-bottleneck-alignment.pdf"
format: "academic-paper"
---

## DO CONCEPT BOTTLENECK MODELS LEARN AS INTENDED?

Andrei Margeloiu ∗ University of Cambridge

am2770@cam.ac.uk

Yanzhi Chen University of Edinburgh Matthew Ashman ∗ University of Cambridge mca39@cam.ac.uk Mateja Jamnik University of Cambridge

## ABSTRACT

Concept bottleneck models map from raw inputs to concepts, and then from concepts to targets. Such models aim to incorporate pre-specified, high-level concepts into the learning procedure, and have been motivated to meet three desiderata: interpretability, predictability, and intervenability. However, we find that concept bottleneck models struggle to meet these goals. Using post hoc interpretability methods, we demonstrate that concepts do not correspond to anything semantically meaningful in input space, thus calling into question the usefulness of concept bottleneck models in their current form.

## 1 INTRODUCTION

Koh et al. (2020) proposed concept bottleneck models (CBMs) as a way to incorporate pre-defined expert concepts (e.g., 'bone spurs present' or 'wing color') into a supervised learning procedure. The approach maps raw inputs ( x ) to concepts ( c ), and then maps concepts ( c ) to targets ( y ). This can be done by specifying an intermediate layer of a neural network model and aligning the layer's activations with concepts when training. As per Koh et al. (2020), a CBM addresses three desiderata:

1. Interpretability: Being able to note which concepts are important for the targets.
2. Predictability: Being able to predict the targets from the concepts alone.
3. Intervenability: Being able to replace predicted concept values with ground truth values to improve predictive performance.

While CBMs are intuitive and have been shown in some settings to provide comparable performance to standard supervised learning approaches (directly learning a mapping between x and y ), it may often be unrealistic to assume that we will have access to concepts which fully capture the relationship between the inputs and targets. Forcing the high-dimensional inputs to pass through a low-dimension concept bottleneck might lose additional information from the inputs that is not captured by the expertdefined concepts. Nonetheless, Koh et al. (2020) propose three methods for training a CBM: training the input-to-concept mapping and the concept-to-target mapping jointly ; training the mappings independently , using the ground truth concepts for the concept-to-target mapping; and training the mappings sequentially , using the predicted concepts for the concept-to-target mapping. While Koh et al. (2020) favor the joint training objective, we call into question its utility for meeting the three desiderata of a CBM. Our results suggest that training a CBM independently may be the only way the three desiderata can be achieved. When we explore how to visualize concepts (as we may want to show domain experts what a concept represents in input space), we find that concepts appear not to correspond to anything semantically meaningful. We highlight our contributions:

1. We investigate training a CBM using constrained bottlenecks, demonstrating that neither the joint nor the sequential training methods satisfy the three desiderata.

∗ Equal Contribution

Umang Bhatt ∗ University of Cambridge

usb20@cam.ac.uk

Adrian Weller University of Cambridge The Alan Turing Institute

2. To analyze the behavior of a CBM, we leverage post hoc interpretability techniques to understand where in input space the concepts lie. We find that pre-specified concepts during training do not appear to correspond to anything meaningful in input space.

## 2 PROBLEM SETUP

Consider a setting where we are given a set of inputs x ∈ R d and corresponding targets y ∈ Y . Suppose we are also given a set of pre-specified concepts c ∈ R k , such that the training set comprises { x n , y n , c n } N n =1 . A CBM is of the form f ( g ( x )) , where g : R d → R k maps from input space to concept space and f : R k → Y maps from concept space to target space. Let L t : Y × Y → R denote a loss function that measures the discrepancy between the predicted and true targets and L c : R k × R k → R denote a loss function that measures the discrepancy between the predicted and true concepts. Koh et al. (2020) propose three distinct methods for learning the functions g and f :

1. The independent CBM learns the mappings g ind = arg min g ∑ N n =1 L c ( g ( x n ); c n ) and f ind = arg min f ∑ N n =1 L t ( f ( c n ); y n ) . That is, we learn a mapping from inputs to groundtruth concepts, and then separately learn a mapping from ground-truth concepts to targets.
2. The sequential CBM learns g in an identical manner as the independent case ( g seq = g ind ), but learns f using the predicted concept values from the learnt g , rather than using the ground-truth concept values: f seq = arg min f ∑ N n =1 L t ( f ( g seq ( x n )); y n ) .
3. The joint CBM learns g and f jointly by minimizing the combined objective: f jnt , g jnt = arg min f,g ∑ N n =1 L t ( f ( g ( x n )); y n ) + λ L c ( g ( x n ); c n ) where λ trades off the two losses.

In all cases, target predictions at test time are made using the combined mapping f ( g ( x test )) . We leverage post hoc interpretability methods to identify (1) which input features are relevant to each concept and (2) which concepts are relevant to each target. We use Integrated Gradients (IG) (Sundararajan et al., 2017) to do this analysis, as IG is popular amongst practitioners (Bhatt et al., 2020) and performs well on multiple explanation evaluation criteria (Ancona et al., 2018). When visualizing saliency maps, we use Gradients (Baehrens et al., 2010) and SmoothGrad (Smilkov et al., 2017) too.

We consider two applications of CBMs: x-ray grading and bird identification. For x-ray grading, we use a dataset from the Osteoarthritis Initiative (OAI) with 10 pre-specified expert concepts that correspond to clinically relevant factors (Peterfy et al., 2008). For bird identification, we use the CUB dataset with 112 concepts that capture the semantic attributes of each image (Wah et al., 2011). These are the same two datasets considered by Koh et al. (2020). See Appendix A for additional details.

## 3 CHECKING THE THREE DESIDERATA

We expect that decreasing the number of concepts in a CBM should lead to a significant drop in predictive performance, since a single concept alone may not be highly predictive of the target. To check the desiderata, we train models where we only use one concept at a time in our concept layer, k = 1 . We compare this one-concept CBM to the predicted performance of a concept oracle model that uses only the single ground truth concept to predict the targets. While the independent CBM and the concept oracle share the same concept-target predictor f ind, the independent CBM obtains targets via f ind ( g ind ( x test )) and the concept oracle obtains targets via f ind ( c test ) .

Table 1 demonstrates that the performance of the joint CBM far exceeds that of the concept oracle, as the joint achieves much lower error. This suggests that the joint CBM is learning useful information for predicting the target beyond simply getting the concept correct, that is, there is no concept bottleneck. Note that the joint CBM will be sensitive to the choice of λ ; however, we use the same λ as Koh et al. (2020) in our experiments. In contrast, the independent and sequential CBMs have no incentive to learn the targets before the concept layer, and thus achieve similar predictive performance to the concept oracle. On reflection, this is not surprising: each concept is represented as a scalar at the concept layer, but is often categorical or binary in the real-world. Thus, extra information about the targets is likely learned in the joint CBM. When all concepts are used (rightmost column of Table 1), the concept oracle achieves low error, suggesting that all 10 concepts together capture the target well. However, it may be difficult to obtain a full set of concepts that well specify the target.

Table 1: Root mean square error (RMSE) of y on OAI. We report the single concept predictive performance for each of ten concepts. Results are averaged over 5 random initializations. All standard deviations were &lt; 0 . 01 .

|  | Concept # | Concept # | Concept # | Concept # | Concept # | Concept # | Concept # | Concept # | Concept # | Concept # |  |
| - | - | - | - | - | - | - | - | - | - | - | - |
| Model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | All concepts |
| Joint | 0.49 | 0.46 | 0.46 | 0.48 | 0.47 | 0.50 | 0.47 | 0.45 | 0.48 | 0.44 | 0.42 |
| Sequential | 0.56 | 0.56 | 0.59 | 0.57 | 0.59 | 0.61 | 0.76 | 0.75 | 0.67 | 0.75 | 0.42 |
| Independent | 0.58 | 0.64 | 0.65 | 0.58 | 0.63 | 0.65 | 0.79 | 0.79 | 0.68 | 0.79 | 0.44 |
| Concept Oracle | 0.62 | 0.64 | 0.59 | 0.59 | 0.63 | 0.70 | 0.80 | 0.78 | 0.72 | 0.80 | 0.16 |

We posit that the dependence between targets and concepts in CBMs should mirror that of the concept oracle. To test our hypothesis, we measure the coefficient of determination R 2 between saliency maps from the concept oracle ( f o = f ind ( c test ) ) and saliency maps (via IG) from f jnt = f jnt ( g jnt ( x test )) and from f seq = f seq ( g seq ( x test )) . For CUB, we find that R 2 ( f o , f seq ) = 0 . 47 ± 0 . 04 , R 2 ( f o , f jnt ) = 0 . 24 ± 0 . 07 , R 2 ( f seq , f jnt ) = 0 . 07 ± 0 . 04 . Note that high correlation does not hold between the joint or sequential CBM and the concept oracle. This agrees with our observation that the predicted concepts are not used in their intended manner but rather as proxies to incorporate target information.

Figure 1 indicates that, since the intervenability of the joint and sequential CBMs is inferior to that of the independent CBM as per Koh et al.

Figure 1: Intervenability on joint and sequential CBMs is better than that of the independent CBM: this is from Figure 4 of (Koh et al., 2020).

<!-- image -->

(2020), the concept may not be used as intended. This is likely an artifact of one-hot encoding each concept values. We suggest that future work studies how the success of intervenability varies for different concept representations: binary, scalar, one-hot encoded categoricals, etc.

We have called into question CBMs trained using the joint objective, as they seem to fall short on all three of our desiderata: (1) they do not provide concept interpretability: post hoc analysis shows that the importance of individual concepts does not correspond to their true importance in predicting the targets; (2) they do not always predict target values based on concepts, thus violating predictability; and (3) they may not intervenable, as the concepts are learned at concept layer.

## 4 CONCEPTS IN INPUT SPACE

A desirable property of concepts is to depend on the relevant parts of the input space. For example, the wing concept ought to correspond to the wing of a bird in input space. Several works suggest that models can assign importance to spurious regions of input space: Ribeiro et al. (2016) showed that in husky versus wolf classification, their model picked up the snow in the background as important since husky images in the training data were more likely to contain snow. We identify regions in input space to which the CBM concepts attend. To investigate this, we construct saliency maps from the concept layer to the input space. Figure 2 illustrates saliency maps for joint and independent bottlenecks. In both cases, the wing concept attends to the entire bird. In Figure 3 and Figure 4, we find that similarly the 'leg color' concept does not attend to the leg of the bird for multiple saliency methods. We believe that existing saliency methods map be ill-equipped to study attribution for concept bottlenecks. Future work can attempt to develop methods to right this. Appendix B provides more results and compares different post hoc methods. Note that if the concepts are correctly predicted, then one can still argue that we can get some understanding of which concepts are important by looking at the intervenability of the concept layer.

Figure 2: Post hoc comparison between the joint and independent CBMs for the concept 'wing pattern.' Columns 2-4 show the saliency map for each discrete value of concept: solid, spotted, stripe. The last column shows the saliency for the entire 'Wing pattern' concept by averaging the saliency maps from columns 2-4. Both CBMs attend to the entire bird instead and not just to the wing. The saliency maps are computed using Integrated Gradients with Gaussian Noise baseline.

<!-- image -->

Figure 3: Joint model 'leg color.' The bird's leg is not attended to by any saliency method.

<!-- image -->

Figure 4: Independent model 'leg color.' The bird's leg is not attended to by any saliency method.

<!-- image -->

## 5 DISCUSSION

In this work, we study concept bottleneck models (CBMs). Our results suggest that CBMs may not learn as intended. We call into question a CBM's ability to meet three desiderata: interpretability, predictability, and intervenability. While the joint CBM learns information about the target before the concept layer, we find that the independent CBM may be the only variant that achieves all three desiderata given current methods for CBMs. Using existing saliency map methods, we find that none of the concepts learned to map semantically meaningful representations in input space. We hope that future work attempts to show concepts to domain experts: were concepts to map to something meaningful in input space, such methods could be used to discover concepts that are important to targets but have not been pre-specified by experts.

## ACKNOWLEDGMENTS

The authors thank Pang Wei Koh, Been Kim, and Percy Liang for their helpful comments. AM acknowledges support from the Cambridge ESRC Doctoral Training Partnership. MA acknowledges support from the George and Lilian Schiff Foundation. UB acknowledges support from DeepMind and the Leverhulme Trust via the Leverhulme Centre for the Future of Intelligence (CFI) and from the Mozilla Foundation. AW acknowledges support from a Turing AI Fellowship under grant EP/V025379/1, The Alan Turing Institute under EPSRC grant EP/N510129/1 and TU/B/000074, and the Leverhulme Trust via CFI.

## REFERENCES

- Marco Ancona, Enea Ceolini, Cengiz Oztireli, and Markus Gross. Towards better understanding of gradient-based attribution methods for deep neural networks. In 6th International Conference on Learning Representations (ICLR 2018) , 2018.
- David Baehrens, Timon Schroeter, Stefan Harmeling, Motoaki Kawanabe, Katja Hansen, and KlausRobert Müller. How to explain individual classification decisions. Journal of Machine Learning Research , 11(Jun):1803-1831, 2010.
- Umang Bhatt, Alice Xiang, Shubham Sharma, Adrian Weller, Ankur Taly, Yunhan Jia, Joydeep Ghosh, Ruchir Puri, José MF Moura, and Peter Eckersley. Explainable machine learning in deployment. In Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency , pages 648-657, 2020.
- JH Kellgren and JS Lawrence. Radiological assessment of osteo-arthrosis. Annals of the rheumatic diseases , 16(4):494, 1957.
- Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. Concept bottleneck models. In International Conference on Machine Learning , pages 5338-5348. PMLR, 2020.
- Narine Kokhlikyan, Vivek Miglani, Miguel Martin, Edward Wang, Bilal Alsallakh, Jonathan Reynolds, Alexander Melnikov, Natalia Kliushkina, Carlos Araya, Siqi Yan, et al. Captum: A unified and generic model interpretability library for pytorch. arXiv preprint arXiv:2009.07896 , 2020.
- Charles G Peterfy, Erika Schneider, and M Nevitt. The osteoarthritis initiative: report on the design rationale for the magnetic resonance imaging protocol for the knee. Osteoarthritis and cartilage , 16(12):1433-1441, 2008.
- Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. Why should I trust you?: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining , pages 1135-1144. ACM, 2016.
- Daniel Smilkov, Nikhil Thorat, Been Kim, Fernanda Viégas, and Martin Wattenberg. Smoothgrad: removing noise by adding noise. arXiv preprint arXiv:1706.03825 , 2017.
- Mukund Sundararajan, Ankur Taly, and Qiqi Yan. Axiomatic attribution for deep networks. In Proceedings of the 34th International Conference on Machine Learning-Volume 70 (ICML 2017) , pages 3319-3328. Journal of Machine Learning Research, 2017.
- Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.

## A EXPERIMENTAL SET-UP

## A.1 DATASETS AND HYPERPARAMETERS

For the x-grading task, we use 36 , 369 x-rays from the Osteoarthritis Initiative (OAI) to predict the Kellgren-Lawrence grade (KLG), a common scale used by radiologists to measure the severity of osteoarthritis (Peterfy et al., 2008); like Koh et al. (2020), we aim to predict KLG from ten ordinal, clinically relevant concepts: these concepts are instance-specific, which means that examples with the same y can have different c . Our ten concepts can be used by medical practitioners to assess KLG (Kellgren and Lawrence, 1957).

For the bird identification task, we use 11 , 788 photographs of birds to predict the correct bird species from 200 possible options: this dataset is known as CUB (Wah et al., 2011). We use the 112 concepts used from (Koh et al., 2020). These concepts are class-level, which means that all training data of the same class y have the same c . The concepts for CUB are also one-hot encoded, which means the concept 'wing color' is transformed from a categorical to a one hot ('has\_wing\_color\_red,' 'has\_wing\_color\_brown,' etc.).

When training a CBM, we admit that our results will highly depend on the λ we use to weight our concept loss: the larger, the more enforced our expert-specified concepts are. For CUB, we use λ = 0 . 01 . For OAI, we use λ = 1 .

## A.2 POST HOC INTERPRETABILITY

We leverage post hoc interpretability tools to analyze where feature space our CBMs attend. We use Integrated Gradients (Sundararajan et al., 2017) and SmoothGrad (Smilkov et al., 2017) to obtain saliency maps of which features are important to a model's predictions. Since we have two models ( g from inputs to concepts and f from concepts to targets), we get two types of saliency maps. First, we identify which concepts are important to a specific target by applying post hoc interpretability techniques to f , i.e., we can confirm if an American Goldfinch has a black wing. We also identify where concepts lie in input space by applying the same techniques to g , i.e., we can check if the tibia concept corresponds to the tibia in the x-ray itself.

When obtaining saliency maps from g , we have to overcome the one-hot-nature of our concepts. We obtain a separate saliency map for each 'beak color' with the CUB data. While it may be elucidating to show concepts for each color, we may want a summary saliency map for 'beak color.' We use two potential methods to overcome this hurdle: (1) we take the mean of the saliency maps for each of the categories of a given concept; or (2) we take a weighted average of the saliency maps, wherein we weigh the saliency map by the softmax of the one-hot-encoded concepts themselves. We leverage out-of-box implementations for all saliency map experiments (Kokhlikyan et al., 2020).

Appendix B shows additional post hoc interpretability results.

## B ADDITIONAL POST HOC RESULTS

Additional post hoc results using three saliency methods: Saliency (or Gradient), Integrated Gradients (Sundararajan et al., 2017) and Smoothgrad (Smilkov et al., 2017). All saliency maps are computed on images from the training set on which both the Joint and Independent models predicts correctly the shown concepts. The saliency maps of Concept Bottleneck Models consistently focus on areas outside the concept of interest.

Figure 5: Independent model 'bill shape' concept.

<!-- image -->

Figure 6: Joint model 'bill shape' concept.

<!-- image -->

Figure 7: Independent model 'wing pattern' concept.

<!-- image -->

Figure 8: Joint model 'wing pattern' concept.

<!-- image -->
