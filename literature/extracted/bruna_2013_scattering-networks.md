---
id: "bruna_2013_scattering-networks"
source_pdf: "../pdf/bruna_2013_scattering-networks.pdf"
source_filename: "bruna_2013_scattering-networks.pdf"
format: "academic-paper"
---

## Invariant Scattering Convolution Networks

Joan Bruna and St´ ephane Mallat CMAP, Ecole Polytechnique, Palaiseau, France

✦

Abstract -A wavelet scattering network computes a translation invariant image representation, which is stable to deformations and preserves high frequency information for classification. It cascades wavelet transform convolutions with non-linear modulus and averaging operators. The first network layer outputs SIFT-type descriptors whereas the next layers provide complementary invariant information which improves classification. The mathematical analysis of wavelet scattering networks explain important properties of deep convolution networks for classification.

A scattering representation of stationary processes incorporates higher order moments and can thus discriminate textures having same Fourier power spectrum. State of the art classification results are obtained for handwritten digits and texture discrimination, with a Gaussian kernel SVM and a generative PCA classifier.

## 1 INTRODUCTION

A major difficulty of image classification comes from the considerable variability within image classes and the inability of Euclidean distances to measure image similarities. Part of this variability is due to rigid translations, rotations or scaling. This variability is often uninformative for classification and should thus be eliminated. In the framework of kernel classifiers [31], metrics are defined as a Euclidean distance applied on a representation Φ( x ) of signals x . The operator Φ must therefore be invariant to these rigid transformations.

Non-rigid deformations also induce important variability within object classes [3], [15], [34]. For instance, in handwritten digit recognition, one must take into account digit deformations due to different writing styles. However, a full deformation invariance would reduce discrimination since a digit can be deformed into a different digit, for example a one into a seven. The representation must therefore not be deformation invariant but continuous to deformations, to handle small deformations with a kernel classifier. A small deformation of an image x into x ′ should correspond to a small Euclidean distance ‖ Φ( x ) - Φ( x ′ ) ‖ in the representation space, as further explained in Section 2.

Translation invariant representations can be constructed with registration algorithms [32] or with the Fourier transform modulus. However, Section 2.1 explains why these invariants are not stable to deformations and hence not adapted to image classification. Trying to avoid Fourier transform instabilities suggests replacing sinusoidal waves by localized waveforms such as wavelets. However, wavelet transforms are not invariant to translations. Building invariant representations from wavelet coefficients requires introducing non-linear operators, which leads to a convolution network architecture.

Deep convolution networks have the ability to build large-scale invariants which are stable to deformations [18]. They have been applied to a wide range of image classification tasks. Despite the remarkable successes of this neural network architecture, the properties and optimal configurations of these networks are not well understood because of cascaded non-linearities. Why use multiple layers ? How many layers ? How to optimize filters and pooling non-linearities ? How many internal and output neurons ? These questions are mostly answered through numerical experimentations that require significant expertise.

Deformation stability is obtained with localized wavelet filters which separate the image variations at multiple scales and orientations [22]. Computing a nonzero translation invariant representation from wavelet coefficients requires introducing a non-linearity, which is chosen to be a modulus to optimize stability [6]. Wavelet scattering networks, introduced in [23], [22], build translation invariant representations with average poolings of wavelet modulus coefficients. The output of the first network layer is similar to SIFT [21] or Daisy [33] type descriptors. However, this limited set of locally invariant coefficients is not sufficiently informative to discriminate complex structures over large-size domains. The information lost by the averaging is recovered by computing a next layer of invariant coefficients, with the same wavelet convolutions and average modulus poolings. A wavelet scattering is thus a deep convolution network which cascades wavelet transforms and modulus operators. The mathematical properties of scattering operators [22] explain how these deep network coefficients relate to image sparsity and geometry. The network architecture is optimized in Section 3, to retain important information while avoiding useless computations.

A scattering representation of stationary processes is introduced for texture discrimination. As opposed to the Fourier power spectrum, it provides information on higher order moments and can thus discriminate non-Gaussian textures having the same power spectrum. Classification applications are studied in Section 4.1. Scattering classification properties are demonstrated with a Gaussian kernel SVM and a generative classifier, which selects affine space models computed with a PCA. State-of-the-art results are obtained for handwritten digit recognition on MNIST and USPS databes, and for texture discrimination. Software is available at www.cmap.polytechnique.fr/scattering .

## 2 TOWARDS A CONVOLUTION NETWORK

Section 2.1 formalizes the deformation stability condition as a Lipschitz continuity property, and explains why high Fourier frequencies are source of unstabilites. Section 2.2 introduces a wavelet-based scattering transform, which is translation invariant and stable to deformations, and section 2.3 describes its convolutional network architecture.

## 2.1 Fourier and Registration Invariants

A representation Φ( x ) is invariant to global translations L c x ( u ) = x ( u - c ) by c = ( c 1 , c 2 ) ∈ R 2 if

$$\Phi ( L _ { c } x ) = \Phi ( x ) \ . & & ( 1 ) & \stackrel { A \ S } { \ w i t h }$$

A canonical invariant [15], [32] Φ( x ) = x ( u - a ( x )) registers x with an anchor point a ( x ) , which is translated when x is translated: a ( L c x ) = a ( x ) + c . It is therefore invariant: Φ( L c x ) = Φ( x ) . For example, the anchor point may be a filtered maxima a ( x ) = arg max u | x ⋆ h ( u ) | , for some filter h ( u ) .

The Fourier transform modulus is another example of translation invariant representation. Let ˆ x ( ω ) be the Fourier transform of x ( u ) . Since ̂ L c x ( ω ) = e - ic.ω ˆ x ( ω ) , it results that | L c x | = | ˆ x | does not depend upon c .

̂ To obtain appropriate similarity measurements between images which have undergone non-rigid transformations, the representation must also be stable to small deformations. A small deformation can be written L τ x ( u ) = x ( u - τ ( u )) where τ ( u ) depends upon u and thus deforms the image. The deformation gradient tensor ∇ τ ( u ) is a matrix whose norm |∇ τ ( u ) | measures the deformation amplitude at u . A small deformation is an invertible transformation if |∇ τ ( u ) | &lt; 1 [2], [34]. Stability to deformations is expressed as a Lipschitz continuity condition relative to this deformation metric:

$$\| \Phi ( L _ { \tau } x ) - \Phi ( x ) \| \leq C \, \| x \| \sup _ { u } | \nabla \tau ( u ) | \ , \quad ( 2 ) \quad \mathbb { R } ^ { 2 } . \ A$$

where ‖ x ‖ 2 = ∫ | x ( u ) | 2 du . This property implies global translation invariance, because if τ ( u ) = c then ∇ τ ( u ) = 0 , but it is much stronger.

A Fourier modulus is translation invariant but unstable with respect to deformations at high frequencies. Indeed, | | ˆ x ( ω ) | - | ̂ L τ x ( ω ) | | can be arbitrarily large at a high frequency ω , even for small deformations and in particular small dilations. As a result, Φ( x ) = | ˆ x | does not satisfy the deformation continuity condition (2) [22]. A Fourier modulus also loses too much information.

For example, a Dirac δ ( u ) and a linear chirp e iu 2 are totally different signals having Fourier transforms whose moduli are equal and constant. Very different signals may not be discriminated from their Fourier modulus.

A registration invariant Φ( x ) = x ( u - a ( x )) carries more information than a Fourier modulus, and characterizes x up to a global absolute position information [32]. However, it has the same high-frequency instability as a Fourier transform. Indeed, for any choice of anchor point a ( x ) , applying the Plancherel formula proves that

$$\text {condition} \quad \| x ( u - a ( x ) ) - x ^ { \prime } ( u - a ( x ^ { \prime } ) ) \| \geq ( 2 \pi ) ^ { - 1 } \left \| | \hat { x } ( \omega ) | - | \hat { x } ^ { \prime } ( \omega ) | \right \| . \\ \text {ns why} \quad ( 3 ) \\ \text {us case} \quad \mathbb { F } _ { 1 } \ \{ \begin{array} { c c } U & U \\ \end{array} \} \quad \mathbb { F } _ { 2 } \ \cdot \quad \mathbb { F } _ { 1 } \ \cdot \quad \mathbb { F } _ { 2 } \ \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot \quad \cdot$$

If x ′ = L τ x , the Fourier transform instability at high frequencies implies that Φ( x ) = x ( u - a ( x )) is also unstable with respect to deformations.

## 2.2 Scattering Wavelets

A wavelet is a localized waveform and is thus stable to deformation, as opposed to the Fourier sinusoidal waves. A wavelet transform computes convolutions with wavelets. It is thus translation covariant, not invariant. A scattering transform computes non-linear invariants with modulus and averaging pooling functions.

Two-dimensional directional wavelets are obtained by scaling and rotating a single band-pass filter ψ . Let G be a discrete, finite rotation group in R 2 . Multiscale directional wavelet filters are defined for any j ∈ Z and rotation r ∈ G by

$$\psi _ { 2 ^ { j } r } ( u ) = 2 ^ { 2 j } \psi ( 2 ^ { j } r ^ { - 1 } u ) \ .$$

If the Fourier transform ˆ ψ ( ω ) is centered at a frequency η then ˆ ψ 2 j r ( ω ) = ˆ ψ (2 - j r - 1 ω ) has a support centered at 2 j rη , with a bandwidth proportional to 2 j . To simplify notations, we denote λ = 2 j r ∈ Λ = G × Z , and | λ | = 2 j .

A wavelet transform filters x using a family of wavelets: { x⋆ψ λ ( u ) } λ . It is computed with a filter bank of dilated and rotated wavelets having no orthogonality property. As further explained in Section 3.1, it is stable and invertible if the rotated and scaled wavelet filters cover the whole frequency plane. On discrete images, to avoid aliasing, we only capture frequencies in the circle | ω | ≤ π inscribed in the image frequency square. However, most digital natural images and textures have negligible energy outside this frequency circle.

Let u.u ′ and | u | denote the inner product and norm in R 2 . A Morlet wavelet ψ is an example of wavelet given by

where C 2 is adjusted so that ∫ ψ ( u ) du = 0 . Figure 1 shows the Morlet wavelet with σ = 0 . 85 and ξ = 3 π/ 4 , used in all classification experiments.

$$\psi ( u ) & = C _ { 1 } \left ( e ^ { i u . \xi } - C _ { 2 } \right ) e ^ { - | u | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \ , \\ \intertext { c h i l l } \ A _ { \ } k a r { \sigma } ^ { \prime } \colon \quad & = C _ { 1 } \left ( e ^ { i u . \xi } - C _ { 2 } \right ) e ^ { - | u | ^ { 2 } / ( 2 \sigma ^ { 2 } ) } \ ,$$

A wavelet transform commutes with translations, and is therefore not translation invariant. To build a translation invariant representation, it is necessary to introduce a non-linearity. If R is a linear or non-linear operator which commutes with translations, R ( L c x ) = L c Rx , then the integral ∫ Rx ( u ) du is translation invariant. Applying this to Rx = x ⋆ ψ λ gives a trivial invariant ∫ x ⋆ ψ λ ( u ) du = 0 for all x because ∫ ψ λ ( u ) du = 0 . If Rx = M ( x ⋆ ψ λ ) but M is linear and commutes with translations then the integral still vanishes, which imposes choosing a non-linear M . Taking advantage of the wavelet transform stability to deformations, to obtain integrals which are also stable to deformations we also impose that M commutes with deformations

Fig. 1. Complex Morlet wavelet. (a): Real part of ψ . (b): Imaginary part of ψ . (c): Fourier modulus | ˆ ψ | .

<!-- image -->

$$\forall \tau ( u ) \ , \ M \ L _ { \tau } = L _ { \tau } \ M \ .$$

By adding a weak differentiability condition, one can prove [6] that M must necessarily be a pointwise operator, which means that Mx ( u ) only depends on the value x ( u ) . If we also impose an L 2 ( R 2 ) stability

$$\forall ( x , y ) \in L ^ { 2 } ( \mathbb { R } ^ { 2 } ) ^ { 2 } \, , \, \| M x \| = \| x \| \text { and } \| M x - M y \| \leq \| x - y \| , \quad \text {scattering}$$

then one can verify [6] that necessarily Mx = e iα | x | , and we set α = 0 . The resulting translation invariant coefficients are therefore L 1 ( R 2 ) norms: ‖ x⋆ψ λ ‖ 1 = ∫ | x⋆ ψ λ ( u ) | du .

$$\text {for all } \lambda _ { 1 } \text { and } \lambda _ { 2 } . \\ \| | x * \psi _ { \lambda _ { 1 } } | * \psi _ { \lambda _ { 2 } } \| _ { 1 } = \int | | x * \psi _ { \lambda _ { 1 } } ( u ) | * \psi _ { \lambda _ { 2 } } | \, d u \, .$$

The L 1 ( R 2 ) norms {‖ x ⋆ ψ λ ‖ 1 } λ form a crude signal representation, which measures the sparsity of the wavelet coefficients. For appropriate wavelets, one can prove [36] that x can be reconstructed from {| x⋆ψ λ ( u ) |} λ , up to a multiplicative constant. The information loss thus comes from the integration of | x ⋆ ψ λ ( u ) | , which removes all non-zero frequency components. These nonzero frequencies can be recovered by calculating the wavelet coefficients {| x⋆ψ λ 1 | ⋆ψ λ 2 ( u ) } λ 2 of | x⋆ψ λ 1 | . Their L 1 ( R 2 ) norms define a much larger family of invariants, for all λ 1 and λ 2 :

More translation invariant coefficients can be computed by further iterating on the wavelet transform and modulus operators. Let U [ λ ] x = | x ⋆ ψ λ | . Any sequence p = ( λ 1 , λ 2 , ..., λ m ) defines a path , i.e, the ordered product of non-linear and non-commuting operators

$$U [ p ] x = U [ \lambda _ { m } ] \dots U [ \lambda _ { 2 } ] \, U [ \lambda _ { 1 } ] x = | \, | x * \psi _ { \lambda _ { 1 } } | * \psi _ { \lambda _ { 2 } } | \dots | * \psi _ { \lambda _ { m } } | \, , \quad \text {direction} \quad \text {bins.} \ \text {Th}$$

with U [ ∅ ] x = x . A scattering transform along the path p is defined as an integral, normalized by the response of a Dirac:

$$\text {variant} \quad \text {a Dirac} \colon \quad \text {varianent} \quad \overline { S } x ( p ) = \mu _ { p } ^ { - 1 } \, \int U [ p ] x ( u ) \, d u \text { \ with } \mu _ { p } = \int U [ p ] \delta ( u ) \, d u \ .$$

Each scattering coefficient Sx ( p ) is invariant to a translation of x . We shall see that this transform has many similarities with the Fourier transform modulus, which is also translation invariant. However, a scattering is Lipschitz continuous to deformations as opposed to the Fourier transform modulus.

For classification, it is often better to compute localized descriptors which are invariant to translations smaller than a predefined scale 2 J , while keeping the spatial variability at scales larger than 2 J . This is obtained by localizing the scattering integral with a scaled spatial window φ 2 J ( u ) = 2 - 2 J φ (2 - J u ) . It defines a windowed scattering transform in the neighborhood of u :

$$& \leq \| x - y \| , & \\ & e ^ { i \alpha } \, | x | , \quad S _ { J } [ p ] x ( u ) = U [ p ] x * \phi _ { 2 ^ { J } } ( u ) = \int U [ p ] x ( v ) \phi _ { 2 ^ { J } } ( u - v ) \, d v \ , \\ & \text {variant}$$

and hence

$$S _ { J } [ p ] x ( u ) = | \ | | x * \psi _ { \lambda _ { 1 } } | * \psi _ { \lambda _ { 2 } } | \dots | * \psi _ { \lambda _ { m } } | * \phi _ { 2 ^ { J } } ( u ) \ , \\ g _ { \mu } = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \dots = \$$

with S J [ ∅ ] x = x ⋆ φ 2 J . For each path p , S J [ p ] x ( u ) is a function of the window position u , which can be subsampled at intervals proportional to the window size 2 J . The averaging by φ 2 J implies that S J [ p ] x ( u ) is nearly invariant to translations L c x ( u ) = x ( u - c ) if | c | ≪ 2 J . Section 3.1 proves that it is also stable relatively to deformations.

## 2.3 Scattering Convolution Network

If p is a path of length m then S J [ p ] x ( u ) is called scattering coefficient of order m at the scale 2 J . It is computed at the layer m of a convolution network which is specified. For large scale invariants, several layers are necessary to avoid losing crucial information.

For appropriate wavelets, first order coefficients S J [ λ 1 ] x are equivalent to SIFT coefficients [21]. Indeed, SIFT computes the local sum of image gradient amplitudes among image gradients having nearly the same direction, in a histogram having 8 different direction bins. The DAISY approximation [33] shows that these coefficients are well approximated by S J [2 j r ] x = | x ⋆ ψ 2 j r | ⋆ φ 2 J ( u ) where ψ 2 j r is the partial derivative of a Gaussian computed at the finest image scale 2 j , for 8 different rotations r . The averaging filter φ 2 J is a scaled Gaussian.

Partial derivative wavelets are well adapted to detect edges or sharp transitions but do not have enough frequency and directional resolution to discriminate complex directional structures. For texture analysis, many researchers [19], [30], [28] have been using averaged wavelet coefficient amplitudes | x ⋆ ψ λ | ⋆ φ J ( u ) , but calculated with a complex wavelet ψ having a better frequency and directional resolution.

A scattering transform computes higher-order coefficients by further iterating on wavelet transforms and modulus operators. At a maximum scale 2 J , wavelet coefficients are computed at frequencies 2 j ≥ 2 - J , and lower frequencies are filtered by φ 2 J ( u ) = 2 - 2 J φ (2 - J u ) . Since images are real-valued signals, it is sufficient to consider 'positive' rotations r ∈ G + with angles in [0 , π ) :

$$W _ { J } x ( u ) = \left \{ x * \phi _ { 2 ^ { J } } ( u ) \ , \ x * \psi _ { \lambda } ( u ) \right \} _ { \lambda \in \Lambda _ { J } } \quad ( 5 ) \quad \text {of each} \quad \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \$$

with Λ J = { λ = 2 j r : r ∈ G + , j ≥ - J } . For a Morlet wavelet ψ , the averaging filter φ is chosen to be a Gaussian. Let us emphasize that 2 J is a spatial scale variable whereas λ = 2 j r is assimilated to a frequency variable.

A wavelet modulus propagator keeps the lowfrequency averaging and computes the modulus of complex wavelet coefficients:

$$U _ { J } x ( u ) = \left \{ x * \phi _ { 2 ^ { J } } ( u ) \ , \ | x * \psi _ { \lambda } ( u ) | \right \} _ { \lambda \in \Lambda _ { J } } \ . \quad ( 6 ) \quad \text {each } \ \substack { 6 \\ \text {such } }$$

Let Λ m J be the set of all paths p = ( λ 1 , ..., λ m ) of length m . We denote U [Λ m J ] x = { U [ p ] x } p ∈ Λ J m and S J [Λ m J ] x = { S J [ p ] x } p ∈ Λ m J . Since

$$U _ { J } \, U [ p ] x & = \left \{ U [ p ] x \ast \phi _ { 2 ^ { J } } \, , \, | U [ p ] x \ast \psi _ { \lambda } | \right \} \, , \\$$

and S J [ p ] x = U [ p ] x ⋆ φ 2 J , it results that

$$a n d \ 5 J [ p ] x & = U [ p ] x * \varphi _ { 2 } J , \, i l l \, \text {results} \, i l l \\ U _ { J } \, U [ \Lambda _ { J } ^ { m } ] x & = \{ U _ { J } \, U [ p ] x \} _ { p \in \Lambda _ { J } ^ { m } } = \left \{ S _ { J } [ \Lambda _ { J } ^ { m } ] x \, , \, U [ \Lambda _ { J } ^ { m + 1 } ] x \right \} \, . \quad \text {along the} \\ \intertext { U _ { J } \, U [ \Lambda _ { J } ^ { m } ] x = \{ U _ { J } \, U [ p ] x \} _ { p \in \Lambda _ { J } ^ { m } } = \left \{ S _ { J } [ \Lambda _ { J } ^ { m } ] x \, , \, U [ \Lambda _ { J } ^ { m + 1 } ] x \right \} \, . \quad \text {along the} \\ \intertext { T h i s i m l i o c h t h t \, S _ { J } [ n ] _ { x } \, \text { can be computed} \, \text { along paths} \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { } \, \text { }$$

This implies that S J [ p ] x can be computed along paths of length m ≤ m max by first calculating U J x = { S J [ ∅ ] x, U [Λ 1 J ] x } and iteratively applying U J to each U [Λ m J ] x for increasing m ≤ m max . This algorithm is illustrated in Figure 2.

A scattering transform thus appears to be a deep convolution network [18], with some particularities. As opposed to most convolution networks, a scattering network outputs coefficients S J [ p ] x at all layers m ≤ m max , and not just at the last layer m max [18]. The next section proves that the energy of the deepest layer converges quickly to zero as m max increases.

A second distinction is that filters are not learned from data but are predefined wavelets. Wavelets are stable with respect to deformations and provide sparse image representations. Stability to deformations is a strong condition which imposes a separation of the different image scales [22], hence the use of wavelets.

The modulus operator which recombines real and imaginary parts can be interpreted as a pooling function in the context of convolution networks. The averaging by φ 2 J at the output is also a pooling operator which aggregates coefficients to build an invariant. It has been argued [7] that an average pooling loses information, which has motivated the use of other operators such as hierarchical maxima [8]. The high frequencies lost by the averaging are recovered as wavelet coefficients in the next layers, which explains the importance of using a multilayer network structure. As a result, it only loses the phase of these wavelet coefficients. This phase may however be recovered from the modulus thanks to the wavelet transform redundancy. It has been proved [36] that the wavelet-modulus operator U J x = { x ⋆ φ 2 J , | x ⋆ ψ λ |} λ ∈ Λ J is invertible with a continuous inverse. It means that x and hence the complex phase of each x ⋆ ψ λ can be reconstructed. Although U J is invertible, the scattering transform is not exactly invertible because of instabilities. Indeed, applying U J in (7) for m ≤ m max computes all S J [Λ m J ] x for m ≤ m max but also the last layer of internal network coefficients U [Λ m max +1 J ] x . The next section proves that U [Λ m max +1 J ] x can be neglected because its energy converges to zero as m max increases. However, this introduces a small error which accumulates when iterating on U - 1 J .

Figure 4 shows the Fourier transform of two images, and the amplitude of their scattering coefficients of orders m = 1 and m = 2 , at a maximum scale 2 J equal to the image size. A scattering coefficient over a quadrant Ω[2 j 1 r 1 ] gives an approximation of the Fourier transform energy over the support of ˆ ψ 2 j 1 r 1 . Although the top and bottom images are very different, they have same order m = 1 scattering coefficients. Here, first-order coefficients are not sufficient to discriminate between two very different images. However, coefficients of order m = 2 succeed in discriminating between the two images. The top image has wavelet coefficients which are much more sparse than the bottom image. As a result, Section 3.1 shows that second-order scattering coeffi-

Scattering coefficients can be displayed in the frequency plane. Let { Ω[ p ] } p ∈ Λ J m be a partition of R 2 . To each frequency ω ∈ R 2 we associate the path p ( ω ) such that ω ∈ Ω[ p ] . We display S J [ p ( ω )] x ( u ) , which is a piecewise constant function of ω ∈ R 2 , for each position u and each m = 1 , 2 . For m = 1 , each Ω[2 j 1 r 1 ] is chosen to be a quadrant rotated by r 1 , to approximate the frequency support of ˆ ψ 2 j 1 r 1 , whose size is proportional to ‖ ψ 2 j 1 r 1 ‖ 2 and hence to 2 j 1 . This defines a partition of a dyadic annulus illustrated in Figure 3(a). For m = 2 , Ω[2 j 1 r 1 , 2 j 2 r 2 ] is obtained by subdividing Ω[2 j 1 r 1 ] , as illustrated in Figure 3(b). Each Ω[2 j 1 r 1 ] is subdivided along the radial axis into quadrants indexed by j 2 . Each of these quadrants are themselves subdivided along the angular variable into rotated quadrants Ω[2 j 1 r 1 , 2 j 2 r 2 ] having a surface proportional to ‖| ψ 2 j 1 r 1 | ⋆ ψ 2 j 2 r 2 ‖ 2 .

Fig. 2. A scattering propagator U J applied to x computes each U [ λ 1 ] x = | x ⋆ ψ λ 1 | and outputs S J [ ∅ ] x = x ⋆ φ 2 J (black arrow). Applying U J to each U [ λ 1 ] x computes all U [ λ 1 , λ 2 ] x and outputs S J [ λ 1 ] = U [ λ 1 ] ⋆ φ 2 J (black arrows). Applying U J iteratively to each U [ p ] x outputs S J [ p ] x = U [ p ] x ⋆ φ 2 J (black arrows) and computes the next path layer.

<!-- image -->

Fig. 3. For m = 1 and m = 2 , a scattering is displayed as piecewise constant functions equal to S J [ p ] x ( u ) over each frequency subset Ω[ p ] . (a): For m = 1 , each Ω[2 j 1 r 1 ] is a rotated quadrant of surface proportional to 2 j 1 . (b): For m = 2 , each Ω[2 j 1 r 1 ] is subdivided into a partition of subsets Ω[2 j 1 r 1 , 2 j 2 r 2 ] .

<!-- image -->

cients have a larger amplitude. Higher-order coefficients are not displayed because they have a negligible energy as explained in Section 3.

## 3 SCATTERING PROPERTIES

A convolution network is highly non-linear, which makes it difficult to understand how the coefficient values relate to the signal properties. For a scattering network, Section 3.1 analyzes the coefficient properties and optimizes the network architecture. For texture analysis, the scattering transform of stationary processes is studied in Section 3.2. The regularity of scattering coefficients can be exploited to reduce the size of a scattering representation, by using a cosine transform, as shown in Section 3.3. Finally, Section 3.4 provides a fast computational algorithm.

## 3.1 Energy Conservation and Deformation Stability

A windowed scattering S J is computed with a cascade of wavelet modulus operators U J , and its properties thus depend upon the wavelet transform properties. Conditions are given on wavelets to define a scattering transform which is contracting and preserves the signal norm. This analysis shows that ‖ S J [ p ] x ‖ decreases quickly as the length of p increases, and is non-negligible only over a particular subset of frequency-decreasing paths. Reducing computations to these paths defines a convolution network with much fewer internal and output coefficients.

The norm of a sequence of transformed signals Rx = { g n } n ∈ Ω is defined by ‖ Rx ‖ 2 = ∑ n ∈ Ω ‖ g n ‖ 2 . If x is real and there exists ǫ &gt; 0 such that for all ω ∈ R 2

$$1 - \epsilon \leq | \hat { \phi } ( \omega ) | ^ { 2 } + \frac { 1 } { 2 } \sum _ { j = 1 } ^ { \infty } \sum _ { r \in G } | \hat { \psi } ( 2 ^ { - j } r \omega ) | ^ { 2 } \leq 1 \ , \quad ( 8 )$$

then applying the Plancherel formula proves that W J x =

Fig. 4. Scattering display of two images having the same first order scattering coefficients. (a) Image x ( u ) . (b) Fourier modulus | ˆ x ( ω ) | . (c) Scattering S J x [ p ( ω )] for m = 1 . (d) Scattering S J x [ p ( ω )] for m = 2 .

<!-- image -->

$$\{ x * \phi _ { J } \, , \, x * \psi _ { \lambda } \} _ { \lambda \in \Lambda _ { J } } \, \text {satisfies} & & \text {of the 1} \, \\ & ( 1 - \epsilon ) \, \| x \| ^ { 2 } \leq \| W _ { J } x \| ^ { 2 } \leq \| x \| ^ { 2 } \ , & & \text {increase} \\ & ( 1 - \epsilon ) \| x \| ^ { 2 } \leq \| W _ { J } x \| ^ { 2 } \leq \| x \| ^ { 2 } \ , & & \text {increase} \\ \intertext { w i t h \, \| W _ { J } x \| ^ { 2 } }$$

with ‖ W J x ‖ 2 = ‖ x ⋆ φ J ‖ 2 + ∑ λ ∈ Λ J ‖ x ⋆ ψ λ ‖ 2 . In the following we suppose that ǫ &lt; 1 and hence that the wavelet transform is a contracting and invertible operator, with a stable inverse. If ǫ = 0 then W J is unitary. The Morlet wavelet ψ in Figure 1 satisfies (8) with ǫ = 0 . 25 , together with φ ( u ) = C exp( -| u | 2 / (2 σ 2 0 )) with σ 0 = 0 . 7 and C adjusted so that ∫ φ ( u ) du = 1 . These functions are used in all classification applications. Rotated and dilated cubic spline wavelets are constructed in [22] to satisfy (8) with ǫ = 0 .

The modulus is contracting in the sense that || a |-| b || ≤ | a - b | . Since U J = { x⋆φ J , | x⋆ψ λ |} λ ∈ Λ J is obtained with a wavelet transform W J followed by modulus, which are both contractive, it is also contractive:

$$\| U _ { J } x - U _ { J } y \| & \leq \| x - y \| \ . \\$$

If W J is unitary then U J also preserves the signal norm ‖ U J x ‖ = ‖ x ‖ .

Let P J = ∪ m ≥ 0 Λ m J be the set of all possible paths of any length m ∈ N . The norm of S J [ P J ] x = { S J [ p ] x } p ∈P J is ‖ S J [ P J ] x ‖ 2 = ∑ p ∈P J ‖ S J [ p ] x ‖ 2 . Since S J iteratively applies U J which is contractive, it is also contractive:

$$\| S _ { J } x - S _ { J } y \| & \leq \| x - y \| \ . \\$$

If W J is unitary, ǫ = 0 in (9) and for appropriate wavelets, it is proved in [22] that

$$\| S _ { J } x \| ^ { 2 } & = \sum _ { m = 0 } ^ { \infty } \| S _ { J } [ \Lambda _ { J } ^ { m } ] x \| ^ { 2 } = \sum _ { m = 0 } ^ { \infty } \sum _ { p \in \Lambda _ { J } ^ { m } } \| S _ { J } [ p ] x \| ^ { 2 } = \| x \| ^ { 2 } \, . \quad \text {scattering} \\ \text {This result uses the fact that } U _ { J } \text { preserves the sig-} \, \text { is below}$$

This result uses the fact that U J preserves the signal norm and that U J U [Λ m J ] x = { S J [Λ m J ] x, U [Λ m +1 J ] x } . Proving (10) is thus equivalent to prove that the energy of the last network layer converges to zero when m max increases

$$& \quad ( 9 ) \\ \text {In the } & \quad \lim _ { m _ { \max } \to \infty } \| U [ \Lambda _ { J } ^ { m _ { \max } } ] x \| ^ { 2 } = \lim _ { m _ { \max } \to \infty } \sum _ { m = m _ { \max } } ^ { \infty } \| S _ { J } [ \Lambda _ { J } ^ { m } ] x \| ^ { 2 } = 0 \, . \\ \text {that the } & \quad \text {opera-} & \quad \text {This result is also important for numerical applications}$$

This result is also important for numerical applications because it explains why the network depth can be limited with a negligible loss of signal energy.

The scattering energy conservation also provides a relation between the network energy distribution and the wavelet transform sparsity. For p = ( λ 1 , ..., λ m ) , we denote p + λ = ( λ, λ 1 , ..., λ m ) . Applying (10) to U [ λ ] x = | x ⋆ ψ λ | instead of x , and separating the first term for m = 0 yields

$$\text {th} \ a \quad \| S _ { J } [ \lambda ] x \| ^ { 2 } + \sum _ { m = 1 } ^ { \infty } \sum _ { p \in \Lambda _ { J } ^ { m } } \| S _ { J } [ \lambda + p ] x \| ^ { 2 } = \| x * \psi _ { \lambda } \| ^ { 2 } \ . \ \ ( 1 2 )$$

But S J [ λ ] x = | x ⋆ ψ λ | ⋆ φ 2 J is a local L 1 ( R 2 ) norm and one can prove [22] that lim J →∞ 2 2 J ‖ S J [ λ ] x ‖ 2 = ‖ φ ‖ 2 ‖ x⋆ ψ λ ‖ 2 1 . The more sparse x⋆ψ λ ( u ) the smaller ‖ x⋆ψ λ ‖ 2 1 and (12) implies that the total energy ∑ ∞ m =1 ∑ p ∈ Λ m J ‖ S J [ p + λ ] x ‖ 2 of higher-order scattering terms is then larger. Figure 4 shows two images having same first order scattering coefficients, but the top image is piecewise regular and hence has wavelet coefficients which are much more sparse than the uniform texture at the bottom. As a result the top image has second order scattering coefficients of larger amplitude than at the bottom. For typical images, as in the CalTech101 dataset [10], Table 1 shows that the scattering energy has an exponential decay as a function of the path length m . As proved by (11), the energy of scattering coefficients converges to 0 as m increases and is below 1% for m ≥ 3 .

The energy conservation (10) is proved by showing that the scattering energy ‖ U [ p ] x ‖ 2 propagates towards

## TABLE 1

This table gives the percentage of scattering energy ‖ S J (Λ m J ) x ‖ 2 / ‖ x ‖ 2 captured by frequency-decreasing paths of length m , as a function of J . These are averaged values computed over normalized images with ∫ x ( u ) du = 0 and ‖ x ‖ = 1 , in the Caltech-101 database. The scattering is computed with cubic spline wavelets.

| J | m = 0 | m = 1 | m = 2 | m = 3 | m = 4 | m ≤ 3 |
| - | - | - | - | - | - | - |
| 1 | 95.1 | 4.86 | - | - | - | 99.96 |
| 2 | 87.56 | 11.97 | 0.35 | - | - | 99.89 |
| 3 | 76.29 | 21.92 | 1.54 | 0.02 | - | 99.78 |
| 4 | 61.52 | 33.87 | 4.05 | 0.16 | 0 | 99.61 |
| 5 | 44.6 | 45.26 | 8.9 | 0.61 | 0.01 | 99.37 |
| 6 | 26.15 | 57.02 | 14.4 | 1.54 | 0.07 | 99.1 |
| 7 | 0 | 73.37 | 21.98 | 3.56 | 0.25 | 98.91 |

lower frequencies as the length of p increases. This energy is thus ultimately captured by the low-pass filter φ 2 J which outputs S J [ p ] x = U [ p ] x⋆φ 2 J . This property requires that x⋆ψ λ has a lower-frequency envelope | x⋆ψ λ | . It is valid if ψ ( u ) = e iη.u θ ( u ) where θ is a low-pass filter. To verify this property, we write x⋆ψ λ ( u ) = e iλξ.u x λ ( u ) with

$$x _ { \lambda } ( u ) = ( e ^ { - i \lambda \xi . u } x ( u ) ) + \theta _ { \lambda } ( u ) \ .$$

This signal is filtered by the dilated and rotated low-pass filter θ λ whose Fourier transform is ˆ θ λ ( ω ) = θ ( λ - 1 ω ) . So | x⋆ψ λ ( u ) | = | x λ ( u ) | is the modulus of a regular function and is therefore mostly regular. This result is not valid if ψ is a real because | x ⋆ ψ λ | is singular at each zerocrossing of x ⋆ ψ λ ( u ) .

The modulus appears as a non-linear 'demodulator' which projects wavelet coefficients to lower frequencies. If λ = 2 j r then | x ⋆ ψ λ ( u ) | ⋆ ψ λ ′ for λ ′ = 2 j ′ r ′ is non-negligible only if ψ λ ′ is located at low frequencies and hence if 2 j ′ &lt; 2 j . Iterating on wavelet modulus operators thus propagates the scattering energy along frequency-decreasing paths p = (2 j 1 r 1 , ..., 2 j m r m ) where 2 j k ≤ 2 j k - 1 , for 1 ≤ k &lt; m . Scattering coefficients along other paths have a negligible energy. Over the CalTech101 images database, Table 1 shows that over 99% of the scattering energy is concentrated along frequencydecreasing paths of length m ≤ 3 . Numerically, it is therefore sufficient to compute the scattering transform along this subset of frequency-decreasing paths. It defines a much smaller convolution network. Section 3.4 shows that the resulting coefficients are computed with O ( N log N ) operations.

For classification applications, one of the most important properties of a scattering transform is its stability to deformations L τ x ( u ) = x ( u - τ ( u )) , because wavelets are stable to deformations and the modulus commutes with L τ . Let ‖ τ ‖ ∞ = sup u | τ ( u ) | and ‖∇ τ ‖ ∞ = sup u |∇ τ ( u ) | &lt; 1 . If S J is computed on paths of length m ≤ m max then it is proved in [22] that for signals x of compact support

$$\text {if is provided in } [ 2 2 ] \text { that for signals } x \text { of compact support } \text { wavelet} \\ \| S _ { J } ( L _ { \tau } x ) - S _ { J } x \| & \leq C m _ { \max } \| x \| \left ( 2 ^ { - J } \| \tau \| _ { \infty } + \| \nabla \tau \| _ { \infty } \right ) \, , \quad \text {scattering} \\$$

with a second order Hessian term which is negligible if τ ( u ) is regular. If 2 J ≥ ‖ τ ‖ ∞ / ‖∇ τ ‖ ∞ then the translation term can be neglected and the transform is Lipschitz continuous to deformations:

$$\| S _ { J } ( L _ { \tau } x ) - S _ { J } x \| \leq C m _ { \max } \| x \| \| \nabla \tau \| _ { \infty } \ .$$

## 3.2 Scattering Stationary Processes

Image textures can be modeled as realizations of stationary processes X ( u ) . We denote the expected value of X by E ( X ) , which does not depend upon u . The Fourier spectrum ̂ RX ( ω ) is the Fourier transform of the autocorrelation

$$^ { \ } a t o c o r r e l a t i o n \\ R X ( \tau ) = E \left ( [ X ( u ) - E ( X ) ] [ X ( u - \tau ) - E ( X ) ] \right ) \, . \\$$

Despite the importance of spectral methods, the Fourier spectrum is often not sufficient to discriminate image textures because it does not take into account higherorder moments. Figure 5 shows very different textures having same second-order moments. A scattering representation of stationary processes includes second order and higher-order moment descriptors of stationary processes, which discriminates between such textures.

If X ( u ) is stationary then U [ p ] X ( u ) remains stationary because it is computed with a cascade of convolutions and modulus operators which preserve stationarity. Its expected value thus does not depend upon u and defines the expected scattering transform:

$$\overline { S } X ( p ) = E ( U [ p ] X ) \ .$$

A windowed scattering gives an estimator of SX ( p ) , calculated from a single realization of X , by averaging U [ p ] X with φ 2 J :

$$S _ { J } [ p ] X ( u ) = U [ p ] X * \phi _ { 2 ^ { J } } ( u ) \ .$$

Since ∫ φ 2 J ( u ) du = 1 , this estimator is unbiased: E ( S J [ p ] X ) = E ( U [ p ] X ) .

For appropriate wavelets, it is also proved [22] that

$$\ a p p { \L p l a p l a t e } { \ v a v e l { S } , \L t i s a n s o p l o v e d \ [ 2 2 ] } \, \L t i a t \\ \sum _ { p \in \mathcal { P } _ { J } } E ( | S _ { J } [ p ] X | ^ { 2 } ) = E ( | X | ^ { 2 } ) \ .$$

Replacing X by X ⋆ ψ λ implies that

$$\sum _ { p \in \mathcal { P } _ { J } } E ( | S _ { J } [ p + \lambda ] X | ^ { 2 } ) = E ( | X * \psi _ { \lambda } | ^ { 2 } ) \ .$$

These expected squared wavelet coefficients can also be written as a filtered integration of the Fourier power spectrum ̂ RX ( ω )

These two equations prove that summing scattering coefficients recovers the power spectrum integral over each wavelet frequency support, which only depends upon second-order moments. However, one can also show that scattering coefficients SX ( p ) depend upon moments of X up to the order 2 m if p has a length m . Scattering

$$\text { spectrum } R X \left ( \omega \right ) \\ E ( | X * \psi _ { \lambda } | ^ { 2 } ) = \int \widehat { R } X ( \omega ) \left | \hat { \psi } ( \lambda ^ { - 1 } \omega ) \right | ^ { 2 } d \omega \ . \\ \text {These two equations prove that summing scattering coefficients}$$

Fig. 5. Two different textures having the same Fourier power spectrum. (a) Textures X ( u ) . Top: Brodatz texture. Bottom: Gaussian process. (b) Same estimated power spectrum ̂ RX ( ω ) . (c) Nearly same scattering coefficients S J [ p ] X for m = 1 and 2 J equal to the image width. (d) Different scattering coefficients S J [ p ] X for m = 2 .

<!-- image -->

coefficients can thus discriminate textures having same second-order moments but different higher-order moments. This is illustrated using the two textures in Figure 5, which have the same power spectrum and hence same second order moments. Scattering coefficients S J [ p ] X are shown for m = 1 and m = 2 with the frequency tiling illustrated in Figure 3. The ability to discriminate the top process X 1 from the bottom process X 2 is measured by a scattering distance normalized by the variance:

$$\rho ( m ) = \frac { \| S _ { J } X _ { 1 } [ \Lambda _ { J } ^ { m } ] - E ( S _ { J } X _ { 2 } [ \Lambda _ { J } ^ { m } ] ) \| ^ { 2 } } { E ( \| S _ { J } X _ { 2 } [ \Lambda _ { J } ^ { m } ] - E ( S _ { J } X _ { 2 } [ \Lambda _ { J } ^ { m } ] ) \| ^ { 2 } ) } \ .$$

For m = 1 , scattering coefficients mostly depend upon second-order moments and are thus nearly equal for both textures. One can indeed verify numerically that ρ (1) = 1 so both textures can not be distinguished using first order scattering coefficients. On the contrary, scattering coefficients of order 2 are highly dissimilar because they depend on moments up to order 4 , and ρ (2) = 5 .

For a large class of ergodic processes including most image textures, it is observed numerically that the total scattering variance ∑ p ∈P J E ( | S J [ p ] X - SX ( p ) | 2 ) decreases to zero when 2 J increases. Table 2 shows the decay of the total scattering variance, computed on average over the Brodatz texture dataset. Since E ( | S J [ p ] X | 2 ) = E ( S J [ p ] X ) 2 + E ( | S J [ p ] X - E ( S J [ p ] X ) | 2 ) and E ( S J [ p ] X ) = SX ( p ) , it results from the energy conservation (15) that the expected scattering transform also satisfies

$$\| \overline { S } X \| ^ { 2 } = \sum _ { m = 0 } ^ { \infty } \sum _ { p \in \Lambda _ { \infty } ^ { m } } | \overline { S } X ( p ) | ^ { 2 } = E ( | X | ^ { 2 } ) \ .$$

## TABLE 2

Decay of the total scattering variance ∑ p ∈P J E ( | S J [ p ] X - SX ( p ) | 2 ) /E ( | X | 2 ) in percentage, as a function of J , averaged over the Brodatz dataset. Results obtained using cubic spline wavelets.

| J = 1 | J = 2 | J = 3 | J = 4 | J = 5 | J = 6 | J = 7 |
| - | - | - | - | - | - | - |
| 85 | 65 | 45 | 26 | 14 | 7 | 2.5 |

TABLE 3 Percentage of expected scattering energy ∑ p ∈ Λ m ∞ | SX ( p ) | 2 , as a function of the scattering order m , computed with cubic spline wavelets, over the Brodatz dataset.

| m = 0 | m = 1 | m = 2 | m = 3 m = 4 |
| - | - | - | - |
| 0 | 74 | 19 | 3 0.3 |

Table 3 gives the percentage of expected scattering energy ∑ p ∈ Λ m ∞ | SX ( p ) | 2 carried by paths of length m , for textures in the Brodatz database. Most of the energy is concentrated in paths of length m ≤ 3 .

## 3.3 Cosine Scattering Transform

Natural images have scattering coefficients S J [ p ] X ( u ) which are correlated across paths p = (2 j 1 r 1 , ..., 2 j m r m ) , at any given position u . The strongest correlation is between paths of same length. For each m , scattering coefficients are decorrelated in a Karhunen-Lo` eve basis which diagonalizes their covariance matrix. Figure 6 compares the decay of the sorted variances E ( | S J [ p ] X - E ( S J [ p ] X ) | 2 ) and the variance decay in the KarhunenLo` eve basis computed on paths of length m = 1 , and on paths of length m = 2 , over the Caltech image dataset with a Morlet wavelet. The variance decay is much faster in the Karhunen-Lo` eve basis, which shows that there is a strong correlation between scattering coefficients of same path length.

A change of variables proves that a rotation and scaling X 2 l r ( u ) = X (2 - l ru ) produces a rotation and inverse scaling on the path variable p = (2 j 1 r 1 , ..., 2 j m r m ) :

$$\overline { S } X _ { 2 ^ { l } r } ( p ) = \overline { S } X ( 2 ^ { l } r p ) \text { where } 2 ^ { l } r p = ( 2 ^ { l + j _ { 1 } } r r _ { 1 } , \dots , 2 ^ { l + j _ { m } } r r _ { m } ) \text { elements by }$$

If images are randomly rotated and scaled by 2 l r - 1 then the path p is randomly rotated and scaled [27]. In this case, the scattering transform has stationary variations along the scale and rotation variables. This suggests approximating the Karhunen-Lo` eve basis by a cosine basis along these variables. Let us parameterize each rotation r by its angle θ ∈ [0 , 2 π ] . A path p is then parameterized by ([ j 1 , θ 1 ] , ..., [ j m , θ m ]) .

Since scattering coefficients are computed along frequency decreasing paths for which - J ≤ j k &lt; j k - 1 , to reduce boundary effects, a separable cosine transform is computed along the variables ˜ j 1 = j 1 , ˜ j 2 = j 2 - j 1 , ... , ˜ j m = j m - j m - 1 , and along each angle variable θ 1 , θ 2 , ... , θ m . We define the cosine scattering transform as the coefficients obtained by applying this separable discrete cosine transform along the scale and angle variables of S J [ p ] X ( u ) , for each u and each path length m . Figure 6 shows that the cosine scattering coefficients have variances for m = 1 and m = 2 which decay nearly as fast as the variances in the Karhunen-Loeve basis. It shows that a DCT across scales and orientations is nearly optimal to decorrelate scattering coefficients. Lower-frequency DCT coefficients absorb most of the scattering energy. On natural images, more than 99% of the scattering energy is absorbed by the 1 / 3 lowest frequency cosine scattering coefficients.

## 3.4 Fast Scattering Computations

Section 3.1 shows that the scattering energy is concentrated along frequency-decreasing paths p = (2 j k r k ) k satisfying 2 - J ≤ 2 j k +1 &lt; 2 j k . If the wavelet transform is computed along C directions then the total number of frequency-decreasing paths of length m is C m ( J m ) . Since φ 2 J is a low-pass filter, S J [ p ] x ( u ) = U [ p ] x ⋆ φ 2 J ( u ) can be uniformly sampled at intervals α 2 J , with α = 1 or α = 1 / 2 . If x ( n ) is a discrete image with N pixels, then each S J [ p ] x has 2 - 2 J α - 2 N coefficients. The scattering representation along all frequency-decreasing paths of length at most m thus has a total number of coefficients equal to N J = Nα - 2 2 - 2 J ∑ m q =0 C q ( J q ) . This reduced scattering representation is computed by a cascade of convolutions, modulus, and sub-samplings, with O ( N log N ) operations. The final DCT transform further compresses the resulting representation.

Let us recall from Section 2.3 that scattering coefficients are computed by iteratively applying the one-step propagator U J . To compute subsampled scattering coefficients along frequency-decreasing paths, this propagator is truncated. For any scale 2 k , U k,J transforms a signal x (2 k αn ) into

$$\text {and} \, \ s c a l { - \ U _ { k , J } x } = \left \{ x * \phi _ { J } ( 2 ^ { J } \alpha n ) \ , \ | x * \psi _ { 2 ^ { j } r } ( 2 ^ { j } \alpha n ) | \right \} _ { - J < j \leq k , r \in G ^ { + } } .$$

) . The algorithm computes subsampled scattering coefficients by iterating on this propagator.

## Algorithm 1 Reduced Scattering Transform

```
Compute U 0 ,J ( x ) Output x ⋆ φ 2 J (2 J αn ) for m = 1 to m max - 1 do for all 0 ≥ j 1 > ... > j m > - J do for all ( r 1 , ..., r q ) ∈ G + m do if m = m max - 1 then Compute ||| x⋆ψ 2 j 1 r 1 | ⋆... | ⋆ψ 2 jm r m | ⋆φ 2 J (2 J αn ) else Compute U j m ,J ( ||| x ⋆ ψ 2 j 1 r 1 | ⋆ ... | ⋆ ψ 2 jm r m | ) end if Output ||| x ⋆ ψ j 1 ,γ 1 | ⋆ ... | ⋆ ψ j q ,γ q | ⋆ φ J (2 J αn ) end for end for end for
```

If x is a signal of size P then FFT's compute U k,J x with O ( P log P ) operations. A reduced scattering transform thus computes its N J = Nα - 2 2 - 2 J ∑ m max m =0 C m ( J m ) coefficients with O ( N J log N ) operations. If m max = 2 then N J = Nα - 2 2 - 2 J ( CJ + C 2 J ( J - 1) / 2) . It decreases exponentially when the scale 2 J increases.

Scattering coefficients are decorrelated with a separable DCT along each scale variable ˜ j 1 = j 1 , ˜ j 2 = j 2 - j 1 , ... , ˜ j m = j m - j m - 1 and each rotation angle variable θ 1 , θ 2 , ... , θ m , which also requires O ( N J log N ) operations. For natural images, more than 99.5 % of the total signal energy is carried by the resulting N J / 2 cosine scattering coefficients of lower frequencies.

Numerical computations in this paper are performed by rotating wavelets along C = 6 directions, for scattering representations of maximum order m max = 2 . The resulting size of a reduced cosine scattering representation has at most three times as many coefficients as a dense SIFT representation. SIFT represents small blocks of 4 2 pixels with 8 coefficients. A cosine scattering representation represents each image block of 2 2 J pixels by N J 2 2 J / (2 N ) = ( CJ + C 2 J ( J - 1) / 2) / 2 coefficients, which is equal to 24 for C = 6 and J = 2 . The cosine scattering transform is thus three times the size of SIFT for J = 2 , but as J increases, the relative size decreases. If J = 3 then the size of a cosine scattering representation is twice the size of a SIFT representation but for J = 7 it is about 20 times smaller .

Fig. 6. (A): Sorted variances of scattering coefficients for m = 1 (left) and m = 2 (right). (B): Sorted variances of DCT scattering coefficients. (C): Variances in the scattering Karhunen-Loeve basis.

<!-- image -->

## 4 CLASSIFICATION USING SCATTERING VECTORS

A scattering transform eliminates the image variability due to translation and is stable to deformations. The resulting classification properties are studied with a PCA and an SVM classifier applied to scattering representations computed with a Morlet wavelet. State-of-the-art results are obtained for hand-written digit recognition and for texture discrimination.

## 4.1 PCA Affine Scattering Space Selection

Although discriminant classifiers such as SVM have better asymptotic properties than generative classifiers [26], the situation can be inverted for small training sets. We introduce a simple robust generative classifier based on affine space models computed with a PCA. Applying a DCT on scattering coefficients has no effect on any linear classifier because it is a linear orthogonal transform. However, keeping the 50 % lower frequency cosine scattering coefficients reduces computations and has a negligible effect on classification results. The classification algorithm is described directly on scattering coefficients to simplify explanations. Each signal class is represented by a random vector X k , whose realizations are images of N pixels in the class.

Let E ( S J X ) = { E ( S J [ p ] X ( u )) } p,u be the family of N J expected scattering values, computed along all frequency-decreasing paths of length m ≤ m max and all subsampled positions u = α 2 J n . The difference S J X k - E ( S J X k ) is approximated by its projection in a linear space of low dimension d ≪ N J . The covariance matrix of S J X k is a matrix of size N 2 J . Let V d,k be the linear space generated by the d PCA eigenvectors of this covariance matrix having the largest eigenvalues. Among all linear spaces of dimension d , this is the space which approximates S J X k - E ( S J X k ) with the smallest expected quadratic error. This is equivalent to approximating S J X k by its projection on an affine approximation space:

$$A _ { d , k } = E \{ S _ { J } X _ { k } \} + V _ { d , k } .$$

The resulting classifier associates a signal X to the class ˆ k which yields the best approximation space:

$$\hat { k } ( X ) = \arg \min _ { k \leq K } \| S _ { J } X - P _ { \mathbf A _ { d , k } } ( S _ { J } X ) \| \ .$$

The minimization of this distance has similarities with the minimization of a tangential distance [12] in the sense that we remove the principal scattering directions of variabilities to evaluate the distance. However it is much simpler since it does not evaluate a tangential space which depends upon S J x . Let V ⊥ d,k be the orthogonal complement of V d,k corresponding to directions of lower variability. This distance is also equal to the norm of the difference between S J x and the average class 'template' E ( S J X k ) , projected in V ⊥ d,k :

Minimizing the affine space approximation error is thus equivalent to finding the class centroid E ( S J X k ) which is the closest to S J x , without taking into account the first d principal variability directions. The d principal directions of the space V d,k result from deformations and from structural variability. The projection P A d,k ( S J x ) is the optimum linear prediction of S J x from these d principal modes. The selected class has the smallest prediction error.

$$\begin{array} { r l } { s i f i s } & { \quad \text {class} \ \text {template} \ E ( S _ { J } X _ { k } ) , \, p \text {projected in} \ V _ { d , k } . } \\ { a i n i g h e r } & { \quad \text {with} \quad } \\ { a i n i g h e r } & { \quad \| S _ { J } x - P _ { A _ { d , k } } ( S _ { J } x ) \| = \left \| P _ { V _ { d , k } } \left ( S _ { J } x - E ( S _ { J } X _ { k } ) \right ) \right \| . \, ( 1 8 ) } \\ { s i f i s } & { \quad } \\ { P C A . } & { \quad \min \min i g h e r t h e a f i n e s p a c h o r x i m a t i o n r o r is t h u s } \\ { o f f o t } & { \quad o u q u v a l o n t to t i n g h e r t h e a c l o s t r o d i v . \, F ( S _ { J } Y _ { k } ) \, w h i c h } \end{array}$$

This affine space selection is effective if S J X k - E ( S J X k ) is well approximated by a projection in a lowdimensional space. This is the case if realizations of X k are translations and limited deformations of a single template. Indeed, the Lipschitz continuity condition implies that small deformations are linearized by the scattering transform. Hand-written digit recognition is an example. This is also valid for stationary textures where S J X k has a small variance, which can be interpreted as structural variability.

̸

The dimension d must be adjusted so that S J X k has a better approximation in the affine space A d,k than in affine spaces A d,k ′ of other classes k ′ = k . This is a model selection problem, which requires to optimize the dimension d in order to avoid over-fitting [5].

The invariance scale 2 J must also be optimized. When the scale 2 J increases, translation invariance increases but it comes with a partial loss of information which brings the representations of different signals closer. One can prove [22] that for any x and x ′

$$\| S _ { J + 1 } x - S _ { J + 1 } x ^ { \prime } \| \leq \| S _ { J } x - S _ { J } x ^ { \prime } \| \ .$$

When 2 J goes to infinity, this scattering distance converges to a non-zero value. To classify deformed templates such as hand-written digits, the optimal 2 J is of the order of the maximum pixel displacements due to translations and deformations. In a stochastic framework where x and x ′ are realizations of stationary processes, S J x and S J x ′ converge to the expected scattering transforms Sx and Sx ′ . In order to classify stationary processes such as textures, the optimal scale is the maximum scale equal to the image width, because it minimizes the variance of the windowed scattering estimator.

A cross-validation procedure is used to find the dimension d and the scale 2 J which yield the smallest classification error. This error is computed on a subset of the training images, which is not used to estimate the covariance matrix for the PCA calculations.

As in the case of SVM, the performance of the affine PCA classifier can be improved by equalizing the descriptor space. Table 1 shows that scattering vectors have unequal energy distribution along its path variables, in particular as the order varies. A robust equalization is obtained by re-normalizing each S J [ p ] X ( u ) by the maximum ‖ S J [ p ] X i ‖ = ( ∑ u | S J [ p ] X i ( u ) | 2 ) 1 / 2 over all training signals X i :

$$\frac { S _ { J } [ p ] X ( u ) } { \sup _ { X _ { i } } \| S _ { J } [ p ] X _ { i } \| } \cdot & & ( 1 9 ) & & \ t o \ t \\ & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & & &$$

To simplify notations, we still write S J X for this normalized scattering vector.

Affine space scattering models can be interpreted as generative models computed independently for each class. As opposed to discriminative classifiers such as SVM, they do not estimate cross-terms between classes, besides from the choice of the model dimensionality d . Such estimators are particularly effective for small number of training samples per class. Indeed, if there are few training samples per class then variance terms dominate bias errors when estimating off-diagonal covariance coefficients between classes [4].

An affine space approximation classifier can also be interpreted as a robust quadratic discriminant classifier obtained by coarsely quantizing the eigenvalues of the inverse covariance matrix. For each class, the eigenvalues of the inverse covariance are set to 0 in V d,k and to 1 in V ⊥ d,k , where d is adjusted by cross-validation. This coarse quantization is justified by the poor estimation of covariance eigenvalues from few training samples. These affine space models will typically be applied to distributions of scattering vectors having non-Gaussian distributions, where a Gaussian Fisher discriminant can lead to important errors.

## 4.2 Handwritten Digit Recognition

The MNIST database of hand-written digits is an example of structured pattern classification, where most of the intra-class variability is due to local translations and deformations. It comprises at most 60000 training samples and 10000 test samples. If the training dataset is not augmented with deformations, the state of the art was achieved by deep-learning convolutional networks [29], deformation models [15], and dictionary learning [25]. These results are improved by a scattering classifier.

All computations are performed on the reduced cosine scattering representation described in Section 3.3, which keeps the lower-frequency half of the coefficients. Table 4 computes classification errors on a fixed set of test images, depending upon the size of the training set, for different representations and classifiers. The affine space selection of section 4.1 is compared with an SVM classifier using RBF kernels, which are computed using Libsvm [9], and whose variance is adjusted using standard cross-validation over a subset of the training set. The SVM classifier is trained with a renormalization which maps all coefficients to [ - 1 , 1] . The PCA classifier is trained with the renormalisation (19). The first two columns of Table 4 show that classification errors are much smaller with an SVM than with the PCA algorithm if applied directly on the image. The 3rd and 4th columns give the classification error obtained with a PCA or an SVM classification applied to the modulus of a windowed Fourier transform. The spatial size 2 J of the window is optimized with a cross-validation which yields a minimum error for 2 J = 8 . It corresponds to the largest pixel displacements due to translations or deformations in each class. Removing the complex phase of the windowed Fourier transform yields a locally invariant representation but whose high frequencies are unstable to deformations, as explained in Section 2.1. Suppressing this local translation variability improves the classification rate by a factor 3 for a PCA and by almost 2 for an SVM. The comparison between PCA and SVM confirms the fact that generative classifiers can outperform discriminative classifiers when training samples are scarce [26]. As the training set size increases, the bias-variance trade-off turns in favor of the richer SVM classifiers, independently of the descriptor.

Columns 6 and 8 give the PCA classification result applied to a windowed scattering representation for m max = 1 and m max = 2 . The cross validation also chooses 2 J = 8 . For the digit '3', Figure 7 displays the 4-by-4 array of normalized scattering vectors. For each u = 2 J ( n 1 , n 2 ) with 1 ≤ n i ≤ 4 , the scattering vector S J [ p ] X ( u ) is displayed for paths of length m = 1 and m = 2 , as circular frequency energy distributions following Section 2.3.

Increasing the scattering order from m max = 1 to m max = 2 reduces errors by about 30 %, which shows that second order coefficients carry important information even at a relatively small scale 2 J = 8 . However, third order coefficients have a negligible energy and including them brings marginal classification improvements, while increasing computations by an important factor. As the learning set increases in size, the classification improvement of a scattering transform increases relatively to windowed Fourier transform because the classification is able to incorporate more high frequency structures, which have deformation instabilities in the Fourier domain as opposed to the scattering domain.

Fig. 7. (a): Image X ( u ) of a digit '3'. (b): Array of scattering vectors S J [ p ] X ( u ) , for m = 1 and u sampled at intervals 2 J = 8 . (c): Scattering vectors S J [ p ] X ( u ) , for m = 2 .

<!-- image -->

Table 4 also shows that below 5 · 10 3 training samples, the scattering PCA classifier improves results of a deeplearning convolutional networks, which learns all filter coefficients with a back-propagation algorithm [18]. As more training samples are available, the flexibility of the SVM classifier brings an improvement over the more rigid affine classifier, yielding a 0 . 43% error rate on the original dataset, thus improving upon previous state of the art methods.

To evaluate the precision of the affine space model, we compute the relative affine approximation error, averaged over all classes:

$$\sigma _ { d } ^ { 2 } = K ^ { - 1 } \sum _ { k = 1 } ^ { K } \frac { E ( \| S _ { J } X _ { k } - P _ { A _ { d , k } } ( S _ { J } X _ { k } ) \| ^ { 2 } ) } { E ( \| S _ { J } X _ { k } \| ^ { 2 } ) } \ .$$

For any S J X k , we also calculate the minimum approximation error produced by another affine model A d,k ′ with k ′ = k :

̸

$$\lambda _ { d } = \frac { E ( \min _ { k ^ { \prime } \neq k } \| S _ { J } X _ { k } - P _ { A _ { k ^ { \prime } , d } } ( S _ { J } X _ { k } ) \| ^ { 2 } ) } { E ( \| S _ { J } X _ { k } - P _ { A _ { d , k } } ( S _ { J } X _ { k } ) \| ^ { 2 } ) } \ . \\ \text {For} \, o \, c \, s t t o r i n g \, r o p r o s t e n t \, w i t h \, m \quad = 2 \, \text { Tablo}$$

For a scattering representation with m max = 2 , Table 5 gives the dimension d of affine approximation spaces optimized with a cross validation, with the corresponding values of σ 2 d and λ d . When the training set size increases, the model dimension d increases because there are more samples to estimate each intra-class covariance matrix. The approximation model becomes more precise so σ 2 d decreases and the relative approximation error λ d produced by wrong classes increases. This explains the reduction of the classification error rate observed in Table 4 as the training size increases.

̸

## TABLE 5

Values of the dimension d of affine approximation models on MNIST classification, of the intra class normalized approximation error σ 2 d , and of the ratio λ d between inter class and intra class approximation errors, as a function of the training size.

| Training | d | σ 2 d | λ d |
| - | - | - | - |
| 300 | 5 | 3 · 10 - 1 | 2 |
| 5000 | 100 | 4 · 10 - 2 | 3 |
| 40000 | 140 | 2 · 10 - 2 | 4 |

TABLE 6 Percentage of errors for the whole USPS database.

| Tang. Kern. | Scat. m max = 2 SVM | Scat. m max = 1 PCA | Scat. m max = 2 PCA |
| - | - | - | - |
| 2 . 4 | 2 . 7 | 3 . 24 | 2 . 6 / 2 . 3 |

The US-Postal Service is another handwritten digit dataset, with 7291 training samples and 2007 test images 16 × 16 pixels. The state of the art is obtained with tangent distance kernels [12]. Table 6 gives results obtained with a scattering transform with the PCA classifier for m max = 1 , 2 . The cross-validation sets the scattering scale to 2 J = 8 . As in the MNIST case, the error is reduced when going from m max = 1 to m max = 2 but remains stable for m max = 3 . Different renormalization strategies can bring marginal improvements on this dataset. If the renormalization is performed by equalizing using the standard deviation of each component, the classification error is 2 . 3% whereas it is 2 . 6% if the supremum is normalized.

The scattering transform is stable but not invariant to rotations. Stability to rotations is demonstrated over the MNIST database in the setting defined in [16]. A database with 12000 training samples and 50000 test images is constructed with random rotations of MNIST digits. The PCA affine space selection takes into account the rotation variability by increasing the dimension d of the affine approximation space. This is equivalent to projecting the distance to the class centroid on a smaller orthogonal space, by removing more principal components. The error rate in Table 7 is much smaller with a scattering PCA than with a convolution network [16]. Much better results are obtained for a scattering with m max = 2 than with m max = 1 because second order coefficients maintain enough discriminability despite the removal of a larger number d of principal directions. In this case, m max = 3 marginally reduces the error.

TABLE 4 MNIST classification results.

| Training | x | x | Wind. Four. | Wind. Four. | Scat. m max = 1 | Scat. m max = 1 | Scat. m max = 2 | Scat. m max = 2 | Conv. |
| - | - | - | - | - | - | - | - | - | - |
| size | PCA | SVM | PCA | SVM | PCA | SVM | PCA | SVM | Net. |
| 300 | 14 . 5 | 15 . 4 | 7 . 35 | 7 . 4 | 5 . 7 | 8 | 4 . 7 | 5 . 6 | 7 . 18 |
| 1000 | 7 . 2 | 8 . 2 | 3 . 74 | 3 . 74 | 2 . 35 | 4 | 2 . 3 | 2 . 6 | 3 . 21 |
| 2000 | 5 . 8 | 6 . 5 | 2 . 99 | 2 . 9 | 1 . 7 | 2 . 6 | 1 . 3 | 1 . 8 | 2 . 53 |
| 5000 | 4 . 9 | 4 | 2 . 34 | 2 . 2 | 1 . 6 | 1 . 6 | 1 . 03 | 1 . 4 | 1 . 52 |
| 10000 | 4 . 55 | 3 . 11 | 2 . 24 | 1 . 65 | 1 . 5 | 1 . 23 | 0 . 88 | 1 | 0 . 85 |
| 20000 | 4 . 25 | 2 . 2 | 1 . 92 | 1 . 15 | 1 . 4 | 0 . 96 | 0 . 79 | 0 . 58 | 0 . 76 |
| 40000 | 4 . 1 | 1 . 7 | 1 . 85 | 0 . 9 | 1 . 36 | 0 . 75 | 0 . 74 | 0 . 53 | 0 . 65 |
| 60000 | 4 . 3 | 1 . 4 | 1 . 80 | 0 . 8 | 1 . 34 | 0 . 62 | 0 . 7 | 0 . 43 | 0 . 53 |

TABLE 7 Percentage of errors on an MNIST rotated dataset [16].

| Scat. m max = 1 PCA | Scat. m max = 2 PCA | Conv. Net. |
| - | - | - |
| 8 | 4 . 4 | 8 . 8 |

TABLE 8 Percentage of errors on scaled and/or rotated MNIST digits

| Transformed Images | Scat. m max = 1 PCA | Scat. m max = 2 PCA |
| - | - | - |
| Without | 1 . 6 | 0 . 8 |
| Rotation | 6 . 7 | 3 . 3 |
| Scaling | 2 | 1 |
| Rot. + Scal. | 12 | 5 . 5 |

Scaling invariance is studied by introducing a random scaling factor uniformly distributed between 1 / √ 2 and √ 2 . In this case, the digit '9' is removed from the database as to avoid any indetermination with the digit '6' when rotated. The training set has 9000 samples ( 1000 samples per class). Table 8 gives the error rate on the original MNIST database and including either rotation, scaling, or both in the training and testing samples. Scaling has a smaller impact on the error rate than rotating digits because scaled scattering vectors span an invariant linear space of lower dimension. Secondorder scattering outperforms first-order scattering, and the difference becomes more significant when rotation and scaling are combined, because it provides interaction coefficients which are discriminative even in presence of scaling and rotation variability.

## 4.3 Texture Discrimination

Visual texture discrimination remains an outstanding image processing problem because textures are realizations of non-Gaussian stationary processes, which cannot be discriminated using the power spectrum. Depending on the imaging conditions, textures undergo transformations due to illumination, rotation, scaling or more complex deformations when mapped on three-dimensional surfaces. The affine PCA space classifier removes most of the variability of S J X - E { S J X } within each class. This variability is due to the residual stochastic variability which decays as J increases and to variability due to illumination, rotation and perspective effects.

Texture classification is tested on the CUReT texture database [19], [35], which includes 61 classes of image textures of N = 200 2 pixels. Each texture class gives images of the same material with different pose and illumination conditions. Specularities, shadowing and surface normal variations make classification challenging. Pose variation requires global rotation and illumination invariance. Figure 8 illustrates the large intraclass variability, after a normalization of the mean and variance of each textured image.

Table 9 compares error rates obtained with different classifiers. The database is randomly split into a training and a testing set, with 46 training images for each class as in [35]. Results are averaged over 10 different splits. A PCA affine space classifier applied directly on the image yields a large classification error of 17% . To estimate the Fourier spectrum, windowed Fourier transforms are computed over half-overlapping windows of size 2 J , and their squared modulus is averaged over the whole image. This averaging is necessary to reduce the spectrum estimator variance, which does not decrease when the window size 2 J increases. The cross-validation sets the optimal window scale to 2 J = 32 , whereas images have a width of 200 pixels. The error drops to 1%. This simple Fourier spectrum yields a smaller error than previously reported state-of-the-art methods. SVM's applied to a dictionary of textons yield an error rate of 1.53% [13], whereas an optimized Markov Random Field model computed with image patches [35] achieves an error of 2.46%.

For the scattering PCA classifier, the cross validation chooses an optimal scale 2 J equal to the image width to reduce the scattering estimation variance. Indeed, contrarly to a power spectrum estimation, the variance of the scattering vector decreases when 2 J increases. Figure 9 displays the scattering coefficients S J [ p ] X of order m = 1 and m = 2 of a CureT textured image X . When m max = 1 , the error drops to 0.5%, although first-order scattering coefficients essentially depend upon second order moments as the Fourier spectrum. This is probably due to the fact that image textures have a spectrum which typically decays like | ω | - α . For such spectrum, an estimation over dyadic frequency bands provide a better bias versus variance trade-off than a windowed Fourier spectrum [1]. For m max = 2 , the error further drops to 0.2%. Indeed, scattering coefficients of order m = 2 depend upon moments of order 4 , which are necessary to differentiate textures having same second order moments as in Figure 5. The dimension of the affine approximation space model is d = 16 , the intraclass normalized approximation error is σ 2 d = 2 . 5 · 10 - 1 and the separation ratio is λ d = 3 for m max = 2 .

Fig. 8. Examples of textures from the CUReT database with normalized mean and variance. Each row corresponds to a different class, showing intra-class variability in the form of stochastic variability and changes in pose and illumination.

<!-- image -->

TABLE 9 Percentage of errors on CUReT for different training sizes.

| Training size | X PCA | Four. Spectr. PCA | Scat. m max = 1 PCA | Scat. m max = 2 PCA | Textons SVM [13] | MRF [35] |
| - | - | - | - | - | - | - |
| 46 | 17 | 1 | 0 . 5 | 0 . 2 | 1 . 53 | 2 . 4 |

Fig. 9. (a): Example of CureT texture X ( u ) . (b): Scattering coefficients S J [ p ] X , for m = 1 and 2 J equal to the image width. (c): Scattering coefficients S J [ p ] X ( u ) , for m = 2 .

<!-- image -->

The PCA classifier provides a partial rotation invariance by removing principal components. It averages scattering coefficients along path rotation parameters, which comes with a loss of discriminability. However, a more efficient rotation invariant texture classification is obtained by cascading this translation invariant scattering with a second rotation invariant scattering [24]. It transforms each layer of the translation invariant scattering network with new wavelet convolutions along rotation parameters, followed by modulus and average pooling operators, which are cascaded. A combined translation and rotation scattering yields a translation and rotation invariant representation which is stable to deformations [22].

## 5 CONCLUSION

A wavelet scattering transform computes a translation invariant representation, which is stable to deformation, using a deep convolution network architecture. The first layer outputs SIFT-type descriptors, which are not sufficiently informative for large-scale invariance. Classification performance is improved by adding other layers providing complementary information. A reduced cosine scattering transform is at most three times larger than a SIFT descriptor and computed with O ( N log N ) operations.

State-of-the-art classification results are obtained for handwritten digit recognition and texture discrimination, with an SVM or a PCA classifier. If the data set has other sources of variability due to the action of other finite Lie groups such as rotations, then this variability can be eliminated with an invariant scattering computed by cascading wavelet transforms defined on these groups [22], [24].

However, signal classes may also include complex sources of variability that can not be approximated by the action of a finite group, as in CalTech101 or Pascal databases. This variability must be taken into account by unsupervised optimizations of the representations from the training data. Deep convolution networks which learn filters from the data [18] have the flexibility to adapt to such variability, but learning translation invariant filters is not necessary. A wavelet scattering transform can be used on the first two network layers, while learning the next layer filters applied to scattering coefficients. Similarly, bag-of-features unsupervised algorithms [37], [7] applied to SIFT can potentially be improved upon by replacing SIFT descriptors by wavelet scattering vectors.

## REFERENCES

- [1] P. Abry, P. Gonc ¸alves, and P. Flandrin, 'Wavelets, spectrum analysis and 1/f processes', Wavelets and statistics, Lecture Notes in Statistics, 1995.
- [2] S. Allassonniere, Y. Amit, A. Trouve, 'Toward a coherent statistical framework for dense deformable template estimation'. Volume 69, part 1 (2007), pages 3-29, of the Journal of the Royal Statistical Society.
- [3] R. Bajcsy and S. Kovacic, 'Multi-resolution elastic matching', Computer Vision Graphics and Image Processing, vol 46, Issue 1, April 1989.
- [4] P. J. Bickel and E. Levina: 'Covariance regularization by thresholding', Annals of Statistics, 2008.
- [5] L. Birge and P. Massart. 'From model selection to adaptive estimation.' In Festschrift for Lucien Le Cam: Research Papers in Probability and Statistics, 55 - 88, Springer-Verlag, New York, 1997.
- [6] J. Bruna, 'Operators commuting with diffeomorphisms', CMAP Tech. Report, Ecole Polytechnique, 2012.
- [7] Y-L. Boureau, F. Bach, Y. LeCun, and J. Ponce. 'Learning MidLevel Features For Recognition'. In IEEE Conference on Computer Vision and Pattern Recognition, 2010.
- [8] J. Bouvrie, L. Rosasco, T. Poggio: 'On Invariance in Hierarchical Models', NIPS 2009.
- [9] C. Chang and C. Lin, 'LIBSVM : a library for support vector machines'. ACM Transactions on Intelligent Systems and Technology, 2:27:1-27:27, 2011
- [10] L. Fei-Fei, R. Fergus and P. Perona. 'Learning generative visual models from few training examples: an incremental Bayesian approach tested on 101 object categories'. IEEE. CVPR 2004
- [11] Z. Guo, L. Zhang, D. Zhang, 'Rotation Invariant texture classification using LBP variance (LBPV) with global matching', Elsevier Journal of Pattern Recognition, Aug. 2009.
- [12] B.Haasdonk, D.Keysers: 'Tangent Distance kernels for support vector machines', 2002.
- [13] E. Hayman, B. Caputo, M. Fritz and J.O. Eklundh, 'On the Significance of Real-World Conditions for Material Classification', ECCV, 2004.
- [14] K. Jarrett, K. Kavukcuoglu, M. Ranzato and Y. LeCun: 'What is the Best Multi-Stage Architecture for Object Recognition?', Proc. of ICCV 2009.
- [15] D.Keysers, T.Deselaers, C.Gollan, H.Ney, 'Deformation Models for image recognition', IEEE trans of PAMI, 2007.
- [16] H. Larochelle, Y. Bengio, J. Louradour, P. Lamblin, 'Exploring Strategies for Training Deep Neural Networks', Journal of Machine Learning Research, Jan. 2009.
- [17] S. Lazebnik, C. Schmid, J.Ponce. 'Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories'. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, New York, June 2006, vol. II, pp. 2169-2178.
- [18] Y. LeCun, K. Kavukvuoglu and C. Farabet: 'Convolutional Networks and Applications in Vision', Proc. of ISCAS 2010.
- [19] T. Leung and J. Malik; 'Representing and Recognizing the Visual Appearance of Materials Using Three-Dimensional Textons'. International Journal of Computer Vision, 43(1), 29-44; 2001.
- [20] W. Lohmiller and J.J.E. Slotine 'On Contraction Analysis for Nonlinear Systems', Automatica, 34(6), 1998.
- [21] D.G. Lowe, 'Distinctive Image Features from Scale-Invariant Keypoints', International Journal of Computer Vision, 60, 2, pp. 91110, 2004
- [22] S. Mallat 'Group Invariant Scattering', to appear in 'Communications in Pure and Applied Mathematics', 2012, http://arxiv.org/abs/1101.2286.
- [23] S. Mallat, 'Recursive Interferometric Representation', Proc. of EUSICO conference, Denmark, August 2010.
- [24] S.Mallat , L. Sifre : 'Combined scattering for rotation invariant texture analysis', submitted to ESANN, 2012.
- [25] J. Mairal, F. Bach, J.Ponce, 'Task-Driven Dictionary Learning', Submitted to IEEE trans. on PAMI, September 2010.
- [26] A. Y. Ng and M. I. Jordan 'On discriminative vs. generative classifiers: A comparison of logistic regression and naive Bayes', in Advances in Neural Information Processing Systems (NIPS) 14, 2002.
- [27] L. Perrinet, 'Role of Homeostasis in Learning Sparse Representations', Neural Computation Journal, 2010.
- [28] J.Portilla, E.Simoncelli, 'A Parametric Texture model based on joint statistics of complex wavelet coefficients', IJCV, 2000.
- [29] M. Ranzato, F.Huang, Y.Boreau, Y. LeCun: 'Unsupervised Learning of Invariant Feature Hierarchies with Applications to Object Recognition', CVPR 2007.
- [30] C. Sagiv, N. A. Sochen and Y. Y. Zeevi, 'Gabor Feature Space Diffusion via the Minimal Weighted Area Method', Springer Lecture Notes in Computer Science, Vol. 2134, pp. 621-635, 2001.
- [31] B. Scholkopf and A. J. Smola, 'Learning with Kernels', MIT Press, 2002.
- [32] S.Soatto, 'Actionable Information in Vision', ICCV, 2009.
- [33] E. Tola, V.Lepetit, P. Fua, 'DAISY: An Efficient Dense Descriptor Applied to Wide-Baseline Stereo', IEEE trans on PAMI, May 2010.
- [34] A. Trouve, L. Younes, 'Local Geometry of Deformable Templates', SIAM Journal on Mathematical Analysis. 2005. Volume: 37, Issue: 1.
- [35] M.Varma, A. Zisserman: 'A Statistical Approach To Material Classification Using Image Patch Exemplars'. IEEE Trans. on PAMI, 31(11):2032-2047, November 2009.
- [36] I. Waldspurger, S. Mallat 'Recovering the phase of a complex wavelet transform', CMAP Tech. Report, Ecole Polytechnique, 2012.
- [37] J.Wang, J.Yang, K.Yu, F.Lv, T.Huang, Y.Gong, 'Localityconstrained Linear Coding for Image Classification', CVPR 2010.
