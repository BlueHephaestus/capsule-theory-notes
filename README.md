# capsule-theory-notes
Notes and resources for my understanding of Geoffrey Hinton's Capsule Theory

Geoffrey Hinton AMA https://www.reddit.com/r/MachineLearning/comments/2lmo0l/ama_geoffrey_hinton/ and http://www.kdnuggets.com/2014/12/geoffrey-hinton-talks-deep-learning-google-everything.html

  Talk on Inverse Graphics - https://www.youtube.com/watch?v=TFIMqt0yT2I and http://cseweb.ucsd.edu/~gary/cs200/s12/Hinton.pdf
  
Thesis on Generating Images http://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf

From this thesis some MIT guys made these
  https://www.reddit.com/r/MachineLearning/comments/35tqvg/understanding_optimizing_neural_networks_that/
  http://willwhitney.github.io/dc-ign/www/
  https://github.com/mrkulk/Unsupervised-Capsule-Network
  
Torch, apparently all the smart institutions and people are using this, so I guess it's time to; wow holy shit this is actually awesome: http://torch.ch/


Notes on Geoffrey Hinton's Inverse Graphics lecture

Equivariance vs. Invariance
  Invariant - the same, regardless of transformations
  Equivariant - different, but with respect to the transformations
  Variant - different in very different ways wrt the transformations
  
  In convnets, we use maxpooling/subsampling to try to make the network invariant for small changes in viewpoint. This is motivated by the idea that the final label needs to be viewpoint-invariant, i.e. an apple is still an apple if it I move it a bit on a desk or get a closer picture of it. 
  
  However, this is wrong, because we do need to care about how it moves and changes in viewpoint to understand it. We need them to be Equivariant.
  
Representing Images Correctly
  The way to do this is to learn features via a generative model.
  
  Since computer vision is inverse computer graphics (graphics = data to image, cv = image to data), the higher levels of a vision system should look like the representations used in graphics.
  
  So, graphics programs use hierarchial models in which the spatial structure is modeled by matrices that represent the transformation from a coordinate frame embedded in the whole image to a coordinate frame embedded in each part of the image. 
  
  This is what we want the higher levels of our vision system to look like. These matrices are completely viewpoint invariant, while also yielding a representation that makes it easy to reverse the process, where we could for example compute the relationship between a part and the retina (in an example of a face) from the relationship between a whole (whole face image, for instance) and the retina.
  
  So it's a matrix multiply, much like how we have a transformation matrix to do linear transforms in linear algebra.
  
