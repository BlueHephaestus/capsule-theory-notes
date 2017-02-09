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
  
  By doing this, our matrices are viewpoint invariant, but the neural activities that represent the positions are highly variant, as they should be.
  
  So we want to both know how each part is related to the whole, and how each part is related to the camera. We want the first to be invariant, and the second to be equivariant.
 
  *A mental image specifies how each node is related to the viewer*

Notes on Mental Rotation
  If we are checking to see if a handwritten R is correctly oriented (see example where R is upside down and rotated slightly), we have a matrix to represent the original, weird R, and a matrix of what an R should look like. We don't get sign(det(A)) of this matrix, as that would be a really heavy thing to compute. Instead, we mentally rotate and transform this image until it is the correct orientation, and then we can easily check if it matches. We use continuous matrix transformations.

Hierarchy of Parts
  So, if we know the position of a mouth in an image, and we know the position of a nose in an image, we can determine what the position of the face in the image should be using matrix part-to-whole transformations on each of these. We can then get two different ideas of the position of the face, a higher level entity, through the positions of the nose and mouth, lower level entities (through our aforementioned matrix transforms). If these higher level positions are in agreement / are close, then we know the higher level visual entity is present.
  
  By doing this, we have knowledge in our weights, and variance through viewpoint in the operations.
  
  If we know it is present, I believe we average the predicted positions together to get the result.
  
