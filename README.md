# capsule-theory-notes
Notes and resources for my understanding of Geoffrey Hinton's Capsule Theory

TODO
  Autoencoders
  Coordinate transforms
  More backprop knowledge
  Investigate github
  torch
  Steerable filters?
  Linear dynamical models?
  Keep doing coursera
  
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

Mental Rotation
  If we are checking to see if a handwritten R is correctly oriented (see example where R is upside down and rotated slightly), we have a matrix to represent the original, weird R, and a matrix of what an R should look like. We don't get sign(det(A)) of this matrix, as that would be a really heavy thing to compute. Instead, we mentally rotate and transform this image until it is the correct orientation, and then we can easily check if it matches. We use continuous matrix transformations.

Hierarchy of Parts
  So, if we know the position of a mouth in an image, and we know the position of a nose in an image, we can determine what the position of the face in the image should be using matrix part-to-whole transformations on each of these. We can then get two different ideas of the position of the face, a higher level entity, through the positions of the nose and mouth, lower level entities (through our aforementioned matrix transforms). If these higher level positions are in agreement / are close, then we know the higher level visual entity is present.
  
  By doing this, we have knowledge in our weights, and variance through viewpoint in the operations.
  
  If we know it is present, I believe we average the predicted positions together to get the result.
  
Factor Analysis
 
  If we were to make images out of five ellipses, like a face or a crude sheep, the shape is determined entirely by the spatial relations between the ellipses because all the parts have the same shape. So we can train a factor analysis model by saying what each ellipse in a face image might represent (left eye ellipse, right eye ellipse, etc), and then get a vector of 30 numbers (presumably 6 for each ellipse) to represent the entire image. We know the relationship between each of these ellipses won't change even if our image does. So we learn a linear model to use just a few numbers for the pose of the whole face to explain the pose of all the parts. 
  Using this, we can generate new samples from our model via extrapolation, with all sorts of sizes and rotations.

  To learn how to assign our vectors to the blocks in our factor analyzer / to train our factor analyzer, we can just iteratively try new combinations until we find the one that makes our factor analyzer happiest - the one that gives the smallest reconstruction error.
  
  By using this, we can recognize shapes even when they are extremely deformed. We can thus make massive generalizations in viewpoint.

From pixels to pose parameters
  So we know what we can do once we have pose parameters, but how can we get from pixels, the lowest/0th level, to the 1st level parts that output explicit pose parameters
  So we have to essentially get from rendered images back to their primitive parts with their poses, "de-render".
  
  We can do this with:

CAPSULES *epic orchestral music playing* 
  One capsule will have
    1. Bunch of recognition neurons at the front - logistic units
    2. These output the following
      a. x coordinate of capsule's visual entity (capsule's thing it likes to find)
      b. y coordinate of capsule's visual entity
      c. i - intensity of the capsule's visual entity
'
  So we want each capsule to go out and find different fragments of the image, but we want them to do so without knowing what the origin of the coordinate frame or the fragment should be; i.e. we want them to find different fragments without being given where to start and what to look for specifically, i.e. (4,15) -> nose.
  So now on to reconstructing the image. Each capsule outputs three values, and we need to decode this.

Decoding the result of capsules
  Domain specific decoders
    This is similar to the idea of un-max-pooling after we are given a pooled representation of an image.
    We say each capsule will learn a fixed template, and then each capsule is allowed to translate that template and scale the intensity (according to it's outputs), and then place that in the result image.

  So i'm pretty sure we encode the original image using the capsules, then we decode by adding together the intensity scaled and translated contributions from each capsule according to their outputs.

  Remember: each capsule is viewpoint specific so it is able to generalize very well to viewpoints

  So we concatenate all our outputs of our capsules together into one big vector (example 10 capsules = length 30 vector), and then, since each capsule has a different factor/part of the image, we run 10 factor analyzers, s.t. each factor analyzer looks for 10 factors in our length 30 vector.

  We can actually apply dropout to a bunch of capsules, however it will be quite computationally expensive since we have to infer a matrix of the whole from different factor parts each time. Fortunately, we can exclude the missing factors when determining this matrix, so that when we do a lot of dropout it actually becomes less computationally expensive than if we were to do just a little bit of dropout. 
  
Transforming Auto Encoders
  We can also learn the 0th level parts in another simple way: use pairs of images that are related by a given coordinate transform / transformation.
  This is intuitive from the fact that we know where our eye moved when we make an eye movement.
  So, given a pair of images related by a known transformation:
    1. Compute capsule outputs for the first image
      - each capsule uses its own set of recognition hidden units to extract the x and y coordinates and probability of existence of its visual entity
      - are these just ReLU neurons???
    2. Apply the known transformation to the outputs of each capsule
      -add delta x to each of our capsule x outputs and do the same for y outputs
    3. Predict transformed image from transformed outputs of the capsule, since each capsule uses its own set of generative hidden units to compute its contribution to the prediction and we can combine these to get a resulting prediction.
      -can we just combine them normally, by concatenation?
'
  We don't necessarily just add delta x, that would really only work for a very small case. Realistically, we would multiply by the corresponding transformation matrix.
  This actually works with 3d just fine because it is fueled by computer graphics methods, and transformations in 3d - which we already know how to do just fine.
  This is opposed to convolutional nets, where it's way harder to switch to 3d.
  
  Because of the way this works, Alex Krizhevsky actually trained networks like this to learn how to transform 3d images.
  
How can we use this for recognition though?
