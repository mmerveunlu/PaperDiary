# Papers

* [HyperNetworks](https://arxiv.org/pdf/1609.09106.pdf)
    * hypernetworks generate the weights of a larger network
    * takes a set of inputs that contain information about the structure of the weights
    and generates the weight of the layer
    * Methods: hypernetworks for deep convolutional networks, dynamic hypernetwors for RNN
    * ExperimentsL: CIFAR10, Penn Treebank Language modelling, Hutter Prize Wikipedia LM, Handwriting sequence learning, Neural ML

## NLP 
* [ Pointer Networks](http://papers.nips.cc/paper/5866-pointer-networks.pdf)
    * PtrNet is a variation of sequence-to-sequence models with attention
    * Baseline: seq2seq and input-attention models
    * The output of PtrNet is discrete and the length depends on the input's length
    * Problems: Convex Hull, Delaunay Triangulation, TSP
    * Additional Info: [Introduction to pointer networks](http://fastml.com/introduction-to-pointer-networks/) 

 
### Machine translation 
* [ Neural  machine  translation  by jointly  learning  to  align  and  translate](https://arxiv.org/pdf/1409.0473.pdf)
    * Previous models: RNN encoder-decoder
        * An encoder reads the input into a vector c
	* A decoder predicts the next word given the context vector c and previously generated words
    * Proposed model: Bidirectional RNN encoder-decoder
        * Encoder: BiRNN. Annotation of each word is the concatenation of the forward and backward hidden states
        * Decoder: BiRNN.
	      * The probability is conditioned on a distinct vector c_i for each target word y_i
	      * A vector c_i depends on a sequence of annotations, each annotation contains information about the whole input sequence
	      * Alignment model: score is based on the RNN hidden state and the annotation of the input sentence
    * Experiment: WMT'14 English-French

* [ Pointing the Unknown Words ](https://arxiv.org/abs/1603.08148)
    * Proposed method: Attention based model with two softmax layers to deal with rare/unknown words
    * Baseline: Neural Translation Modeul with attention
    * Pointer Softmax (PS):
        * can predict whether it is necessary to use the pointing
        * can point any location of the context sequence (length varies)
        * Two softmax Layer :
	      * Shortlist Layer: Typical softmax layer with shortlist
	      * Location Layer: Pointer network, points location in the context
        * At each time step, if the model chooses the shortlist layer a word is generated; if the model chooses the location layer a context word's location is obtained.
        * A switching network to decide with layer to use, a binary variable trained MLP
    * Experiments: Summarization with Gigaword, Translation wirh Europarl
    * Slight improvements on both tasks

### Text Summarization

* [ A Neural Attention Model for Abstractive Sentence Summarization](https://arxiv.org/abs/1509.00685)
    * Model generates each word of the summary conditioned on the input sentence.
    * From all possible summaries, model finds the summary that have the max probability
    given that previously generated words and input.
    * Model consists of three components: Neural language model,encoder and summary generator
        * Neural Language Model: adapted from standard feed-forward LM.
	    * Encoder: Attention-based encoder
	    * Generation: beam-search
        * Training: loss function is negative log likelihood
    * Experiments: DUC 2003-2004, Gigaword


* [ Abstractive Sentence Summarization with Attentive Recurrent Neural Networks](http://www.aclweb.org/anthology/N16-1012)
    * A convolutional attention-based conditional RNN
    * The model called Recurrent Attentive Summarizer (RAS)
    * The model can be seen as an extension of Rush et al. 2015 (ABS)
        * Different from ABS, the encoder is a RNN.
    * RAS has a recurrent decoder,an attentive encoder and a beam search
    * Dataset: Gigaword, DUC 2004
    * RAS achieves better results than ABS

* [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
    * Pointer generator model with coverage
    * Baseline is sequence to sequence attention model
    * The PointerGenerator model is a hybrid between the baseline and the pointer networks
    * The covarge is added to overcome the repetition problem of seq2seq models
    * Dataset: CNN/Daily Mail
    * PointerGenerator is better than baseline and the abstractive models, but extractive models are still better on Rouge score.
    
### Question-Answering / Reading Comprehension

* [R-NET: Machine Reading Comprehension with Self-matching Networks ](https://www.microsoft.com/en-us/research/publication/mrc/)
    * An end-to-end neural network for reading comprehension and question answering
    * Model consists of :
        * A RNN encoder to build the representation for questions and passage: biRNN with GRU, also
	        character and word embedding concatenated for vector representations
        * Gated attention-based RNN: match the question and passage,generates question-aware passage representation
        * Self-matching attention: matches the question-aware passage representation to itself again
        * Output layer: the pointer network to predict the boundary of the answer in the passage
    * Training: Initialization with Glove, 1-layer biGRU for character embeddings, 3-layer RNN for word embeddings
    * Datasets: Squad, MS-Marco

* [Attention-over-Attention Neural Networks for Reading Comprehension(2017)](https://arxiv.org/pdf/1607.04423.pdf)
    * Task: Cloze-stype RC, there are triples as (Document,Query,Answer)
    * Attention over document-level attention
    * Model consists of:
        * Embedding: shared with document and query, biRNN, GRU
	* Matching between context vectors: Pairwise mathing between one document word and one query word by dot product
	* Document-to-query attention: column-wise softmax applied on the matching matrix
	* Query-to-document attention: row-wise softmax applied on the matching matrix
	* N-best reranking
    * Datasets: CNN/Daily Mail, Children's book RC

* [Machine Comprehension using MatchLSM and Answer Pointer(2017)](https://arxiv.org/pdf/1608.07905.pdf)
    * Data: A passage, embedded into dxP matrix where P is the length of the passage. A question, embedded into dxQ matrix where Q is the length of the question and d is the embedding dim. The answer can be : 
        * A sequence of integers which indicate the positions of the answer's words in the passage ==> Sequence Model
        * Two integeres which indicate the start and end positions of the answer in the passage ==> Boundary Model
    * Model has 3 layers: LSTM preprocessing (for embeddings), Match LSTM (shows the degree of matching between a token of a passage and a token of the question), Answer Pointer (based PointerNetworks)
    * Experiments: Initialization with glove, embeddings are not learned 
    * Dataset: Squad  ==> Results: F1 77%, exact Match 67.6%
    
# Vision/Image

* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    * proposed to handle degradation problem (deeper networks have higher training errors)
    * Residual Learning:
        * rather than expect stacked layers to approximate H(x), desired underlying mappings,
	we explicitly let these layers approximate a residual function F(x)=H(x)-x
        * If the identity mappings are optimal, the solvers may simply drive the weights
	of the multiple nonlinear layers towards zero to approach identity mappings.
    * Experiments: ImageNet
        * Comparison between PlainNet (18 and 34 layers) and ResNet (18 and 34 Layers )
        * 34Layer Resnet is better than 18Layer Resnet
        * 18Layer Resnet converges faster than 18Layer PlainNet
        * 34Layer Plain Net is worse than 18 Layer Plain Net
    * Experiments: CIFAR10, MSCOCO,PASCAL
    * 1st place in ILSVRC (2015)

* [Highway Networks](https://arxiv.org/pdf/1505.00387.pdf)
    * The proposed method enables the optimization of the deep networks with a learned gating mechanism for regulating the information flow.
    * Additionally added two nonlinear transforms to the input, called Transform gate and Carry gate, shows how much of the input is transformed and carried to form the output
    * Experiments: MNIST, CIFAR 
        * Optimization: Highway is better than plain network when depth is increased
        * Classification: Highway is easier to train than FitNets  
    
* [Densely Connected Convolutional Networks](http://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.pdf)
    * Proposed method DenseNets
    * connects all layers directly each other to ensure maximum information flow
    * Each layer obtains additionnal inputs from all preceding layers and passes its own features maps to all others
    * For x_l is the output of the layer l, x_l = H_l([x_0x_1...x_(l-1)]) where H_l is the non-linear transformation
        * In DenseNets: H_l is the composite of thre functions: Batch Normalization, Relu and 3x3 ConvNet
    * Experiments: CIFAR, SVHN, ImageNet
        * state-of-the-art results in CIFAR10 and CIFAR100
        * DenseNets perform better as L and k increase (L number of layers, k number of feature maps)
    * Conclusion: Deep Supervision (??)
