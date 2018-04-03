# Papers

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

* [Pointing the Unknown Words] (https://arxiv.org/abs/1603.08148)
    * Proposed method: Attention based model with two softmax layers to deal with rare/unknown words
    * Baseline: Neural Translation Modeul with attention
    * Pointer Softmax (PS):
        * can predict whether it is necessary to use the pointing
	* can point any location of the context sequence (length varies)
	* Two softmax Layer :
	    * Shortlist Layer: Typical softmax layer with shortlist
	    * Location Layer: Pointer network, points location in the context
	* At each time step, if the model chooses the shortlist layer a word is
	generated; if the model chooses the location layer a context word's location is obtained.
	    * A switching network to decide with layer to use, a binary variable trained MLP
    * Experiments: Summarization with Gigaword, Translation wirh Europarl
    * Slight imporevements on both tasks

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


### Squad

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

# Vision/Image

* [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf)
    * proposed to handle degragation problem (deeper networks have higher training error)
    * Residual Learning:
        * rather than expect stacked layers to approximate H(x), we explicitly let these
	layers approxinmate a residual function F(x)=H(x)-x
	* If the identity mappings are optimal, the solvers may simply drive the weights
	of the multiple nonlinear layers toward zero to approach identity mappings.
    * Experiments: compared with PlainNet and ResNet
        * ImageNet dataset: 34-layer Resnet is better than 18Layer Resnet and PlainNet
	    * 18Resnet converges faster than 18PlainNet
	* CIFAR 10, Pascal, MSCOCO
    * 1st place in ILSVCR (2015)