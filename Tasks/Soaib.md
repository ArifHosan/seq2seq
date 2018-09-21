# What is Encoder/ Decoder?

The Encoder-Decoder architecture with recurrent neural networks has become an effective and standard approach for both neural machine translation (NMT) and sequence-to-sequence (seq2seq) prediction in general.

As the name suggests, encoder-decoder models consist of two parts: an encoder and a decoder. The encoder network is that part of the network that takes the input sequence and maps it to an encoded representation of the sequence. The encoded representation is then used by the decoder network to generate an output sequence.

Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.
Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.
	
We feed in the input sequence into the encoder which will generate a final hidden state that will feed into a decoder. The final hidden state from the encoder is the new initial state for the decoder. We use the decoder outputs with softmax and compare it to the targets to calculate our loss.


# What is RNN?

A recurrent neural network (RNN) is a class of artificial neural network where connections between nodes form a directed graph along a sequence. This allows it to exhibit temporal dynamic behavior for a time sequence. This is a neural network that processes sequential data, and takes in as input both the new input at the current timestep and the output (or a hidden layer) of the net in the previous timestep. The most popular type of RNN is probably the LSTM, which has a ‘cell state’ at each time step that changes with new input.

Unlike feed-forward neural networks, recurrent neural networks have a backward connection between hidden layers. Therefore, they have some kind of memory in them.


# What is Attention?

Attention is a mechanism that was developed to improve the performance of the Encoder-Decoder RNN on machine translation.

A potential issue with this encoder–decoder approach is that a neural network needs to be able to compress all the necessary information of a source sentence into a fixed-length vector. This may make it difficult for the neural network to cope with long sentences, especially those that are longer than the sentences in the training corpus.

Instead of decoding the input sequence into a single fixed context vector, the attention model develops a context vector that is filtered specifically for each output time step.


# What is Tensor?

A tensor is a generalization of vectors and matrices and is easily understood as a multidimensional array. In the general case, an array of numbers arranged on a regular grid with a variable number of axes is known as a tensor.

A vector is a one-dimensional or first order tensor and a matrix is a two-dimensional or second order tensor.

Tensor notation is much like matrix notation with a capital letter representing a tensor and lowercase letters with subscript integers representing scalar values within the tensor.


# What is epoch?

In the neural network terminology:

- one epoch = one forward pass and one backward pass of all the training examples
- batch size = the number of training examples in one forward/backward pass. The higher the batch size, the more memory space you'll need.
- number of iterations = number of passes, each pass using [batch size] number of examples. To be clear, one pass = one forward pass + one backward pass (we do not count the forward pass and backward pass as two different passes).

###### Example: if you have 1000 training examples, and your batch size is 500, then it will take 2 iterations to complete 1 epoch.	


# What is BLEU? How is BLEU calculated?

BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality
of text which has been machine-translated from one natural language to another. Quality is considered to be the correspondence between a machine's output and that of a human: "the closer a machine translation is to a professional human translation, the better it is" – this is the central idea behind BLEU. 

The approach works by counting matching n-grams in the candidate translation to n-grams in the reference text, where 1-gram or unigram would be each token and a bigram comparison would be each word pair. The comparison is made regardless of word order.

The primary programming task for a BLEU implementor is to compare n-grams of the candidate with the n-grams of the reference translation and count the number of matches. These matches are position-independent. The more the matches, the better the candidate translation is.

A perfect score is not possible in practice as a translation would have to match the reference exactly. This is not even possible by human translators. The number and quality of the references used to calculate the BLEU score means that comparing scores across datasets can be troublesome.


# BLEU vs Accruacy.

# What is preprocessing/ postprocessing?

# How to improve accuracy? And most important, workflow of the code.