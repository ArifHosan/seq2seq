# What is Encoder/ Decoder?

Encoder is 


# What is RNN?

Humans don’t start their thinking from scratch every second. As you read this essay, you understand each word based on your understanding of previous words.
 It represent words as dense vectors which allows sharing information between words. The representation is learned through 'back propagation'. 
 
 The Recurrent Neural Network (RNN) is a natural generalization of feedforward neural
networks to sequences. Given a sequence of inputs (x1, . . . , xT ), a standard RNN computes a
sequence of outputs (y1, . . . , yT ).


 
The RNN can easily map sequences to sequences whenever the alignment between the inputs the
outputs is known ahead of time. However, it is not clear how to apply an RNN to problems whose
input and the output sequences have different lengths with complicated and non-monotonic relationships.


A simple strategy for general sequence learning is to map the input sequence to a fixed-sized vector using one RNN, and then to map the vector to the target sequence with another RNN.  While it could work in principle since the RNN is provided with all the relevant information, it would be difficult to train the RNNs due to the resulting long term dependencies. However, the Long Short-Term Memory (LSTM) is known to learn problems with long range temporal dependencies, so an LSTM may succeed in this setting
 
# What is Attention?

# What is Tensor?

# What is epoch?

# What is BLEU? How is BLEU calculated?

Developers of machine translation systems need to monitor the effect of daily changes to their systems in order to evaluate the systems.To judge the quality of a machine translation, one measures its closeness to one or more reference human translations according to a numerical metric.

BLEU (bilingual evaluation understudy) is an algorithm for evaluating the quality of text which has been machine-translated from one natural language to another. Few translations will attain a score of 1 unless they are identical to a reference translation.

Translations may vary in word choice or in word order even when they use the same words. And yet humans can clearly distinguish a good translation from a bad one. For example, consider these two candidate translations of a Chinese source sentence:

``
Candidate 1: It is a guide to action which ensures that the military always obeys the commands of the party.
``

``
Candidate 2: It is to insure the troops
forever hearing the activity guidebook
that party direct.
``



# BLEU vs Accruacy.

# What is preprocessing/ postprocessing?

# How to improve accuracy? And most important, workflow of the code.

# What is LSTM?

LSTM (Long Sort Term Memory) architecture can able to solve general sequence to sequence problems. The idea of LSTM is to read the input sequence, one timestamp at a time and to obtain a fixed dimensional vector representation, and then to use another LSTM to extract the output sequence from that vector.

The second LSTM is essentially a recurrent neural network language model except that it is conditioned on the input sequence. The LSTM’s ability to successfully learn on data with long range temporal dependencies makes it a natural choice for this application due to the considerable time lag between the inputs and their corresponding outputs.

Surprisingly, the LSTM did not suffer on very long sentences, despite the recent experience of other researchers with related architectures. A useful property of the LSTM is that it learns to map an input sentence of variable length into a fixed-dimensional vector representation.


The goal of the LSTM is to estimate the conditional probability p(y1, . . . , yT′ |x1, . . . , xT ) where (x1, . . . , xT ) is an input sequence and y1, . . . , yT′ is its corresponding output sequence whose length T′ may differ from T. 

The LSTM computes this conditional probability by first obtaining the fixeddimensional
representation v of the input sequence (x1, . . . , xT ) given by the last hidden state of the
LSTM, and then computing the probability of y1, . . . , yT′ with a standard LSTM-LM formulation. But it is difficult to train long term dependecies in RNN. But LSTM can resolve this long term dependencies.



##### Reference
https://colah.github.io/posts/2015-08-Understanding-LSTMs/