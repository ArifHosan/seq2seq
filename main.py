#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Util import *
from AttnDecoderRNN import AttnDecoderRNN
from EncoderRNN import EncoderRNN
from evaluate import evaluateRandomly, evaluate, evaluateAndShowAttention
from trainer import trainIters

input_lang, output_lang, pairs = prepareData('eng', 'ben', False)
# print(random.choice(pairs))

hidden_size = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(DEVICE)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(DEVICE)
#trainIters(encoder1, attn_decoder1, input_lang, output_lang, pairs, 1000000, print_every=100)

#save_model_param(encoder1,"model/en_sust")
#save_model_param(attn_decoder1, "model/de_sust")
#save_model(encoder1,"model/en_final_data")
#save_model(attn_decoder1,"model/de_final_data")

encoder1 = load_model("model/en_final_data")
attn_decoder1 = load_model("model/de_final_data")

#evaluateRandomly(encoder1, attn_decoder1, input_lang, output_lang, pairs,100)
#output_words, attentions = evaluate(encoder1, attn_decoder1, "you worried ?", input_lang, output_lang)
#plt.matshow(attentions.numpy())
#plt.savefig("plots/attentions")

evaluateAndShowAttention("i don't agree with you .", encoder1,attn_decoder1,input_lang,output_lang)
# evaluateAndShowAttention("take good care of yourself .", encoder1,attn_decoder1,input_lang,output_lang)
# evaluateAndShowAttention("i accepted her invitation .", encoder1,attn_decoder1,input_lang,output_lang)
# evaluateAndShowAttention("we made mistakes .", encoder1,attn_decoder1,input_lang,output_lang)
# evaluateAndShowAttention("don't speak ill of others", encoder1,attn_decoder1,input_lang,output_lang)