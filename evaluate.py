import torch
import random

from BLEU import calculate_bleu
from Helper import showAttention
from Util import tensorFromSentence, MAX_LENGTH, DEVICE, SOS_token, EOS_token, check_if_unk, word_dict


def evaluate(encoder, decoder, sentence, input_lang, output_lang, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=DEVICE)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, input_lang, output_lang, pairs, n=10):
    total_bleu = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang,output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        output_words.pop(-1)
        total_bleu += calculate_bleu(pair[1], ' '.join(output_words))
        print('')
    print("Average BLEU: ",total_bleu/n)

def evaluateAndShowAttention(input_sentence, encoder1, attn_decoder1, input_lang, output_lang):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence, input_lang, output_lang)

    replace_index = 0
    if len(output_words) == 4:
        replace_index = 1
    elif len(output_words) > 4:
        replace_index = 2

    unk = check_if_unk(input_lang, input_sentence)
    if unk != '' and unk in word_dict:
        output_words[replace_index] = word_dict[unk]
    elif unk != '':
        output_words[replace_index] = "UNK"

    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)


def evaluate_all_test(encoder, decoder, input_lang, output_lang, pairs):
    total_bleu = 0
    n = len(pairs)
    for i in range(n):
        pair = pairs[i]
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0], input_lang,output_lang)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        output_words.pop(-1)
        total_bleu += calculate_bleu(pair[1], ' '.join(output_words))
        print('')
    print("Average BLEU: ",total_bleu/n)