model = Transformer(
    input_vocab_size=len(integer_input.word_counts),
    target_vocab_size=len(integer_output.word_counts),
    encoder_input_size=20,
    decoder_input_size=320
)

sentence = x_test[0]


start_token = integer_output.word_index["[START]"]
end_token = integer_output.word_index["[END]"]

def translate(sentence):

    sentence = re.sub(r'\s*\([^)]*\)$', '', sentence)

    encoder_input = integer_input.texts_to_sequences([sentence])

    encoder_input = pad_sequences(
        encoder_input,
        maxlen=20,
        padding="post"
    )

    decoder_input = np.zeros((1, 320), dtype=np.int32)

    decoder_input[0, 0] = start_token

    for i in range(319):

        pred = model.predict(
            [encoder_input, decoder_input],
            verbose=0
        )

        next_token = np.argmax(pred[0, i, :])

        decoder_input[0, i + 1] = next_token

        if next_token == end_token:
            break

    words = []

    for token in decoder_input[0]:

        if token == 0:
            continue

        word = integer_output.index_word.get(int(token), "")

        if word in ["start", "end", "[START]", "[END]"]:
            continue

        words.append(word)

    return " ".join(words)