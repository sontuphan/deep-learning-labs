from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string], '')
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()


def main():

    dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    encoder = info.features['text'].encoder

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.padded_batch(
        BATCH_SIZE, train_dataset.output_shapes)

    test_dataset = test_dataset.padded_batch(
        BATCH_SIZE, test_dataset.output_shapes)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(encoder.vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    history = model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset,
                        validation_steps=30)

    plot_graphs(history, 'accuracy')

    test_loss, test_acc = model.evaluate(test_dataset)
    print('Test Loss: {}'.format(test_loss))
    print('Test Accuracy: {}'.format(test_acc))

    # # predict on a sample text with padding

    # sample_pred_text = ('The movie was cool. The animation and the graphics '
    #                     'were out of this world. I would recommend this movie.')
    # predictions = sample_predict(sample_pred_text, pad=True)
    # print(predictions)

    # def pad_to_size(vec, size):
    #     zeros = [0] * (size - len(vec))
    #     vec.extend(zeros)
    #     return vec

    # def sample_predict(sentence, pad):
    #     encoded_sample_pred_text = encoder.encode(sample_pred_text)

    #     if pad:
    #         encoded_sample_pred_text = pad_to_size(
    #             encoded_sample_pred_text, 64)
    #     encoded_sample_pred_text = tf.cast(
    #         encoded_sample_pred_text, tf.float32)
    #     predictions = model.predict(
    #         tf.expand_dims(encoded_sample_pred_text, 0))

    #     return (predictions)


if __name__ == "__main__":
    main()
