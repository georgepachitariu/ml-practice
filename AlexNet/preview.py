def show_batch(image_batch, label_batch):
    nr_images = 20
    images = []
    with tf.Session().as_default():
        images = image_batch[:nr_images].numpy().astype(float)

    plt.figure(figsize=(20, 20))
    for n in range(nr_images):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(images[n] + 0.5)
        # plt.title(label_batch[n])
        plt.title(CLASS_NAMES[np.nonzero(label_batch[n].numpy())[0]][0])
        plt.axis('off')


def preview_augmentation():
    # we take only 1 to see how the augmentation worked
    prepared_train_ds = prepare(train_ds.take(1), is_train=True)
    image_batch, label_batch = next(iter(prepared_train_ds))
    show_batch(image_batch, label_batch)


# preview_augmentation()
prepared_train_ds = prepare(train_ds.take(1), is_train=True)
next(iter(prepared_train_ds))
# image_batch, label_batch = next(iter(prepared_train_ds))
# show_batch(image_batch, label_batch)


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    # plt.xlabel("{} {:2.0f}% ({})".format(CLASS_NAMES[predicted_label],
    #                            100*np.max(predictions_array),
    #                            CLASS_NAMES[np.nonzero(true_label.numpy())[0]][0]),
    #                            color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, np.argmax(true_label[i])
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 10
num_cols = 4
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions[i], label_batch, image_batch + 0.5)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions[i], label_batch.numpy())
plt.tight_layout()
plt.show()