import alexnet
import numpy as np
import matplotlib.pyplot as plt

label_dict = alexnet.Data.load_labelid_to_names()

def get_top_k_predictions(batch_predictions, k=10):
    # This pivots all elements around the position -k (like in quicksort)
    # and this makes the highest k elements be on the right side of the result
    topk_label_ids = np.argpartition(batch_predictions, -k)[-k:]
    
    topk_label_names = [label_dict[label_id] for label_id in topk_label_ids]
    topk_label_probabilities = batch_predictions[topk_label_ids]
    
    return topk_label_names, topk_label_probabilities


def display_images_with_predictions(ds, model):
    image_batch, label_batch=next(iter(ds))
    batch_predictions = model.predict(image_batch)    

    image_batch_denormalized = alexnet.Preprocessing.denormalize(image_batch) 

    nrows=1 
    ncols=4
    plt.figure(figsize=(16,4))
    
    for i, (image, label, prediction) in \
            enumerate(zip(image_batch_denormalized, label_batch, batch_predictions)):
        # image
        index=i+1
        plt.subplot(nrows, ncols, i*2+1)
        plt.imshow(image.numpy().astype(np.uint8))         
        plt.title(label_dict[label.numpy()])
        
        # predictions
        ax = plt.subplot(nrows, ncols, i*2+2)
        topk_label_names, topk_label_probabilities = get_top_k_predictions(prediction)
        
        plt.yticks([0, max(topk_label_probabilities)])
        plt.ylim([0, max(topk_label_probabilities)]) # the value for an empty bar and a full bar   
        
        thisplot = plt.bar(x=range(10), height=topk_label_probabilities, 
                           tick_label=topk_label_names, color="#777777")
        plt.xticks(rotation=90)
        
        if i*2+2 >= nrows*ncols: break

    