# -*- coding: utf-8 -*-
"""6_Understanding_EmbeddingLayers.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OmVRKWN-hUdHuwLQhYxx5VOvWBwrp004

# <Font color = 'indianred'>**Embeddings in NLP**</font>

**Introduction to Embeddings**

In Natural Language Processing (NLP), embeddings are a pivotal concept used to translate words, phrases, or entire documents into numerical vectors. This translation is essential because machine learning models, especially neural networks, can't process raw text. Instead, they require numerical input for various tasks like classification, sentiment analysis, or language translation.

**The Concept of Word Embeddings**

Word embeddings are a specific type of embedding that converts individual words into vectors of real numbers.

- **Why Use Embeddings?**
  - Machine learning models can't understand text, so embeddings convert words into a format these models can work with.
  - They capture semantic relationships between words, meaning that words with similar meanings will have similar representations.


**Simple Word Embeddin Example**

The accompanying table and graph provide a tangible illustration of word embeddings. In the table, words like "apple," "banana," "car," and others are represented by vectors in a 2-dimensional space - Dimension 1 and Dimension 2. This simplification helps visualize how embeddings work.

In the graph, each word is plotted according to its vector values. Words with similar meanings or contexts tend to be closer together. For instance, "bus" and "truck," both modes of transportation, are plotted near each other. This proximity signifies their semantic similarity in the embedding space.

<table>
<tr>
<td>
<td><img src='https://drive.google.com/uc?export=view&id=1HKk6DfRx0Q8lhyzVblH9OCh_bVxUsl1y' width='500'/></td>
</td>
<td><img src='https://drive.google.com/uc?export=view&id=1Tk0jfFVKV2moEljHVzCZofjSRr9hr7Wx' width='500'/></td>
</tr>
</table>

 *In real-world NLP tasks, embeddings usually have a much higher dimension (often in the hundreds) to capture the complex and nuanced relationships between words*. High-dimensional embeddings can represent an extensive range of semantic and syntactic word properties.

**Learning and Using Embeddings**
- **Embedding Layers in Neural Networks**: In deep learning models, embedding layers are used to learn word representations. These layers can be initialized randomly and then learned from data during training.
- **Pre-trained Embeddings**: Often, pre-trained embeddings like Word2Vec, GloVe, or BERT embeddings are used, which have been trained on large text corpora and can capture rich language semantics.

**Conclusion**

Embeddings are a foundational technique in NLP, enabling machines to process and understand textual data. Through the course, we will explore different ways to learn and use embeddings, starting with understanding basic embedding layers in Pytorch `nn.Embedding` and `nn.EmbeddingBag`.

# <Font color = 'indianred'>**PyTorch Embedding Layers**

`nn.Embedding` and `nn.EmbeddingBag` are two modules provided by PyTorch, a popular library for machine learning and neural network research. These modules are used for dealing with embeddings in neural networks. Here's a brief description of each:

1. <Font color = 'indianred'> **nn.Embedding**
  - `nn.Embedding` in PyTorch is a simple lookup table that stores embeddings of a fixed dictionary and size. It serves as a layer in neural networks specifically for handling embeddings.
  - This module maps indices of words in a vocabulary to their corresponding embedding vectors.
  - Upon initialization, `nn.Embedding` creates a weight matrix of size `V x D`, where `V` is the size of the vocabulary (number of unique words) and `D` is the dimensionality of the embeddings. These weights are learnable parameters that the neural network will adjust during training.
  - The primary arguments for `nn.Embedding` are the size of the dictionary (V) and the size of each embedding vector (D). For instance, `nn.Embedding(10000, 300)` creates an embedding layer for a vocabulary of 10,000 words, each represented by a 300-dimensional vector.
  - When the embedding layer receives an input index (or indices), it outputs the corresponding embedding vector (or vectors) from this matrix. Essentially, the layer transforms the input indices into dense vectors that capture the semantic properties of the words.
  - `nn.Embedding` is commonly used in various natural language processing tasks, enabling models to process words (tokens) as meaningful vectors and learn rich representations of language in their hidden layers.


   <img src='https://drive.google.com/uc?export=view&id=1HLcqG5ZI7lQEFv3X2f2VzA7JSpO2PYfq' width='800'/>



2. <Font color = 'indianred'>**nn.EmbeddingBag**
   - `nn.EmbeddingBag` is similar to `nn.Embedding` but with an additional feature: it computes the mean or sum of 'bags' of embeddings, without instantiating the intermediate embeddings.
   - This module is particularly useful for efficiently computing the embeddings for a dataset with varying-length sequences of indices, such as sentences in a paragraph.
   - It's optimized for certain use cases where you want to aggregate the embeddings of multiple items (like words in a sentence). For instance, it's beneficial in applications like text classification, where you might want to take the average of all word embeddings in a sentence as its representation.
   - `nn.EmbeddingBag` also takes the size of the dictionary and the size of each embedding vector as its primary arguments.

In summary, while both `nn.Embedding` and `nn.EmbeddingBag` are used for handling embeddings in PyTorch, `nn.Embedding` is more straightforward and suitable for individual word embeddings, whereas `nn.EmbeddingBag` is optimized for aggregating embeddings over a sequence of data, making it more efficient for certain tasks.

# <Font color = 'indianred'>**Understanding Embedding Layers with a simple example**

# <Font color = 'indianred'>**Understanding Embedding Layers with a simple example**

**Objective**: Understanding the inputs and outputs of nn.Embedding and nn.EmbeddingBag Layers

The embedding layers  in neural networks is essential for converting words in a vocabulary into dense vectors. We will understand two important embedding layers in Pytorch.

1. nn.Embedding() - useful in scenarios where we need word level embeddings
2. nn.EmbeddingBag - useful in scenarios (like sentiment analysis) where we can use average of word level vectors to represent documents (e.g. reviews, posts etc.)

**Plan**:
1. Set Environment
2. Load Data
3. Create Dataset
4. Create Vocab: We need to create vocab (dictionary) to create mapping of words to numbers (indices).
5. Understanding nn.Embedding() layer
  1. Preparing inputs for nn.Embedding layer: Typically the process of creating batch of inputs is handled by DataLoaders. Since, the inputs is now text we need to pass a function to dataloader that can convert the text into indices based on predefined vocab. In this section, we will learn how to create this function (this function is typically referred to as collate function) and use this function in creating Dataloaders.
  2. Instantiate embedding layer
  3. Understand the output of embedding layer
6. Understanding nn.EmbeddingBag layer()
  1. Preparing inputs for nn.EmbeddingBag layer: Change the collate function to meet nn.Embeddingbag layer input requirements.
  2. Instantiate embedding layer
  3. Undertand the output of embedding layer

## <font color = 'indianred'> **1. Set Environment**
"""

# Torchtext is a PyTorch library specifically designed for text processing tasks.
if 'google.colab' in str(get_ipython()):
  !pip install torchtext -q

import torch
import torch.nn as nn
import pandas as pd

from torchtext.vocab import vocab
from collections import Counter

from torch.utils.data import Dataset, DataLoader

"""* Import `vocab` from `torchtext`, which is a PyTorch library specifically designed for text processing tasks. The `vocab` class from torchtext is used to create a vocabulary object that maps tokens (words) to indices (numbers).
* `Counter` is used to count elements in collections like lists or tuples. It's particularly useful for counting frequency of tokens in text, which is a crucial step in building a vocabulary.
* Dataset is an abstract class for representing a dataset.

## <Font color = 'indianred'>**2. Load Data**
"""

# data example
data = {
    "label": [0, 1, 1, 0],
    "data": [
        "Movie was bad",
        "Movie was good",
        "It was thrilling.",
        "It was horrible. "
    ]
}

df = pd.DataFrame(data)

df.head()

"""## <Font color = 'indianred'>**3. Create Dataset** </font>"""

X = df['data']
y = df['label']

class CustomDataset(Dataset):
    """
    Custom Dataset class inheriting from PyTorch's Dataset class.
    Intended to handle custom text and label data.

    Attributes:
        X (pd.Series): The input features (text).
        y (pd.Series): The labels corresponding to the input features.
    """

    def __init__(self, X, y):
        """
        Initialize the dataset with input features and labels.

        Parameters:
            X (pd.Series): Input features.
            y (pd.Series): Labels corresponding to input features.
        """
        self.X = X  # Input features (text)
        self.y = y  # Corresponding labels

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.X)  # Return the length of the dataset

    def __getitem__(self, idx):
        """
        Fetch and return a single sample from the dataset at the given index.

        Parameters:
            idx (int): Index of the sample to fetch.

        Returns:
            tuple: A tuple containing the label and the input feature (text) at the index.
        """
        text = self.X.iloc[idx]  # Fetch the input feature at the given index
        labels = self.y.iloc[idx]  # Fetch the corresponding label
        sample = (labels, text)  # Create a tuple of label and input feature

        return sample  # Return the sample as a tuple

# Create an instance of CustomDataset with the input features X and labels y for training
train_dataset = CustomDataset(X, y)

# Retrieve the sample at index 2 from train_dataset using the __getitem__ method
train_dataset.__getitem__(2)

# Retrieve the sample at index 2 from train_dataset using Python's built-in indexing syntax
train_dataset[2]

"""* `train_dataset[2]` internally calls the `__getitem__` method.

## <Font color = 'indianred'>**4. Create Vocab**
Creating a vocabulary is a crucial step in the process of handling text data for NLP tasks. A vocabulary in this context is a mapping between words and their corresponding indices. It forms the basis for encoding text data into numerical format that can be processed by machine learning models. To build an effective vocabulary, not only do we need to identify unique words in our dataset, but we also often need to consider the frequency of each word. This frequency information can be instrumental in filtering out rare words or limiting the size of the vocabulary for efficiency.

To begin constructing our vocabulary, we start with creating a Counter object. A Counter is a specialized dictionary provided by Python's collections module, used for counting hashable objects. In our case, it's used to keep track of the frequency of each word in the dataset.
"""

# Initialize an empty Counter object to hold the word frequencies
counter = Counter()

# Loop through each sample in train_dataset to count word occurrences
for (label, line) in train_dataset:
    # Split the line into words and update their frequencies in the counter
    counter.update(str(line).split())

"""- `counter = Counter()`: An empty Counter object is initialized. This object will keep track of the frequency of each word in the dataset.

- The `for` loop (`for (label, line) in train_dataset:`) iterates through each sample in `train_dataset`. `train_dataset` is assumed to be a collection of text samples along with their labels.

- Inside the loop, each text sample (referred to as `line`) is split into words using `str(line).split()`. The `split()` method divides the text into individual words.

- `counter.update(...)`: The Counter object is updated with the words from each line. This step counts the occurrences of each word, aggregating these counts across all samples in the dataset.

By the end of this process, the Counter object holds a comprehensive count of all words in the dataset, which serves as the foundational step in creating a vocabulary.

"""

counter

# Create a vocabulary using the word frequencies stored in the counter, with a minimum frequency of 1 for inclusion
my_vocab = vocab(counter, min_freq=1)

"""- `vocab` initializes a vocabulary object using the word frequencies gathered so far.
- Words are included in the vocabulary only if their frequency is at least 1, as specified by the `min_freq` parameter.
"""

# Retrieve the word-to-index mapping from the my_vocab object
my_vocab.get_stoi()

# Insert the '<unk>' token at index 0 in my_vocab to represent any unknown words
my_vocab.insert_token('<unk>', 0)

"""- In natural language processing (NLP), it's common to encounter words in the test data that were not present in the training data. These words are unknown to the model's vocabulary (often referred to as 'out-of-vocabulary' words).
- To handle such cases, NLP models often include a special token in the vocabulary, typically `<unk>`, which stands for 'unknown'. This token is used to represent any word not found in the vocabulary.
- The above code adds a special token `<unk>` to the vocabulary at index 0.
"""

# check mapping of words to index
my_vocab.get_stoi()

# Let us convert a sentence to a number vector by replacing words to their respective indices
[my_vocab[token] for token in 'Movie was bad'.split()]

"""- First, the sentence is `split` into words `(['Movie', 'was', 'bad'])`, and then each word is looked up in `my_vocab` to find its corresponding index."""

# check whether word hello is in dictionary
'hello' in my_vocab

# get the index for  the word hello
# since this word is not in the dictionary we should get an error
try:
    my_vocab['hello']
except RuntimeError:
    print('token not found in vocab')

"""* `try:` This is the beginning of a try-except block. The code inside the try block is the part that might raise an exception.
* `my_vocab['hello']` attempts to access the value associated with the key 'hello' in the dictionary `my_vocab`.
* `except RuntimeError:` specifies what should happen a RuntimeError exception is raised. If any other type of exception occurs, it won't be caught by this except block.
* `print('token not found in vocab')`: If a RuntimeError exception is raised during the execution of the code inside the try block (e.g., if the key 'hello' is not found in my_vocab), then this line of code will be executed. It prints the message 'token not found in vocab' to the console.
"""

# set the default index to zero
# thus any uknown word will be represented b index 0 or token '<unk>'
my_vocab.set_default_index(0)

"""* `set_default_index(0)` sets the default index for any word that is not present in `my_vocab` as `0`, which corresponds to the special `<unk>` token.
- This ensures that the text-to-index conversion process is robust and error-free. It can handle any text input, even if it contains words not seen during the training phase.
"""

# get the index for  the word hello
# since we set default index to 0, now it should return 0 for the word hello
my_vocab['hello']

"""## <Font color = 'indianred'>**5. Understanding `nn.Embedding()` Layer**

### <Font color = 'indianred'>**5.1. Preparing and Understanding Inputs for Embedding layer** </font>

<Font color = 'indianred'>*Role of the collate Function* </font>

The `collate` function serves a vital role in PyTorch's data handling pipeline, particularly when working with batched data in machine learning models. Its primary responsibilities include not only aggregating individual data samples into coherent batches but also performing crucial preprocessing steps at the batch level. This functionality is especially important in scenarios where preprocessing the entire dataset into tensors beforehand is impractical or inefficient. Key aspects of the `collate_batch` function's role include:

1. *Dynamic Preprocessing*: The function allows for dynamic preprocessing, which is tailored to the specific samples present in each batch. This is particularly important for tasks like padding text sequences to a uniform length, where the required padding varies from one batch to another.

2. *Memory Efficiency*: By preprocessing data at the batch level, `collate_batch` helps in managing memory more efficiently. This is crucial for large datasets, as it avoids the need to load the entire dataset into memory.



3. *Handling Variable-Sized Data*: In cases of variable-sized inputs, such as different lengths of text or varying image sizes, `collate_batch` provides the flexibility to perform custom preprocessing like padding, truncation, or resizing, ensuring that each batch has a consistent format suitable for model training.
"""

#Lets wrap a function around the code for converting text to indices
  # it helps with code readability, modularity and reuse.
def tokenizer(x, vocab):
    """
    Converts a text string into a list of vocabulary indices.

    Parameters:
        x (str): The input text string to be converted.
        vocab (vocab object): The vocabulary object containing the word-to-index mapping.

    Returns:
        list: A list of integers representing the vocabulary indices of the words in the input string.
    """
    # Tokenize the input string, then map each token to its corresponding index in the given vocabulary
    return [vocab[token] for token in str(x).split()]

# check the function
tokenizer('Movie was bad', my_vocab)

def collate_batch_emb(batch):
    """
    Collates a batch of samples into tensors of labels and texts.

    Parameters:
        batch (list): A list of tuples, each containing a label and a text.

    Returns:
        tuple: A tuple containing two tensors, one for labels and one for texts.
    """
    # Unpack the batch into separate lists for labels and texts
    labels, texts = zip(*batch)

    # Convert the list of labels into a tensor of dtype int32
    labels = torch.tensor(labels, dtype=torch.int32)

    # Convert the list of texts into a tensor; each text is transformed into a list of vocabulary indices using tokenizer
    indices = torch.tensor([tokenizer(text, my_vocab) for text in texts], dtype=torch.int32)

    return labels, indices

"""- `collate_batch` accepts a list of tuples, each containing a `label` and a `text`.
- `zip(*batch)` separates the list of tuples into two distinct lists: one for `labels` and another for `texts`. Check out example provided below for more explanation on `zip`.
- The list of `labels` is converted to a PyTorch tensor using `torch.tensor`. The data type (`dtype`) is explicitly set to `torch.int64` to ensure compatibility with PyTorch's requirements.
- The `texts` undergo a transformation via the `tokenizer` function within a list comprehension. This results in a list of lists, where each inner list is a sequence of integer indices representing words. This list is then converted into a PyTorch tensor, also with the dtype set to `torch.int64` (not `torch.int32` as previously mentioned).
- Finally, the function returns a tuple consisting of the `labels` and `texts` tensors, ready for further processing.

**----Digression Understanding zip, zip(*)-----**
"""

x = [1, 2, 3]
y = [11, 12, 13]
z = zip(x, y)
print(x, y, z)

temp = list(z)

temp

temp[0]

x1, y1 = zip(*temp)

print(x1, y1)

"""**----END of Digression-----**"""

# check the function by passing complete dataset
collate_batch_emb(train_dataset)

"""As we can see we got the labels along with indices of words."""

# create DataLoader now
torch.manual_seed(0)
batch_size = 2
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_batch_emb,
                                           )

# iterate over the dataloader
torch.manual_seed(0)
for label, text in train_loader:
    print(label, text)

"""### <Font color = 'indianred'>**5.2. Instantiate Embedding Layer**"""

# Instantiating embedding layer with total number of embeddings and dimension of embedding i.e. size of vector
torch.manual_seed(0)
model = nn.Embedding(num_embeddings=len(my_vocab), embedding_dim=5)

"""- The above code creates an instance of the `nn.Embedding` layer.
- `num_embeddings=len(my_vocab)`: This specifies the total number of embeddings in the layer, which is set to the length of `my_vocab`. Essentially, it determines how many unique words the embedding layer can represent.
- `embedding_dim=5`: This sets the size of each embedding vector to 5. In other words, each word in the vocabulary will be represented as a 5-dimensional vector.
"""

# check the weights associated with the embedding layer
model.weight

"""- The output of `model.weight` is a PyTorch tensor containing the initialized embedding vectors for each word in the vocabulary (`my_vocab`). Since the random seed is set to 0, these initializations are deterministic in this context.

  - The tensor is of size `[num_embeddings x embedding_dim]`, which in this case translates to `[len(my_vocab) x 5]`. The first row is the embedding for the first word in the vocab;  the second row is the embedding for the second word in the vocab; similarly, the last row is the embedding for the last word in the vocab.

  - The values for embeddings (tensors) are randomly initialized and will be updated during the training process. The goal of training is to adjust these embeddings so that they capture meaningful semantic relationships between words.

  - `requires_grad=True` indicates that these embeddings are learnable parameters, and gradients will be computed for them during the backpropagation step in training, enabling their optimization.

### <Font color = 'indianred'>**5.3. Understanding output of Embedding Layer**

The figure illustrates the process of converting indices from input text data into embedding vectors using a pre-trained embedding layer in a neural network.

<img src='https://drive.google.com/uc?export=view&id=1HPGKeaKZCPFy8JolveYF1hQ7BrQOUUhT' width='800'/>

<Font color = 'indianred'>*Inputs*
- `x`: This is a batch of input data, represented as a tensor containing two sequences of word indices: `[1, 2, 3]` and `[1, 2, 4]`. Each number corresponds to a specific word in the vocabulary.

<Font color = 'indianred'>*Embedding Matrix*
- The embedding matrix is a table where each row corresponds to the embedding of a word in the vocabulary. Each word is represented by a 5-dimensional vector. The word at index 1 has its embedding on the row labeled `1`, and so forth for other indices.

<Font color = 'indianred'>*Output*
- The output tensor shows the embedding vectors retrieved for each word in the input sequences. For the first sequence `[1, 2, 3]`, the embedding vectors corresponding to indices 1, 2, and 3 are selected from the embedding matrix and displayed in the same order in the output tensor. The same process occurs for the second sequence `[1, 2, 4]`, where the vectors for indices 1, 2, and 4 are selected.
- The embeddings for each index in the input sequences are stacked to form the output tensor, preserving the sequence order. This results in a 3-dimensional tensor where the first dimension corresponds to the batch size (number of sequences), the second dimension corresponds to the sequence length (number of words in each sequence), and the third dimension is the embedding size (dimensionality of the embedding vectors).
"""

# iterate over the dataloader and check the output of the model (embedding layer)
for y, x in train_loader:
    output = model(x)
    print('\nx\n', x)
    print('\ny\n', y)
    print('\nOutput Shape \n', output.shape)
    print('\nOutput\n', output)
    sentence_embedding = torch.mean(output, dim=1)
    print('-'*75)
    print('sentence_embedding')
    print(sentence_embedding)
    print('='*75)

"""* `for y, x in train_loader` iterates through train_loader, which yields batches of data during each iteration.  
  * y represents the labels in a batch
  * x represents the text data (token indices) for a batch.

* `output = model(x)`uses the neural network `model` to perform forward pass inference on the input data `x`, the token indices for the text data in a batch. The result, `output`, is a tensor containing embeddings for the text data in the batch.
  * `batch_size` was 2, which means there would be 2 lines in a batch. That's why there are 2 set of tensors in output.  
  * We chose to represent each token index with 5 numbers, there are 5 numbers in each embedding for a token index.
  * Since each sentence had 3 words, there are three rows in each tensor set. Just to remind you, the first batch has lines with indices ([5, 2, 6], [1, 2, 4]), which maps to:
    * It was thrilling.
    * Movie was good
  Note: 2nd word in both sentences is `was`, and hence, the embedding for the 2nd word is the same across the 2 sentences in the first batch.

"""

# check the model output for a random indices (sentence)
output = model(torch.tensor([5, 3, 4, 5]))
output

output.shape

"""## <Font color = 'indianred'>**6. Understanding `nn.EmbeddingBag()` Layer**
- `nn.EmbeddingBag` differs from `nn.Embedding` in that it computes the mean or sum of embeddings for a bag (sequence) of inputs, providing a single aggregated embedding vector per sequence, rather than individual vectors for each input token.

- `nn.EmbeddingBag` is particularly useful for sequence classification tasks. By averaging (or summing) the embeddings at the document level, it effectively aggregates information across the entire sequence into a single representation. This makes it ideal for handling batches of sequences of unequal lengths, streamlining the process and enabling efficient batch processing in neural network models, even when the input sequences vary in size.

### <Font color = 'indianred'>**6.1. Preparing and Understanding Inputs for EmbeddingBag layer** </font>

`nn.EmbeddingBag` requires two main inputs:

1. **Input Tensor**: A 1D containing the indices of the tokens of all the tokens in a whole batch.

2. **Offsets Tensor** A 1D tensor specifying the starting index of each bag (sequence) in the input tensor.

The input tensor provides the indices for lookup in the embedding matrix, and the offsets tensor helps in identifying the boundaries of each sequence or bag in the case of a concatenated 1D input.

We will now modify the collate function to meet these input requirements.
"""

def collate_batch_emb_bag(batch):
    """
    Collates a batch of samples into tensors of labels, texts, and offsets.

    Parameters:
        batch (list): A list of tuples, each containing a label and a text.

    Returns:
        tuple: A tuple containing three tensors:
               - Labels tensor
               - Concatenated texts tensor
               - Offsets tensor indicating the start positions of each text in the concatenated tensor
    """
    # Unpack the batch into separate lists for labels and texts
    labels, texts = zip(*batch)

    # Convert the list of labels into a tensor of dtype int32
    labels = torch.tensor(labels, dtype=torch.int32)

    # Convert the list of texts into a list of lists; each inner list contains the vocabulary indices for a text
    list_of_list_of_indices = [tokenizer(text, my_vocab) for text in texts]

    # Compute the offsets for each text in the concatenated tensor
    offsets = [0] + [len(i) for i in list_of_list_of_indices]
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    # Concatenate all text indices into a single tensor
    indices = torch.cat([torch.tensor(i, dtype=torch.int64) for i in list_of_list_of_indices])

    return labels, indices, offsets

"""- `zip(*batch)` is utilized to separate the batch into two distinct lists: one for `labels` and another for `texts`.

- The list of `labels` is promptly converted into a PyTorch tensor with data type set to `torch.int32`.

- For each text string in `texts`, `tokenizer` is invoked to transform it into a list of vocabulary indices. These lists are then stored in another list named `list_of_list_of_indices`.

- The individual lists within `list_of_list_of_indices` are concatenated into a single PyTorch tensor using `torch.cat`. This tensor holds the entire batch of text data in index form.

- To manage the original boundary of each document within the concatenated tensor, an `offsets` tensor is computed. It starts with a zero and is followed by the cumulative sum of the lengths of the individual text index lists.

- The tensors for `labels`, `texts`, and `offsets` are packaged into a tuple and returned as the final output of `collate_batch`.


"""

# create data loader now
torch.manual_seed(0)
batch_size = 2
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           collate_fn=collate_batch_emb_bag,
                                           )

# iterate over the data loader to see the output
torch.manual_seed(0)
for label, text, offsets in train_loader:
    print(label, text, offsets)

"""### <Font color = 'indianred'>**6.2. Instantiate Embedding Bag Layer**"""

# Instantiating EmbeddingBag layer with total number of embeddings and dimension of embedding
# i.e. dimension of vector

torch.manual_seed(0)
model = nn.EmbeddingBag(len(my_vocab), 5)

model.weight

"""### <Font color = 'indianred'>**6.3. Understand output of EmbeddingBag Layer**"""

for label, text, offsets in train_loader:
    output = model(text, offsets)
    print('Output')
    print(output)
    print(output.shape)
    print('='*75)

