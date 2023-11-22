# BERT: Pre-training of Deep Bidirectional Transformers forLanguage Understanding

### Overview:

The paper tackles the problematic area of question answering tasks from passages in Wikipedia, addressing the specific prediction of an answer text span within a given passage and tackling unanswerable questions. BERT, a novel, conceptually simple yet potent model, is presented as the paper's primary contribution. BERT is optimized for deep bidirectional architectures, offering applicability across a wide range of Natural Language Processing tasks, exemplified by achieving state-of-the-art results across eleven such tasks. The paper utilizes various datasets, such as the GLUE benchmark, SQuAD v1.1, and the SWAG dataset. Ablation studies were conducted to explore aspects of BERT and the importance of its deep bidirectionality was a notable discovery. Despite various model alterations, BERT's effectiveness remained evident, particularly for Named Entity Recognition tasks. The paper concludes by affirming BERT's simplicity and efficacy in achieving exceptional results on multiple NLP tasks, emphasizing the crucial role of unsupervised pre-training in today's language understanding systems.

### Detailed Method:

BERT (Bidirectional Encoder Representations from Transformers) is a language model that has two key stages: pre-training and fine-tuning. In the pre-training phase, the model is trained on a range of tasks without labeled data. The fine-tuning phase involves using labeled data from specific tasks, which is where the pre-trained parameters of the BERT model are fine-tuned for targeted performance. 

A key feature of BERT is its model architecture, which is a multi-layer bidirectional Transformer encoder. The number of layers is represented as L, the hidden size as H, and the number of self-attention heads as A. Two model sizes are used: BERT BASE and BERT LARGE. The main distinction between these is that BERT uses bidirectional self-attention while other models use self-attention that only conditions to the context on the left of each token.

BERT's input representation allows for representation of both single sentences or pairs of sentences. The first token in every sequence is a special classification token ([CLS]) that helps understand aggregated sequence representation. Sentence pairs have a special token ([SEP]) to separate them and to identify their separate embeddings (Segment A or B), as shown in Figure 2.

During pre-training, BERT uses two unsupervised tasks. The first involves masking a certain portion of the input tokens and then predicting those masked tokens — this approach is called "Masked LM". BERT also trains for a "Next Sentence Prediction (NSP)" task, which helps the model understand the relationship between sentences. The final hidden vector of the special [CLS] token (C) is used for this prediction.

To enhance performance on various tasks like question answering and natural language inference, the authors 'fine-tuned' BERT. The advantage of BERT's method is that a wide range of tasks can be handled by simply changing the inputs and outputs even in task-specific areas. Fine-tuning is much quicker and cheaper than pre-training; it can be completed in just an hour on a cloud TPU or a few hours on a GPU. 

Overall, BERT's approach combines unlabeled pre-training with task-specific fine-tuning, leading to a powerful and versatile model for NLP tasks.
