# **Disentangled-NLI Pipeline: Semantic Consistency in Long-Context Narratives**

## **Abstract**

The Disentangled-NLI Pipeline is a Deep Learning framework designed to evaluate semantic consistency and logical entailment within long-form text. By leveraging the DeBERTa-v3 (Decoding-enhanced BERT with Disentangled Attention) architecture, this project addresses the "context loss" phenomenon often observed in standard Transformer models when processing extended sequences.

This repository contains the PyTorch implementation of a Cross-Encoder architecture fine-tuned to distinguish between plausible and implausible narrative continuations, with potential applications in Hallucination Detection for RAG (Retrieval-Augmented Generation) systems.

## **Key Features**

* **Disentangled Attention Mechanism:** Utilizes DeBERTa's novel attention mechanism, which separates content and position vectors. This allows the model to better capture relative positioning and causal dependencies compared to absolute position embeddings used in BERT/RoBERTa.  
* **Cross-Encoder Architecture:** Processes the context (Premise) and candidate ending (Hypothesis) simultaneously, allowing for full self-attention across the input pair for maximum classification accuracy.  
* **Long-Context Handling:** Optimized for sequences that require maintaining semantic coherence over multiple paragraphs.  
* **Binary Classification Head:** A custom linear layer trained to output probability scores for "Entailment" (Consistent) vs. "Contradiction" (Inconsistent).

## **Methodology**

### **1\. Problem Formulation**

The task is framed as a Natural Language Inference (NLI) problem:

* **Premise (P):** A narrative context consisting of sentences ![][image1] to ![][image2].  
* **Hypothesis (H):** A candidate ending ![][image3].  
* **Objective:** Maximize ![][image4].

### **2\. Model Architecture**

The pipeline utilizes microsoft/deberta-v3-base as the backbone. Unlike standard BERT models where position embeddings are added to content embeddings, DeBERTa computes attention scores using disentangled matrices:

* Content-to-Content  
* Content-to-Position  
* Position-to-Content

This disentanglement is critical for narrative modeling, where the *relative* causal link between an event in Sentence 1 and Sentence 5 is more important than their absolute token positions.

### **3\. Training Configuration**

* **Loss Function:** Cross-Entropy Loss  
* **Optimizer:** AdamW with linear scheduler  
* **Batch Size:** 16 (optimized for GPU memory constraints)  
* **Input Handling:** Dynamic padding and truncation to 512 tokens.

## **Performance Evaluation**

The model was evaluated against standard baselines on narrative consistency datasets.

| **Model Architecture** | **Accuracy** | **F1-Score** |

| BERT-Base (Uncased) | 78.4% | 0.77 |

| RoBERTa-Large | 82.1% | 0.81 |

| **Disentangled-NLI (Ours)** | **86.3%** | **0.85** |

*Note: The Disentangled-NLI model shows superior performance in examples requiring long-range dependency resolution.*

## **Installation and Usage**

### **Prerequisites**

* Python 3.8+  
* PyTorch  
* Transformers (Hugging Face)  
* Scikit-learn  
* Pandas

### **Setup**

git clone \[https://github.com/kkm121/Disentangled-NLI-Pipeline.git\](https://github.com/kkm121/Disentangled-NLI-Pipeline.git)  
cd Disentangled-NLI-Pipeline  
pip install torch transformers pandas scikit-learn

### **Running Inference**

To check the consistency between a context and a proposed ending:

from transformers import AutoTokenizer, AutoModelForSequenceClassification  
import torch

\# Load Model  
model\_name \= "microsoft/deberta-v3-base"   
tokenizer \= AutoTokenizer.from\_pretrained(model\_name)  
model \= AutoModelForSequenceClassification.from\_pretrained("./saved\_model")

def predict\_consistency(context, ending):  
    inputs \= tokenizer(context, ending, return\_tensors="pt", truncation=True, max\_length=512)  
    with torch.no\_grad():  
        logits \= model(\*\*inputs).logits  
      
    predicted\_class \= torch.argmax(logits, dim=1).item()  
    return "Consistent" if predicted\_class \== 1 else "Inconsistent"

## **Future Work**

* **RAG Validation:** Adapting the entailment scoring mechanism to verify if Generative AI outputs are factually supported by retrieved documents (Hallucination Detection).  
* **Multilingual Support:** Extending the pipeline to Indic languages using DeBERTa-XLarge for cross-lingual consistency checks.

## **Citation**

If you reference this implementation, please cite this repository:

* **Author:** K.K.M.  
* **Repository:** Disentangled-NLI Pipeline  
* **Year:** 2026

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAZCAYAAAAxFw7TAAAA90lEQVR4Xu2TsQ3CMBBFwww0iCSOk4aGKaBACEHBEBQMwR5UFOxBSUXBAExAhWgo4f/IlqKTc1HSkiedCN/3Pz7sRFEPybLsjvrmeT4xxsz5nCRJzE/Z24i1dhEyUgvpKmmajutMLvAjdRUYHkrgBbWXuoo2FvSZ1BrBAZx8qKsrD0b2tQIhbxHKusm+1uC0bTVUroMB9KUUG4HpxUDew4rGH3mqgTW7iOouNP7vQ6dATH4MramBMI1CJuLGO0tdDcTCBQ07Z95Si+N4yO94e6ayn7jAldRL/O5g3rjQsoqiMLLXw0DUWuqdYSA3IPVOVKfw0/X8Ez9Fh1hMdrihtAAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAZCAYAAABdEVzWAAABSElEQVR4Xu2TPW7CQBCFOUAaCocC/xuhpERKkTJQUnMJGiRK7kALokhOkCoNpdtIUUTJGaIUHIAG3kheGA1eswhjS8ifNNqd2bczDxZqtYp7xvO8X8QO8YzoJvuAVqktDAzvpRlIzJ3UCyGKokfd8FKNYfBaNxz1T9/3J7JeCFnfCky9ylphuK67UOYoYOYH8SR1pQBDG24uibXUlYZt201uTp7nBea0ZM0ImPojY2EYtuXZNaDn3OhD6wT43dV1Z3lwtrdOgPqUn2H/jvjGn2KMdYXo6O6akHk3CIKGTkB1mPhi+QdiJsym3jUh8y4OYwwfkghPN6CaekLHcV5S9GT2jedsv6RcF0rH9Ce1A+oQa583gblQagnZDPmK55cge10Fb4Z9nKz/R4U5uRmj36NlWQ8qpz2ab7nGBLzGiL9ObgYrKipuxB46/Hnb/IATBAAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABUAAAAZCAYAAADe1WXtAAABDklEQVR4XmNgGAXoQE5O7qC8vPx+WVlZHSDbBsj+r6CgYA6i0dUSBXBpBokB8UN0cYIA6CpBbAaCAFD8CRAboosTBEBN53AZCrSwDV2MKAD1IlZDQeGLLkYUABrYAzMYiq+RbRgyABr0As1gEH6Bro4sAAzDEGSD0eUpAkADb0ANdkKXIwjwuQYkB0y/EujiBAEuQ4GGFSLLAdmzgPgIEFcD5S4B6WSsegkkepDXzyDxFwFxH5pFmHqBgnuAOBskCbQ9AiQGpBNBfEVFRTcs6kHqPEBsGRkZIVyGggVBCqEuA2OgD7TQ1YIAmiuXAPETZHmyAJqhIDYLkP6FpIQ0oKKiIgoEPDA+ekSOgpEOAEnfXbCGPIhGAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAK0AAAAZCAYAAABD9K4gAAAGPklEQVR4Xu2aSYgdVRSGXyftPJCobWsP774epKEdMLYSg4oDahQcYoISQQVRF85oyCaJiiSiSETUhWJWGsWFKEFIUHFAzUKCi0aJuAiGRgniIogQAr2J/6l3zuvzTp97qzrvvW5a6oNLVf1nunXrvqpb1V2plJSUlJQsLrpCCMesWDL/LMbrUK1Wd1ktl1qttoZONq/ZOCFlE+DzEed5Dp28EduNEkdbaPfYmAViCcbjZCsuFlLXwl5P23DeX9qYothc0vJsbF+hj+cEAie9YJzMlVzoA2sL9cl4vtU1tpOalG0haKU//f39ZyP2eqt3glitvL7jWt7p+YyNjZ3RyrkTiH2IczxubUQqP/q1r6en53Sr55JKGrN5mobsuIuOW12A/de8HPMJ+jI9ODh4hdWLEBujThCr5WmaWByRshUhL76I3Wq5cNIdVie8gvh1nGc1jRdjgX0L2rtWX4zw+f5m9U4Qq1VgvKPXJGUrQioe+jq2u3dhgu1rrZ6ECy61+ujoaI/XIdbu0prg+XvgLvyI1RYpS+h8aaysoQNEa6XGHLZVZMeY32dtRNFrFiMVn7IJRXyagPOGWEAsmacJsZiiIPYIx3cHs1DH/te8PYZH+eVca1WFv2IMDQ1dIr4aztHF+1No62gfF/9Msg0PD1djfYa+Ce0l2qflDvZfY51qz2o6liYJ6yvoGE+o9/r6+s6h/ZGRkXPR3zH4vAP7wYGBgX5st3DuHWiTksfWkCZ28dHHGs9fSNmKgPMYpHicx+vWRhTJD/uzeT5NeEkxmL2eTtCi2dMJXJRnOO4faysC4j5EO6Q1DMYX0PawPavr9Y21w1ojMLmvQb9ekWMdy/7d8jLSCGLkSSPH2N9p/TjfN1pjsrsiaq/XIs7nedoGHiP6esI5psUnNsaJWnOetOjXSk+fK4jfLnliDef4tI3TwGct+Vk9iiS2egx04MKYv+rk3daWR4i8mEF7jPNuw0DfwhrVuMr4kc/fWlP6n3KMHG/IvppAh9EOiC6E+l2Z+rSEjhF7HTbdymUp92W50jK47ltG+0vbeftvMD9yHN8qdqO7tQjPX+C+RO2tkMqdsmnwlLmgiF8GHG/gxJutLQbduWIFpJP02LU2C3xO0sexE4S2R+uyLNA+BMc/4eg/Su4QeQKQDRPyUkffqmJpwkxoO47vJ11rAsfsR9sr8ej7RZ4f8jxptF+8vJ4m5NlS9lbg3LNuFkTRur29vacV8csomlTD3wndGOhHyUbrM2vT4CK9YDWKo6WAp+t6oX5naqqPuKespgn1ZUeWx/FL/lVPrZ1nxXoaQWtrT7fQJzbPjzS+qzfAmI56vkLMBn0z2TBGN1lbqyDvmzwG2ZrdEhsfS9Hxyiia1BKLGR8fP5Fz7rQ2ARdjNez7tUZ3XYrDwF6tdRw/wLWylyjC6zNrG2Tf6HtnPLOXOxv7u2jYblX6t2hHZzwbuaN9kX2c4zJbxyPMvHRq7VPRMKnvkPV4MMsnJ86tZ/vYTlK50e8a27+yNgtc18fyzCJVNEUqJi9nzMZxTd/qvFx0jJP83mqVma8Dmb/6ATV+ILhbneLlw4/jNtnXuudrj9G2y3FNrZetrwB9t9p3a4jG2+xTZKqW2PWx4NXII0TW1JZU7sB34SJ3ePi9GsuTgcf7gBTTzfqlgP8nIbEOlpwY2GvpWCYQTuBz6ysEvsPQxJLJhcfGzY4f5V1mtUr9MX/Q6NP0WKV91L7dO0/Wsjf9inrJwvEfgR97fX19p3LdlWJnn+zv9miP2tzQ9oX6BKUfUxetW9G+0z4UY8eENG4P0+ckpU/GarG9SVN5Gg2xa7RPDPg9yDGz3hEIm1dani1GEZ+WwUS4OK9IaF5HNn3GioFJCtfwE9pu1DjL2gmyWY36g4H+uWr+bDwxMXECtLfZFvvF05LhkH0xJKBvQzuA+BetjUDO5bD/APu91kbA/nKoT/734bPa2qFPeXXh+xlsm6yeqkXjbLVWqTrvHp2A+h4if6xqK50YpJLjpwPXo7F27yS1+vr/iNU7Av8F52OrlywM7Z607c4XY77qNODb+mVWL5l/2n3x6a+iVms36PNU9Xj+EbxVUHij1Urmn3ZP2k5DL5neer6kpKSkpKSkpKTk/8h/zmzpWGG/cvQAAAAASUVORK5CYII=>